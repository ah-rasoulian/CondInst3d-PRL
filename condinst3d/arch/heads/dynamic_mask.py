from condinst3d.utils.detection import InstanceList
from condinst3d.utils.anchors import compute_locations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple
from torch import Tensor


class DynamicMaskHead(nn.Module):
    """
        A Dynamic Segmentation head for use in DynUNetDetection.
    """

    def __init__(
            self,
            in_channels,
            channels,
            num_layers: int,
            kernel_size: Union[int, List[int], List[List[int]]],
            size_of_interest: int,
            stride: Tuple[int, int, int],
            op_device: str = 'auto',
            max_batch_size: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.size_of_interest = size_of_interest
        self.stride = stride
        self.op_device = op_device
        self.max_batch_size = max_batch_size
        self.locations = {}

        # init dynamic conv params
        if isinstance(kernel_size, int):
            kernel_size = [[kernel_size, kernel_size, kernel_size] for _ in range(num_layers)]
        else:  # isinstance(kernel_size, list):
            if len(kernel_size) != num_layers:
                raise ValueError(f"The number of kernel-sizes passed {len(kernel_size)} is not equal to "
                                 f"the number of convolutions {num_layers}.")
            if isinstance(kernel_size[0], list):
                for i, layer_ksize in enumerate(kernel_size):
                    if len(layer_ksize) != 3:
                        raise ValueError(f"layer {i} ksize length is not 3. {layer_ksize}")
            else:
                kernel_size = [[layer_ksize] * 3 for layer_ksize in kernel_size]
        self.kernel_size = kernel_size

        weights_num_params = []
        weights_shapes = []
        biases_num_params = []
        biases_shapes = []
        for i, k in zip(range(num_layers), self.kernel_size):
            in_ch = in_channels + 3 if i == 0 else channels
            out_ch = channels if i < num_layers - 1 else 1
            weight = torch.empty(out_ch, in_ch, k[0], k[1], k[2])
            bias = torch.empty(out_ch)

            weights_num_params.append(weight.numel())
            weights_shapes.append(weight.shape)
            biases_num_params.append(bias.numel())
            biases_shapes.append(bias.shape)
        self.weights_num_params = weights_num_params
        self.weights_shapes = weights_shapes
        self.biases_num_params = biases_num_params
        self.biases_shapes = biases_shapes
        self.num_params = sum(weights_num_params) + sum(biases_num_params)

    def parse_dynamic_params(self, params: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """parse the dynamic params for dynamic conv."""
        n_insts = len(params)
        params_splits = list(torch.split_with_sizes(
            params, self.weights_num_params + self.biases_num_params, dim=-1
        ))
        weight_splits = params_splits[:self.num_layers]
        bias_splits = params_splits[self.num_layers:]

        for l in range(self.num_layers):
            ko, ki, kx, ky, kz = self.weights_shapes[l]
            bo, = self.biases_shapes[l]

            weight_splits[l] = weight_splits[l].reshape(ko * n_insts, ki, kx, ky, kz)
            bias_splits[l] = bias_splits[l].reshape(bo * n_insts)
        return weight_splits, bias_splits

    def mask_heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        original_device = features.device
        device = features.device if self.op_device == 'auto' else self.op_device
        features = features.to(device)
        weights = [w.to(device) for w in weights]
        biases = [b.to(device) for b in biases]
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding="same",
                groups=num_insts
            )
            if i < self.num_layers - 1:
                x = F.leaky_relu(x)
        return x.to(original_device)

    def forward(
        self,
        f_mask: Tensor,                 # [B, C, W, H, D]
        controller_logits: Tensor,       # [B, A, P]
        instance_list: "InstanceList",
    ) -> Tensor:
        """
        Returns:
            mask_logits: [M, 1, W, H, D] where M = number of sampled instances
        """
        B, _, W, H, D = f_mask.shape
        n_inst = len(instance_list)

        if n_inst == 0:
            return f_mask.new_zeros((0, 1, W, H, D))

        # cached grid locations for this feature size
        key = (W, H, D)
        locations = self.locations.get(key)
        if locations is None or locations.device != f_mask.device:
            locations = compute_locations(W, H, D, stride=self.stride, device=f_mask.device)  # [W*H*D, 3]
            self.locations[key] = locations

        # flattened per-instance data
        im_inds = instance_list.get_image_indices()            # [M]
        inst_points = instance_list.get_points()               # [M, 3]
        inst_strides = instance_list.get_level_strides()             # [M, 3]
        mask_head_params = instance_list.gather_mask_head_params(controller_logits)  # [M, P]

        # size-of-interest scaling (broadcast safe)
        soi = inst_strides * float(self.size_of_interest)      # [M, 3]

        # output prealloc
        out = []

        # batch processing to limit grouped conv size
        max_bs = min(self.max_batch_size, n_inst)

        # reshape locations once for reuse
        loc = locations.view(1, -1, 3)  # [1, W*H*D, 3]

        for start in range(0, n_inst, max_bs):
            end = min(start + max_bs, n_inst)
            bs = end - start

            pts = inst_points[start:end].view(bs, 1, 3)     # [bs, 1, 3]
            soi_bs = soi[start:end].view(bs, 3, 1)          # [bs, 3, 1]

            # relative coords: (pt - grid) / soi
            rel = (pts - loc).permute(0, 2, 1).float()      # [bs, 3, W*H*D]
            rel = rel / soi_bs
            rel = rel.view(bs, 3, W, H, D).to(dtype=f_mask.dtype)

            # gather per-instance feature maps from their image
            feat = f_mask[im_inds[start:end]]               # [bs, C, W, H, D]

            # concat coords + features and pack into grouped-conv format
            mask_in = torch.cat([feat, rel], dim=1)         # [bs, C+3, W, H, D]
            mask_in = mask_in.view(1, -1, W, H, D)          # [1, bs*(C+3), W, H, D]

            weights, biases = self.parse_dynamic_params(mask_head_params[start:end])
            mask_logits = self.mask_heads_forward(mask_in, weights, biases, bs)  # [1, bs, W, H, D] (grouped)

            out.append(mask_logits.view(bs, 1, W, H, D))

        return torch.cat(out, dim=0)