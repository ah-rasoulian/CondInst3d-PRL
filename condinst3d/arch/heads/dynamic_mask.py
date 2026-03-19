from condinst3d.utils.detection import InstanceList
from condinst3d.utils.anchors import compute_locations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple
from torch import Tensor
from collections import defaultdict
from torch.utils.checkpoint import checkpoint


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
        op_device: str = "auto",
        max_batch_size: int = 64,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.num_layers = num_layers
        self.size_of_interest = size_of_interest
        self.stride = stride
        self.op_device = op_device
        self.max_batch_size = max_batch_size
        self.use_checkpoint = use_checkpoint

        # cache of locations by (W,H,D,device)
        self.locations = {}

        if isinstance(kernel_size, int):
            kernel_size = [[kernel_size, kernel_size, kernel_size] for _ in range(num_layers)]
        else:
            if len(kernel_size) != num_layers:
                raise ValueError(
                    f"The number of kernel-sizes passed {len(kernel_size)} is not equal to "
                    f"the number of convolutions {num_layers}."
                )
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
            weights_shapes.append(tuple(weight.shape))
            biases_num_params.append(bias.numel())
            biases_shapes.append(tuple(bias.shape))

        self.weights_num_params = weights_num_params
        self.weights_shapes = weights_shapes
        self.biases_num_params = biases_num_params
        self.biases_shapes = biases_shapes
        self.num_params = sum(weights_num_params) + sum(biases_num_params)

    def _get_locations(self, W: int, H: int, D: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (W, H, D, str(device))
        locations = self.locations.get(key)
        if locations is None:
            locations = compute_locations(W, H, D, stride=self.stride, device=device)  # [W*H*D, 3]
            self.locations[key] = locations
        return locations.to(dtype=dtype)

    def parse_dynamic_params(self, params: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        params: [N, P]
        Returns:
            weights: list of tensors shaped for grouped conv
            biases:  list of tensors shaped for grouped conv
        """
        n_insts = params.shape[0]
        splits = list(torch.split_with_sizes(
            params,
            self.weights_num_params + self.biases_num_params,
            dim=-1,
        ))
        weight_splits = splits[:self.num_layers]
        bias_splits = splits[self.num_layers:]

        for l in range(self.num_layers):
            ko, ki, kx, ky, kz = self.weights_shapes[l]
            (bo,) = self.biases_shapes[l]

            weight_splits[l] = weight_splits[l].reshape(n_insts * ko, ki, kx, ky, kz).contiguous()
            bias_splits[l] = bias_splits[l].reshape(n_insts * bo).contiguous()

        return weight_splits, bias_splits

    def _mask_heads_forward_impl(self, x: Tensor, weights: List[Tensor], biases: List[Tensor], num_insts: int) -> Tensor:
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv3d(
                x,
                w,
                bias=b,
                stride=1,
                padding="same",
                groups=num_insts,
            )
            if i < self.num_layers - 1:
                x = F.leaky_relu(x, negative_slope=0.01, inplace=True)
        return x

    def mask_heads_forward(self, x: Tensor, weights: List[Tensor], biases: List[Tensor], num_insts: int) -> Tensor:
        # assumes everything is already on the right device
        if self.use_checkpoint and self.training:
            def custom_forward(inp):
                return self._mask_heads_forward_impl(inp, weights, biases, num_insts)
            return checkpoint(custom_forward, x, use_reentrant=False)
        return self._mask_heads_forward_impl(x, weights, biases, num_insts)

    def forward(
        self,
        f_mask: Tensor,              # [B, C, W, H, D]
        controller_logits: Tensor,   # [B, A, P]
        instance_list: "InstanceList",
    ) -> Tensor:
        """
        Returns:
            mask_logits: [M, 1, W, H, D] where M = number of sampled instances
        """
        B, C, W, H, D = f_mask.shape
        n_inst = len(instance_list)

        if n_inst == 0:
            return f_mask.new_zeros((0, 1, W, H, D))

        device = f_mask.device if self.op_device == "auto" else torch.device(self.op_device)
        if f_mask.device != device:
            f_mask = f_mask.to(device, non_blocking=True)

        dtype = f_mask.dtype
        locations = self._get_locations(W, H, D, device=device, dtype=dtype)   # [W*H*D, 3]
        loc = locations.view(1, -1, 3)  # [1, WHD, 3]

        im_inds = instance_list.get_image_indices().to(device=device)
        inst_points = instance_list.get_points().to(device=device, dtype=dtype)
        inst_strides = instance_list.get_level_strides().to(device=device, dtype=dtype)
        mask_head_params = instance_list.gather_mask_head_params(controller_logits).to(device=device, dtype=dtype)

        soi = inst_strides * float(self.size_of_interest)  # [M, 3]

        # preallocate final output once
        out = torch.empty((n_inst, 1, W, H, D), device=device, dtype=dtype)

        # group instance indices by image id to avoid copying the same feature map many times
        per_image_indices = defaultdict(list)
        for global_idx, img_idx in enumerate(im_inds.tolist()):
            per_image_indices[img_idx].append(global_idx)

        for img_idx, inst_ids_list in per_image_indices.items():
            inst_ids_all = torch.as_tensor(inst_ids_list, device=device, dtype=torch.long)
            feat_img = f_mask[img_idx:img_idx + 1]  # [1, C, W, H, D]

            for start in range(0, len(inst_ids_list), self.max_batch_size):
                end = min(start + self.max_batch_size, len(inst_ids_list))
                inst_ids = inst_ids_all[start:end]
                bs = inst_ids.numel()

                pts = inst_points[inst_ids].view(bs, 1, 3)      # [bs,1,3]
                soi_bs = soi[inst_ids].view(bs, 3, 1)           # [bs,3,1]

                # [bs, WHD, 3] -> [bs, 3, WHD]
                rel = (pts - loc).permute(0, 2, 1)
                rel = rel / soi_bs.clamp_min(1e-6)
                rel = rel.reshape(bs, 3, W, H, D)

                # expand image feature instead of fancy indexing from full batch repeatedly
                feat = feat_img.expand(bs, -1, -1, -1, -1)      # [bs, C, W, H, D]

                # concat once for this chunk
                mask_in = torch.cat((feat, rel), dim=1).reshape(1, bs * (C + 3), W, H, D)

                weights, biases = self.parse_dynamic_params(mask_head_params[inst_ids])
                mask_logits = self.mask_heads_forward(mask_in, weights, biases, bs)  # [1, bs, W, H, D]

                out[inst_ids] = mask_logits.reshape(bs, 1, W, H, D)

                # aggressively drop refs
                del pts, soi_bs, rel, feat, mask_in, weights, biases, mask_logits

        return out