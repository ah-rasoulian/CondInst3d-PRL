from typing import List, Sequence
import torch
import torch.nn.functional as F
import numbers
from omegaconf import ListConfig
from torch import Tensor
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.box_utils import box_centers, centers_in_boxes, box_iou
import warnings
from .info import  DeviceType, COMPUTE_DTYPE, INF


def boxes_center_distance(
    boxes1: NdarrayOrTensor, boxes2: NdarrayOrTensor, euclidean: bool = True, spacing: NdarrayOrTensor | None = None
) -> tuple[NdarrayOrTensor, NdarrayOrTensor, NdarrayOrTensor]:
    """
    Distance of center points between two sets of boxes

    Args:
        boxes1: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        boxes2: bounding boxes, Mx4 or Mx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        euclidean: computed the euclidean distance otherwise it uses the l1 distance
        spacing: the physical space over each dimension, (3,) torch tensor or ndarray.

    Returns:
        - The pairwise distances for every element in boxes1 and boxes2,
          with size of (N,M) and same data type as ``boxes1``.
        - Center points of boxes1, with size of (N,spatial_dims) and same data type as ``boxes1``.
        - Center points of boxes2, with size of (M,spatial_dims) and same data type as ``boxes1``.

    Reference:
        https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/core/boxes/ops.py

    """

    if not isinstance(boxes1, type(boxes2)):
        warnings.warn(f"boxes1 is {type(boxes1)}, while boxes2 is {type(boxes2)}. The result will be {type(boxes1)}.")

    # convert numpy to tensor if needed
    boxes1_t, *_ = convert_data_type(boxes1, torch.Tensor)
    boxes2_t, *_ = convert_data_type(boxes2, torch.Tensor)

    center1 = box_centers(boxes1_t.to(COMPUTE_DTYPE))  # (N, spatial_dims)
    center2 = box_centers(boxes2_t.to(COMPUTE_DTYPE))  # (M, spatial_dims)

    diff = center1[:, None] - center2[None]
    if spacing is not None:
        spacing_t, *_ = convert_data_type(spacing, torch.Tensor)
        diff = diff * spacing_t[None, None].to(COMPUTE_DTYPE)
    if euclidean:
        dists = diff.pow(2).sum(-1).sqrt()  # type: ignore
    else:
        # before sum: (N, M, spatial_dims)
        dists = diff.sum(-1)

    # convert tensor back to numpy if needed
    (dists, center1, center2), *_ = convert_to_dst_type(src=(dists, center1, center2), dst=boxes1)
    return dists, center1, center2

def boxes_center_distance(
    boxes1: NdarrayOrTensor, boxes2: NdarrayOrTensor, euclidean: bool = True, spacing: NdarrayOrTensor | None = None
) -> tuple[NdarrayOrTensor, NdarrayOrTensor, NdarrayOrTensor]:
    """
    Distance of center points between two sets of boxes

    Args:
        boxes1: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        boxes2: bounding boxes, Mx4 or Mx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        euclidean: computed the euclidean distance otherwise it uses the l1 distance
        spacing: the physical space over each dimension, (3,) torch tensor or ndarray.

    Returns:
        - The pairwise distances for every element in boxes1 and boxes2,
          with size of (N,M) and same data type as ``boxes1``.
        - Center points of boxes1, with size of (N,spatial_dims) and same data type as ``boxes1``.
        - Center points of boxes2, with size of (M,spatial_dims) and same data type as ``boxes1``.

    Reference:
        https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/core/boxes/ops.py

    """

    if not isinstance(boxes1, type(boxes2)):
        warnings.warn(f"boxes1 is {type(boxes1)}, while boxes2 is {type(boxes2)}. The result will be {type(boxes1)}.")

    # convert numpy to tensor if needed
    boxes1_t, *_ = convert_data_type(boxes1, torch.Tensor)
    boxes2_t, *_ = convert_data_type(boxes2, torch.Tensor)

    center1 = box_centers(boxes1_t.to(COMPUTE_DTYPE))  # (N, spatial_dims)
    center2 = box_centers(boxes2_t.to(COMPUTE_DTYPE))  # (M, spatial_dims)

    diff = center1[:, None] - center2[None]
    if spacing is not None:
        spacing_t, *_ = convert_data_type(spacing, torch.Tensor, device=diff.device)
        diff = diff * spacing_t[None, None].to(COMPUTE_DTYPE)
    if euclidean:
        dists = diff.pow(2).sum(-1).sqrt()  # type: ignore
    else:
        # before sum: (N, M, spatial_dims)
        dists = diff.sum(-1)

    # convert tensor back to numpy if needed
    (dists, center1, center2), *_ = convert_to_dst_type(src=(dists, center1, center2), dst=boxes1)
    return dists, center1, center2

def _as_3tuple_factor(factor):
    """Return (fw, fh, fd) as ints. Accepts int-like or sequence of len 3."""
    # OmegaConf ListConfig -> python list
    if isinstance(factor, ListConfig):
        factor = list(factor)

    # scalar
    if isinstance(factor, numbers.Number):
        f = int(factor)
        return (f, f, f)

    # sequence
    if isinstance(factor, (list, tuple)):
        if len(factor) != 3:
            raise ValueError(f"factor must be an int or a sequence of length 3, got: {factor}")
        fw, fh, fd = factor
        if isinstance(fw, ListConfig): fw = list(fw)
        if isinstance(fh, ListConfig): fh = list(fh)
        if isinstance(fd, ListConfig): fd = list(fd)
        return (int(fw), int(fh), int(fd))

    raise TypeError(f"factor must be int-like or length-3 sequence, got {type(factor)}: {factor}")


def aligned_trilinear(tensor, factor):
    """
    tensor: [B, C, W, H, D]  (matches your current code; if you use [B,C,D,H,W] see note below)
    factor: int or (fw, fh, fd) to scale each spatial dim independently.
    """
    assert tensor.dim() == 5

    fw, fh, fd = _as_3tuple_factor(factor)
    if fw < 1 or fh < 1 or fd < 1:
        raise ValueError(f"factors must be >= 1, got {(fw, fh, fd)}")

    if fw == 1 and fh == 1 and fd == 1:
        return tensor

    # current code assumes spatial dims order is [W, H, D] (i.e., tensor.size()[2:]).
    w, h, d = tensor.size()[2:]

    # replicate-pad +1 at the end of each dim (like your original)
    tensor = F.pad(tensor, pad=(0, 1, 0, 1, 0, 1), mode="replicate")

    ow = fw * w + 1
    oh = fh * h + 1
    od = fd * d + 1

    tensor = F.interpolate(
        tensor,
        size=(ow, oh, od),
        mode="trilinear",
        align_corners=True,
    )

    # pad on the "left" side by floor(f/2) per dim (like your original)
    tensor = F.pad(
        tensor,
        pad=(fd // 2, 0, fh // 2, 0, fw // 2, 0),
        mode="replicate",
    )

    return tensor[:, :, :ow - 1, :oh - 1, :od - 1]

def get_patch_spatial_shapes(inputs: Tensor, start: int, end: int) -> Tensor:
    """
    Helper: extract per-patch spatial shapes from meta if present.
    Expected meta shape patterns are messy; adapt to your actual meta.
    Returns Tensor [num_patches,3] (ph,pw,pd) for each patch.
    """
    # old code used: patches.meta['spatial_shape'][:, i] when iterating
    sp = inputs.meta.get("spatial_shape", None)
    if sp is None:
        # fallback: assume all patches same size as inputs patches
        ph, pw, pd = inputs.shape[-3:]
        return torch.tensor([[ph, pw, pd]] * (end - start), device=inputs.device)
    sp = torch.as_tensor(sp, device=inputs.device)
    # common: [3, total_patches] -> transpose
    if sp.ndim == 2 and sp.shape[0] == 3:
        sp = sp.T
    return sp[start:end]

def merge_patch_logits_per_instance(
    patches: List[Tensor],                 # list of [Ki,1,Ph,Pw,Pd] (often fixed patch size)
    like_shape: Sequence[int],             # (K_total,1,out_h,out_w,out_d)
    patch_offsets: Tensor,                 # [n_patches,3] (x,y,z)
    patch_spatial_shapes: Tensor,          # [n_patches,3] (ph,pw,pd) actual valid region in the big image
    device: torch.device,
) -> Tensor:
    """
    Places each patch's instance logits into a global canvas.

    IMPORTANT:
    - Instances are assumed independent across patches, so this is assignment, not averaging.
    """
    K_total, C, out_h, out_w, out_d = map(int, like_shape)
    dtype = patches[0].dtype if len(patches) else torch.float32
    out = torch.zeros((K_total, C, out_h, out_w, out_d), device=device, dtype=dtype)

    if len(patches) == 0:
        return out

    # ensure tensors are on CPU for indexing extraction, but values assigned on device
    patch_offsets = patch_offsets.to("cpu")
    patch_spatial_shapes = patch_spatial_shapes.to("cpu")

    total = 0
    for pi, patch_logits in enumerate(patches):
        if patch_logits is None or patch_logits.numel() == 0:
            continue

        Ki = int(patch_logits.shape[0])

        x, y, z = [int(v) for v in patch_offsets[pi].tolist()]
        ph, pw, pd = [int(v) for v in patch_spatial_shapes[pi].tolist()]

        # clamp to canvas bounds (robust to any meta inconsistencies)
        H = max(0, min(ph, out_h - x))
        W = max(0, min(pw, out_w - y))
        D = max(0, min(pd, out_d - z))

        if H == 0 or W == 0 or D == 0:
            continue

        # crop patch logits to match the region we're writing into
        # patch_logits might be larger (e.g., 128x128x64) while H/W/D may be smaller (e.g., D=42)
        patch_logits = patch_logits.to(device)
        patch_logits = patch_logits[..., :H, :W, :D]  # [Ki,1,H,W,D]

        end = total + Ki
        if end > K_total:
            # prevent overflow if caller miscomputed K_total
            Ki = K_total - total
            if Ki <= 0:
                break
            patch_logits = patch_logits[:Ki]
            end = total + Ki

        out[total:end, :, x:x + H, y:y + W, z:z + D] = patch_logits
        total = end

        if total >= K_total:
            break

    return out
