import torch
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

def aligned_trilinear(tensor, factor):
    assert tensor.dim() == 5
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    w, h, d = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1, 0, 1), mode="replicate")
    ow = factor * w + 1
    oh = factor * h + 1
    od = factor * d + 1
    tensor = F.interpolate(
        tensor, size=(ow, oh, od),
        mode='trilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :ow - 1, :oh - 1, :od - 1]
