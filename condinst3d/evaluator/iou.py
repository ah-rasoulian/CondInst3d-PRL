import torch
from torch import Tensor
from monai.data import MetaTensor
from typing import List, Sequence, Tuple, Optional, Union, Dict, Any, Literal, Callable


def mask_intersection_over_union(
    preds: torch.Tensor,
    targets: torch.Tensor,
    max_chunk_size: int = 24,
):
    """
    Calculate IoU between prediction and target masks in chunks on GPU,
    avoiding large broadcasted [n, m, D, H, W] temporaries.

    preds:   [N, D, H, W]
    targets: [M, D, H, W]
    returns: [N, M] pairwise IoU matrix
    """
    if preds.dim() != 4 or targets.dim() != 4:
        raise ValueError(
            f"preds and targets dims must be 4 but are {preds.dim()} and {targets.dim()}"
        )
    if preds.shape[-3:] != targets.shape[-3:]:
        raise ValueError(
            f"preds and targets spatial shapes must be similar but are "
            f"{preds.shape[-3:]} and {targets.shape[-3:]}"
        )

    if isinstance(preds, MetaTensor):
        preds = preds.as_tensor()
    if isinstance(targets, MetaTensor):
        targets = targets.as_tensor()

    device = preds.device

    preds = preds.detach().bool()
    targets = targets.detach().bool()

    N, M = preds.size(0), targets.size(0)

    preds_flat = preds.flatten(1).float()    # [N, V]
    targets_flat = targets.flatten(1).float()  # [M, V]

    pred_areas = preds_flat.sum(dim=1)       # [N]
    target_areas = targets_flat.sum(dim=1)   # [M]

    iou_matrix = torch.zeros((N, M), device=device, dtype=torch.float32)

    for i in range(0, N, max_chunk_size):
        p = preds_flat[i:i + max_chunk_size]         # [n, V]
        p_area = pred_areas[i:i + max_chunk_size]    # [n]

        for j in range(0, M, max_chunk_size):
            t = targets_flat[j:j + max_chunk_size]         # [m, V]
            t_area = target_areas[j:j + max_chunk_size]    # [m]

            intersection = p @ t.T                         # [n, m]
            union = p_area[:, None] + t_area[None, :] - intersection

            iou_matrix[
                i:i + p.size(0),
                j:j + t.size(0)
            ] = intersection / (union + 1e-9)

    return iou_matrix


def mask_intersection_over_minimum(
    preds: torch.Tensor,
    targets: torch.Tensor,
    max_chunk_size: int = 24,
):
    """
    Calculate IoM (intersection over minimum area) between prediction and target masks
    in chunks on GPU, avoiding large broadcasted [n, m, D, H, W] temporaries.

    preds:   [N, D, H, W]
    targets: [M, D, H, W]
    returns: [N, M] pairwise IoM matrix
    """
    if preds.dim() != 4 or targets.dim() != 4:
        raise ValueError(
            f"preds and targets dims must be 4 but are {preds.dim()} and {targets.dim()}"
        )
    if preds.shape[-3:] != targets.shape[-3:]:
        raise ValueError(
            f"preds and targets spatial shapes must be similar "
            f"but are {preds.shape[-3:]} and {targets.shape[-3:]}"
        )

    if isinstance(preds, MetaTensor):
        preds = preds.as_tensor()
    if isinstance(targets, MetaTensor):
        targets = targets.as_tensor()

    device = preds.device

    preds = preds.detach().bool()
    targets = targets.detach().bool()

    N, M = preds.size(0), targets.size(0)

    preds_flat = preds.flatten(1).float()      # [N, V]
    targets_flat = targets.flatten(1).float()  # [M, V]

    pred_areas = preds_flat.sum(dim=1)         # [N]
    target_areas = targets_flat.sum(dim=1)     # [M]

    iom_matrix = torch.zeros((N, M), device=device, dtype=torch.float32)

    for i in range(0, N, max_chunk_size):
        p = preds_flat[i:i + max_chunk_size]          # [n, V]
        p_area = pred_areas[i:i + max_chunk_size]     # [n]

        for j in range(0, M, max_chunk_size):
            t = targets_flat[j:j + max_chunk_size]          # [m, V]
            t_area = target_areas[j:j + max_chunk_size]     # [m]

            intersection = p @ t.T                          # [n, m]
            min_area = torch.minimum(p_area[:, None], t_area[None, :])

            iom_matrix[
                i:i + p.size(0),
                j:j + t.size(0)
            ] = intersection / (min_area + 1e-9)

    return iom_matrix


def box_intersection_over_union(preds, targets):
    """
    Compute the IoU (Intersection over Union) for 3D bounding boxes.
    box1, box2: Tensor of shape [N, 6] or [M, 6] where N or M is the number of boxes.
    Each box is represented by (x1, y1, z1, x2, y2, z2) coordinates.
    """
    # Compute intersection
    inter_min = torch.max(preds[:, :3].unsqueeze(1), targets[:, :3].unsqueeze(0))  # [N, M, 3]
    inter_max = torch.min(preds[:, 3:].unsqueeze(1), targets[:, 3:].unsqueeze(0))  # [N, M, 3]
    inter_dim = torch.clamp(inter_max - inter_min, min=0)  # Intersection dimensions
    inter_vol = inter_dim.prod(dim=-1)  # Intersection volume

    # Compute volume of boxes
    vol_pred_boxes = (preds[:, 3:] - preds[:, :3]).prod(dim=-1)  # Volume of pred boxes
    vol_target_boxes = (targets[:, 3:] - targets[:, :3]).prod(dim=-1)  # Volume of target boxes

    # Compute union
    union_vol = vol_pred_boxes.unsqueeze(1) + vol_target_boxes.unsqueeze(0) - inter_vol

    # IoU calculation
    iou = inter_vol / (union_vol + 1e-9)
    return iou

def box_intersection_over_minimum(preds, targets):
    """
    Compute the IoMin (Intersection over Minimum) for 3D bounding boxes.
    Each box is represented by (x1, y1, z1, x2, y2, z2) coordinates.
    preds: Tensor of shape [N, 6]
    targets: Tensor of shape [M, 6]
    Returns:
        Tensor of shape [N, M] with pairwise IoMin values.
    """
    # Compute intersection
    inter_min = torch.max(preds[:, :3].unsqueeze(1), targets[:, :3].unsqueeze(0))  # [N, M, 3]
    inter_max = torch.min(preds[:, 3:].unsqueeze(1), targets[:, 3:].unsqueeze(0))  # [N, M, 3]
    inter_dim = torch.clamp(inter_max - inter_min, min=0)  # [N, M, 3]
    inter_vol = inter_dim.prod(dim=-1)  # [N, M]

    # Compute individual volumes
    vol_pred_boxes = (preds[:, 3:] - preds[:, :3]).prod(dim=-1)  # [N]
    vol_target_boxes = (targets[:, 3:] - targets[:, :3]).prod(dim=-1)  # [M]

    # Compute minimum volume between each pair
    min_vol = torch.min(vol_pred_boxes.unsqueeze(1), vol_target_boxes.unsqueeze(0))  # [N, M]

    # IoMin calculation
    iomin = inter_vol / (min_vol + 1e-9)  # [N, M]
    return iomin
