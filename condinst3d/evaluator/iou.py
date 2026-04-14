import torch
from torch import Tensor
from monai.data import MetaTensor
from typing import List, Sequence, Tuple, Optional, Union, Dict, Any, Literal, Callable


def mask_intersection_over_union(preds: torch.Tensor, targets: torch.Tensor, max_chunk_size=24):
    """
    Calculate IoU between prediction and target masks in chunks to avoid CUDA OOM.
    preds: predicted 3D masks [N, D, H, W] where N is number of instances
    targets: ground truth 3D masks [M, D, H, W] where M is number of instances
    :returns [N, M] pairwise IoU matrix
    """
    if not preds.dim() == targets.dim() == 4:
        raise ValueError("preds and targets dims must be 4 but are {} and {}".format(preds.dim(), targets.dim()))
    if not preds.shape[-3:] == targets.shape[-3:]:
        raise ValueError("preds and targets spatial shapes must be similar "
                         "but are {} and {}".format(preds.shape[-3:], targets.shape[-3:]))
    preds, targets = preds.type(torch.bool), targets.type(torch.bool)

    device = preds.device
    N, M = preds.size(0), targets.size(0)
    iou_matrix = torch.zeros((N, M), device=device)
    if isinstance(preds, MetaTensor):
        preds = preds.as_tensor()
    if isinstance(targets, MetaTensor):
        targets = targets.as_tensor()

    # Process in chunks along N or M to fit within memory limits
    for i in range(0, N, max_chunk_size):
        for j in range(0, M, max_chunk_size):
            preds_chunk = preds[i:i + max_chunk_size].unsqueeze(1)
            targets_chunk = targets[j:j + max_chunk_size].unsqueeze(0)

            intersection = (preds_chunk & targets_chunk).float().sum((2, 3, 4))
            union = (preds_chunk | targets_chunk).float().sum((2, 3, 4))
            iou_matrix[i:i + max_chunk_size, j:j + max_chunk_size] = intersection / (union + 1e-9)

    return iou_matrix


def mask_intersection_over_minimum(preds: torch.Tensor, targets: torch.Tensor, max_chunk_size=24):
    """
    Calculate IoM (intersection over minimum) between prediction and target masks in chunks to avoid CUDA OOM.
    preds: predicted 3D masks [N, D, H, W] where N is number of instances
    targets: ground truth 3D masks [M, D, H, W] where M is number of instances
    :returns [N, M] pairwise IoM matrix
    """
    if not preds.dim() == targets.dim() == 4:
        raise ValueError("preds and targets dims must be 4 but are {} and {}".format(preds.dim(), targets.dim()))
    if not preds.shape[-3:] == targets.shape[-3:]:
        raise ValueError("preds and targets spatial shapes must be similar "
                         "but are {} and {}".format(preds.shape[-3:], targets.shape[-3:]))
    preds, targets = preds.bool(), targets.bool()

    device = preds.device
    N, M = preds.size(0), targets.size(0)
    iom_matrix = torch.zeros((N, M), device=device)

    # Process in chunks along N or M to fit within memory limits
    for i in range(0, N, max_chunk_size):
        for j in range(0, M, max_chunk_size):
            preds_chunk = preds[i:i + max_chunk_size].unsqueeze(1)
            targets_chunk = targets[j:j + max_chunk_size].unsqueeze(0)

            intersection = (preds_chunk & targets_chunk).float().sum((2, 3, 4))
            min_area = torch.minimum(
                preds_chunk.float().sum((2, 3, 4)),
                targets_chunk.float().sum((2, 3, 4))
            )
            iom_matrix[i:i + max_chunk_size, j:j + max_chunk_size] = intersection / (min_area + 1e-9)

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
