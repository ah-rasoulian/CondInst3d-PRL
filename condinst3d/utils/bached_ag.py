from typing import Literal, Dict, Optional, Callable, List, Tuple
import torch
from torch import Tensor
from condinst3d.utils.detection import get_onehot_instance_mask_boxes
from condinst3d.evaluator.iou import (box_intersection_over_union, box_intersection_over_minimum,
                                      mask_intersection_over_union, mask_intersection_over_minimum)


def _as_int_offset(offset: Tensor) -> Tuple[int, int, int]:
    o = offset.detach().to(torch.float32).round().to(torch.int64)
    return int(o[0].item()), int(o[1].item()), int(o[2].item())


def _ensure_phw_pd(mask: Tensor) -> Tensor:
    # Accept [ph,pw,pd] or [1,ph,pw,pd] -> [ph,pw,pd]
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 3:
        raise ValueError(f"mask must be [ph,pw,pd] or [1,ph,pw,pd]. Got shape={tuple(mask.shape)}")
    return mask


def _place_patch_mask_into_global(
    patch_mask: Tensor,  # [ph,pw,pd] or [1,ph,pw,pd]
    offset_xyz: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
) -> Tensor:
    patch_mask = _ensure_phw_pd(patch_mask)

    ph, pw, pd = patch_mask.shape
    X, Y, Z = output_shape
    ox, oy, oz = offset_xyz

    x0 = max(0, ox)
    y0 = max(0, oy)
    z0 = max(0, oz)

    x1 = min(X, ox + ph)
    y1 = min(Y, oy + pw)
    z1 = min(Z, oz + pd)

    if (x1 <= x0) or (y1 <= y0) or (z1 <= z0):
        return patch_mask.new_zeros(size=(X, Y, Z))

    px0 = x0 - ox
    py0 = y0 - oy
    pz0 = z0 - oz

    px1 = px0 + (x1 - x0)
    py1 = py0 + (y1 - y0)
    pz1 = pz0 + (z1 - z0)

    out = patch_mask.new_zeros(size=(X, Y, Z))
    out[x0:x1, y0:y1, z0:z1] = patch_mask[px0:px1, py0:py1, pz0:pz1]
    return out


def _find_connected_components(grouped: torch.Tensor):
    """
    grouped: [N, N] bool tensor (adjacency matrix)

    Returns:
        List[List[int]] – connected components (indices)
    """
    N = grouped.shape[0]
    visited = torch.zeros(N, dtype=torch.bool, device=grouped.device)
    components = []

    def dfs(node, component):
        visited[node] = True
        component.append(node)

        for neighbor in range(N):
            if grouped[node, neighbor] and not visited[neighbor]:
                dfs(neighbor, component)

    for i in range(N):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)

    return components


@torch.no_grad()
def merge_prediction(
    det: Dict[str, Tensor],
    output_shape: Tuple[int, int, int],
    mask_thresh: float,
) -> Dict[str, Tensor]:
    """
    Group if:
      (1) close anchor centers (per-axis |d| < threshold derived from strides), AND

    Output `onehot_prob` is [G,X,Y,Z] bool.
    """

    # -------------------- empty --------------------
    if det["scores"].numel() == 0:
        X, Y, Z = output_shape
        device = det["anchor_centers"].device
        return {
            "anchor_centers": det["anchor_centers"].reshape(0, 3),
            "anchor_strides": det["anchor_strides"].reshape(0, 3),
            "classes": det["classes"].reshape(0),
            "scores": det["scores"].reshape(0),
            "onehot_prob": torch.zeros((0, X, Y, Z), dtype=torch.bool, device=device),
        }
    onehot_probs = torch.sigmoid(det["onehot_logits"])
    offsets = det["offsets"]

    onehot_probs_gb = torch.stack([
        _place_patch_mask_into_global(oh, _as_int_offset(of), output_shape) for oh, of in zip(onehot_probs, offsets)
    ])
    onehot_probs_gb = onehot_probs_gb.unsqueeze(1)

    bboxes = get_onehot_instance_mask_boxes(onehot_probs_gb >= mask_thresh)
    centers = (bboxes[:, :3] + bboxes[:, 3:]) / 2
    strides = (bboxes[:, 3:] - bboxes[:, :3])
    anchor_centers = det["anchor_centers"] + offsets
    anchor_strides = det["anchor_strides"]
    scores  = det["scores"]
    classes = det["classes"]

    # Step 1: Group nearby predictions (within stride distance) and Prevent large lesions from absorbing small lesions
    pairwise_anchor_dist = (anchor_centers[None] - anchor_centers[:, None]).abs()
    pairwise_center_dist = (centers[None] - centers[:, None]).abs()
    max_anchor_dist = (anchor_strides[None] + anchor_strides[:, None]) / 2
    max_center_dist = (strides[None] + strides[:, None]) / 2
    is_anchors_close = torch.all(pairwise_anchor_dist < max_anchor_dist, dim=-1)
    is_anchors_neighbor = torch.all(pairwise_anchor_dist <= max_anchor_dist, dim=-1)

    is_centers_close = torch.all(pairwise_center_dist < max_center_dist, dim=-1)
    is_close = is_anchors_close | (is_anchors_neighbor & is_centers_close)

    max_ratio = 1.5
    anchor_stride_ratio = (torch.max(anchor_strides[None], anchor_strides[:, None]) /
                           torch.min(anchor_strides[None], anchor_strides[:, None]))
    same_scale = torch.all(anchor_stride_ratio < max_ratio, dim=-1)


    # Step 2: Combine grouping conditions
    proximity_grouped = is_close & same_scale
    components = _find_connected_components(proximity_grouped)

    out = {
        "anchor_centers": [],
        "anchor_strides": [],
        "classes": [],
        "scores": [],
        "onehot_prob": [],
    }
    for group_indices in components:
        group_anchor_centers = anchor_centers[group_indices]
        group_anchor_strides = anchor_strides[group_indices]
        group_classes = classes[group_indices]
        group_scores = scores[group_indices]
        group_onehot_probs = onehot_probs_gb[group_indices]

        weights = torch.softmax(group_scores, dim=0)
        group_mask = (group_onehot_probs * weights.view(-1, 1, 1, 1, 1)).sum(dim=0) / weights.sum()

        out['anchor_centers'] += [group_anchor_centers.mean(dim=0)]
        out['anchor_strides'] += [group_anchor_strides.mean(dim=0)]
        out['classes'] += [torch.mode(group_classes)[0]]
        out['scores'] += [group_scores.mean()]
        out['onehot_prob'] += [group_mask]

    for key in out:
        out[key] = torch.stack(out[key])

    return out


def batched_nms(
    preds: Tensor,
    scores: Tensor,
    threshold: float,
    classes: Optional[Tensor] = None,
    metric: Callable | Literal["iou", "iom"] = "iou",
    device: str | None = None,
) -> Tensor:
    assert 0.0 <= threshold < 1.0
    if preds.numel() == 0:
        return torch.empty((0,), device=scores.device, dtype=torch.long)

    if device is not None:
        preds = preds.to(device)
        scores = scores.to(device)
        if classes is not None:
            classes = classes.to(device)

    # pairwise metric
    if isinstance(metric, Callable):
        pairwise = metric(preds, preds)
    else:
        if preds.dim() == 2:
            if metric == "iou":
                pairwise = box_intersection_over_union(preds, preds)
            else:
                pairwise = box_intersection_over_minimum(preds, preds)
        else:
            preds = preds.squeeze(1) if preds.dim() == 5 else preds  # N, C, W, H, D
            if metric == "iou":
                pairwise = mask_intersection_over_union(preds, preds)
            else:
                pairwise = mask_intersection_over_minimum(preds, preds)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]
        ious = pairwise[i, rest]

        if classes is not None:
            same = classes[rest] == classes[i]
            suppress = (ious > threshold) & same
        else:
            suppress = (ious > threshold)

        order = rest[~suppress]

    return torch.stack(keep).to(dtype=torch.long)
