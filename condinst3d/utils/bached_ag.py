from typing import Literal, Dict, Optional, Callable, List
import torch
from torch import Tensor
from condinst3d.evaluator.iou import (mask_intersection_over_union, mask_intersection_over_minimum,
                                      box_intersection_over_union, box_intersection_over_minimum)
from condinst3d.utils.detection import get_onehot_instance_mask_boxes


def _connected_components(adj: Tensor) -> List[Tensor]:
    """
    adj: [N,N] bool adjacency
    returns list of 1D LongTensor indices for each component
    """
    N = adj.shape[0]
    visited = torch.zeros(N, device=adj.device, dtype=torch.bool)
    comps = []

    for i in range(N):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            nbrs = torch.where(adj[u] & ~visited)[0]
            if nbrs.numel():
                visited[nbrs] = True
                stack.extend(nbrs.tolist())
                comp.extend(nbrs.tolist())
        comps.append(torch.as_tensor(comp, device=adj.device, dtype=torch.long))
    return comps


def batched_lbe(d: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Input keys expected:
      - anchor_centers [K,3]
      - anchor_strides [K,3]
      - classes [K]
      - scores [K]
      - onehot_logits [K,1,H,W,D]  (probabilities)
      - bboxes [K,6]
    Output same keys but grouped+fused.
    """
    K = d["scores"].numel()
    if K == 0:
        return d
    if K == 1:
        return d

    anchor_centers = d["anchor_centers"]
    anchor_strides = d["anchor_strides"]

    # ---- grouping: close in anchor space, similar scale ----
    pairwise_anchor_dist = (anchor_centers[:, None, :] - anchor_centers[None, :, :]).abs()  # [K,K,3]
    max_anchor_dist = (anchor_strides[:, None, :] + anchor_strides[None, :, :]) * 0.5       # [K,K,3]
    is_close = torch.all(pairwise_anchor_dist <= max_anchor_dist, dim=-1)                   # [K,K]

    max_ratio = 1.5
    stride_ratio = torch.max(anchor_strides[:, None, :], anchor_strides[None, :, :]) / torch.clamp_min(
        torch.min(anchor_strides[:, None, :], anchor_strides[None, :, :]), 1e-6
    )
    same_scale = torch.all(stride_ratio < max_ratio, dim=-1)                                # [K,K]

    adj = is_close & same_scale
    comps = _connected_components(adj)

    # ---- fuse groups ----
    out_centers, out_strides, out_classes, out_scores, out_logits, out_boxes = [], [], [], [], [], []

    for idx in comps:
        scores = d["scores"][idx]
        classes = d["classes"][idx]
        logits = d["onehot_logits"][idx]  # [g,1,H,W,D]

        w = torch.softmax(scores, dim=0)  # [g]
        fused = (logits * w.view(-1, 1, 1, 1, 1)).sum(dim=0) / w.sum().clamp_min(1e-6)  # [1,H,W,D]

        out_logits.append(fused)
        out_centers.append(anchor_centers[idx].mean(dim=0))
        out_strides.append(anchor_strides[idx].mean(dim=0))
        out_classes.append(torch.mode(classes)[0])
        out_scores.append(scores.mean())

        bb = get_onehot_instance_mask_boxes((fused >= 0.5).unsqueeze(0))  # [1,6]
        out_boxes.append(bb.squeeze(0))

    out = {
        "anchor_centers": torch.stack(out_centers, dim=0),
        "anchor_strides": torch.stack(out_strides, dim=0),
        "classes": torch.stack(out_classes, dim=0),
        "scores": torch.stack(out_scores, dim=0),
        "onehot_logits": torch.stack(out_logits, dim=0),  # [G,1,H,W,D]
        "bboxes": torch.stack(out_boxes, dim=0),          # [G,6]
    }
    return out


def batched_nms(
    preds: Tensor,
    scores: Tensor,
    threshold: float,
    classes: Optional[Tensor] = None,
    metric: Callable | Literal["iou", "iom"] = "iou",
) -> Tensor:
    assert 0.0 <= threshold < 1.0
    if preds.numel() == 0:
        return torch.empty((0,), device=scores.device, dtype=torch.long)

    # pairwise metric
    if callable(metric):
        pairwise = metric(preds, preds)
    else:
        # boxes only here (your call site uses boxes)
        if metric == "iou":
            pairwise = box_intersection_over_union(preds, preds)
        else:
            pairwise = box_intersection_over_minimum(preds, preds)

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
