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
def merge_patch_prediction(
    det: Dict[str, Tensor],
    output_shape: Tuple[int, int, int],
    mask_thresh: float,
    group_iom_thresh: float,
    iom_chunk: int = 128,
) -> Dict[str, Tensor]:
    """
    Merge patch-level duplicate detections into full-image instance candidates.

    Expected input keys
    -------------------
    det["anchor_centers"] : [M, 3]
    det["anchor_strides"] : [M, 3]
    det["classes"]        : [M]
    det["scores"]         : [M]
    det["onehot_logits"]  : [M, 1, px, py, pz] or [M, px, py, pz]
    det["offsets"]        : [M, 3]

    Returns
    -------
    dict with:
      - anchor_centers : [G, 3]
      - anchor_strides : [G, 3]
      - classes        : [G]
      - scores         : [G]
      - onehot_prob    : [G, 1, X, Y, Z]
    """
    X, Y, Z = output_shape
    device = det["scores"].device

    def _empty_output() -> Dict[str, Tensor]:
        return {
            "anchor_centers": torch.empty((0, 3), device=device, dtype=det["anchor_centers"].dtype),
            "anchor_strides": torch.empty((0, 3), device=device, dtype=det["anchor_strides"].dtype),
            "classes": torch.empty((0,), device=device, dtype=torch.long),
            "scores": torch.empty((0,), device=device, dtype=det["scores"].dtype),
            "onehot_prob": torch.zeros((0, 1, X, Y, Z), device=device, dtype=torch.float32),
        }

    # -------------------- empty input --------------------
    if det["scores"].numel() == 0:
        return _empty_output()

    scores = det["scores"].reshape(-1)
    classes = det["classes"].to(torch.long).reshape(-1)
    offsets = det["offsets"]

    onehot_probs = torch.sigmoid(det["onehot_logits"])
    if onehot_probs.ndim == 4:
        onehot_probs = onehot_probs.unsqueeze(1)  # [M,1,px,py,pz]

    # -------------------- paste patch probabilities to global --------------------
    onehot_probs_gb = torch.stack([
        _place_patch_mask_into_global(oh, _as_int_offset(of), output_shape)
        for oh, of in zip(onehot_probs, offsets)
    ], dim=0)

    if onehot_probs_gb.ndim == 4:
        onehot_probs_gb = onehot_probs_gb.unsqueeze(1)  # [M,1,X,Y,Z]

    onehot_probs_gb = onehot_probs_gb.to(torch.float32)

    # threshold only for geometry / grouping decisions
    onehot_bin_gb = onehot_probs_gb >= float(mask_thresh)  # [M,1,X,Y,Z] bool

    # -------------------- remove empty masks early --------------------
    keep_nonempty = onehot_bin_gb[:, 0].flatten(1).any(dim=1)
    if not keep_nonempty.any():
        return _empty_output()

    scores = scores[keep_nonempty]
    classes = classes[keep_nonempty]
    offsets = offsets[keep_nonempty]
    onehot_probs_gb = onehot_probs_gb[keep_nonempty]
    onehot_bin_gb = onehot_bin_gb[keep_nonempty]
    anchor_centers = det["anchor_centers"][keep_nonempty] + offsets
    anchor_strides = det["anchor_strides"][keep_nonempty]

    M = int(scores.numel())
    if M == 0:
        return _empty_output()

    # geometry from thresholded full-image masks
    bboxes = get_onehot_instance_mask_boxes(onehot_bin_gb)  # [M,6]
    centers = (bboxes[:, :3] + bboxes[:, 3:]) / 2.0
    strides = (bboxes[:, 3:] - bboxes[:, :3]).clamp_min(1.0)

    if M == 1:
        return {
            "anchor_centers": anchor_centers,
            "anchor_strides": anchor_strides,
            "classes": classes,
            "scores": scores,
            "onehot_prob": onehot_probs_gb,
        }

    # -------------------- Step 1: coarse grouping by proximity --------------------
    pairwise_anchor_dist = (anchor_centers[None] - anchor_centers[:, None]).abs()
    pairwise_center_dist = (centers[None] - centers[:, None]).abs()

    max_anchor_dist = (anchor_strides[None] + anchor_strides[:, None]) / 2.0
    max_center_dist = (strides[None] + strides[:, None]) / 2.0

    is_anchors_close = torch.all(pairwise_anchor_dist < max_anchor_dist, dim=-1)
    is_anchors_neighbor = torch.all(pairwise_anchor_dist <= max_anchor_dist, dim=-1)
    is_centers_close = torch.all(pairwise_center_dist < max_center_dist, dim=-1)

    is_close = is_anchors_close | (is_anchors_neighbor & is_centers_close)

    max_ratio = 1.5
    anchor_stride_ratio = (
        torch.maximum(anchor_strides[None], anchor_strides[:, None]) /
        torch.minimum(
            anchor_strides[None].clamp_min(1e-6),
            anchor_strides[:, None].clamp_min(1e-6),
        )
    )
    same_scale = torch.all(anchor_stride_ratio < max_ratio, dim=-1)

    proximity_grouped = is_close & same_scale
    proximity_grouped = proximity_grouped & (classes[None] == classes[:, None])
    proximity_grouped.fill_diagonal_(True)

    coarse_components = _find_connected_components(proximity_grouped)

    # -------------------- Step 2: refine each coarse group by IoM --------------------
    flat = onehot_bin_gb[:, 0].flatten(1)              # [M,V] bool
    area = flat.sum(dim=1).to(torch.float32)           # [M]

    out = {
        "anchor_centers": [],
        "anchor_strides": [],
        "classes": [],
        "scores": [],
        "onehot_prob": [],
    }

    for group in coarse_components:
        if len(group) == 1:
            idx = int(group[0])
            out["anchor_centers"].append(anchor_centers[idx])
            out["anchor_strides"].append(anchor_strides[idx])
            out["classes"].append(classes[idx])
            out["scores"].append(scores[idx])
            out["onehot_prob"].append(onehot_probs_gb[idx])
            continue

        gi = torch.as_tensor(group, device=device, dtype=torch.long)
        k = int(gi.numel())

        adj = torch.eye(k, device=device, dtype=torch.bool)
        ii, jj = torch.triu_indices(k, k, offset=1, device=device)

        for s in range(0, ii.numel(), iom_chunk):
            a = ii[s:s + iom_chunk]
            b = jj[s:s + iom_chunk]

            mi = flat[gi[a]]  # [B,V]
            mj = flat[gi[b]]  # [B,V]

            inter = (mi & mj).sum(dim=1).to(torch.float32)
            denom = torch.minimum(area[gi[a]], area[gi[b]]).clamp_min(1.0)
            iom = inter / denom

            keep_edge = iom >= float(group_iom_thresh)
            if keep_edge.any():
                aa = a[keep_edge]
                bb = b[keep_edge]
                adj[aa, bb] = True
                adj[bb, aa] = True

        refined_local_components = _find_connected_components(adj)

        # -------------------- reduce each refined component to one merged instance --------------------
        for comp_local in refined_local_components:
            li = torch.as_tensor(comp_local, device=device, dtype=torch.long)
            idxs = gi[li]

            best_idx = idxs[torch.argmax(scores[idxs])]

            # max-fusion preserves support across duplicates without blurring
            group_prob = onehot_probs_gb[idxs].max(dim=0).values  # [1,X,Y,Z]

            out["anchor_centers"].append(anchor_centers[best_idx])
            out["anchor_strides"].append(anchor_strides[best_idx])
            out["classes"].append(classes[best_idx])
            out["scores"].append(scores[best_idx])
            out["onehot_prob"].append(group_prob)

    if len(out["scores"]) == 0:
        return _empty_output()

    return {
        "anchor_centers": torch.stack(out["anchor_centers"], dim=0),
        "anchor_strides": torch.stack(out["anchor_strides"], dim=0),
        "classes": torch.stack(out["classes"], dim=0),
        "scores": torch.stack(out["scores"], dim=0),
        "onehot_prob": torch.stack(out["onehot_prob"], dim=0),
    }

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


def merge_semantic_logits(
        patch_logits: List[Tensor],
        patch_offsets: List[Tensor],
        output_shape: tuple[int, int, int],
) -> Tensor:
    """
    Merge semantic patch logits into one full-size logit volume by averaging overlaps.
    Safely handles boundary patches by clipping to output_shape.
    """
    if len(patch_logits) != len(patch_offsets):
        raise ValueError(
            f"patch_logits and patch_offsets must have same length, got "
            f"{len(patch_logits)} and {len(patch_offsets)}."
        )

    if len(output_shape) != 3:
        raise ValueError(f"output_shape must be length 3, got {output_shape}")

    if len(patch_logits) == 0:
        raise ValueError("patch_logits must not be empty.")

    device = patch_logits[0].device
    dtype = patch_logits[0].dtype

    merged = torch.zeros((1, *output_shape), device=device, dtype=dtype)
    counts = torch.zeros((1, *output_shape), device=device, dtype=dtype)

    for logit_patch, offset in zip(patch_logits, patch_offsets):
        if logit_patch.ndim == 3:
            logit_patch = logit_patch.unsqueeze(0)
        elif logit_patch.ndim != 4:
            raise ValueError(
                f"Each patch logit must have shape [H,W,D] or [1,H,W,D], got {tuple(logit_patch.shape)}"
            )

        if logit_patch.shape[0] != 1:
            raise ValueError(
                f"Expected a single semantic channel, got patch shape {tuple(logit_patch.shape)}"
            )

        offset = offset.to(device=device, dtype=torch.long).view(-1)
        if offset.numel() != 3:
            raise ValueError(f"Each patch offset must have 3 values, got shape {tuple(offset.shape)}")

        h0, w0, d0 = offset.tolist()
        _, ph, pw, pd = logit_patch.shape

        h1 = min(h0 + ph, output_shape[0])
        w1 = min(w0 + pw, output_shape[1])
        d1 = min(d0 + pd, output_shape[2])

        if h0 >= h1 or w0 >= w1 or d0 >= d1:
            continue

        use_h = h1 - h0
        use_w = w1 - w0
        use_d = d1 - d0

        patch_crop = logit_patch[:, :use_h, :use_w, :use_d]

        merged[:, h0:h1, w0:w1, d0:d1] += patch_crop
        counts[:, h0:h1, w0:w1, d0:d1] += 1

    counts = torch.clamp(counts, min=1)
    return merged / counts
