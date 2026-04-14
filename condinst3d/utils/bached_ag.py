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
    topk_candidates: int = 100,
) -> Dict[str, Tensor]:
    """
    Memory-efficient merge of patch-level duplicate detections into full-image instance candidates.

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

    def _to_offset_int_tensor(offsets_: Tensor) -> Tensor:
        out = torch.as_tensor(
            [_as_int_offset(of) for of in offsets_],
            dtype=torch.long,
            device=device,
        )
        if out.ndim != 2 or out.shape[1] != 3:
            raise ValueError(f"Expected offsets of shape [M, 3], got {tuple(out.shape)}")
        return out

    def _compute_valid_crop_slices(
        offset_xyz: Tensor,
        patch_shape_xyz: Tuple[int, int, int],
        output_shape_xyz: Tuple[int, int, int],
    ):
        """
        Returns slices describing the valid overlap between a patch placed at offset_xyz
        and the global output volume.

        Coordinates are assumed to be [X, Y, Z].
        """
        px, py, pz = patch_shape_xyz
        ox, oy, oz = [int(v) for v in offset_xyz.tolist()]
        Xg, Yg, Zg = output_shape_xyz

        gx0 = max(0, ox)
        gy0 = max(0, oy)
        gz0 = max(0, oz)

        gx1 = min(Xg, ox + px)
        gy1 = min(Yg, oy + py)
        gz1 = min(Zg, oz + pz)

        if gx1 <= gx0 or gy1 <= gy0 or gz1 <= gz0:
            return None

        lx0 = gx0 - ox
        ly0 = gy0 - oy
        lz0 = gz0 - oz

        lx1 = lx0 + (gx1 - gx0)
        ly1 = ly0 + (gy1 - gy0)
        lz1 = lz0 + (gz1 - gz0)

        return (gx0, gy0, gz0, gx1, gy1, gz1), (lx0, ly0, lz0, lx1, ly1, lz1)

    def _global_bbox_from_local_bbox(
        local_bbox_xyz: Tensor,
        offset_xyz: Tensor,
        patch_shape_xyz: Tuple[int, int, int],
        output_shape_xyz: Tuple[int, int, int],
    ) -> Tensor:
        """
        Shift local bbox to global space and clip to valid image bounds.
        bbox format: [x0, y0, z0, x1, y1, z1]
        """
        crop_info = _compute_valid_crop_slices(offset_xyz, patch_shape_xyz, output_shape_xyz)
        if crop_info is None:
            return torch.zeros((6,), device=device, dtype=local_bbox_xyz.dtype)

        (gx0v, gy0v, gz0v, gx1v, gy1v, gz1v), (lx0v, ly0v, lz0v, lx1v, ly1v, lz1v) = crop_info

        gb = local_bbox_xyz.clone()
        gb[:3] += offset_xyz.to(dtype=gb.dtype)
        gb[3:] += offset_xyz.to(dtype=gb.dtype)

        # clip to the valid pasted region for this patch
        gb[0] = gb[0].clamp(min=gx0v, max=gx1v)
        gb[1] = gb[1].clamp(min=gy0v, max=gy1v)
        gb[2] = gb[2].clamp(min=gz0v, max=gz1v)
        gb[3] = gb[3].clamp(min=gx0v, max=gx1v)
        gb[4] = gb[4].clamp(min=gy0v, max=gy1v)
        gb[5] = gb[5].clamp(min=gz0v, max=gz1v)

        return gb

    def _crop_local_mask_to_valid_global_extent(mask_1xyz: Tensor, offset_xyz: Tensor) -> Tensor:
        """
        Crop a local [1,px,py,pz] or [px,py,pz] mask to only the region that actually lands
        inside the output image.
        """
        if mask_1xyz.ndim == 4:
            mask_xyz = mask_1xyz[0]
        else:
            mask_xyz = mask_1xyz

        px, py, pz = mask_xyz.shape
        crop_info = _compute_valid_crop_slices(offset_xyz, (px, py, pz), output_shape)
        if crop_info is None:
            return mask_xyz.new_zeros((0, 0, 0), dtype=mask_xyz.dtype)

        _, (lx0, ly0, lz0, lx1, ly1, lz1) = crop_info
        return mask_xyz[lx0:lx1, ly0:ly1, lz0:lz1]

    def _pairwise_iom_from_local_masks(
        mask_i_1xyz: Tensor,
        offset_i_xyz: Tensor,
        area_i: Tensor,
        bbox_i_xyz: Tensor,
        mask_j_1xyz: Tensor,
        offset_j_xyz: Tensor,
        area_j: Tensor,
        bbox_j_xyz: Tensor,
    ) -> Tensor:
        """
        Compute IoM between two thresholded masks using only their overlapping global ROI.
        No full-image materialization.
        """
        # intersection of clipped global boxes
        sx = max(int(bbox_i_xyz[0].item()), int(bbox_j_xyz[0].item()))
        sy = max(int(bbox_i_xyz[1].item()), int(bbox_j_xyz[1].item()))
        sz = max(int(bbox_i_xyz[2].item()), int(bbox_j_xyz[2].item()))

        ex = min(int(bbox_i_xyz[3].item()), int(bbox_j_xyz[3].item()))
        ey = min(int(bbox_i_xyz[4].item()), int(bbox_j_xyz[4].item()))
        ez = min(int(bbox_i_xyz[5].item()), int(bbox_j_xyz[5].item()))

        if ex <= sx or ey <= sy or ez <= sz:
            return torch.zeros((), device=device, dtype=torch.float32)

        mi = mask_i_1xyz[0] if mask_i_1xyz.ndim == 4 else mask_i_1xyz
        mj = mask_j_1xyz[0] if mask_j_1xyz.ndim == 4 else mask_j_1xyz

        oi = [int(v) for v in offset_i_xyz.tolist()]
        oj = [int(v) for v in offset_j_xyz.tolist()]

        # global ROI -> local ROI for i
        i_lx0, i_ly0, i_lz0 = sx - oi[0], sy - oi[1], sz - oi[2]
        i_lx1, i_ly1, i_lz1 = ex - oi[0], ey - oi[1], ez - oi[2]

        # global ROI -> local ROI for j
        j_lx0, j_ly0, j_lz0 = sx - oj[0], sy - oj[1], sz - oj[2]
        j_lx1, j_ly1, j_lz1 = ex - oj[0], ey - oj[1], ez - oj[2]

        crop_i = mi[i_lx0:i_lx1, i_ly0:i_ly1, i_lz0:i_lz1]
        crop_j = mj[j_lx0:j_lx1, j_ly0:j_ly1, j_lz0:j_lz1]

        inter = (crop_i & crop_j).sum().to(torch.float32)
        denom = torch.minimum(area_i, area_j).clamp_min(1.0)
        return inter / denom

    # -------------------- empty input --------------------
    if det["scores"].numel() == 0:
        return _empty_output()

    scores = det["scores"].reshape(-1)
    classes = det["classes"].to(torch.long).reshape(-1)
    offsets = det["offsets"]

    onehot_probs = torch.sigmoid(det["onehot_logits"])
    if onehot_probs.ndim == 4:
        onehot_probs = onehot_probs.unsqueeze(1)  # [M,1,px,py,pz]

    # Keep local masks in local patch space
    onehot_probs = onehot_probs.to(torch.float32)
    onehot_bin = onehot_probs >= float(mask_thresh)  # [M,1,px,py,pz] bool

    # -------------------- remove empty masks early (local space) --------------------
    keep_nonempty = onehot_bin[:, 0].flatten(1).any(dim=1)
    if not keep_nonempty.any():
        return _empty_output()

    scores = scores[keep_nonempty]
    classes = classes[keep_nonempty]
    offsets = offsets[keep_nonempty]
    onehot_probs = onehot_probs[keep_nonempty]
    onehot_bin = onehot_bin[keep_nonempty]
    anchor_centers = det["anchor_centers"][keep_nonempty] + offsets
    anchor_strides = det["anchor_strides"][keep_nonempty]

    M = int(scores.numel())
    if M == 0:
        return _empty_output()

    # -------------------- geometry from local masks --------------------
    # local boxes in patch coordinates
    local_bboxes = get_onehot_instance_mask_boxes(onehot_bin)  # [M,6]
    offsets_int = _to_offset_int_tensor(offsets)

    # clipped global boxes and valid clipped areas
    global_bboxes: List[Tensor] = []
    valid_areas: List[Tensor] = []

    px, py, pz = onehot_bin.shape[-3:]
    patch_shape = (px, py, pz)

    for i in range(M):
        gb = _global_bbox_from_local_bbox(
            local_bbox_xyz=local_bboxes[i],
            offset_xyz=offsets_int[i],
            patch_shape_xyz=patch_shape,
            output_shape_xyz=output_shape,
        )
        global_bboxes.append(gb)

        cropped_mask = _crop_local_mask_to_valid_global_extent(onehot_bin[i], offsets_int[i])
        area_i = cropped_mask.sum().to(torch.float32)
        valid_areas.append(area_i)

    global_bboxes = torch.stack(global_bboxes, dim=0)  # [M,6]
    valid_areas = torch.stack(valid_areas, dim=0)      # [M]

    # Remove masks that become empty after valid-image clipping
    keep_valid = valid_areas > 0
    if not keep_valid.any():
        return _empty_output()

    scores = scores[keep_valid]
    classes = classes[keep_valid]
    offsets = offsets[keep_valid]
    offsets_int = offsets_int[keep_valid]
    onehot_probs = onehot_probs[keep_valid]
    onehot_bin = onehot_bin[keep_valid]
    anchor_centers = anchor_centers[keep_valid]
    anchor_strides = anchor_strides[keep_valid]
    local_bboxes = local_bboxes[keep_valid]
    global_bboxes = global_bboxes[keep_valid]
    valid_areas = valid_areas[keep_valid]

    M = int(scores.numel())
    if M == 0:
        return _empty_output()

    centers = (global_bboxes[:, :3] + global_bboxes[:, 3:]) / 2.0
    strides = (global_bboxes[:, 3:] - global_bboxes[:, :3]).clamp_min(1.0)

    if M == 1:
        only_prob = _place_patch_mask_into_global(
            onehot_probs[0], offsets_int[0], output_shape
        ).to(torch.float32)

        if only_prob.ndim == 3:
            only_prob = only_prob.unsqueeze(0)

        return {
            "anchor_centers": anchor_centers,
            "anchor_strides": anchor_strides,
            "classes": classes,
            "scores": scores,
            "onehot_prob": only_prob.unsqueeze(0),  # [1,1,X,Y,Z]
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

            prob_gb = _place_patch_mask_into_global(
                onehot_probs[idx], offsets_int[idx], output_shape
            ).to(torch.float32)

            if prob_gb.ndim == 3:
                prob_gb = prob_gb.unsqueeze(0)  # [1,X,Y,Z]

            out["anchor_centers"].append(anchor_centers[idx])
            out["anchor_strides"].append(anchor_strides[idx])
            out["classes"].append(classes[idx])
            out["scores"].append(scores[idx])
            out["onehot_prob"].append(prob_gb)
            continue

        gi = torch.as_tensor(group, device=device, dtype=torch.long)
        k = int(gi.numel())

        adj = torch.eye(k, device=device, dtype=torch.bool)
        ii, jj = torch.triu_indices(k, k, offset=1, device=device)

        for s in range(0, ii.numel(), iom_chunk):
            a = ii[s:s + iom_chunk]
            b = jj[s:s + iom_chunk]

            for aa, bb in zip(a.tolist(), b.tolist()):
                i = int(gi[aa].item())
                j = int(gi[bb].item())

                iom = _pairwise_iom_from_local_masks(
                    mask_i_1xyz=onehot_bin[i],
                    offset_i_xyz=offsets_int[i],
                    area_i=valid_areas[i],
                    bbox_i_xyz=global_bboxes[i],
                    mask_j_1xyz=onehot_bin[j],
                    offset_j_xyz=offsets_int[j],
                    area_j=valid_areas[j],
                    bbox_j_xyz=global_bboxes[j],
                )

                if iom >= float(group_iom_thresh):
                    adj[aa, bb] = True
                    adj[bb, aa] = True

        refined_local_components = _find_connected_components(adj)

        # -------------------- reduce each refined component to one merged instance --------------------
        for comp_local in refined_local_components:
            li = torch.as_tensor(comp_local, device=device, dtype=torch.long)
            idxs = gi[li]

            best_idx = idxs[torch.argmax(scores[idxs])]

            # Only materialize globals for this final refined component
            comp_probs_gb = []
            for idx in idxs.tolist():
                prob_gb = _place_patch_mask_into_global(
                    onehot_probs[idx], offsets_int[idx], output_shape
                ).to(torch.float32)

                if prob_gb.ndim == 3:
                    prob_gb = prob_gb.unsqueeze(0)  # [1,X,Y,Z]

                comp_probs_gb.append(prob_gb)

            group_prob = torch.stack(comp_probs_gb, dim=0).max(dim=0).values  # [1,X,Y,Z]

            out["anchor_centers"].append(anchor_centers[best_idx])
            out["anchor_strides"].append(anchor_strides[best_idx])
            out["classes"].append(classes[best_idx])
            out["scores"].append(scores[best_idx])
            out["onehot_prob"].append(group_prob)

    if len(out["scores"]) == 0:
        return _empty_output()

    # stack only the lightweight tensors first
    scores = torch.stack(out["scores"], dim=0)
    k = min(int(topk_candidates), int(scores.numel()))
    if k < int(scores.numel()):
        keep = torch.topk(scores, k=k, largest=True, sorted=True).indices.tolist()
    else:
        keep = list(range(int(scores.numel())))

    return {
        "anchor_centers": torch.stack([out["anchor_centers"][i] for i in keep], dim=0),
        "anchor_strides": torch.stack([out["anchor_strides"][i] for i in keep], dim=0),
        "classes": torch.stack([out["classes"][i] for i in keep], dim=0),
        "scores": torch.stack([out["scores"][i] for i in keep], dim=0),
        "onehot_prob": torch.stack([out["onehot_prob"][i] for i in keep], dim=0),
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
