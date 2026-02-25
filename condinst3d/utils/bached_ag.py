from typing import Literal, Dict, Optional, Callable, List, Tuple
import torch
from torch import Tensor
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
    return mask.to(torch.bool)


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
        return torch.zeros((X, Y, Z), dtype=torch.bool, device=patch_mask.device)

    px0 = x0 - ox
    py0 = y0 - oy
    pz0 = z0 - oz

    px1 = px0 + (x1 - x0)
    py1 = py0 + (y1 - y0)
    pz1 = pz0 + (z1 - z0)

    out = torch.zeros((X, Y, Z), dtype=torch.bool, device=patch_mask.device)
    out[x0:x1, y0:y1, z0:z1] = patch_mask[px0:px1, py0:py1, pz0:pz1]
    return out


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


@torch.no_grad()
def merge_prediction(
    det: Dict[str, Tensor],
    output_shape: Tuple[int, int, int],
    *,
    group_thresh: float,
    stride_mode: str = "min",  # {"min","mean","max"} per-axis thresholding
) -> Dict[str, Tensor]:
    """
    Group if BOTH:
      (1) close anchor centers (per-axis |d| < threshold derived from strides), AND
      (2) IoM(binary(onehot_prob_i), binary(onehot_prob_j)) > group_thresh,
          where IoM = inter / min(|A|,|B|) computed in GLOBAL coordinates (offset-aware).

    The *group mask* is taken ONLY from the highest-score prediction in the group
    (pasted to global space). No union.

    Output `onehot` is [G,X,Y,Z] bool.
    """

    def _thr(si: Tensor, sj: Tensor) -> Tensor:
        if stride_mode == "min":
            return torch.minimum(si, sj)
        if stride_mode == "mean":
            return 0.5 * (si + sj)
        if stride_mode == "max":
            return torch.maximum(si, sj)
        raise ValueError("stride_mode must be one of {'min','mean','max'}")

    def _normalize_to_Mphwpd(x: Tensor, name: str) -> Tensor:
        # accept [M,1,ph,pw,pd] or [M,ph,pw,pd]
        if x.ndim == 5 and x.shape[1] == 1:
            return x[:, 0]
        if x.ndim == 4:
            return x
        raise ValueError(f"det['{name}'] must be [M,ph,pw,pd] or [M,1,ph,pw,pd]. Got {tuple(x.shape)}")

    def _binarize_mask(x: Tensor) -> Tensor:
        # If already bool, keep. If float/prob, binarize at 0.5 (since caller asked for "binaries").
        if x.dtype == torch.bool:
            return x
        return x > 0.5

    def _iom_shifted(
        mi: Tensor, off_i_xyz: Tensor,
        mj: Tensor, off_j_xyz: Tensor,
    ) -> float:
        """
        IoM between two patch masks placed in global space, computed without allocating a full global volume.
        mi/mj: [ph,pw,pd] bool
        off_*: [3] int offsets (x,y,z) into global
        """
        # patch sizes (in x,y,z order consistent with your codepath)
        ph, pw, pd = mi.shape

        # offsets as Python ints for slicing math
        oi = [int(off_i_xyz[0]), int(off_i_xyz[1]), int(off_i_xyz[2])]
        oj = [int(off_j_xyz[0]), int(off_j_xyz[1]), int(off_j_xyz[2])]

        # overlap box in global coords
        gx0 = max(oi[0], oj[0]); gy0 = max(oi[1], oj[1]); gz0 = max(oi[2], oj[2])
        gx1 = min(oi[0] + ph, oj[0] + ph)
        gy1 = min(oi[1] + pw, oj[1] + pw)
        gz1 = min(oi[2] + pd, oj[2] + pd)

        if gx1 <= gx0 or gy1 <= gy0 or gz1 <= gz0:
            return 0.0

        # local slices inside each patch
        i_x0 = gx0 - oi[0]; i_x1 = gx1 - oi[0]
        i_y0 = gy0 - oi[1]; i_y1 = gy1 - oi[1]
        i_z0 = gz0 - oi[2]; i_z1 = gz1 - oi[2]

        j_x0 = gx0 - oj[0]; j_x1 = gx1 - oj[0]
        j_y0 = gy0 - oj[1]; j_y1 = gy1 - oj[1]
        j_z0 = gz0 - oj[2]; j_z1 = gz1 - oj[2]

        mi_crop = mi[i_x0:i_x1, i_y0:i_y1, i_z0:i_z1]
        mj_crop = mj[j_x0:j_x1, j_y0:j_y1, j_z0:j_z1]

        inter = (mi_crop & mj_crop).sum().item()
        if inter == 0:
            return 0.0

        ai = mi.sum().item()
        aj = mj.sum().item()
        denom = min(ai, aj)
        if denom <= 0:
            return 0.0
        return float(inter) / float(denom)

    # -------------------- empty --------------------
    if det["scores"].numel() == 0:
        X, Y, Z = output_shape
        device = det["anchor_centers"].device
        return {
            "anchor_centers": det["anchor_centers"].reshape(0, 3),
            "anchor_strides": det["anchor_strides"].reshape(0, 3),
            "classes": det["classes"].reshape(0) if det["classes"].ndim == 1 else det["classes"][:0],
            "scores": det["scores"].reshape(0),
            "onehot": torch.zeros((0, X, Y, Z), dtype=torch.bool, device=device),
            "offset": torch.zeros((0, 3), dtype=torch.float32, device=device),
        }

    centers = det["anchor_centers"].to(torch.float32)   # [M,3] global
    strides = det["anchor_strides"].to(torch.float32)   # [M,3]
    scores  = det["scores"].to(torch.float32)           # [M]
    classes = det["classes"]
    offsets = det["offset"].to(torch.float32)           # [M,3] patch origin

    # Use onehot_prob if present for grouping IoM; fallback to onehot.
    # (Caller asked specifically: "iom of onehot_prob binaries".)
    if "onehot_prob" in det and det["onehot_prob"] is not None:
        masks_for_grouping = det["onehot_prob"]
    else:
        masks_for_grouping = det["onehot"]

    masks_for_grouping = _normalize_to_Mphwpd(masks_for_grouping, "onehot_prob" if "onehot_prob" in det else "onehot")
    masks_for_grouping = _binarize_mask(masks_for_grouping)

    # onehot used for final paste (best in group)
    onehot_patch = _normalize_to_Mphwpd(det["onehot"].to(torch.bool), "onehot")

    M = int(scores.numel())

    # -------------------- grouping --------------------
    uf = _UnionFind(M)
    for i in range(M):
        ci, si = centers[i], strides[i]
        off_i = _as_int_offset(offsets[i])

        mi = masks_for_grouping[i]
        for j in range(i + 1, M):
            d = (ci - centers[j]).abs()   # [3]
            t = _thr(si, strides[j])      # [3]
            if not bool((d < t).all().item()):
                continue

            off_j = _as_int_offset(offsets[j])
            mj = masks_for_grouping[j]

            iom = _iom_shifted(mi, off_i, mj, off_j)
            if iom > float(group_thresh):
                uf.union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(M):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    # -------------------- merge: take best mask per group --------------------
    X, Y, Z = output_shape
    device = centers.device

    out_onehot: List[Tensor] = []
    out_scores: List[Tensor] = []
    out_classes: List[Tensor] = []
    out_centers: List[Tensor] = []
    out_strides: List[Tensor] = []

    for idxs in groups.values():
        idx = torch.as_tensor(idxs, device=device, dtype=torch.long)

        g_scores = scores[idx]
        best_local = int(torch.argmax(g_scores).item())
        kbest = idx[best_local]

        off_xyz = _as_int_offset(offsets[kbest])
        best_global = _place_patch_mask_into_global(onehot_patch[kbest], off_xyz, output_shape)
        out_onehot.append(best_global.unsqueeze(0))

        out_scores.append(scores[kbest].unsqueeze(0))
        out_classes.append(classes[kbest].unsqueeze(0) if classes.ndim == 1 else classes[kbest:kbest + 1])
        out_centers.append(centers[kbest].unsqueeze(0))
        out_strides.append(strides[kbest].unsqueeze(0))

    onehot_global = torch.cat(out_onehot, dim=0)  # [G,X,Y,Z]

    return {
        "anchor_centers": torch.cat(out_centers, dim=0),
        "anchor_strides": torch.cat(out_strides, dim=0),
        "classes": torch.cat(out_classes, dim=0),
        "scores": torch.cat(out_scores, dim=0),
        "onehot": onehot_global,
        "offset": torch.zeros((onehot_global.shape[0], 3), dtype=torch.float32, device=device),
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