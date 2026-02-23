from typing import Literal, Dict, Optional, Callable, List, Tuple
import torch
from torch import Tensor
from condinst3d.evaluator.iou import (box_intersection_over_union, box_intersection_over_minimum,
                                      mask_intersection_over_union, mask_intersection_over_minimum)
import math


def _encode_xyz_to_key(xyz: Tensor, shape_xyz: Tuple[int, int, int]) -> Tensor:
    """
    xyz: [P,3] int64 (x,y,z) in the same axis order as output_shape.
    returns: [P] int64 keys
    """
    X, Y, Z = shape_xyz
    return xyz[:, 0] + X * (xyz[:, 1] + Y * xyz[:, 2])


def _decode_key_to_xyz(keys: Tensor, shape_xyz: Tuple[int, int, int]) -> Tensor:
    """
    keys: [P] int64
    returns xyz: [P,3] int64 (x,y,z)
    """
    X, Y, Z = shape_xyz
    z = keys // (X * Y)
    rem = keys - z * (X * Y)
    y = rem // X
    x = rem - y * X
    return torch.stack([x, y, z], dim=1)


def _intersect_count_sorted(a: Tensor, b: Tensor) -> int:
    """
    Count intersection size of two sorted 1D int64 tensors (two-pointer).
    """
    i = j = 0
    na = a.numel()
    nb = b.numel()
    cnt = 0
    while i < na and j < nb:
        av = int(a[i].item())
        bv = int(b[j].item())
        if av == bv:
            cnt += 1
            i += 1
            j += 1
        elif av < bv:
            i += 1
        else:
            j += 1
    return cnt


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


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

def merge_prediction(
        det: Dict[str, Tensor],
        output_shape: Tuple[int, int, int],
        group_thresh: float,
    ) -> Dict[str, Tensor]:
    """
            Per-image merge logic:
              - Group detections if:
                  (1) ||c_i - c_j|| < 0.5 * (||s_i|| + ||s_j||)
                  (2) IoM(mask_i, mask_j) > group_thresh, IoM = inter / min(|A|,|B|)
              - Merge each group by union of masks in original image space
              - Set center/stride to group average
            Returns a per-image dict with:
              onehot: [G,1,X,Y,Z] bool  (global/original image space)
              offset: [G,3] zeros
            """
    X, Y, Z = output_shape
    device = det["onehot"].device

    anchor_centers = det["anchor_centers"].float()  # [M,3] global
    anchor_strides = det["anchor_strides"].float()  # [M,3]
    scores = det["scores"].float()  # [M]
    classes = det["classes"].long()  # [M]
    masks_local = det["onehot"][:, 0].bool()  # [M,96,96,48] patch-local
    offsets = det["offset"].long()  # [M,3] patch origin (global)

    M = int(anchor_centers.shape[0])
    if M == 0:
        return {
            "anchor_centers": anchor_centers.new_zeros((0, 3)),
            "anchor_strides": anchor_strides.new_zeros((0, 3)),
            "classes": classes.new_zeros((0,)),
            "scores": scores.new_zeros((0,)),
            "onehot": masks_local.new_zeros((0, 1, X, Y, Z)),
            "offset": offsets.new_zeros((0, 3)),
        }

    # -------- Step A: build sparse global keys for each mask (offset used ONLY here) --------
    keys: List[Tensor] = [None] * M  # type: ignore[assignment]
    vols = torch.empty((M,), dtype=torch.int64, device="cpu")  # keep vols on cpu ints

    for i in range(M):
        idx = torch.nonzero(masks_local[i], as_tuple=False)  # [P,3] patch coords
        if idx.numel() == 0:
            keys[i] = torch.empty((0,), dtype=torch.int64, device=device)
            vols[i] = 0
            continue
        gxyz = idx + offsets[i].view(1, 3)  # [P,3] global coords
        k = _encode_xyz_to_key(gxyz, output_shape)  # [P]
        k = torch.unique(k)
        k, _ = torch.sort(k)
        keys[i] = k
        vols[i] = int(k.numel())

    # -------- Step 1: pairwise proximity gating (your batched_lbe style) --------
    # axis-wise anchor distance vs half-sum strid
    # pairwise_anchor_dist: [M,M,3]
    pairwise_anchor_dist = (anchor_centers[None] - anchor_centers[:, None]).abs()
    max_anchor_dist = 0.5 * (anchor_strides[None] + anchor_strides[:, None])
    is_close = torch.all(pairwise_anchor_dist < max_anchor_dist, dim=-1)  # [M,M]
    # remove self edges
    is_close.fill_diagonal_(False)

    # -------- Step 2: IoM mask criterion (sparse, computed only for close pairs) --------
    # We'll build adjacency as union-find to avoid an MxM boolean build based on expensive IoM.
    uf = _UnionFind(M)

    # grouped = is_close & (IoM > group_thresh)
    # we compute IoM lazily for (i,j) where is_close is True.

    close_pairs = torch.nonzero(is_close, as_tuple=False)  # [E,2]
    # To avoid double-checking (i,j) and (j,i), only keep i<j
    close_pairs = close_pairs[close_pairs[:, 0] < close_pairs[:, 1]]

    for ij in close_pairs:
        i = int(ij[0].item())
        j = int(ij[1].item())

        vi = int(vols[i].item())
        vj = int(vols[j].item())
        if vi == 0 or vj == 0:
            continue

        inter = _intersect_count_sorted(keys[i], keys[j])
        if inter == 0:
            continue

        iom = inter / max(1, min(vi, vj))
        if iom > group_thresh:
            uf.union(i, j)

    # -------- Step 3: connected components (your find_connected_components analogue) --------
    groups: Dict[int, List[int]] = {}
    for i in range(M):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    group_list = list(groups.values())
    # Sort groups by best score (helps if you later resolve overlaps)
    group_list.sort(key=lambda ids: float(scores[torch.tensor(ids, device=device)].max().item()), reverse=True)

    # -------- Step 4: merge each group (union masks) + average center/stride --------
    G = len(group_list)
    out_centers = torch.empty((G, 3), dtype=torch.float32, device=device)
    out_strides = torch.empty((G, 3), dtype=torch.float32, device=device)
    out_scores = torch.empty((G,), dtype=torch.float32, device=device)
    out_classes = torch.empty((G,), dtype=torch.long, device=device)

    out_onehot = torch.zeros((G, 1, X, Y, Z), dtype=torch.bool, device=device)
    out_offset = torch.zeros((G, 3), dtype=torch.long, device=device)  # global masks => offset=0

    for gi, ids in enumerate(group_list):
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)

        out_centers[gi] = anchor_centers[ids_t].mean(dim=0)
        out_strides[gi] = anchor_strides[ids_t].mean(dim=0)
        out_scores[gi] = scores[ids_t].max()
        out_classes[gi] = classes[ids_t][0]  # single-class

        # union keys -> paint global mask
        kcat = torch.cat([keys[k] for k in ids], dim=0)
        kuniq = torch.unique(kcat)
        xyz = _decode_key_to_xyz(kuniq, output_shape)  # [P,3]

        x = xyz[:, 0].clamp_(0, X - 1)
        y = xyz[:, 1].clamp_(0, Y - 1)
        z = xyz[:, 2].clamp_(0, Z - 1)
        out_onehot[gi, 0, x, y, z] = True

    return {
        "anchor_centers": out_centers,
        "anchor_strides": out_strides,
        "classes": out_classes,
        "scores": out_scores,
        "onehot": out_onehot,
        "offset": out_offset,
    }