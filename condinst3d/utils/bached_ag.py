from typing import Literal, Dict, Optional, Callable, List, Tuple
import torch
from torch import Tensor
from condinst3d.evaluator.iou import box_intersection_over_union, box_intersection_over_minimum
from monai.transforms.utils import get_unique_labels
from condinst3d.utils.mask import relabel_sequential


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

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


def aggregate_per_patch_detections(
    x: Dict[str, List[Tensor]],
    iou_thr: float,
    mask_thr: float,  # <-- probability threshold in [0,1]
    output_shape: Tuple[int, int, int],
) -> Dict[str, Tensor]:
    """
    Aggregates per-patch detections into per-image detections.

    Inputs (per image):
      - x["bboxes"][p]         : [Kp,6] patch-local boxes (x1,y1,z1,x2,y2,z2)
      - x["scores"][p]         : [Kp]
      - x["onehot_logits"][p]  : [Kp,1,ph,pw,pd] logits in patch space
      - x["offset"][p]         : [3] patch origin in global space (ox,oy,oz)
      - x["anchor_centers"][p] : [Kp,3] patch-local centers
      - x["anchor_strides"][p] : [Kp,3]

    Output (per image):
      - instance_mask:        [1,H,W,D] int64 (0=bg, 1..G)
      - onehot_instance_mask: [G,1,H,W,D] bool
      - bboxes:               [G,6] global boxes
      - centers:              [G,3] global centers
      - strides:              [G,3]
      - scores:               [G] aggregated scores (max over grouped boxes)
    """
    # -------------------- basic setup --------------------
    P = len(x.get("scores", []))
    device = x["scores"][0].device if P > 0 else torch.device("cpu")
    H, W, D = output_shape

    # -------------------- flatten all instances across patches --------------------
    all_boxes: List[Tensor] = []
    all_scores: List[Tensor] = []
    all_logits: List[Tensor] = []
    all_offsets: List[Tensor] = []
    all_centers: List[Tensor] = []
    all_strides: List[Tensor] = []

    for p in range(P):
        scores_p = x["scores"][p]
        if scores_p.numel() == 0:
            continue

        bboxes_p  = x["bboxes"][p]                 # [Kp,6] local
        centers_p = x["anchor_centers"][p]         # [Kp,3] local
        strides_p = x["anchor_strides"][p]         # [Kp,3]
        logits_p  = x["onehot_logits"][p]          # [Kp,1,ph,pw,pd]
        offset_p  = x["offset"][p].view(1, 3)      # [1,3]

        Kp = int(bboxes_p.shape[0])
        if Kp == 0:
            continue

        # shift boxes/centers to global
        bboxes_g = bboxes_p.clone()
        bboxes_g[:, :3] += offset_p
        bboxes_g[:, 3:] += offset_p

        centers_g = centers_p + offset_p
        offsets_g = offset_p.expand(Kp, 3)

        all_boxes.append(bboxes_g)
        all_scores.append(scores_p)
        all_logits.append(logits_p)
        all_offsets.append(offsets_g)
        all_centers.append(centers_g)
        all_strides.append(strides_p)

    # empty (no detections at all)
    if len(all_scores) == 0:
        instance_mask = torch.zeros((1, H, W, D), dtype=torch.int64, device=device)
        onehot = torch.zeros((0, 1, H, W, D), dtype=torch.bool, device=device)
        return {
            "instance_mask": instance_mask,
            "onehot_instance_mask": onehot,
            "bboxes": torch.zeros((0, 6), dtype=torch.float32, device=device),
            "centers": torch.zeros((0, 3), dtype=torch.float32, device=device),
            "strides": torch.zeros((0, 3), dtype=torch.float32, device=device),
            "scores": torch.zeros((0,), dtype=torch.float32, device=device),
        }

    boxes   = torch.cat(all_boxes, dim=0).float()        # [N,6]
    scores  = torch.cat(all_scores, dim=0).float()       # [N]
    logits  = torch.cat(all_logits, dim=0)               # [N,1,ph,pw,pd]
    offsets = torch.cat(all_offsets, dim=0).long()       # [N,3] (global origin of each patch for that instance)
    centers = torch.cat(all_centers, dim=0).float()      # [N,3]
    strides = torch.cat(all_strides, dim=0).float()      # [N,3]
    N = int(scores.numel())

    # -------------------- group instances by IoM/IoU threshold --------------------
    iom = box_intersection_over_minimum(boxes, boxes)  # [N,N], on device

    uf = _UnionFind(N)

    triu = torch.triu(iom, diagonal=1)
    edges = (triu >= float(iou_thr)).nonzero(as_tuple=False)  # [E,2] on device

    # UnionFind is python-side; move edges once
    for i, j in edges.cpu().tolist():
        uf.union(int(i), int(j))

    # Build groups: root -> member indices
    groups: Dict[int, List[int]] = {}
    for i in range(N):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    # -------------------- aggregate group metadata --------------------
    # We'll build a "group struct" list so sorting keeps members aligned.
    group_list = []
    for members in groups.values():
        idx = torch.as_tensor(members, device=device, dtype=torch.long)

        b = boxes[idx]  # [m,6]
        merged_box = torch.stack(
            [b[:, 0].min(), b[:, 1].min(), b[:, 2].min(),
             b[:, 3].max(), b[:, 4].max(), b[:, 5].max()],
            dim=0,
        )  # float [6]

        s = scores[idx]                       # [m]
        rep = idx[torch.argmax(s)]            # representative (highest score)
        s_agg = s.max()                       # group score = max

        group_list.append({
            "members": members,               # list[int] (indices into flattened N)
            "box": merged_box,                # [6] float
            "center": centers[rep],           # [3] float
            "stride": strides[rep],           # [3] float
            "score": s_agg,                   # scalar float
        })

    # Sort ascending score so higher scores overwrite later in instance_mask
    group_list.sort(key=lambda g: float(g["score"]))

    G = len(group_list)

    # -------------------- build global instance masks --------------------
    instance_mask = torch.zeros((1, H, W, D), dtype=torch.int64, device=device)
    onehot_global = torch.zeros((G, 1, H, W, D), dtype=torch.bool, device=device)

    # We'll keep outputs aligned with this sorted order: instance id = k+1
    out_boxes   = torch.zeros((G, 6), dtype=torch.float32, device=device)
    out_centers = torch.zeros((G, 3), dtype=torch.float32, device=device)
    out_strides = torch.zeros((G, 3), dtype=torch.float32, device=device)
    out_scores  = torch.zeros((G,),   dtype=torch.float32, device=device)

    mask_thr = float(mask_thr)  # prob threshold

    for k, g in enumerate(group_list):
        members = g["members"]

        out_boxes[k] = g["box"].to(out_boxes.dtype)
        out_centers[k] = g["center"].to(out_centers.dtype)
        out_strides[k] = g["stride"].to(out_strides.dtype)
        out_scores[k] = float(g["score"])

        # crop region from merged global box
        bx = g["box"].to(torch.long)  # [6]
        x1, y1, z1, x2, y2, z2 = bx.tolist()

        # clamp to canvas bounds
        gx1, gy1, gz1 = max(0, x1), max(0, y1), max(0, z1)
        gx2, gy2, gz2 = min(H, x2), min(W, y2), min(D, z2)

        if gx2 <= gx1 or gy2 <= gy1 or gz2 <= gz1:
            continue

        # accumulate probabilities and counts in the cropped region
        crop_shape = (gx2 - gx1, gy2 - gy1, gz2 - gz1)
        acc = torch.zeros(crop_shape, dtype=torch.float32, device=device)
        cnt = torch.zeros(crop_shape, dtype=torch.float32, device=device)

        for m in members:
            # local logits -> probs
            m_prob = torch.sigmoid(logits[m, 0]).to(torch.float32)  # [ph,pw,pd]
            ox, oy, oz = offsets[m].tolist()

            ph, pw, pd = m_prob.shape
            px1, py1, pz1 = ox, oy, oz
            px2, py2, pz2 = ox + ph, oy + pw, oz + pd

            # intersection with group crop
            ix1, iy1, iz1 = max(gx1, px1), max(gy1, py1), max(gz1, pz1)
            ix2, iy2, iz2 = min(gx2, px2), min(gy2, py2), min(gz2, pz2)

            if ix2 <= ix1 or iy2 <= iy1 or iz2 <= iz1:
                continue

            # map to acc indices
            ax1, ay1, az1 = ix1 - gx1, iy1 - gy1, iz1 - gz1
            ax2, ay2, az2 = ix2 - gx1, iy2 - gy1, iz2 - gz1

            # map to local indices
            lx1, ly1, lz1 = ix1 - px1, iy1 - py1, iz1 - pz1
            lx2, ly2, lz2 = ix2 - px1, iy2 - py1, iz2 - pz1

            patch_crop = m_prob[lx1:lx2, ly1:ly2, lz1:lz2]
            acc[ax1:ax2, ay1:ay2, az1:az2] += patch_crop
            cnt[ax1:ax2, ay1:ay2, az1:az2] += 1.0

        # average probs where covered
        mean_prob = acc / cnt.clamp_min(1.0)
        tmp = mean_prob >= mask_thr  # boolean mask in crop

        if not torch.any(tmp):
            continue

        # write onehot + instance id (safe, no chained indexing bug)
        onehot_global[k, 0, gx1:gx2, gy1:gy2, gz1:gz2] = tmp
        instance_mask[0, gx1:gx2, gy1:gy2, gz1:gz2].masked_fill_(tmp, k + 1)

    # -------------------- keep only actually-present instances --------------------
    remained = list(get_unique_labels(instance_mask, is_onehot=False, discard=0))  # labels in {1..G}
    if len(remained) == 0:
        # nothing survived thresholding
        empty_onehot = torch.zeros((0, 1, H, W, D), dtype=torch.bool, device=device)
        return {
            "instance_mask": torch.zeros((1, H, W, D), dtype=torch.int64, device=device),
            "onehot_instance_mask": empty_onehot,
            "bboxes": torch.zeros((0, 6), dtype=torch.float32, device=device),
            "centers": torch.zeros((0, 3), dtype=torch.float32, device=device),
            "strides": torch.zeros((0, 3), dtype=torch.float32, device=device),
            "scores": torch.zeros((0,), dtype=torch.float32, device=device),
        }

    remained_idx = torch.as_tensor([int(x) - 1 for x in remained], device=device, dtype=torch.long)

    return {
        "instance_mask": relabel_sequential(instance_mask, exclude_background=True), # [1,H,W,D]
        "onehot_instance_mask": onehot_global[remained_idx],  # [G',1,H,W,D]
        "bboxes": out_boxes[remained_idx],                    # [G',6]
        "centers": out_centers[remained_idx],                 # [G',3]
        "strides": out_strides[remained_idx],                 # [G',3]
        "scores": out_scores[remained_idx],                   # [G']
    }