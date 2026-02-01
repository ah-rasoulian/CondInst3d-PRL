from typing import Literal, Dict, Optional, Callable, List
import torch
from torch import Tensor
from condinst3d.evaluator.iou import box_intersection_over_union, box_intersection_over_minimum
from monai.transforms.utils import get_unique_labels


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
    mask_thr: float,
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
      - x["output_shape"]      : (H,W,D) global image shape

    Output (per image):
      - instance_mask:        [1,H,W,D] int64 (0=bg, 1..G)
      - onehot_instance_mask: [G,1,H,W,D] bool
      - bboxes:               [G,6] global boxes
      - centers:              [G,3] global centers
      - strides:              [G,3]
      - scores:               [G] aggregated scores (max over grouped boxes)
    """
    device = x["scores"][0].device if len(x["scores"]) else torch.device("cpu")
    H, W, D = map(int, x["output_shape"])
    P = len(x["scores"])

    # -------------------- Flatten all instances across patches --------------------
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

        bboxes_p = x["bboxes"][p]                # [Kp,6] local
        centers_p = x["anchor_centers"][p]       # [Kp,3] local
        strides_p = x["anchor_strides"][p]       # [Kp,3]
        logits_p = x["onehot_logits"][p]         # [Kp,1,ph,pw,pd]
        offset_p = x["offset"][p].view(1, 3)     # [1,3]

        Kp = bboxes_p.shape[0]

        # Shift boxes/centers to global coordinates
        bboxes_g = bboxes_p.clone()
        bboxes_g[:, :3] += offset_p
        bboxes_g[:, 3:] += offset_p
        centers_g = centers_p + offset_p  # broadcast

        # Per-instance offsets [Kp,3]
        offsets_g = offset_p.expand(Kp, 3)

        all_boxes.append(bboxes_g)
        all_scores.append(scores_p)
        all_logits.append(logits_p)
        all_offsets.append(offsets_g)
        all_centers.append(centers_g)
        all_strides.append(strides_p)

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

    boxes = torch.cat(all_boxes, dim=0)       # [N,6] global
    scores = torch.cat(all_scores, dim=0)     # [N]
    logits = torch.cat(all_logits, dim=0)     # [N,1,ph,pw,pd]
    offsets = torch.cat(all_offsets, dim=0)   # [N,3]
    centers = torch.cat(all_centers, dim=0)   # [N,3]
    strides = torch.cat(all_strides, dim=0)   # [N,3]
    N = scores.numel()

    # -------------------- Group boxes by IoM/IoU threshold --------------------
    # Uses your existing helper (assumed returns [N,N]).
    iom = box_intersection_over_minimum(boxes.float(), boxes.float())  # [N,N]

    uf = _UnionFind(N)

    # Collect edges once (upper-triangular) to avoid per-i GPU->CPU sync
    triu = torch.triu(iom, diagonal=1)
    edges = (triu >= iou_thr).nonzero(as_tuple=False)  # [E,2] on device

    edges_cpu = edges.cpu()
    for i, j in edges_cpu.tolist():
        uf.union(int(i), int(j))

    # Build groups: root -> member indices
    groups: Dict[int, List[int]] = {}
    for i in range(N):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    # -------------------- Aggregate groups metadata --------------------
    group_boxes: List[Tensor] = []
    group_centers: List[Tensor] = []
    group_strides: List[Tensor] = []
    group_scores: List[Tensor] = []
    group_members: List[List[int]] = []

    for idxs in groups.values():
        idx = torch.tensor(idxs, device=device, dtype=torch.long)

        b = boxes[idx]  # [m,6]
        merged_box = torch.stack(
            [b[:, 0].min(), b[:, 1].min(), b[:, 2].min(),
             b[:, 3].max(), b[:, 4].max(), b[:, 5].max()],
            dim=0,
        ).to(boxes.dtype)

        s = scores[idx]
        rep = idx[torch.argmax(s)]   # representative (highest score)
        s_agg = s.max()              # <-- requested: MAX aggregation

        group_boxes.append(merged_box[None, :])
        group_centers.append(centers[rep][None, :])
        group_strides.append(strides[rep][None, :])
        group_scores.append(s_agg[None])
        group_members.append(idxs)

    group_boxes = torch.cat(group_boxes, dim=0)        # [G,6]
    group_centers = torch.cat(group_centers, dim=0)    # [G,3]
    group_strides = torch.cat(group_strides, dim=0)    # [G,3]
    group_scores = torch.cat(group_scores, dim=0)      # [G]
    G = group_scores.numel()

    # -------------------- Build global instance masks --------------------
    # instance_mask must be [1,H,W,D]
    instance_mask = torch.zeros((1, H, W, D), dtype=torch.int64, device=device)
    onehot_global = torch.zeros((G, 1, H, W, D), dtype=torch.bool, device=device)

    # Sort by score ascending so higher scores overwrite later
    order = torch.argsort(group_scores)  # ascending
    # We'll write outputs aligned with sorted order (k -> instance_id k+1)
    group_boxes = group_boxes[order]
    group_centers = group_centers[order]
    group_strides = group_strides[order]
    group_scores = group_scores[order]

    # Within-group merge improvement:
    #   - accumulate logits into a GLOBAL logit canvas (sum)
    #   - threshold once at the end in logit space (mask_thr is assumed logit threshold)
    #
    for k in range(G):
        # g_idx refers to the original group index before sorting
        g_idx = int(order[k].item())
        members = group_members[g_idx]

        # Use a cropped region to reduce work: use merged global bbox for this group
        bx = group_boxes[k].to(torch.long)  # [6]
        x1, y1, z1, x2, y2, z2 = bx.tolist()

        # Clamp to canvas
        gx1, gy1, gz1 = max(0, x1), max(0, y1), max(0, z1)
        gx2, gy2, gz2 = min(H, x2), min(W, y2), min(D, z2)

        if gx2 <= gx1 or gy2 <= gy1 or gz2 <= gz1:
            continue

        # Accumulate logits over the group's region only
        # Start at 0 so sum-aggregation works naturally
        acc = torch.zeros((gx2 - gx1, gy2 - gy1, gz2 - gz1), dtype=logits.dtype, device=device)

        for m in members:
            # local logits [ph,pw,pd]
            m_log = logits[m, 0]  # [ph,pw,pd]
            off = offsets[m].to(torch.long)
            ox, oy, oz = off.tolist()

            ph, pw, pd = m_log.shape
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

            acc[ax1:ax2, ay1:ay2, az1:az2] += m_log[lx1:lx2, ly1:ly2, lz1:lz2]

        # Threshold once (logit threshold)
        tmp = acc >= mask_thr  # [cropH,cropW,cropD] bool

        onehot_global[k, 0, gx1:gx2, gy1:gy2, gz1:gz2] = tmp
        instance_mask[0, gx1:gx2, gy1:gy2, gz1:gz2][tmp] = (k + 1)

    remained_instances = list(get_unique_labels(instance_mask, is_onehot=False, discard=0))
    remained_indices = [x - 1 for x in remained_instances]  # undo the effect of background class

    return {
        "instance_mask": instance_mask,             # [1,H,W,D]
        "onehot_instance_mask": onehot_global[remained_indices],      # [G,1,H,W,D]
        "bboxes": group_boxes[remained_indices],                      # [G,6] (sorted low->high score)
        "centers": group_centers[remained_indices],                   # [G,3]
        "strides": group_strides[remained_indices],                   # [G,3]
        "scores": group_scores[remained_indices],                     # [G]
    }
