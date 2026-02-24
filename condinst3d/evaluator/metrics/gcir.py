from __future__ import annotations

from typing import Any, Sequence
import torch
from torch import Tensor
from torchmetrics import Metric


class GlobalConfluentInstanceRecall(Metric):
    """
    Global recall over GT instances that belong to confluent clusters (cluster size >= 2).

    For each IoU threshold:
      - Greedy match predictions to GT instances (COCO-style, by descending score)
      - Each GT can be matched at most once (duplicates don't help)
      - Only GTs with gt_cluster_ids >= 0 are counted (i.e., confluent GT instances)

    update inputs (per image):
      - iou_matrix: Tensor [N_pred, N_gt]
      - scores:     Tensor [N_pred]
      - gt_cluster_ids: LongTensor [N_gt]
            -1 => GT instance is NOT part of a confluent cluster
             0..C-1 => cluster id for confluent GT instance
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        iou_thresholds: float | Sequence[float] = 0.25,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if isinstance(iou_thresholds, (float, int)):
            iou_thresholds = [float(iou_thresholds)]
        else:
            iou_thresholds = [float(x) for x in iou_thresholds]
        self.iou_thresholds = torch.tensor(iou_thresholds, dtype=torch.float32)

        self.add_state("matched", default=torch.zeros(len(iou_thresholds), dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(len(iou_thresholds), dtype=torch.long), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, iou_matrix: Tensor, scores: Tensor, gt_cluster_ids: Tensor) -> None:
        if iou_matrix.dim() != 2:
            raise ValueError(f"iou_matrix must be [N_pred, N_gt], got {tuple(iou_matrix.shape)}")
        if scores.dim() != 1 or scores.numel() != iou_matrix.size(0):
            raise ValueError(f"scores must be [N_pred], got {tuple(scores.shape)}")
        if gt_cluster_ids.dim() != 1 or gt_cluster_ids.numel() != iou_matrix.size(1):
            raise ValueError(f"gt_cluster_ids must be [N_gt], got {tuple(gt_cluster_ids.shape)}")

        device = self.matched.device
        iou_matrix = iou_matrix.to(device=device, dtype=torch.float32)
        scores = scores.to(device=device, dtype=torch.float32)
        gt_cluster_ids = gt_cluster_ids.to(device=device, dtype=torch.long)
        thr = self.iou_thresholds.to(device=device)

        n_pred, n_gt = iou_matrix.shape
        if n_gt == 0:
            return

        # Select confluent GT instances
        confluent_mask = gt_cluster_ids >= 0
        if not torch.any(confluent_mask):
            return

        gt_idx = torch.nonzero(confluent_mask, as_tuple=False).squeeze(1)  # indices into GT axis
        sub_iou = iou_matrix[:, gt_idx]  # [N_pred, N_conf_gt]
        n_conf_gt = sub_iou.size(1)

        T = thr.numel()
        self.total += torch.full((T,), n_conf_gt, device=device, dtype=torch.long)

        if n_pred == 0:
            return

        # Sort preds by score desc
        order = torch.argsort(scores, descending=True)
        sub_iou = sub_iou[order]  # [N_pred, N_conf_gt]

        # For each threshold, greedy match to GTs (one-to-one)
        for ti in range(T):
            t = thr[ti]
            gt_matched = torch.zeros((n_conf_gt,), device=device, dtype=torch.bool)
            cnt = 0
            for pi in range(n_pred):
                ious = sub_iou[pi].masked_fill(gt_matched, -1.0)
                best_iou, best_j = torch.max(ious, dim=0)
                if best_iou >= t:
                    gt_matched[best_j] = True
                    cnt += 1
                    if cnt == n_conf_gt:
                        break
            self.matched[ti] += cnt

    def compute(self) -> Any:
        return self.matched.to(torch.float32) / (self.total.to(torch.float32) + 1e-8)
