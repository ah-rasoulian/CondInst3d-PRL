from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor
from torchmetrics import Metric


class DetectionConfusionMatrix(Metric):
    """
    COCO-style matching confusion matrix for a single class.

    Computes per-IoU-threshold counts: [TP, FP, FN], where
      - TP: number of matched GT instances (each GT can be matched at most once)
      - FP: number of unmatched predictions (duplicate detections count as FP)
      - FN: number of unmatched GT instances

    Inputs to update():
      - iou_matrix: Tensor [N_pred, N_gt]
      - scores:     Tensor [N_pred]  (needed for COCO-style greedy matching by score)

    If you truly cannot provide scores for confusion-matrix computation, you *can* set
    scores to all-ones, but then duplicates are broken arbitrarily.
    """

    is_differentiable = False
    higher_is_better = None
    full_state_update = False

    def __init__(
        self,
        iou_thresholds: float | Sequence[float] = 0.5,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if isinstance(iou_thresholds, (float, int)):
            iou_thresholds = [float(iou_thresholds)]
        else:
            iou_thresholds = [float(x) for x in iou_thresholds]

        self.iou_thresholds = torch.tensor(iou_thresholds, dtype=torch.float32)

        # cfm[t] = [TP, FP, FN]
        self.add_state(
            "cfm",
            default=torch.zeros((len(iou_thresholds), 3), dtype=torch.long),
            dist_reduce_fx="sum",
        )

    @torch.no_grad()
    def update(self, iou_matrix: Tensor, scores: Tensor | None = None) -> None:
        """
        Update confusion matrix with one image.

        Args:
          iou_matrix: [N_pred, N_gt]
          scores:     [N_pred] (recommended; if None, all scores assumed equal)
        """
        if iou_matrix.dim() != 2:
            raise ValueError(f"Expected iou_matrix 2D [N_pred, N_gt], got {tuple(iou_matrix.shape)}")

        device = self.cfm.device
        iou_matrix = iou_matrix.to(device=device, dtype=torch.float32)
        thr = self.iou_thresholds.to(device=device)

        n_pred, n_gt = iou_matrix.shape

        if scores is None:
            scores = torch.ones((n_pred,), device=device, dtype=torch.float32)
        else:
            if scores.dim() != 1 or scores.numel() != n_pred:
                raise ValueError(f"Expected scores [N_pred], got {tuple(scores.shape)} (N_pred={n_pred})")
            scores = scores.to(device=device, dtype=torch.float32)

        # Score order (COCO-style greedy matching)
        if n_pred > 0:
            order = torch.argsort(scores, descending=True)
            iou_sorted = iou_matrix[order]  # [N_pred, N_gt]
        else:
            iou_sorted = iou_matrix

        T = thr.numel()
        tp = torch.zeros((T,), device=device, dtype=torch.long)
        fp = torch.zeros((T,), device=device, dtype=torch.long)
        fn = torch.zeros((T,), device=device, dtype=torch.long)

        # Handle trivial cases
        if n_gt == 0:
            # No GT: everything predicted is FP
            fp[:] = n_pred
            # tp, fn stay 0
            self.cfm += torch.stack([tp, fp, fn], dim=1)
            return

        if n_pred == 0:
            # No preds: everything GT is FN
            fn[:] = n_gt
            self.cfm += torch.stack([tp, fp, fn], dim=1)
            return

        # For each IoU threshold, greedy match by score so each GT matched at most once
        for ti in range(T):
            t = thr[ti]
            gt_matched = torch.zeros((n_gt,), device=device, dtype=torch.bool)

            matched_count = 0
            for pi in range(n_pred):
                ious = iou_sorted[pi]  # [N_gt]
                # only consider unmatched GTs
                ious = ious.masked_fill(gt_matched, -1.0)
                best_iou, best_gt = torch.max(ious, dim=0)

                if best_iou >= t:
                    gt_matched[best_gt] = True
                    matched_count += 1
                # else unmatched pred -> FP (counted later)

            tp[ti] = matched_count
            fp[ti] = n_pred - matched_count
            fn[ti] = n_gt - matched_count

        self.cfm += torch.stack([tp, fp, fn], dim=1)

    def compute(self) -> Any:
        return self.cfm
