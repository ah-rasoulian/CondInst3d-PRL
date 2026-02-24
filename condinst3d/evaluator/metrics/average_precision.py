from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor
from torchmetrics import Metric
import matplotlib.pyplot as plt
import numpy as np


class AveragePrecision(Metric):
    """
    COCO-style AP for a single class, given per-image IoU matrices.

    DDP-safe fix:
      TorchMetrics' built-in distributed sync can try to `stack()` per-rank states.
      Since different ranks have different numbers of predictions, tensors are ragged
      and stacking fails.

      This implementation avoids TorchMetrics' state syncing for variable-length
      prediction lists by setting `dist_sync_on_step=False` and `sync_on_compute=False`
      and performing manual all_gather with padding inside `compute()`.

    Inputs to update():
      - iou_matrix: Tensor [N_pred, N_gt]
      - scores:     Tensor [N_pred]

    Returns:
      - Tensor [T] AP for each IoU threshold
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        iou_thresholds: float | Sequence[float] = 0.5,
        interpolation: None | int = 101,  # COCO default
        dist_sync_on_step: bool = False,  # keep False; we do manual sync in compute()
        eps: float = 1e-8,
    ):
        # IMPORTANT: sync_on_compute=False prevents TorchMetrics from trying to sync/stack states.
        super().__init__(dist_sync_on_step=dist_sync_on_step, sync_on_compute=False)

        if isinstance(iou_thresholds, (float, int)):
            iou_thresholds = [float(iou_thresholds)]
        else:
            iou_thresholds = [float(x) for x in iou_thresholds]

        self.iou_thresholds = torch.tensor(iou_thresholds, dtype=torch.float32)
        self.interpolation = 101 if interpolation is None else int(interpolation)
        self.eps = float(eps)

        # Store per-rank data locally (no TorchMetrics distributed reduction)
        self.add_state("scores_all", default=torch.empty(0, dtype=torch.float32), dist_reduce_fx=None)
        self.add_state("tp_all", default=torch.empty(0, 0, dtype=torch.uint8), dist_reduce_fx=None)  # uint8 saves comm
        self.add_state("fp_all", default=torch.empty(0, 0, dtype=torch.uint8), dist_reduce_fx=None)
        self.add_state("num_gt", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx=None)

        # For plotting
        self.precision_curve: Tensor | None = None  # [T, R]
        self.recall_curve: Tensor | None = None     # [R]
        self._last_ap: Tensor | None = None         # [T]

    @staticmethod
    def _dist_available() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @staticmethod
    @torch.no_grad()
    def _all_gather_padded_1d(x: Tensor) -> tuple[Tensor, Tensor]:
        """
        All-gather 1D tensors of different lengths by padding to max length.
        Returns:
          gathered: [W, max_len]
          lengths:  [W]
        """
        if not AveragePrecision._dist_available():
            return x.unsqueeze(0), torch.tensor([x.numel()], device=x.device, dtype=torch.long)

        W = torch.distributed.get_world_size()
        device = x.device

        length = torch.tensor([x.numel()], device=device, dtype=torch.long)
        lengths = [torch.zeros_like(length) for _ in range(W)]
        torch.distributed.all_gather(lengths, length)
        lengths = torch.cat(lengths, dim=0)
        max_len = int(lengths.max().item())

        padded = torch.zeros((max_len,), device=device, dtype=x.dtype)
        if x.numel() > 0:
            padded[: x.numel()] = x

        gathered = [torch.zeros_like(padded) for _ in range(W)]
        torch.distributed.all_gather(gathered, padded)
        return torch.stack(gathered, dim=0), lengths

    @staticmethod
    @torch.no_grad()
    def _all_gather_padded_2d(x: Tensor) -> tuple[Tensor, Tensor]:
        """
        All-gather 2D tensors [N, T] of different N by padding along dim=0.
        Returns:
          gathered: [W, max_N, T]
          lengths:  [W]
        """
        if not AveragePrecision._dist_available():
            return x.unsqueeze(0), torch.tensor([x.size(0)], device=x.device, dtype=torch.long)

        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor [N,T], got {tuple(x.shape)}")

        W = torch.distributed.get_world_size()
        device = x.device
        N, T = x.shape

        length = torch.tensor([N], device=device, dtype=torch.long)
        lengths = [torch.zeros_like(length) for _ in range(W)]
        torch.distributed.all_gather(lengths, length)
        lengths = torch.cat(lengths, dim=0)
        max_N = int(lengths.max().item())

        padded = torch.zeros((max_N, T), device=device, dtype=x.dtype)
        if N > 0:
            padded[:N] = x

        gathered = [torch.zeros_like(padded) for _ in range(W)]
        torch.distributed.all_gather(gathered, padded)
        return torch.stack(gathered, dim=0), lengths

    @torch.no_grad()
    def update(self, iou_matrix: Tensor, scores: Tensor) -> None:
        if iou_matrix.dim() != 2:
            raise ValueError(f"Expected iou_matrix to be 2D [N_pred, N_gt], got shape {tuple(iou_matrix.shape)}")
        if scores.dim() != 1:
            raise ValueError(f"Expected scores to be 1D [N_pred], got shape {tuple(scores.shape)}")
        if iou_matrix.size(0) != scores.numel():
            raise ValueError(
                f"Mismatch: iou_matrix has N_pred={iou_matrix.size(0)} but scores has {scores.numel()}"
            )

        device = self.scores_all.device
        iou_matrix = iou_matrix.to(device=device, dtype=torch.float32)
        scores = scores.to(device=device, dtype=torch.float32)
        thr = self.iou_thresholds.to(device=device)

        n_pred, n_gt = iou_matrix.shape
        self.num_gt += int(n_gt)

        if n_pred == 0:
            return

        # Sort predictions by score desc
        order = torch.argsort(scores, descending=True)
        scores_sorted = scores[order]
        iou_sorted = iou_matrix[order]

        T = thr.numel()

        # Prepare per-image tp/fp in score order
        tp_img = torch.zeros((n_pred, T), device=device, dtype=torch.uint8)
        fp_img = torch.zeros((n_pred, T), device=device, dtype=torch.uint8)

        if n_gt == 0:
            fp_img[:] = 1
            self._append(scores_sorted, tp_img, fp_img)
            return

        for ti in range(T):
            t = thr[ti]
            gt_matched = torch.zeros((n_gt,), device=device, dtype=torch.bool)
            for pi in range(n_pred):
                ious = iou_sorted[pi].masked_fill(gt_matched, -1.0)
                best_iou, best_gt = torch.max(ious, dim=0)
                if best_iou >= t:
                    tp_img[pi, ti] = 1
                    gt_matched[best_gt] = True
                else:
                    fp_img[pi, ti] = 1

        self._append(scores_sorted, tp_img, fp_img)

    def _append(self, scores_sorted: Tensor, tp_img: Tensor, fp_img: Tensor) -> None:
        # Initialize tp_all/fp_all second dimension if empty
        if self.tp_all.numel() == 0:
            self.tp_all = tp_img
            self.fp_all = fp_img
        else:
            if self.tp_all.size(1) != tp_img.size(1):
                raise RuntimeError(
                    f"Threshold count changed: stored T={self.tp_all.size(1)} but got T={tp_img.size(1)}"
                )
            self.tp_all = torch.cat([self.tp_all, tp_img], dim=0)
            self.fp_all = torch.cat([self.fp_all, fp_img], dim=0)

        self.scores_all = torch.cat([self.scores_all, scores_sorted], dim=0)

    @torch.no_grad()
    def compute(self) -> Any:
        device = self.scores_all.device
        thr = self.iou_thresholds.to(device=device)
        T = thr.numel()

        R = int(self.interpolation)
        recall_levels = torch.linspace(0.0, 1.0, R, device=device)

        # ---- manual DDP sync ----
        # gather num_gt
        num_gt = self.num_gt.clone()
        if self._dist_available():
            torch.distributed.all_reduce(num_gt, op=torch.distributed.ReduceOp.SUM)
        num_gt_int = int(num_gt.item())

        # gather ragged per-rank tensors
        scores_g, scores_len = self._all_gather_padded_1d(self.scores_all.to(torch.float32))
        tp_g, tp_len = self._all_gather_padded_2d(self.tp_all.to(torch.uint8)) if self.tp_all.numel() else (
            torch.zeros((1, 0, T), device=device, dtype=torch.uint8),
            torch.tensor([0], device=device, dtype=torch.long),
        )
        fp_g, fp_len = self._all_gather_padded_2d(self.fp_all.to(torch.uint8)) if self.fp_all.numel() else (
            torch.zeros((1, 0, T), device=device, dtype=torch.uint8),
            torch.tensor([0], device=device, dtype=torch.long),
        )

        # Trim and concatenate across ranks
        all_scores = []
        all_tp = []
        all_fp = []
        W = scores_g.size(0)
        for r in range(W):
            n = int(scores_len[r].item())
            if n == 0:
                continue
            all_scores.append(scores_g[r, :n])
            all_tp.append(tp_g[r, :n])
            all_fp.append(fp_g[r, :n])

        if num_gt_int == 0 or len(all_scores) == 0:
            ap = torch.zeros((T,), device=device, dtype=torch.float32)
            self._last_ap = ap
            self.recall_curve = recall_levels
            self.precision_curve = torch.zeros((T, R), device=device, dtype=torch.float32)
            return ap

        scores_all = torch.cat(all_scores, dim=0)             # [N]
        tp_all = torch.cat(all_tp, dim=0).to(torch.float32)   # [N,T]
        fp_all = torch.cat(all_fp, dim=0).to(torch.float32)   # [N,T]

        # Global sort by score
        global_order = torch.argsort(scores_all, descending=True)
        tp = tp_all[global_order]
        fp = fp_all[global_order]

        # Cumulative sums
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)

        precision = tp_cum / (tp_cum + fp_cum + self.eps)  # [N,T]
        recall = tp_cum / (float(num_gt_int) + self.eps)   # [N,T]

        # COCO-style interpolation
        precision_curve = torch.zeros((T, R), device=device, dtype=torch.float32)
        for ti in range(T):
            rec_t = recall[:, ti]
            prec_t = precision[:, ti]
            prec_env = torch.flip(torch.cummax(torch.flip(prec_t, dims=[0]), dim=0).values, dims=[0])

            idx = torch.searchsorted(rec_t, recall_levels, right=False)
            valid = idx < rec_t.numel()
            sampled = torch.zeros((R,), device=device, dtype=torch.float32)
            sampled[valid] = prec_env[idx[valid]]
            precision_curve[ti] = sampled

        ap = precision_curve.mean(dim=1)
        self.recall_curve = recall_levels
        self.precision_curve = precision_curve
        self._last_ap = ap
        return ap

    def plot(self) -> Any:
        if self.precision_curve is None or self.recall_curve is None:
            _ = self.compute()

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, int(self.iou_thresholds.numel())))

        recall_np = self.recall_curve.detach().cpu().numpy()
        prec_np = self.precision_curve.detach().cpu().numpy()
        thr_np = self.iou_thresholds.detach().cpu().numpy()

        for i, t in enumerate(thr_np):
            ax.plot(recall_np, prec_np[i], color=colors[i], label=f"IoU={t:.2f}")

        ax.set_xlabel("Recall")
        ax.set_xlim([0, 1])
        ax.set_ylabel("Precision")
        ax.set_ylim([0, 1])
        ax.set_title(f"COCO-style Precision/Recall ({self.interpolation}-point)")
        ax.legend()
        ax.grid(True)
        return fig, ax
