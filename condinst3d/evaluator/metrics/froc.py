from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torchmetrics


@torch.no_grad()
def _greedy_match_sorted(
    iou_sorted: torch.Tensor,  # [K, G], predictions already sorted by score desc
    iou_thr: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Greedy 1-1 matching along the sorted predictions.
    Returns:
      tp_inc: [K] {0,1} per prediction rank
      fp_inc: [K] {0,1} per prediction rank
      G: number of GT
    """
    device = iou_sorted.device
    K, G = int(iou_sorted.shape[0]), int(iou_sorted.shape[1])

    if K == 0:
        return torch.zeros((0,), device=device), torch.zeros((0,), device=device), G

    if G == 0:
        # everything is FP
        tp_inc = torch.zeros((K,), device=device, dtype=torch.float32)
        fp_inc = torch.ones((K,), device=device, dtype=torch.float32)
        return tp_inc, fp_inc, 0

    matched_gt = torch.zeros((G,), device=device, dtype=torch.bool)
    tp_inc = torch.zeros((K,), device=device, dtype=torch.float32)
    fp_inc = torch.zeros((K,), device=device, dtype=torch.float32)

    for k in range(K):
        best_iou, best_g = torch.max(iou_sorted[k], dim=0)
        if (best_iou >= iou_thr) and (not matched_gt[best_g]):
            matched_gt[best_g] = True
            tp_inc[k] = 1.0
        else:
            fp_inc[k] = 1.0

    return tp_inc, fp_inc, G


class DetectionFROC(torchmetrics.Metric):
    """
    FROC: Sensitivity vs FPPI (false positives per image)

    DDP-safe: accumulates fixed-size TP/FP/FN arrays over score thresholds.
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        iou_thr: float = 0.1,
        n_thresholds: int = 200,
        score_min: float = 0.0,
        score_max: float = 1.0,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if n_thresholds < 2:
            raise ValueError("n_thresholds must be >= 2")

        self.iou_thr = float(iou_thr)
        self.n_thresholds = int(n_thresholds)
        self.score_min = float(score_min)
        self.score_max = float(score_max)

        # fixed threshold grid
        thresholds = torch.linspace(self.score_min, self.score_max, self.n_thresholds)
        self.register_buffer("thresholds", thresholds)

        # states reduced across ranks by sum
        self.add_state("tp", default=torch.zeros(self.n_thresholds), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(self.n_thresholds), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(self.n_thresholds), dist_reduce_fx="sum")
        self.add_state("num_images", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, pairwise_iou: torch.Tensor, scores: torch.Tensor) -> None:
        """
        pairwise_iou: [K, G]
        scores:       [K]
        """
        if pairwise_iou.ndim != 2:
            raise ValueError(f"pairwise_iou must be [K,G], got {tuple(pairwise_iou.shape)}")
        if scores.ndim != 1:
            raise ValueError(f"scores must be [K], got {tuple(scores.shape)}")
        if pairwise_iou.shape[0] != scores.shape[0]:
            raise ValueError("K mismatch: pairwise_iou.shape[0] must equal scores.shape[0]")

        device = self.tp.device
        pairwise_iou = pairwise_iou.to(device)
        scores = scores.to(device)

        # Sort by score desc once
        if scores.numel() > 0:
            order = torch.argsort(scores, descending=True)
            scores_sorted = scores[order]
            iou_sorted = pairwise_iou[order]
        else:
            scores_sorted = scores
            iou_sorted = pairwise_iou

        tp_inc, fp_inc, G = _greedy_match_sorted(iou_sorted, self.iou_thr)

        # cumulative counts for top-k predictions
        tp_cum = torch.cumsum(tp_inc, dim=0)  # [K]
        fp_cum = torch.cumsum(fp_inc, dim=0)  # [K]

        # For each threshold t, let k(t) = #preds with score >= t.
        # scores_sorted is descending. Create ascending version for searchsorted.
        if scores_sorted.numel() == 0:
            k_per_t = torch.zeros_like(self.thresholds, dtype=torch.long)
        else:
            scores_asc = torch.flip(scores_sorted, dims=(0,))  # ascending
            # # of scores < t in ascending order
            num_lt = torch.searchsorted(scores_asc, self.thresholds, right=False)
            k_per_t = scores_sorted.numel() - num_lt  # # >= t

        # map k -> tp/fp using cumulative arrays (tp_cum[k-1])
        # if k==0 -> 0
        tp_t = torch.zeros((self.n_thresholds,), device=device, dtype=torch.float32)
        fp_t = torch.zeros((self.n_thresholds,), device=device, dtype=torch.float32)

        nonzero = k_per_t > 0
        if nonzero.any():
            k_nz = k_per_t[nonzero] - 1
            tp_t[nonzero] = tp_cum[k_nz]
            fp_t[nonzero] = fp_cum[k_nz]

        fn_t = float(G) - tp_t  # [T]

        self.tp += tp_t
        self.fp += fp_t
        self.fn += fn_t
        self.num_images += 1.0

    @torch.no_grad()
    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          thresholds: [T]
          fppi: [T]
          sensitivity: [T]
        """
        thresholds = self.thresholds

        # avoid div by 0
        num_images = torch.clamp(self.num_images, min=1.0)
        denom = torch.clamp(self.tp + self.fn, min=1e-12)

        sensitivity = self.tp / denom
        fppi = self.fp / num_images

        # Sort by FPPI increasing for nicer plotting/integration
        order = torch.argsort(fppi)
        return {
            "thresholds": thresholds[order],
            "fppi": fppi[order],
            "sensitivity": sensitivity[order],
        }

    @torch.no_grad()
    def auc(self, fppi_max: float = 8.0) -> torch.Tensor:
        curve = self.compute()
        x = torch.clamp(curve["fppi"], 0.0, float(fppi_max))
        y = curve["sensitivity"]

        # ensure endpoints for stable trapz
        if x.numel() == 0:
            return torch.tensor(0.0, device=self.tp.device)

        if x[0] > 0:
            x = torch.cat([x.new_tensor([0.0]), x])
            y = torch.cat([y[:1], y])
        if x[-1] < fppi_max:
            x = torch.cat([x, x.new_tensor([float(fppi_max)])])
            y = torch.cat([y, y[-1:]])

        area = torch.trapz(y, x) / float(fppi_max)
        return area

    @torch.no_grad()
    def sensitivity_at_fppi(self, targets: Sequence[float] = (0.5, 1, 2, 4, 8)) -> Dict[float, torch.Tensor]:
        curve = self.compute()
        x = curve["fppi"]
        y = curve["sensitivity"]
        out: Dict[float, torch.Tensor] = {}

        if x.numel() == 0:
            for t in targets:
                out[float(t)] = torch.tensor(0.0, device=self.tp.device)
            return out

        for t in targets:
            t = float(t)
            tt = x.new_tensor([t])
            if t <= x[0].item():
                out[t] = y[0]
            elif t >= x[-1].item():
                out[t] = y[-1]
            else:
                idx = torch.searchsorted(x, tt, right=True).item() - 1
                x0, x1 = x[idx], x[idx + 1]
                y0, y1 = y[idx], y[idx + 1]
                w = torch.where(x1 == x0, x0.new_tensor(0.0), (tt[0] - x0) / (x1 - x0))
                out[t] = y0 + w * (y1 - y0)

        return out
