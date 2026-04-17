from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torchmetrics


@torch.no_grad()
def _greedy_match_sorted(
    iou_sorted: torch.Tensor,  # [K, G], predictions already sorted by score desc
    iou_thr: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Greedy 1-1 matching along score-sorted predictions.

    Returns:
      tp_inc: [K] float tensor with per-prediction TP increments {0,1}
      fp_inc: [K] float tensor with per-prediction FP increments {0,1}
      G:      number of GT objects
    """
    device = iou_sorted.device
    K, G = int(iou_sorted.shape[0]), int(iou_sorted.shape[1])

    if K == 0:
        return (
            torch.zeros((0,), device=device, dtype=torch.float32),
            torch.zeros((0,), device=device, dtype=torch.float32),
            G,
        )

    if G == 0:
        tp_inc = torch.zeros((K,), device=device, dtype=torch.float32)
        fp_inc = torch.ones((K,), device=device, dtype=torch.float32)
        return tp_inc, fp_inc, 0

    matched_gt = torch.zeros((G,), device=device, dtype=torch.bool)
    tp_inc = torch.zeros((K,), device=device, dtype=torch.float32)
    fp_inc = torch.zeros((K,), device=device, dtype=torch.float32)

    for k in range(K):
        ious = iou_sorted[k].masked_fill(matched_gt, -1.0)
        best_iou, best_g = torch.max(ious, dim=0)

        if best_iou >= iou_thr:
            matched_gt[best_g] = True
            tp_inc[k] = 1.0
        else:
            fp_inc[k] = 1.0

    return tp_inc, fp_inc, G


class DetectionFROC(torchmetrics.Metric):
    """
    FROC: Sensitivity vs FPPI (false positives per image)

    Accumulates fixed-size TP / FP / FN arrays over a score threshold grid.
    DDP-safe because states have fixed size and use sum reduction.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        iou_thr: float = 0.10,
        n_thresholds: int = 200,
        score_min: float = 0.0,
        score_max: float = 1.0,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if n_thresholds < 2:
            raise ValueError("n_thresholds must be >= 2")
        if score_max <= score_min:
            raise ValueError("score_max must be > score_min")

        self.iou_thr = float(iou_thr)
        self.n_thresholds = int(n_thresholds)
        self.score_min = float(score_min)
        self.score_max = float(score_max)

        thresholds = torch.linspace(self.score_min, self.score_max, self.n_thresholds)
        self.register_buffer("thresholds", thresholds)

        self.add_state("tp", default=torch.zeros(self.n_thresholds, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(self.n_thresholds, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(self.n_thresholds, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("num_images", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, pairwise_iou: torch.Tensor, scores: torch.Tensor) -> None:
        """
        Args:
            pairwise_iou: [K, G]
            scores:       [K]
        """
        if pairwise_iou.ndim != 2:
            raise ValueError(f"pairwise_iou must be [K, G], got {tuple(pairwise_iou.shape)}")
        if scores.ndim != 1:
            raise ValueError(f"scores must be [K], got {tuple(scores.shape)}")
        if pairwise_iou.shape[0] != scores.shape[0]:
            raise ValueError("pairwise_iou.shape[0] must equal scores.shape[0]")

        device = self.tp.device
        pairwise_iou = pairwise_iou.to(device=device, dtype=torch.float32)
        scores = scores.to(device=device, dtype=torch.float32)

        K = int(scores.numel())

        if K > 0:
            order = torch.argsort(scores, descending=True)
            scores_sorted = scores[order]
            iou_sorted = pairwise_iou[order]
        else:
            scores_sorted = scores
            iou_sorted = pairwise_iou

        tp_inc, fp_inc, G = _greedy_match_sorted(iou_sorted, self.iou_thr)

        tp_cum = torch.cumsum(tp_inc, dim=0)  # [K]
        fp_cum = torch.cumsum(fp_inc, dim=0)  # [K]

        if K == 0:
            k_per_t = torch.zeros_like(self.thresholds, dtype=torch.long)
        else:
            scores_asc = torch.flip(scores_sorted, dims=(0,))  # ascending
            num_lt = torch.searchsorted(scores_asc, self.thresholds, right=False)
            k_per_t = K - num_lt  # number of predictions with score >= t

        tp_t = torch.zeros((self.n_thresholds,), device=device, dtype=torch.float32)
        fp_t = torch.zeros((self.n_thresholds,), device=device, dtype=torch.float32)

        nonzero = k_per_t > 0
        if nonzero.any():
            idx = k_per_t[nonzero] - 1
            tp_t[nonzero] = tp_cum[idx]
            fp_t[nonzero] = fp_cum[idx]

        fn_t = float(G) - tp_t

        self.tp += tp_t
        self.fp += fp_t
        self.fn += fn_t
        self.num_images += 1.0

    @torch.no_grad()
    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Returns:
          {
            "thresholds": [M],
            "fppi": [M],
            "sensitivity": [M],
          }

        The returned curve is cleaned for plotting:
        - sorted by FPPI
        - duplicate FPPI values collapsed by max sensitivity
        - sensitivity made monotone non-decreasing
        - (0, 0) prepended when needed
        """
        num_images = torch.clamp(self.num_images, min=1.0)
        denom = torch.clamp(self.tp + self.fn, min=1e-12)

        sensitivity = self.tp / denom
        fppi = self.fp / num_images
        thresholds = self.thresholds

        order = torch.argsort(fppi)
        fppi = fppi[order]
        sensitivity = sensitivity[order]
        thresholds = thresholds[order]

        if fppi.numel() == 0:
            return {
                "thresholds": thresholds.reshape(-1),
                "fppi": fppi.reshape(-1),
                "sensitivity": sensitivity.reshape(-1),
            }

        unique_fppi, inverse = torch.unique_consecutive(fppi, return_inverse=True)
        sens_collapsed = torch.zeros_like(unique_fppi)
        thr_collapsed = torch.zeros_like(unique_fppi)

        for i in range(unique_fppi.numel()):
            mask = inverse == i
            sens_i = sensitivity[mask]
            thr_i = thresholds[mask]
            best_idx = torch.argmax(sens_i)
            sens_collapsed[i] = sens_i[best_idx]
            thr_collapsed[i] = thr_i[best_idx]

        sens_collapsed = torch.cummax(sens_collapsed, dim=0).values

        if unique_fppi[0] > 0:
            unique_fppi = torch.cat([unique_fppi.new_tensor([0.0]), unique_fppi], dim=0)
            sens_collapsed = torch.cat([sens_collapsed.new_tensor([0.0]), sens_collapsed], dim=0)
            thr_collapsed = torch.cat([thr_collapsed[:1], thr_collapsed], dim=0)

        return {
            "thresholds": thr_collapsed.reshape(-1),
            "fppi": unique_fppi.reshape(-1),
            "sensitivity": sens_collapsed.reshape(-1),
        }

    @torch.no_grad()
    def auc(self, fppi_max: float = 8.0) -> torch.Tensor:
        curve = self.compute()
        x = curve["fppi"]
        y = curve["sensitivity"]

        if x.numel() == 0:
            return torch.tensor(0.0, device=self.tp.device)

        keep = x <= float(fppi_max)
        x = x[keep]
        y = y[keep]

        if x.numel() == 0:
            x = torch.tensor([0.0], device=self.tp.device)
            y = torch.tensor([0.0], device=self.tp.device)

        if x[0] > 0:
            x = torch.cat([x.new_tensor([0.0]), x], dim=0)
            y = torch.cat([y.new_tensor([0.0]), y], dim=0)

        if x[-1] < fppi_max:
            x = torch.cat([x, x.new_tensor([float(fppi_max)])], dim=0)
            y = torch.cat([y, y[-1:]], dim=0)

        return torch.trapz(y, x) / float(fppi_max)

    @torch.no_grad()
    def sensitivity_at_fppi(
            self,
            targets: Sequence[float] = (0.5, 1, 2, 4, 8),
    ) -> Dict[float, torch.Tensor]:
        curve = self.compute()
        x = curve["fppi"].reshape(-1)
        y = curve["sensitivity"].reshape(-1)

        out: Dict[float, torch.Tensor] = {}

        if x.numel() == 0:
            zero = torch.tensor(0.0, device=self.tp.device)
            for t in targets:
                out[float(t)] = zero
            return out

        if y.numel() != x.numel():
            raise RuntimeError(
                f"FROC curve shape mismatch: fppi has shape {tuple(x.shape)}, "
                f"sensitivity has shape {tuple(y.shape)}"
            )

        for t in targets:
            t = float(t)
            tt = x.new_tensor([t])

            if t <= float(x[0].item()):
                out[t] = y[0]
            elif t >= float(x[-1].item()):
                out[t] = y[-1]
            else:
                idx = int(torch.searchsorted(x, tt, right=True).item()) - 1
                idx = max(0, min(idx, x.numel() - 2))

                x0, x1 = x[idx], x[idx + 1]
                y0, y1 = y[idx], y[idx + 1]

                if float(x1.item()) == float(x0.item()):
                    out[t] = y0
                else:
                    w = (tt[0] - x0) / (x1 - x0)
                    out[t] = y0 + w * (y1 - y0)

        return out