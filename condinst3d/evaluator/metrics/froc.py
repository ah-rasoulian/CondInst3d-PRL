from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


def _greedy_match_tp_fp(
    iou: torch.Tensor,          # [K, G]
    scores: torch.Tensor,       # [K]
    iou_thr: float,
    score_thr: float,
) -> Tuple[int, int, int]:
    """
    Greedy one-to-one matching:
      - sort predictions by descending score
      - each prediction matches best currently-unmatched GT
      - TP if best IoU >= iou_thr else FP
    Returns: (tp, fp, fn) for this image at this score_thr
    """
    device = iou.device
    K = int(iou.shape[0])
    G = int(iou.shape[1])

    # filter by score threshold
    keep = scores >= score_thr
    if keep.any():
        sel_scores = scores[keep]
        sel_iou = iou[keep]
    else:
        sel_scores = scores.new_zeros((0,))
        sel_iou = iou.new_zeros((0, G))

    # no GT edge case
    if G == 0:
        tp = 0
        fp = int(sel_scores.numel())
        fn = 0
        return tp, fp, fn

    # no predictions edge case
    if sel_scores.numel() == 0:
        tp = 0
        fp = 0
        fn = G
        return tp, fp, fn

    order = torch.argsort(sel_scores, descending=True)
    sel_iou = sel_iou[order]

    matched_gt = torch.zeros((G,), dtype=torch.bool, device=device)

    tp = 0
    fp = 0
    for p in range(sel_iou.shape[0]):
        # best GT for this prediction
        best_iou, best_g = torch.max(sel_iou[p], dim=0)
        if best_iou >= iou_thr and (not matched_gt[best_g]):
            matched_gt[best_g] = True
            tp += 1
        else:
            fp += 1

    fn = int((~matched_gt).sum().item())
    return tp, fp, fn


@dataclass
class DetectionFROC:
    """
    FROC metric for detection/instance segmentation.
    Stores per-image pairwise IoU and scores, then computes:
      - FPPI (false positives per image) vs Sensitivity (TPR) curve
      - Optional AUC over FPPI range
    """
    iou_thr: float = 0.1
    n_thresholds: int = 200
    score_min: float = 0.0
    score_max: float = 1.0
    device: str = "cpu"

    # internal buffers (kept on CPU by default)
    _ious: List[torch.Tensor] = field(default_factory=list)
    _scores: List[torch.Tensor] = field(default_factory=list)
    _num_gt: List[int] = field(default_factory=list)

    def reset(self) -> None:
        self._ious.clear()
        self._scores.clear()
        self._num_gt.clear()

    @torch.no_grad()
    def update(self, pairwise_iou: torch.Tensor, scores: torch.Tensor) -> None:
        """
        pairwise_iou: [K, G] IoU between predicted instances and GT instances for ONE image
        scores:       [K]    confidence scores for predicted instances for ONE image
        """
        if pairwise_iou.ndim != 2:
            raise ValueError(f"pairwise_iou must be [K,G], got {tuple(pairwise_iou.shape)}")
        if scores.ndim != 1:
            raise ValueError(f"scores must be [K], got {tuple(scores.shape)}")
        if pairwise_iou.shape[0] != scores.shape[0]:
            raise ValueError("K mismatch: pairwise_iou.shape[0] must equal scores.shape[0]")

        # move to metric device (CPU recommended)
        self._ious.append(pairwise_iou.detach().to(self.device))
        self._scores.append(scores.detach().to(self.device))
        self._num_gt.append(int(pairwise_iou.shape[1]))

    @torch.no_grad()
    def compute_curve(self) -> Dict[str, torch.Tensor]:
        """
        Returns:
          thresholds: [T]
          fppi:       [T]
          sensitivity:[T]
        """
        if len(self._ious) == 0:
            return {
                "thresholds": torch.empty((0,)),
                "fppi": torch.empty((0,)),
                "sensitivity": torch.empty((0,)),
            }

        thresholds = torch.linspace(self.score_min, self.score_max, self.n_thresholds, device=self.device)

        total_images = len(self._ious)
        total_gt = sum(self._num_gt)

        # Handle case with no GT at all: sensitivity undefined; return zeros
        if total_gt == 0:
            fppi = torch.zeros_like(thresholds)
            sens = torch.zeros_like(thresholds)
            return {"thresholds": thresholds, "fppi": fppi, "sensitivity": sens}

        fppi_list = []
        sens_list = []

        for thr in thresholds:
            tp = 0
            fp = 0
            fn = 0
            for iou, sc in zip(self._ious, self._scores):
                tpi, fpi, fni = _greedy_match_tp_fp(iou, sc, self.iou_thr, float(thr.item()))
                tp += tpi
                fp += fpi
                fn += fni

            # Sensitivity = TP / (TP + FN) = TP / total_gt
            sensitivity = tp / (tp + fn + 1e-12)
            fppi = fp / float(total_images)

            fppi_list.append(fppi)
            sens_list.append(sensitivity)

        fppi_t = torch.tensor(fppi_list, device=self.device, dtype=torch.float32)
        sens_t = torch.tensor(sens_list, device=self.device, dtype=torch.float32)

        # Sort by FPPI increasing (nice for plotting/AUC)
        order = torch.argsort(fppi_t)
        return {
            "thresholds": thresholds[order],
            "fppi": fppi_t[order],
            "sensitivity": sens_t[order],
        }

    @torch.no_grad()
    def compute_auc(self, fppi_max: float = 8.0) -> float:
        """
        Area under FROC curve from FPPI=0 to FPPI=fppi_max using trapezoidal rule.
        """
        curve = self.compute_curve()
        x = curve["fppi"]
        y = curve["sensitivity"]
        if x.numel() == 0:
            return 0.0

        # clamp to [0, fppi_max]
        x_clamped = torch.clamp(x, 0.0, float(fppi_max))
        # ensure monotonic increasing x (already sorted)
        # remove duplicates in x for stable integration
        # (optional) small epsilon jitter:
        # but simplest: unique via grouping
        xu, idx = torch.unique_consecutive(x_clamped, return_inverse=False, return_counts=False, dim=0)
        if xu.numel() != x_clamped.numel():
            # recompute y by taking max sensitivity for same FPPI (conservative)
            # group consecutive identical x
            new_x = []
            new_y = []
            start = 0
            while start < x_clamped.numel():
                end = start + 1
                while end < x_clamped.numel() and x_clamped[end] == x_clamped[start]:
                    end += 1
                new_x.append(x_clamped[start])
                new_y.append(torch.max(y[start:end]))
                start = end
            x_clamped = torch.stack(new_x)
            y = torch.stack(new_y)

        # append (0, y_at_0) and (fppi_max, y_at_max) if needed
        if x_clamped[0] > 0:
            x_clamped = torch.cat([torch.tensor([0.0], device=self.device), x_clamped])
            y = torch.cat([y[:1], y])
        if x_clamped[-1] < fppi_max:
            x_clamped = torch.cat([x_clamped, torch.tensor([fppi_max], device=self.device)])
            y = torch.cat([y, y[-1:]])

        auc = torch.trapz(y, x_clamped).item() / float(fppi_max)
        return float(auc)

    @torch.no_grad()
    def sensitivity_at_fppi(self, targets: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0)) -> Dict[float, float]:
        """
        Interpolate sensitivity at given FPPI points.
        """
        curve = self.compute_curve()
        x = curve["fppi"]
        y = curve["sensitivity"]
        if x.numel() == 0:
            return {float(t): 0.0 for t in targets}

        out: Dict[float, float] = {}
        for t in targets:
            t = float(t)
            if t <= x[0].item():
                out[t] = float(y[0].item())
            elif t >= x[-1].item():
                out[t] = float(y[-1].item())
            else:
                # find rightmost x <= t
                idx = torch.searchsorted(x, torch.tensor([t], device=self.device), right=True).item() - 1
                x0, x1 = x[idx].item(), x[idx + 1].item()
                y0, y1 = y[idx].item(), y[idx + 1].item()
                # linear interp
                w = 0.0 if x1 == x0 else (t - x0) / (x1 - x0)
                out[t] = float(y0 + w * (y1 - y0))
        return out
