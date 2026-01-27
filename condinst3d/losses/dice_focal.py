from __future__ import annotations

from dataclasses import dataclass
import torch

from .focal import FocalLoss
from .dice import dice_coefficient  # <-- your existing function


@dataclass
class DiceFocalLoss:
    lambda_dice: float = 1.0
    lambda_focal: float = 1.0
    fg_weight: float = 1.0
    reduction: str = "mean"  # none|mean|sum

    # focal params
    alpha: float = 0.5
    gamma: float = 2.0

    def __post_init__(self):
        # keep focal reduction "none" to match your per-sample averaging behavior
        self._focal = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")

    def __call__(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_logits = pred_logits.float()
        target = target.float()

        # dice on probabilities (same as your old code)
        dice = dice_coefficient(torch.sigmoid(pred_logits), target, fg_weight=self.fg_weight)

        # focal on logits
        focal = self._focal(pred_logits, target)  # same shape as pred/target
        # average over spatial dims (B, C, W, H, D) -> (B,)
        # adjust dims if your layout differs
        focal = torch.mean(focal, dim=tuple(range(1, focal.ndim)))

        loss = self.lambda_dice * dice + self.lambda_focal * focal

        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()

        raise ValueError(f"Invalid reduction={self.reduction}. Use: none|mean|sum")
