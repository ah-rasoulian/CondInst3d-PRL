import torch
import torch.nn as nn
from dataclasses import dataclass
from condinst3d.losses.dice import dice_coefficient


@dataclass
class DiceBCELoss(nn.Module):
    """
    Dice + BCEWithLogits loss for instance masks.

    pred_logits: [N, 1, W, H, D] (logits)  OR probabilities if from_logits=False
    target:      same shape, binary {0,1}

    Returns scalar by default (reduction="mean") or per-instance vector (reduction="none").
    """
    lambda_dice: float = 1.0
    lambda_bce: float = 0.2

    # dice params
    fg_weight: float = 1.0
    eps: float = 1e-5

    # bce params
    pos_weight: float | None = None  # >1 boosts positives; None disables

    reduction: str = "mean"          # none|mean|sum
    from_logits: bool = True         # True if pred are logits (recommended)

    def __post_init__(self):
        super().__init__()
        if self.reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be one of none|mean|sum, got {self.reduction}")

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_logits = pred_logits.float()
        target = target.float()

        if pred_logits.shape != target.shape:
            raise ValueError(f"pred and target must have same shape. Got {pred_logits.shape} vs {target.shape}")

        # ---- Dice on probabilities ----
        probs = torch.sigmoid(pred_logits) if self.from_logits else pred_logits
        dice = dice_coefficient(probs, target, fg_weight=self.fg_weight, eps=self.eps)  # [N]

        # ---- BCE (per-voxel), reduced to per-instance ----
        if self.from_logits:
            if self.pos_weight is None:
                bce_vox = nn.functional.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
            else:
                # pos_weight must be a tensor on the right device
                pw = torch.tensor(float(self.pos_weight), device=pred_logits.device, dtype=pred_logits.dtype)
                bce_vox = nn.functional.binary_cross_entropy_with_logits(
                    pred_logits, target, reduction="none", pos_weight=pw
                )
        else:
            # if from_logits=False, treat input as probabilities
            bce_vox = nn.functional.binary_cross_entropy(pred_logits, target, reduction="none")

        # average BCE over spatial dims -> [N]
        bce = bce_vox.flatten(1).mean(dim=1)

        loss = self.lambda_dice * dice + self.lambda_bce * bce  # [N]

        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()