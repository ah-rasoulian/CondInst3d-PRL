from __future__ import annotations
import torch
import torch.nn as nn


def dice_coefficient(x, target, fg_weight=1.0):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)

    weight_mask = torch.ones_like(target)
    weight_mask[target == 1] = fg_weight

    intersection = (weight_mask * x * target).sum(dim=1)
    union = (weight_mask * (x ** 2.0)).sum(dim=1) + (weight_mask * (target ** 2.0)).sum(dim=1)
    loss = 1. - (2 * intersection + eps) / (union + eps)
    return loss


class DiceLoss(nn.Module):
    """
    Dice loss for instance masks.

    Expects:
      x:      Tensor of shape [N, ...] (probabilities in [0,1] OR logits if from_logits=True)
      target: Tensor of shape [N, ...] (binary {0,1})

    Returns:
      scalar if reduction in {"mean","sum"} else tensor of shape [N]
    """
    def __init__(
        self,
        fg_weight: float = 1.0,
        eps: float = 1e-5,
        reduction: str = "mean",
        from_logits: bool = False,
    ):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be one of: none|mean|sum, got {reduction}")
        self.fg_weight = float(fg_weight)
        self.eps = float(eps)
        self.reduction = reduction
        self.from_logits = bool(from_logits)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            x = torch.sigmoid(x)

        if x.shape != target.shape:
            raise ValueError(f"x and target must have the same shape. Got x={x.shape}, target={target.shape}")

        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1).to(dtype=x.dtype)

        weight_mask = torch.ones_like(target)
        if self.fg_weight != 1.0:
            weight_mask = weight_mask + (target == 1).to(weight_mask.dtype) * (self.fg_weight - 1.0)

        intersection = (weight_mask * x * target).sum(dim=1)
        union = (weight_mask * (x ** 2.0)).sum(dim=1) + (weight_mask * (target ** 2.0)).sum(dim=1)

        loss = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
