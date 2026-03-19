import torch
from torch import Tensor
import torchmetrics


class SemanticDice(torchmetrics.Metric):
    """
    Binary semantic Dice accumulated over the whole epoch.

    Expects binary masks:
      pred:   [H, W, D] or [B, H, W, D]
      target: [H, W, D] or [B, H, W, D]
    """
    full_state_update = False

    def __init__(self, empty_score: float = 1.0):
        super().__init__()
        self.empty_score = float(empty_score)

        self.add_state(
            "intersection",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred_sum",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "target_sum",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )

    def update(self, pred: Tensor, target: Tensor) -> None:
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)

        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have same shape, got {tuple(pred.shape)} vs {tuple(target.shape)}"
            )

        pred = pred.bool()
        target = target.bool()

        self.intersection += (pred & target).sum().to(torch.float64)
        self.pred_sum += pred.sum().to(torch.float64)
        self.target_sum += target.sum().to(torch.float64)

    def compute(self) -> Tensor:
        denom = self.pred_sum + self.target_sum
        if denom.item() == 0:
            return torch.tensor(
                self.empty_score,
                device=self.intersection.device,
                dtype=torch.float32,
            )
        return (2.0 * self.intersection / denom).to(torch.float32)
