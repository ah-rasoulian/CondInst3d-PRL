import torch
import torch.nn as nn
from monai.losses import GeneralizedDiceLoss


class GeneralizedDiceBCELoss(nn.Module):
    """
    Combination of MONAI GeneralizedDiceLoss and BCEWithLogitsLoss
    for binary or multi-channel segmentation.

    Expected shapes:
        input:  [B, C, H, W, D] logits
        target: [B, 1, H, W, D] or [B, C, H, W, D]

    Notes:
    - For binary sigmoid segmentation with one output channel, use:
        sigmoid=True, softmax=False
    - For multi-class segmentation, use:
        sigmoid=False, softmax=True
    - BCE term is applied channel-wise and expects target to match input shape.
      If target has shape [B,1,...] and input has C>1, it will be one-hot encoded.
    """

    def __init__(
        self,
        include_background: bool = True,
        sigmoid: bool = True,
        softmax: bool = False,
        w_type: str = "square",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        lambda_gdice: float = 1.0,
        lambda_bce: float = 1.0,
        pos_weight: float | list[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if sigmoid and softmax:
            raise ValueError("sigmoid and softmax cannot both be True.")

        if lambda_gdice < 0 or lambda_bce < 0:
            raise ValueError("lambda_gdice and lambda_bce must be non-negative.")

        self.sigmoid = sigmoid
        self.softmax = softmax
        self.lambda_gdice = float(lambda_gdice)
        self.lambda_bce = float(lambda_bce)
        self.reduction = reduction
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

        self.gdice = GeneralizedDiceLoss(
            include_background=include_background,
            sigmoid=sigmoid,
            softmax=softmax,
            w_type=w_type,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )

        if pos_weight is None:
            self.register_buffer("_pos_weight", None, persistent=False)
        else:
            pw = torch.as_tensor(pos_weight, dtype=torch.float32)
            if pw.ndim == 0:
                pw = pw.view(1)
            self.register_buffer("_pos_weight", pw, persistent=False)

    def _expand_target(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Convert target to the shape expected by BCE:
        - if input is single-channel, target becomes float with same shape
        - if input has multiple channels and target is [B,1,...], convert to one-hot
        """
        if target.ndim != input.ndim:
            raise ValueError(
                f"target ndim ({target.ndim}) must match input ndim ({input.ndim}). "
                f"Got target shape={tuple(target.shape)}, input shape={tuple(input.shape)}."
            )

        if input.shape[1] == target.shape[1]:
            return target.float()

        if target.shape[1] != 1:
            raise ValueError(
                "Target channel dimension must be 1 or match input channels. "
                f"Got input shape={tuple(input.shape)}, target shape={tuple(target.shape)}."
            )

        if input.shape[1] == 1:
            return target.float()

        # Multi-class one-hot expansion: [B,1,...] -> [B,C,...]
        target_long = target.long().squeeze(1)
        one_hot = torch.nn.functional.one_hot(target_long, num_classes=input.shape[1])

        # Move class axis to channel axis
        dims = list(range(one_hot.ndim))
        one_hot = one_hot.permute(0, one_hot.ndim - 1, *dims[1:-1]).contiguous()
        return one_hot.float()

    def _compute_bce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = self._expand_target(input, target)

        pos_weight = self._pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=input.device, dtype=input.dtype)

            # For binary single-channel case, length-1 tensor is fine.
            # For multi-channel BCE, length must match channel count.
            if pos_weight.numel() == 1 and input.shape[1] > 1:
                pos_weight = pos_weight.repeat(input.shape[1])

            if pos_weight.numel() != input.shape[1]:
                raise ValueError(
                    f"pos_weight must have length 1 or match input channels ({input.shape[1]}). "
                    f"Got shape {tuple(pos_weight.shape)}."
                )

        return torch.nn.functional.binary_cross_entropy_with_logits(
            input,
            target,
            pos_weight=pos_weight,
            reduction=self.reduction,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = input.new_tensor(0.0)

        if self.lambda_gdice > 0:
            gdice_loss = self.gdice(input, target)
            loss = loss + self.lambda_gdice * gdice_loss

        if self.lambda_bce > 0:
            bce_loss = self._compute_bce(input, target)
            loss = loss + self.lambda_bce * bce_loss

        return loss
