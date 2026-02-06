from __future__ import annotations

from typing import Sequence, Hashable, Mapping, Any, Tuple, Optional, Union, List

import numpy as np
import torch
from monai.transforms import MapTransform
from monai.config import KeysCollection
from monai.utils import ensure_tuple_rep


Percentiles = Optional[Tuple[float, float]]  # e.g. (0.5, 99.5)


def _as_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    # Accept {0,1} or probabilities; treat >0 as brain
    if mask.dtype != torch.bool:
        mask = mask > 0
    return mask


def _masked_percentile_clip_torch(
    x: torch.Tensor,
    mask: torch.Tensor,
    p: Tuple[float, float],
) -> torch.Tensor:
    """Clip x using percentiles computed from x[mask]."""
    vals = x[mask]
    if vals.numel() < 10:
        return x
    lo_p, hi_p = p
    # torch.quantile expects [0,1]
    lo = torch.quantile(vals, lo_p / 100.0)
    hi = torch.quantile(vals, hi_p / 100.0)
    if torch.isfinite(lo) and torch.isfinite(hi) and (hi > lo):
        return x.clamp(lo, hi)
    return x


class MaskedPercentileNormalizeIntensityd(MapTransform):
    """
    Like MONAI NormalizeIntensityd, but:
      - uses a brain mask from another key (mask_key)
      - optionally clips to percentiles computed inside the mask
      - normalizes using mean/std computed inside the mask

    Works with torch.Tensor / MetaTensor. Expects image tensors shaped:
      - [C, H, W, D] or [C, H, W] (channel-first), OR
      - [H, W, D] / [H, W] (treated as single-channel)

    Args:
        keys: image keys to normalize.
        mask_key: key in the dict that contains the brain mask (same spatial size).
        percentiles: e.g. (0.5, 99.5) to clip within mask before computing mean/std.
                     Pass None to disable clipping.
        channel_wise: if True, compute stats per channel; else across all channels jointly.
        eps: numerical stability.
        z_clamp: optional clamp after z-norm, e.g. (-5, 5). None disables.
        dtype: output dtype.
    """

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: Hashable,
        percentiles: Percentiles = (0.5, 99.5),
        channel_wise: bool = True,
        eps: float = 1e-6,
        z_clamp: Optional[Tuple[float, float]] = (-5.0, 5.0),
        dtype: torch.dtype = torch.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.percentiles = percentiles
        self.channel_wise = channel_wise
        self.eps = float(eps)
        self.z_clamp = z_clamp
        self.dtype = dtype

        if self.percentiles is not None:
            lo, hi = self.percentiles
            if not (0.0 <= lo < hi <= 100.0):
                raise ValueError(f"percentiles must satisfy 0 <= lo < hi <= 100, got {self.percentiles}")

    def _normalize_1ch(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # x: [H,W,(D)] and m: [H,W,(D)]
        if self.percentiles is not None:
            x = _masked_percentile_clip_torch(x, m, self.percentiles)

        vals = x[m]
        if vals.numel() < 10:
            return x  # fallback if mask is empty/bad

        mean = vals.mean()
        std = vals.std(unbiased=False).clamp_min(self.eps)
        x = (x - mean) / std

        if self.z_clamp is not None:
            x = x.clamp(self.z_clamp[0], self.z_clamp[1])
        return x

    def _normalize_chw(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # x: [C,H,W,(D)], m: [H,W,(D)]
        c = int(x.shape[0])
        out = x

        if self.channel_wise:
            # Per-channel stats inside same mask
            for ci in range(c):
                out[ci] = self._normalize_1ch(out[ci], m)
            return out

        # Joint stats across all channels but still masked spatially
        # Expand mask to [C,H,W,(D)]
        mC = m.unsqueeze(0).expand_as(out)
        if self.percentiles is not None:
            vals = out[mC]
            if vals.numel() >= 10:
                lo = torch.quantile(vals, self.percentiles[0] / 100.0)
                hi = torch.quantile(vals, self.percentiles[1] / 100.0)
                if torch.isfinite(lo) and torch.isfinite(hi) and (hi > lo):
                    out = out.clamp(lo, hi)
                    mC = m.unsqueeze(0).expand_as(out)

        vals = out[mC]
        if vals.numel() < 10:
            return out

        mean = vals.mean()
        std = vals.std(unbiased=False).clamp_min(self.eps)
        out = (out - mean) / std

        if self.z_clamp is not None:
            out = out.clamp(self.z_clamp[0], self.z_clamp[1])
        return out

    def __call__(self, data: Mapping[Hashable, Any]) -> dict:
        d = dict(data)

        if self.mask_key not in d:
            raise KeyError(f"mask_key='{self.mask_key}' not found in data dict keys={list(d.keys())}")

        mask = d[self.mask_key]
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)

        mask = _as_bool_mask(mask).squeeze()

        for key in self.key_iterator(d):
            x = d[key]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)

            x = x.to(dtype=self.dtype)

            # Handle shapes
            if x.ndim in (2, 3):  # [H,W] or [H,W,D]
                if mask.shape != x.shape:
                    raise ValueError(f"Mask shape {tuple(mask.shape)} != image shape {tuple(x.shape)} for key={key}")
                x = self._normalize_1ch(x, mask)

            elif x.ndim in (3, 4):  # [C,H,W] or [C,H,W,D]
                if x.ndim == 3:  # [C,H,W]
                    spatial = x.shape[1:]
                else:  # [C,H,W,D]
                    spatial = x.shape[1:]
                if tuple(mask.shape) != tuple(spatial):
                    raise ValueError(
                        f"Mask shape {tuple(mask.shape)} != image spatial shape {tuple(spatial)} for key={key}"
                    )
                x = self._normalize_chw(x, mask)

            else:
                raise ValueError(f"Unsupported image ndim={x.ndim} for key={key}, shape={tuple(x.shape)}")

            d[key] = x

        return d
