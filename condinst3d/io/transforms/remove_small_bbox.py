from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union, Mapping, Hashable, Any

import torch
from monai.transforms import Transform, MapTransform
from monai.config import KeysCollection


def _bbox_3d_torch(fg: torch.Tensor):
    """
    fg: bool tensor [H,W,D]
    Returns (h_size, w_size, d_size) or None if empty.
    """
    coords = torch.nonzero(fg, as_tuple=False)
    if coords.numel() == 0:
        return None

    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values + 1

    sizes = maxs - mins
    return sizes  # (H, W, D)


@dataclass
class RemoveSmallBBox(Transform):
    min_size: Union[int, Tuple[int, int, int]]
    threshold: float = 0.0
    channel_wise: bool = False

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        x = img

        if not torch.is_tensor(x):
            raise TypeError("RemoveSmallBBox expects torch.Tensor input")

        min_h, min_w, min_d = self._parse_min_size()

        if x.ndim == 3:
            fg = x > self.threshold
            sizes = _bbox_3d_torch(fg)
            if sizes is None:
                return x
            h, w, d = sizes
            if (h < min_h) or (w < min_w) or (d < min_d):
                return torch.zeros_like(x)
            return x

        elif x.ndim == 4:  # [C,H,W,D]
            if self.channel_wise:
                out = x.clone()
                for c in range(x.shape[0]):
                    fg = x[c] > self.threshold
                    sizes = _bbox_3d_torch(fg)
                    if sizes is None:
                        continue
                    h, w, d = sizes
                    if (h < min_h) or (w < min_w) or (d < min_d):
                        out[c] = 0
                return out
            else:
                fg = (x > self.threshold).any(dim=0)
                sizes = _bbox_3d_torch(fg)
                if sizes is None:
                    return x
                h, w, d = sizes
                if (h < min_h) or (w < min_w) or (d < min_d):
                    return torch.zeros_like(x)
                return x

        else:
            raise ValueError(f"Expected [H,W,D] or [C,H,W,D], got {x.shape}")

    def _parse_min_size(self):
        if isinstance(self.min_size, int):
            return self.min_size, self.min_size, self.min_size
        if isinstance(self.min_size, (tuple, list)) and len(self.min_size) == 3:
            return int(self.min_size[0]), int(self.min_size[1]), int(self.min_size[2])
        raise ValueError("min_size must be int or 3-tuple")


class RemoveSmallBBoxd(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            min_size: Union[int, Tuple[int, int, int]],
            threshold: float = 0.0,
            channel_wise: bool = False,
            allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.t = RemoveSmallBBox(
            min_size=min_size,
            threshold=threshold,
            channel_wise=channel_wise,
        )

    def __call__(self, data: Mapping[Hashable, Any]):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.t(d[key])
        return d
