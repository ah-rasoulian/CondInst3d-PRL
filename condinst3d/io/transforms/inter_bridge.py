import torch
import torch.nn.functional as F
from torch import Tensor

class InterSliceBridgeBandLimited:
    """
    Channel-first [C,X,Y,Z] inter-slice bridging along Z, but ONLY near the current mask.
    This prevents filling ring holes.

    add = bridge & ~curr
    add = add & dilate(curr, band_k)   # band constraint
    optional: tiny 2D closing on add (usually keep small or off)
    """

    def __init__(
        self,
        mode: str = "union",           # "union" or "intersection"
        band_k: int = 5,               # dilation kernel on curr to define where we are allowed to add
        add_close_k: int = 1,          # closing kernel on add-mask (1 disables)
        min_neighbor_area: int = 0,
    ):
        assert mode in ("union", "intersection")
        self.mode = mode
        self.band_k = int(band_k)
        self.add_close_k = int(add_close_k)
        self.min_neighbor_area = int(min_neighbor_area)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.ndim != 4:
            raise ValueError(f"Expected [C,X,Y,Z], got {tuple(x.shape)}")

        C, X, Y, Z = x.shape
        if Z < 3:
            return x

        orig_dtype = x.dtype
        m = x > 0
        out = m.clone()

        for c in range(C):
            for z in range(1, Z - 1):
                prev = m[c, :, :, z - 1]
                curr = m[c, :, :, z]
                nxt  = m[c, :, :, z + 1]

                if self.min_neighbor_area > 0:
                    if int(prev.sum()) < self.min_neighbor_area and int(nxt.sum()) < self.min_neighbor_area:
                        continue

                bridge = (prev | nxt) if self.mode == "union" else (prev & nxt)
                add = bridge & (~curr)

                if not bool(add.any()):
                    continue

                # Band-limit: only allow additions near the current mask
                if bool(curr.any()):
                    band = self._binary_dilate_2d(curr, k=self.band_k)
                    add = add & band
                else:
                    # if curr is empty, do nothing (prevents hallucinating a slice)
                    continue

                if self.add_close_k > 1 and bool(add.any()):
                    add = self._binary_closing_2d(add, k=self.add_close_k)

                out[c, :, :, z] = curr | add

        if orig_dtype == torch.bool:
            return out
        return out.to(orig_dtype)

    @staticmethod
    def _binary_dilate_2d(mask_xy: Tensor, k: int) -> Tensor:
        if k <= 1:
            return mask_xy
        pad = k // 2
        x = mask_xy[None, None].float()
        w = torch.ones((1, 1, k, k), device=mask_xy.device, dtype=x.dtype)
        y = F.conv2d(x, w, padding=pad)
        return (y[0, 0] > 0)

    @staticmethod
    def _binary_erode_2d(mask_xy: Tensor, k: int) -> Tensor:
        if k <= 1:
            return mask_xy
        pad = k // 2
        x = mask_xy[None, None].float()
        w = torch.ones((1, 1, k, k), device=mask_xy.device, dtype=x.dtype)
        y = F.conv2d(x, w, padding=pad)
        return (y[0, 0] >= (k * k))

    @classmethod
    def _binary_closing_2d(cls, mask_xy: Tensor, k: int) -> Tensor:
        return cls._binary_erode_2d(cls._binary_dilate_2d(mask_xy, k), k)
