from __future__ import annotations

from typing import Hashable, Sequence, Tuple
import torch
from monai.transforms import MapTransform
from monai.config import KeysCollection
from monai.utils import ensure_tuple, ensure_tuple_rep


class SpatialPadWithMind(MapTransform):
    """
    Pads tensors to spatial_size.
    - Image keys: pad with per-sample minimum value
    - Mask keys: pad with 0
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size,
        mask_keys=(),
        allow_missing_keys=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.spatial_size = ensure_tuple(spatial_size)
        self.mask_keys = set(mask_keys)

    def __call__(self, data):
        d = dict(data)

        for k in self.key_iterator(d):
            x = d[k]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)

            # x shape: [C,X,Y,Z] or [X,Y,Z]
            has_c = (x.ndim == 4)
            spatial = x.shape[-3:]
            target = self.spatial_size

            pad_x = max(0, target[0] - spatial[0])
            pad_y = max(0, target[1] - spatial[1])
            pad_z = max(0, target[2] - spatial[2])

            if pad_x == pad_y == pad_z == 0:
                d[k] = x
                continue

            px0, px1 = pad_x // 2, pad_x - pad_x // 2
            py0, py1 = pad_y // 2, pad_y - pad_y // 2
            pz0, pz1 = pad_z // 2, pad_z - pad_z // 2

            pad = (pz0, pz1, py0, py1, px0, px1)  # torch order

            if k in self.mask_keys:
                pad_val = 0
            else:
                pad_val = float(x.min().item())

            x = torch.nn.functional.pad(x, pad, mode="constant", value=pad_val)
            d[k] = x

        return d


class SymmetricGridPadWithMind(MapTransform):
    """
    Symmetrically pad a volume so GridPatchd produces patches with equal padding at beginning/end.

    Assumes spatial order: (X, Y, Z) and tensor shape [C, X, Y, Z] or [X, Y, Z].
    Pads with per-sample minimum value (like a hypothetical pad_mode="minimum").

    Padding amount is chosen so that:
        (dim_padded - patch) is divisible by stride,
    where stride = patch - overlap.

    Args:
        keys: keys to pad (usually ["inputs"]).
        patch_size: patch size (x,y,z).
        overlap: overlap size (x,y,z). stride = patch - overlap.
        mode: currently only "min" supported (pads with min). Kept for clarity.
    """

    def __init__(
        self,
        keys,
        patch_size: Sequence[int],
        overlap: Sequence[int],
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.patch_size = ensure_tuple_rep(patch_size, 3)
        self.overlap = ensure_tuple_rep(overlap, 3)

        # stride = patch - overlap
        self.stride = tuple(int(p - o) for p, o in zip(self.patch_size, self.overlap))
        if any(s <= 0 for s in self.stride):
            raise ValueError(f"Invalid stride={self.stride}. Need patch_size > overlap per axis.")

    @staticmethod
    def _pad_needed(dim: int, patch: int, stride: int) -> int:
        # If dim < patch, pad up to patch.
        if dim <= patch:
            return patch - dim
        # Need (dim_padded - patch) % stride == 0
        rem = (dim - patch) % stride
        return 0 if rem == 0 else (stride - rem)

    def __call__(self, data):
        d = dict(data)
        for k in self.key_iterator(d):
            x = d[k]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)

            has_c = (x.ndim == 4)
            spatial = x.shape[-3:]  # (X,Y,Z)
            X, Y, Z = map(int, spatial)

            pad_x = self._pad_needed(X, int(self.patch_size[0]), int(self.stride[0]))
            pad_y = self._pad_needed(Y, int(self.patch_size[1]), int(self.stride[1]))
            pad_z = self._pad_needed(Z, int(self.patch_size[2]), int(self.stride[2]))

            # split symmetrically
            px0, px1 = pad_x // 2, pad_x - pad_x // 2
            py0, py1 = pad_y // 2, pad_y - pad_y // 2
            pz0, pz1 = pad_z // 2, pad_z - pad_z // 2

            if pad_x == pad_y == pad_z == 0:
                d[k] = x
                continue

            pad_val = float(x.min().item())
            pad = (pz0, pz1, py0, py1, px0, px1)  # torch pad order for 3D

            x = torch.nn.functional.pad(x, pad, mode="constant", value=pad_val)
            d[k] = x

        return d