from __future__ import annotations

from typing import Hashable, Sequence, Tuple, Union
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


def _to_overlap_voxels(patch: Tuple[int, int, int],
                      overlap: Union[float, Sequence[Union[int, float]]]) -> Tuple[int, int, int]:
    """
    Convert overlap to integer voxel overlap per axis.

    overlap can be:
      - float in (0,1): ratio of patch size (same for all axes)
      - sequence of 3 floats in (0,1): per-axis ratios
      - sequence of 3 ints: per-axis voxel overlap
    """
    # scalar ratio
    if isinstance(overlap, float):
        if 0.0 < overlap < 1.0:
            return tuple(int(round(p * overlap)) for p in patch)
        # treat as voxel overlap (rare / not recommended)
        return (int(round(overlap)),) * 3

    ov = tuple(overlap)
    if len(ov) != 3:
        raise ValueError(f"overlap must be float or length-3 sequence, got {overlap}")

    # per-axis ratios
    if all(isinstance(o, float) and 0.0 < o < 1.0 for o in ov):
        return tuple(int(round(p * o)) for p, o in zip(patch, ov))

    # per-axis voxels
    return tuple(int(round(o)) for o in ov)


class SymmetricGridPadWithMind(MapTransform):
    """
    Symmetrically pad so that (dim_padded - patch) is divisible by stride,
    where stride = patch - overlap_voxels.

    Spatial order: (X, Y, Z).
    Tensor: [C,X,Y,Z] or [X,Y,Z].
    Pads with per-sample minimum value.
    """

    def __init__(
        self,
        keys,
        patch_size: Sequence[int],
        overlap: Union[float, Sequence[Union[int, float]]],
        mask_keys=(),
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.mask_keys = mask_keys

        self.patch_size = tuple(int(x) for x in ensure_tuple_rep(patch_size, 3))
        self.overlap_vox = _to_overlap_voxels(self.patch_size, overlap)
        self.stride = tuple(p - o for p, o in zip(self.patch_size, self.overlap_vox))

        if any(s <= 0 for s in self.stride):
            raise ValueError(
                f"Invalid stride={self.stride}. Need patch_size > overlap per axis.\n"
                f"patch_size={self.patch_size}, overlap_vox={self.overlap_vox}"
            )

    @staticmethod
    def _pad_needed(dim: int, patch: int, stride: int) -> int:
        # Need dim_padded >= patch and (dim_padded - patch) % stride == 0 with minimal padding
        if dim < patch:
            return patch - dim
        rem = (dim - patch) % stride
        return 0 if rem == 0 else (stride - rem)

    def __call__(self, data):
        d = dict(data)
        for k in self.key_iterator(d):
            x = d[k]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)

            X, Y, Z = map(int, x.shape[-3:])  # (X,Y,Z)
            px = self._pad_needed(X, self.patch_size[0], self.stride[0])
            py = self._pad_needed(Y, self.patch_size[1], self.stride[1])
            pz = self._pad_needed(Z, self.patch_size[2], self.stride[2])

            if px == py == pz == 0:
                d[k] = x
                continue

            # symmetric split (left/right)
            px0, px1 = px // 2, px - px // 2
            py0, py1 = py // 2, py - py // 2
            pz0, pz1 = pz // 2, pz - pz // 2

            if k in self.mask_keys:
                pad_val = 0
            else:
                pad_val = float(x.min().item())
            pad = (pz0, pz1, py0, py1, px0, px1)  # torch pad order for 3D

            d[k] = torch.nn.functional.pad(x, pad, mode="constant", value=pad_val)

        return d
