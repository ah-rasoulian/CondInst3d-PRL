from typing import Optional, Sequence, Hashable, Mapping, Dict, Any
import torch
from monai.transforms import MapTransform
from monai.config import KeysCollection


class InstanceMaskToOneHotd(MapTransform):
    """
    Convert an instance label map (H,W,D) or (1,H,W,D) with integer instance IDs into
    onehot masks: (K,1,H,W,D) by default.

    - instance ID 0 is treated as background.
    - Output is bool by default (saves memory).
    """

    def __init__(
        self,
        keys: KeysCollection,
        out_key: str = "onehot",
        include_background: bool = False,
        max_instances: int = -1,
        dtype: torch.dtype = torch.bool,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.out_key = out_key
        self.include_background = include_background
        self.max_instances = max_instances
        self.dtype = dtype

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)

        # this transform supports a single key in `keys`
        key = self.keys[0] if isinstance(self.keys, (list, tuple)) else self.keys
        inst = d[key]

        # inst may be MetaTensor; torch ops still work
        inst_t = torch.as_tensor(inst)

        # accept [H,W,D] or [1,H,W,D]; output will always be [H,W,D]
        if inst_t.ndim == 4 and inst_t.shape[0] == 1:
            inst_t = inst_t[0]
        elif inst_t.ndim != 3:
            raise ValueError(f"Expected instance mask of shape [H,W,D] or [1,H,W,D], got {tuple(inst_t.shape)}")

        # unique ids (on same device)
        ids = torch.unique(inst_t)

        # drop background if requested
        if not self.include_background:
            ids = ids[ids != 0]

        # if empty -> create empty onehot with correct shape
        H, W, D = inst_t.shape
        if ids.numel() == 0:
            d[self.out_key] = torch.empty((0, H, W, D), device=inst_t.device, dtype=self.dtype)
            return d

        # optionally limit instances (keep smallest or first N ids)
        if 0 < self.max_instances < ids.numel():
            ids = ids[: self.max_instances]

        # build onehot: [K, H, W, D] then add channel: [K,1,H,W,D]
        onehot = (inst_t[None, ...] == ids[:, None, None, None])
        onehot = onehot.to(dtype=self.dtype)

        d[self.out_key] = onehot
        return d
