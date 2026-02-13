from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Optional

import numpy as np
import torch
from monai.transforms import MapTransform
from monai.config import KeysCollection
from monai.utils import convert_to_numpy

try:
    from skimage.measure import label as cc_label
except Exception:
    cc_label = None


@dataclass
class SemanticToInstanced(MapTransform):
    """
    Convert a *binary* semantic mask of shape [1, X, H, W] into an instance mask
    using connected-component labeling.

    Requirements / assumptions:
      - Input is 4D: [1, X, H, W] (channel-first).
      - Only ONE foreground class is allowed:
          background = 0, foreground = 1 (or any positive value).
        If more than one non-zero label value is found (e.g. {0,1,2}), raises an error.

    Output:
      - Instance mask with same shape [1, X, H, W] (channel preserved),
        dtype int32 (torch.int32 if input is torch).
      - Values: 0 = background, 1..K = instance ids.
    """

    keys: KeysCollection
    out_key: Optional[Hashable] = None
    connectivity: int = 1  # for 3D: 1=6-neigh, 2=18, 3=26
    allow_missing_keys: bool = False
    strict: bool = True  # if False, set output to None on failure

    def __post_init__(self):
        super().__init__(self.keys, allow_missing_keys=self.allow_missing_keys)
        if cc_label is None:
            raise ImportError(
                "SemanticToInstanced requires scikit-image. Install it with `pip install scikit-image`."
            )

    def _check_shape_and_get_spatial(self, x_np: np.ndarray) -> np.ndarray:
        if x_np.ndim != 4:
            raise ValueError(f"Expected input of shape [1, X, H, W], got {tuple(x_np.shape)}")
        if x_np.shape[0] != 1:
            raise ValueError(f"Expected channel dimension to be 1, got C={x_np.shape[0]} in {tuple(x_np.shape)}")
        return x_np[0]  # [X, H, W]

    def _check_single_class(self, sem_spatial: np.ndarray) -> None:
        # Allow background=0 and a single foreground value (>0).
        uniq = np.unique(sem_spatial.astype(np.int64))
        fg = uniq[uniq != 0]
        if fg.size == 0:
            return  # all background is ok
        if fg.size > 1:
            raise ValueError(
                f"Expected only one foreground class (binary mask). Found multiple non-zero labels: {fg.tolist()}"
            )

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)

        for key in self.key_iterator(d):
            try:
                x = d.get(key, None)
                if x is None:
                    d[self.out_key or key] = None
                    continue

                # to numpy
                if isinstance(x, torch.Tensor):
                    x_np = convert_to_numpy(x)
                else:
                    x_np = np.asarray(x)

                sem = self._check_shape_and_get_spatial(x_np)  # [X, H, W]
                self._check_single_class(sem)

                # binary foreground: any > 0
                fg = sem > 0

                inst_spatial = cc_label(fg.astype(np.uint8), connectivity=self.connectivity).astype(np.int32)  # [X,H,W]
                inst = inst_spatial[None, ...]  # [1,X,H,W]

                if isinstance(x, torch.Tensor):
                    d[self.out_key or key] = torch.as_tensor(inst, device=x.device, dtype=torch.int32)
                else:
                    d[self.out_key or key] = inst

            except Exception:
                if self.strict:
                    raise
                d[self.out_key or key] = None

        return d
