from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Optional

import numpy as np
import torch
from monai.transforms import MapTransform
from monai.config import KeysCollection
from monai.utils import convert_to_numpy


@dataclass
class InstanceScoresFromSoftmaxd(MapTransform):
    """
    Compute per-instance scores from:
      - instance mask: [1, X, Y, Z] with labels 0..N (0=background)
      - semantic softmax: [2, X, Y, Z] (channel 0=background prob, channel 1=foreground prob)

    Output:
      - scores: [N] float array/tensor where scores[i-1] is the mean foreground prob
        over voxels belonging to instance id i.

    Notes:
      - Instances with 0 voxels (shouldn't happen) get score 0.0.
      - Returns float32 scores.
    """

    instance_key: Hashable
    softmax_key: Hashable
    out_key: Hashable = "instance_scores"
    foreground_channel: int = 1
    allow_missing_keys: bool = False
    strict: bool = True  # if False, output None on failure

    def __post_init__(self):
        super().__init__([self.instance_key, self.softmax_key], allow_missing_keys=self.allow_missing_keys)

    @staticmethod
    def _check_shapes(inst: np.ndarray, sm: np.ndarray) -> None:
        if inst.ndim != 4 or inst.shape[0] != 1:
            raise ValueError(f"instance_mask must be [1,X,Y,Z], got {tuple(inst.shape)}")
        if sm.ndim != 4 or sm.shape[0] != 2:
            raise ValueError(f"semantic softmax must be [2,X,Y,Z], got {tuple(sm.shape)}")
        if inst.shape[1:] != sm.shape[1:]:
            raise ValueError(f"spatial shapes must match: inst={tuple(inst.shape)}, softmax={tuple(sm.shape)}")

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)

        try:
            inst_in = d.get(self.instance_key, None)
            sm_in = d.get(self.softmax_key, None)
            if inst_in is None or sm_in is None:
                d[self.out_key] = None
                return d

            # Convert to numpy
            inst_np = convert_to_numpy(inst_in) if isinstance(inst_in, torch.Tensor) else np.asarray(inst_in)
            sm_np = convert_to_numpy(sm_in) if isinstance(sm_in, torch.Tensor) else np.asarray(sm_in)

            self._check_shapes(inst_np, sm_np)

            inst = inst_np[0].astype(np.int64)             # [X,Y,Z]
            fg_prob = sm_np[self.foreground_channel].astype(np.float32)  # [X,Y,Z]

            max_id = int(inst.max())
            if max_id <= 0:
                scores = np.zeros((0,), dtype=np.float32)  # no instances
            else:
                flat_ids = inst.reshape(-1)
                flat_fg = fg_prob.reshape(-1)

                # sum fg probs per instance id (0..max_id)
                sums = np.bincount(flat_ids, weights=flat_fg, minlength=max_id + 1).astype(np.float64)
                counts = np.bincount(flat_ids, minlength=max_id + 1).astype(np.int64)

                # exclude background id=0, compute means for ids 1..max_id
                denom = np.maximum(counts[1:], 1)
                scores = (sums[1:] / denom).astype(np.float32)

                # If an instance id is "missing" (count==0), set score to 0
                scores[counts[1:] == 0] = 0.0

            # Return torch if either input was torch (common MONAI behavior)
            if isinstance(inst_in, torch.Tensor) or isinstance(sm_in, torch.Tensor):
                device = inst_in.device if isinstance(inst_in, torch.Tensor) else sm_in.device
                d[self.out_key] = torch.as_tensor(scores, device=device, dtype=torch.float32)
            else:
                d[self.out_key] = scores

        except Exception:
            if self.strict:
                raise
            d[self.out_key] = None

        return d
