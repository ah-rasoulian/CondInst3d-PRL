from typing import Any, Dict, Hashable, Mapping, Optional
import torch
from torch import Tensor
from monai.transforms import MapTransform
from monai.config import KeysCollection


class OneHotToBoxesd(MapTransform):
    """
    Convert instance onehot masks to boxes/classes for detection.

    Input:
        onehot: [N,1,H,W,D] or [N,H,W,D] (bool/int)
    Output:
        boxes:  [N,6] float32 (x1,y1,z1,x2,y2,z2)
        classes:[N]   int64

    Notes:
      - Empty masks are removed.
      - Coordinates are voxel indices. x is dim -3, y dim -2, z dim -1.
      - x2/y2/z2 are set to max_index+1 (exclusive end), which plays well with many box ops.
    """

    def __init__(
        self,
        onehot_key: str = "onehot",
        boxes_key: str = "boxes",
        classes_key: str = "classes",
        default_class: int = 0,
        max_instances: int = -1,
        allow_missing_keys: bool = False,
    ):
        super().__init__([onehot_key], allow_missing_keys=allow_missing_keys)
        self.onehot_key = onehot_key
        self.boxes_key = boxes_key
        self.classes_key = classes_key
        self.default_class = int(default_class)
        self.max_instances = int(max_instances)

    @staticmethod
    def _mask_to_box(mask: Tensor) -> Optional[Tensor]:
        """
        mask: [H,W,D] bool
        return: [6] (x1,y1,z1,x2,y2,z2) or None if empty
        """
        # Find nonzero voxels
        nz = mask.nonzero(as_tuple=False)  # [M, 3] with dims (h,w,d) == (x,y,z) in your convention?
        if nz.numel() == 0:
            return None

        mins = nz.min(dim=0).values
        maxs = nz.max(dim=0).values

        # Here nz columns correspond to (x,y,z) if mask is [X,Y,Z].
        # In torch, a 3D tensor index order is (dim0, dim1, dim2).
        x1, y1, z1 = mins.tolist()
        x2, y2, z2 = (maxs + 1).tolist()  # exclusive

        return mask.new_tensor([x1, y1, z1, x2, y2, z2], dtype=torch.float32)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        onehot = d[self.onehot_key]
        onehot_t = torch.as_tensor(onehot)

        # Accept [N,1,H,W,D] or [N,H,W,D]
        if onehot_t.ndim == 5 and onehot_t.shape[1] == 1:
            onehot_t = onehot_t[:, 0]  # [N,H,W,D]
        elif onehot_t.ndim != 4:
            raise ValueError(
                f"Expected {self.onehot_key} as [N,1,H,W,D] or [N,H,W,D], got {tuple(onehot_t.shape)}"
            )

        N = onehot_t.shape[0]
        if N == 0:
            d[self.boxes_key] = torch.empty((0, 6), device=onehot_t.device, dtype=torch.float32)
            d[self.classes_key] = torch.empty((0,), device=onehot_t.device, dtype=torch.long)
            return d

        boxes = []
        keep_ids = []
        # Loop is OK since N is usually small (lesion count)
        for i in range(N):
            box = self._mask_to_box(onehot_t[i].bool())
            if box is not None:
                boxes.append(box)
                keep_ids.append(i)

        if len(boxes) == 0:
            d[self.boxes_key] = torch.empty((0, 6), device=onehot_t.device, dtype=torch.float32)
            d[self.classes_key] = torch.empty((0,), device=onehot_t.device, dtype=torch.long)
            return d

        boxes_t = torch.stack(boxes, dim=0)  # [K,6]

        # Optional cap
        if self.max_instances > 0 and boxes_t.shape[0] > self.max_instances:
            boxes_t = boxes_t[: self.max_instances]

        classes_t = torch.full(
            (boxes_t.shape[0],),
            self.default_class,
            device=onehot_t.device,
            dtype=torch.long,
        )

        d[self.boxes_key] = boxes_t
        d[self.classes_key] = classes_t
        return d
