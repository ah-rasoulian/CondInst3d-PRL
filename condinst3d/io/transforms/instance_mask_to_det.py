from __future__ import annotations

from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, List

import torch
from torch import Tensor

from monai.transforms import MapTransform
from monai.transforms.utils import generate_spatial_bounding_box


class InstanceMaskToDetd(MapTransform):
    """
    Convert an instance (label) mask to boxes/classes AND onehot instance masks.

    Input:
        instance: [1,H,W,D] or [H,W,D] integer labels (0 = background, 1..K = instances)
                 (bool is also accepted but then it's effectively one instance)

    Output:
        onehot:  [K,1,H,W,D] (int/bool)  (K instances kept)
        boxes:   [K,6] float32 (x1,y1,z1,x2,y2,z2)  (x2/y2/z2 are exclusive end)
        classes: [K]   int64

    Notes:
      - Empty masks are removed (no voxels for that label).
      - Coordinates are voxel indices. Tensor indexing order is (x,y,z) == (dim0,dim1,dim2).
      - If box_margin is provided, boxes are expanded (clipped to image bounds by MONAI util).
    """

    def __init__(
        self,
        instance_key: str = "instance",
        onehot_key: str = "onehot",
        boxes_key: str = "boxes",
        classes_key: str = "classes",
        default_class: int = 0,
        # If True, class id will be the instance label value (useful if labels encode classes)
        class_from_label: bool = False,
        box_margin: int | Sequence[int] = 0,
        include_background: bool = False,
        max_instances: int = -1,
        # If True, onehot dtype is torch.long (0/1). Else bool.
        onehot_long: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__([instance_key], allow_missing_keys=allow_missing_keys)
        self.instance_key = instance_key
        self.onehot_key = onehot_key
        self.boxes_key = boxes_key
        self.classes_key = classes_key

        self.default_class = int(default_class)
        self.class_from_label = bool(class_from_label)

        if isinstance(box_margin, int):
            self.box_margin = (box_margin, box_margin, box_margin)
        else:
            bm = tuple(int(x) for x in box_margin)
            if len(bm) != 3:
                raise ValueError(f"box_margin must be int or len-3 sequence, got {box_margin}")
            self.box_margin = bm

        self.include_background = bool(include_background)
        self.max_instances = int(max_instances)
        self.onehot_long = bool(onehot_long)

    @staticmethod
    def _as_3d_instance_mask(x: Any) -> Tensor:
        t = torch.as_tensor(x)
        # Accept [1,H,W,D] or [H,W,D]
        if t.ndim == 4:
            if t.shape[0] != 1:
                raise ValueError(f"Expected instance mask as [1,H,W,D], got {tuple(t.shape)}")
            t = t[0]
        elif t.ndim != 3:
            raise ValueError(f"Expected instance mask as [H,W,D] or [1,H,W,D], got {tuple(t.shape)}")

        # If bool, make it int labels {0,1}
        if t.dtype == torch.bool:
            t = t.to(torch.int64)

        return t

    @staticmethod
    def _labels_from_instance_mask(inst: Tensor, include_background: bool) -> Tensor:
        # unique sorted labels
        labels = torch.unique(inst)
        if not include_background:
            labels = labels[labels != 0]
        return labels

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        inst = self._as_3d_instance_mask(d[self.instance_key])  # [H,W,D]
        device = inst.device

        labels = self._labels_from_instance_mask(inst, self.include_background)

        if labels.numel() == 0:
            d[self.onehot_key] = torch.zeros((0, 1, *inst.shape), device=device, dtype=torch.long if self.onehot_long else torch.bool)
            d[self.boxes_key] = torch.empty((0, 6), device=device, dtype=torch.float32)
            d[self.classes_key] = torch.empty((0,), device=device, dtype=torch.long)
            return d

        boxes: List[Tensor] = []
        onehots: List[Tensor] = []
        classes: List[int] = []

        # Loop is fine: instance count usually small
        for lab in labels.tolist():
            m = (inst == lab)  # [H,W,D] bool
            if not m.any():
                continue

            # MONAI expects channel-first, so make [1,H,W,D]
            m_ch = m.unsqueeze(0)

            # generate_spatial_bounding_box returns (start, end) inclusive indices by default,
            # but end is typically "max index" (inclusive). We'll convert to exclusive end (+1),
            # to match your original transform convention.
            box_start, box_end = generate_spatial_bounding_box(
                m_ch,
                margin=self.box_margin,
                allow_smaller=True,
            )
            # box_start/end are sequences length 3
            x1, y1, z1 = map(int, box_start)
            x2, y2, z2 = map(int, box_end)
            # Convert inclusive end -> exclusive end
            x2, y2, z2 = x2 + 1, y2 + 1, z2 + 1

            boxes.append(torch.tensor([x1, y1, z1, x2, y2, z2], device=device, dtype=torch.float32))

            if self.onehot_long:
                onehots.append(m_ch.to(torch.long))
            else:
                onehots.append(m_ch)

            if self.class_from_label:
                classes.append(int(lab))
            else:
                classes.append(self.default_class)

            if self.max_instances > 0 and len(boxes) >= self.max_instances:
                break

        if len(boxes) == 0:
            d[self.onehot_key] = torch.zeros((0, 1, *inst.shape), device=device, dtype=torch.long if self.onehot_long else torch.bool)
            d[self.boxes_key] = torch.empty((0, 6), device=device, dtype=torch.float32)
            d[self.classes_key] = torch.empty((0,), device=device, dtype=torch.long)
            return d

        d[self.onehot_key] = torch.stack(onehots, dim=0)  # [K,1,H,W,D]
        d[self.boxes_key] = torch.stack(boxes, dim=0)     # [K,6]
        d[self.classes_key] = torch.tensor(classes, device=device, dtype=torch.long)  # [K]
        return d
