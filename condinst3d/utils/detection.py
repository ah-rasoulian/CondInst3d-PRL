from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from torch import Tensor
from monai.transforms.utils import generate_spatial_bounding_box


@dataclass
class ImageInstancesData:
    """
    Per-image container. Stores only what is needed to build the flattened InstanceList.
    """
    anchor_centers: Tensor               # [A, 3]
    anchor_strides: Tensor               # [A, 3]
    level_strides: Tensor                # [A, 3]

    # train-time
    matched_idx: Optional[Tensor] = None # [A] in [-1..G-1]
    gt_classes: Optional[Tensor] = None  # [A] in {-1,0..C-1}
    gt_boxes: Optional[Tensor] = None    # [A, 6]
    fg_mask: Optional[Tensor] = None     # [A] bool

    # infer-time
    keep_idxs: Optional[Tensor] = None   # [K] anchor indices of selected candidates

    @classmethod
    def from_targets(
        cls,
        anchors: Tensor,               # [A,6]
        level_strides: Tensor,         # [A,3]
        matched_idx: Tensor,           # [A]
        targets: Dict[str, Tensor],    # boxes [G,6], classes [G]
    ):
        assert targets["classes"].ndim == 1, f'targets["classes"] must be [G], got {targets["classes"].shape}'
        assert targets["boxes"].ndim == 2 and targets["boxes"].shape[1] == 6,  f'targets["boxes"] must be [G,6], got {targets["boxes"].shape}'

        centers = (anchors[:, :3] + anchors[:, 3:]) / 2
        strides = anchors[:, 3:] - anchors[:, :3]

        A = matched_idx.numel()
        device = anchors.device

        if targets["classes"].numel() == 0:
            gt_classes = torch.full((A,), -1, dtype=torch.long, device=device)
            gt_boxes   = torch.zeros((A, 6), dtype=anchors.dtype, device=device)
        else:
            idx = matched_idx.clamp(min=0)
            gt_classes = targets["classes"][idx].to(torch.long)
            gt_boxes = targets["boxes"][idx].to(dtype=anchors.dtype)

            bg = matched_idx < 0
            gt_classes[bg] = -1
            gt_boxes[bg] = 0

        fg_mask = gt_classes >= 0
        return cls(
            anchor_centers=centers,
            anchor_strides=strides,
            level_strides=level_strides,
            matched_idx=matched_idx,
            gt_classes=gt_classes,
            gt_boxes=gt_boxes,
            fg_mask=fg_mask,
        )

    @classmethod
    def from_keep(
        cls,
        anchors: Tensor,          # [A,6]
        level_strides: Tensor,    # [A,3]
        keep_idxs: Tensor,        # [K]
    ):
        centers = (anchors[:, :3] + anchors[:, 3:]) / 2
        strides = anchors[:, 3:] - anchors[:, :3]
        return cls(anchor_centers=centers, anchor_strides=strides, level_strides=level_strides, keep_idxs=keep_idxs)


class InstanceList:
    """
    A flattened view across a batch, designed for fast indexing and gather.
    """
    def __init__(self, per_image: List[ImageInstancesData], max_samples: int = -1):
        self.per_image = per_image
        self.device = per_image[0].anchor_centers.device if len(per_image) else torch.device("cpu")

        # build flat index of foreground (train) OR keep_idxs (infer)
        img_ids, anchor_ids, gt_ids = [], [], []
        centers, strides, level_strides = [], [], []

        for img_i, d in enumerate(per_image):
            if d.keep_idxs is not None:
                aidx = d.keep_idxs
                img = torch.full((aidx.numel(),), img_i, device=aidx.device, dtype=torch.long)
                img_ids.append(img)
                anchor_ids.append(aidx)
                # no gt in inference
                gt_ids.append(torch.full((aidx.numel(),), -1, device=aidx.device, dtype=torch.long))
                centers.append(d.anchor_centers[aidx])
                strides.append(d.anchor_strides[aidx])
                level_strides.append(d.level_strides[aidx])
            else:
                fg = d.fg_mask
                aidx = torch.where(fg)[0]
                img = torch.full((aidx.numel(),), img_i, device=aidx.device, dtype=torch.long)
                img_ids.append(img)
                anchor_ids.append(aidx)
                gt_ids.append(d.matched_idx[aidx])  # index of GT instance in that image
                centers.append(d.anchor_centers[aidx])
                strides.append(d.anchor_strides[aidx])
                level_strides.append(d.level_strides[aidx])

        if len(img_ids) == 0:
            self.img_idx = torch.empty((0,), device=self.device, dtype=torch.long)
            self.anchor_idx = torch.empty((0,), device=self.device, dtype=torch.long)
            self.gt_idx = torch.empty((0,), device=self.device, dtype=torch.long)
            self.centers = torch.empty((0, 3), device=self.device)
            self.strides = torch.empty((0, 3), device=self.device)
            self.level_strides = torch.empty((0, 3), device=self.device)
        else:
            self.img_idx = torch.cat(img_ids, dim=0)
            self.anchor_idx = torch.cat(anchor_ids, dim=0)
            self.gt_idx = torch.cat(gt_ids, dim=0)
            self.centers = torch.cat(centers, dim=0)
            self.strides = torch.cat(strides, dim=0)
            self.level_strides = torch.cat(level_strides, dim=0)

        # sampling
        self.sample_idx = self._build_sample_idx(max_samples)

    def _build_sample_idx(self, max_samples: int) -> Tensor:
        n = self.img_idx.numel()
        if n == 0:
            return torch.empty((0,), device=self.device, dtype=torch.long)

        if max_samples < 0 or max_samples >= n:
            return torch.arange(n, device=self.device)

        return torch.randperm(n, device=self.device)[:max_samples]

    def __len__(self):
        return self.sample_idx.numel()

    # fast getters
    def get_points(self) -> Tensor:
        return self.centers[self.sample_idx]

    def get_strides(self) -> Tensor:
        return self.strides[self.sample_idx]

    def get_level_strides(self) -> Tensor:
        return self.level_strides[self.sample_idx]

    def get_image_indices(self) -> Tensor:
        return self.img_idx[self.sample_idx]

    def gather_mask_head_params(self, controller_logits: Tensor) -> Tensor:
        """
        controller_logits: [B, A, P]
        return: [M, P] for sampled instances
        """
        s = self.sample_idx
        return controller_logits[self.img_idx[s], self.anchor_idx[s]]

    def get_gt_mask(self, targets: List[Dict[str, Tensor]], dtype=torch.float32) -> Tensor:
        """
        Returns GT masks aligned with self.sample_idx order.
        targets[i]["onehot"]: [Gi, W, H, D] or [Gi, 1, W, H, D]
        """
        s = self.sample_idx
        if s.numel() == 0:
            return torch.empty((0, 1, 1, 1, 1), device=self.device, dtype=dtype)

        # Indices for sampled instances
        img = self.img_idx[s]  # [M]
        gt = self.gt_idx[s]  # [M]

        masks_out = []
        # Still group per image to avoid many tiny index ops,
        # BUT write back in the original order using a buffer.
        out = None

        unique_imgs = torch.unique(img)
        # allocate output lazily after reading first mask shape
        for img_i in unique_imgs.tolist():
            sel = (img == img_i).nonzero(as_tuple=True)[0]  # positions within [0..M)
            gt_i = gt[sel]

            gt_onehot = targets[img_i]["onehot"]
            if gt_onehot.ndim == 4:
                gt_onehot = gt_onehot.unsqueeze(1)  # [G,1,W,H,D]

            gathered = gt_onehot[gt_i].to(dtype=dtype)  # [n_i,1,W,H,D]

            if out is None:
                out = torch.empty((img.numel(),) + gathered.shape[1:], device=self.device, dtype=dtype)

            out[sel] = gathered

        return out

def get_onehot_instance_mask_boxes(instance_segmentation: Tensor) -> Tensor:
    """
    instance_segmentation: [N,1,H,W,D] bool or uint8
    returns: [N,6] float32 (x1,y1,z1,x2,y2,z2)
    """
    n = instance_segmentation.shape[0]
    if n == 0:
        return instance_segmentation.new_zeros((0, 6), dtype=torch.float32)

    boxes = []
    for m in instance_segmentation:
        if not torch.any(m):
            # empty -> zeros
            boxes.append(m.new_zeros((6,), dtype=torch.float32))
            continue
        start, end = generate_spatial_bounding_box(m, allow_smaller=True)
        boxes.append(torch.tensor(start + end, device=m.device, dtype=torch.float32))

    return torch.stack(boxes, dim=0)


def priority_based_onehot_to_instance_mask(onehot_mask: Tensor, scores: Tensor) -> Tensor:
    """
    onehot_mask:
      - [N, 1, H, W, D] or [N, H, W, D]   (typical)
      - or [N, C, H, W, D]               (multi-class)
    scores: [N]

    Returns:
      instance_mask: [C, H, W, D] where labels are 1..N (global instance ids by score rank),
      or [1, H, W, D] if C=1. Background = 0.
    """
    if onehot_mask.ndim == 4:
        onehot_mask = onehot_mask.unsqueeze(1)  # [N,1,H,W,D]
    if onehot_mask.ndim != 5:
        raise ValueError(f"onehot_mask must be 4D or 5D, got {onehot_mask.shape}")

    N, C, H, W, D = onehot_mask.shape
    if scores.ndim != 1:
        scores = scores.view(-1)
    if scores.numel() != N:
        raise ValueError(f"N(onehot) != len(scores): {N} != {scores.numel()}")

    if N == 0:
        return onehot_mask.new_zeros((C, H, W, D), dtype=torch.int16)

    # Ensure boolean for correct masking; allow float/bool inputs
    m = onehot_mask > 0  # [N,C,H,W,D] bool

    # Put scores on same device/dtype as we need for broadcast
    s = scores.to(device=m.device, dtype=torch.float32).view(N, 1, 1, 1, 1)

    # Score-weighted masks; background is 0 score
    weighted = m.to(torch.float32) * s  # [N,C,H,W,D]

    # Best instance per voxel
    best_score, best_idx = weighted.max(dim=0)  # each: [C,H,W,D], idx in [0..N-1]
    # Convert to instance labels (idx+1) only where any instance present
    inst = (best_idx + 1).to(torch.int16)
    inst = torch.where(best_score > 0, inst, inst.new_zeros(()).expand_as(inst))

    return inst


def instance_mask_to_onehot(mask: Tensor, return_instance_ids: bool = False):
    """
    mask: [H,W,D] or [1,H,W,D] or [1,1,H,W,D]
    Returns:
      onehot: [N, H, W, D] (dtype = uint8 by default)
    """
    # squeeze optional batch/channel
    if mask.ndim == 5:
        if mask.shape[0] != 1:
            raise ValueError("Only batch size 1 supported.")
        mask = mask[0]
    if mask.ndim == 4:
        if mask.shape[0] != 1:
            raise ValueError("Only channel size 1 supported.")
        mask = mask[0]
    if mask.ndim != 3:
        raise ValueError(f"mask must end up 3D [H,W,D], got {mask.shape}")

    ids = torch.unique(mask)
    ids = ids[ids != 0]  # remove background

    if ids.numel() == 0:
        onehot = mask.new_zeros((0, *mask.shape), dtype=torch.uint8)
        return (onehot, ids.to(torch.long)) if return_instance_ids else onehot

    # [N,1,1,1] == [1,H,W,D] -> [N,H,W,D]
    onehot = (mask.unsqueeze(0) == ids.view(-1, 1, 1, 1)).to(torch.uint8)

    return (onehot, ids.to(torch.long)) if return_instance_ids else onehot


def onehot_to_instance_mask(onehot_mask: Tensor, inds: Optional[Tensor] = None) -> Tensor:
    """
    Convert instance onehot -> instance id mask per class.

    Args:
        onehot_mask:
            [N, C, W, H, D] or [N, W, H, D]
            N = num instances, C = num classes (often 1)
            values are {0,1} (bool/uint8/int/float ok).
        inds:
            optional mapping for instance ids.
            - If provided, should be shape [N] (or list/1D tensor).
            - Output ids (1..N) will be replaced by inds values.
              Background stays 0.

    Returns:
        instance_mask: [C, W, H, D] (dtype long)
            0 = background
            1..N = instance index (or mapped ids if inds provided)
    """
    if onehot_mask.ndim == 4:
        onehot_mask = onehot_mask.unsqueeze(1)  # [N,1,W,H,D]
    if onehot_mask.ndim != 5:
        raise ValueError(f"Expected [N,C,W,H,D] or [N,W,H,D], got {onehot_mask.shape}")

    N, C, W, H, D = onehot_mask.shape
    device = onehot_mask.device

    if N == 0:
        # keep consistent dtype/shape with rest of pipeline
        return torch.zeros((C, W, H, D), device=device, dtype=torch.long)

    # Ensure boolean for cheap reductions
    m = onehot_mask.to(dtype=torch.bool, copy=False)  # [N,C,W,H,D]

    # any_instance[c] tells whether voxel belongs to any instance for that class
    any_instance = m.any(dim=0)  # [C,W,H,D] bool

    # argmax over instance dimension gives 0..N-1, but meaningless where any_instance==False
    # Use int for indexing
    arg = m.to(torch.uint8).argmax(dim=0).to(torch.long)  # [C,W,H,D], 0..N-1

    # convert to 1..N where foreground, else 0
    out = torch.where(any_instance, arg + 1, torch.zeros((), device=device, dtype=torch.long))

    # Optional remap: ids 1..N -> inds values
    if inds is not None:
        if not torch.is_tensor(inds):
            inds = torch.as_tensor(inds, device=device)
        else:
            inds = inds.to(device)

        if inds.ndim != 1 or inds.numel() != N:
            raise ValueError(f"`inds` must be shape [N]={N}, got {inds.shape}")

        # Build lookup table: lut[0]=0 (bg), lut[i]=inds[i-1]
        # dtype: match inds dtype if you want, but long is usually safest
        lut = torch.empty((N + 1,), device=device, dtype=inds.dtype)
        lut[0] = 0
        lut[1:] = inds

        out = lut[out]  # vectorized remap

    return out