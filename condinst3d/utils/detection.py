from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from torch import Tensor


@dataclass
class ImageInstancesData:
    """
    Per-image container. Stores only what is needed to build the flattened InstanceList.
    """
    anchor_centers: Tensor               # [A, 3]
    anchor_strides: Tensor               # [A, 3]

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
            matched_idx=matched_idx,
            gt_classes=gt_classes,
            gt_boxes=gt_boxes,
            fg_mask=fg_mask,
        )

    @classmethod
    def from_keep(
        cls,
        anchors: Tensor,          # [A,6] OR you can pass centers/strides directly
        keep_idxs: Tensor,        # [K]
    ):
        centers = (anchors[:, :3] + anchors[:, 3:]) / 2
        strides = anchors[:, 3:] - anchors[:, :3]
        return cls(anchor_centers=centers, anchor_strides=strides, keep_idxs=keep_idxs)


class InstanceList:
    """
    A flattened view across a batch, designed for fast indexing and gather.
    """
    def __init__(self, per_image: List[ImageInstancesData], max_samples: int = -1):
        self.per_image = per_image
        self.device = per_image[0].anchor_centers.device if len(per_image) else torch.device("cpu")

        # build flat index of foreground (train) OR keep_idxs (infer)
        img_ids, anchor_ids, gt_ids = [], [], []
        centers, strides = [], []

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
            else:
                fg = d.fg_mask
                aidx = torch.where(fg)[0]
                img = torch.full((aidx.numel(),), img_i, device=aidx.device, dtype=torch.long)
                img_ids.append(img)
                anchor_ids.append(aidx)
                gt_ids.append(d.matched_idx[aidx])  # index of GT instance in that image
                centers.append(d.anchor_centers[aidx])
                strides.append(d.anchor_strides[aidx])

        if len(img_ids) == 0:
            self.img_idx = torch.empty((0,), device=self.device, dtype=torch.long)
            self.anchor_idx = torch.empty((0,), device=self.device, dtype=torch.long)
            self.gt_idx = torch.empty((0,), device=self.device, dtype=torch.long)
            self.centers = torch.empty((0, 3), device=self.device)
            self.strides = torch.empty((0, 3), device=self.device)
        else:
            self.img_idx = torch.cat(img_ids, dim=0)
            self.anchor_idx = torch.cat(anchor_ids, dim=0)
            self.gt_idx = torch.cat(gt_ids, dim=0)
            self.centers = torch.cat(centers, dim=0)
            self.strides = torch.cat(strides, dim=0)

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
