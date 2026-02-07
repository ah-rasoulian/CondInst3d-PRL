from typing import Any, Dict, List, Tuple, Optional, Literal, Sequence, Union, Iterable
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from monai.data import MetaTensor
from monai.transforms import Compose
from monai.transforms.utils import get_unique_labels
import pytorch_lightning as pl
import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import random
import numpy as np

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from condinst3d.arch.heads import ClassificationHead, ControllerHead, DynamicMaskHead
from condinst3d.evaluator.metrics import AveragePrecision, DetectionConfusionMatrix
from condinst3d.evaluator.iou import mask_intersection_over_union, box_intersection_over_union
from condinst3d.utils.detection import (ImageInstancesData, InstanceList, get_onehot_instance_mask_boxes,
                                        priority_based_onehot_to_instance_mask, instance_mask_to_onehot,
                                        onehot_to_instance_mask)
from condinst3d.utils.mask import relabel_sequential
from condinst3d.utils.anchors import AnisotropicATSSMatcher, generate_3d_anchors
from condinst3d.arch.backbone.abstract import AbstractBackbone
from condinst3d.utils.spatial import aligned_trilinear, merge_patch_logits_per_instance, get_patch_spatial_shapes
from condinst3d.utils.bached_ag import batched_nms, aggregate_per_patch_detections
from condinst3d.visualization.utils import get_stats
from condinst3d.visualization.list_instance_boxseg_visualizer import ListInstanceBoxSegSliceVisualizer
from condinst3d.evaluator.metrics.cfm_based import compute_precision, compute_fi, compute_recall
from monai.transforms import KeepLargestConnectedComponent, RemoveSmallObjects, Compose


# ---------------- Metric logging helpers ----------------
def _to_float(x: Tensor | float) -> Tensor | float:
    # keep tensors as tensors (Lightning likes tensors), but avoid accidental MetaTensor issues
    return x


def _safe_mean(x: Tensor) -> Tensor:
    # some AP implementations return shape [T] or [T, ...]
    return x.mean() if x.numel() else x.new_tensor(0.0)

def _ensure_boxes_2d(boxes: torch.Tensor, name: str) -> torch.Tensor:
    # Accept empty [0,6], [6], or [G,6]
    if boxes.numel() == 0:
        return boxes.reshape(0, 6)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    assert boxes.ndim == 2 and boxes.shape[1] == 6, f"{name} must be [N,6], got {tuple(boxes.shape)}"
    return boxes


def _squeeze_onehot(onehot: torch.Tensor) -> torch.Tensor:
    # Normalize to [N,H,W,D]
    if onehot.ndim == 5 and onehot.shape[1] == 1:
        return onehot.squeeze(1)
    return onehot

class CondInst3dPRL(pl.LightningModule):
    """
    A 3D Instance Segmentation Network based on CondInst framework.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        self.forward_chunk_size: int = cfg.forward_chunk_size
        self.max_mask_to_train: int = cfg.max_mask_to_train
        self.backbone: AbstractBackbone = instantiate(cfg.backbone)

        # --------------- anchor matching ----------------
        self.matcher = AnisotropicATSSMatcher(
            num_candidates=cfg.matcher.num_candidates,
            center_in_gt=cfg.matcher.center_in_gt,
        )
        self.anchors_sizes = cfg.matcher.anchors_sizes

        # -------------------- heads ---------------------
        self.classification_head = ClassificationHead(
            in_channels=self.backbone.heads_dim,
            num_classes=cfg.heads.classification.num_classes,
            prior_probability=cfg.heads.classification.prior_probability,
        )
        self.instance_segmentation_head = DynamicMaskHead(
            in_channels=self.backbone.out_channels,
            channels=cfg.heads.instance_segmentation.channels,
            num_layers=cfg.heads.instance_segmentation.num_layers,
            kernel_size=cfg.heads.instance_segmentation.kernel_size,
            size_of_interest=cfg.heads.instance_segmentation.size_of_interest,
            stride=self.backbone.mask_stride,
        )
        self.controller_head = ControllerHead(
            in_channels=self.backbone.heads_dim,
            num_params=self.instance_segmentation_head.num_params
        )
        self.semantic_head = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(self.backbone.out_channels, cfg.heads.classification.num_classes + 1, kernel_size=1, padding='same')
        )

        # -------------------- loss functions ---------------------
        self.losses = {
            "classification": instantiate(cfg.losses.classification),
            "instance_segmentation": instantiate(cfg.losses.instance_segmentation),
            "semantic_segmentation": instantiate(cfg.losses.semantic_segmentation),
        }

        # -------------------- postproc params --------------------
        self.postproc_transform = Compose([
            RemoveSmallObjects(min_size=8, connectivity=None),
            KeepLargestConnectedComponent(independent=True, is_onehot=True, num_components=1)
        ])

        # -------------------- inference/matching hyperparams --------------------
        self.inference_hyperparams = cfg.inference

        # -------------------- metrics --------------------
        self.metrics = nn.ModuleDict({
            "validation": self._generate_metrics(cfg.evaluation),
            "test": self._generate_metrics(cfg.evaluation),
        })

        # -------------------- visualization --------------------
        img_channels = cfg.visualization.img_channels
        self.instance_head_visualizer = ListInstanceBoxSegSliceVisualizer(
            crop_size=cfg.visualization.crop_size,
            pred_seg_is_binary=False,
            draw_boxes=False,
            show_slices=cfg.visualization.show_slices,
            figsize=cfg.visualization.figsize,
            img_channels=img_channels + ["semantic_mask"],
            channel_seg_under_image=cfg.visualization.channel_seg_under_image,
        )
        self.images_to_visualize: Dict[int, List[int]] = {}
        self.num_images_to_show = int(cfg.visualization.num_images_to_show)

    def _generate_metrics(self, eval_cfg: Dict):
        metrics = torchmetrics.MetricCollection({
                "mask_cfm": DetectionConfusionMatrix(iou_thresholds=eval_cfg.iou_list),
                "mask_ap": AveragePrecision(iou_thresholds=eval_cfg.iou_list, interpolation=eval_cfg.ap_n_interp),
                "box_cfm": DetectionConfusionMatrix(iou_thresholds=eval_cfg.iou_list),
                "box_ap": AveragePrecision(iou_thresholds=eval_cfg.iou_list, interpolation=eval_cfg.ap_n_interp),
        })
        return metrics

    # ---------------- Viz helpers ----------------
    def _get_single_val_loader(self) -> DataLoader:
        """You said you only have one val dataloader."""
        val_loader = self.trainer.val_dataloaders
        if isinstance(val_loader, list):
            # take the first; you said only 1 exists
            return val_loader[0]
        return val_loader

    def _assign_images_to_visualize(self, seed: int = 42) -> None:
        """
        Pick a fixed set of batch indices to visualize from the (single) val dataloader.
        Stores them in self.images_to_visualize[0] as a sorted list[int].
        """
        if not self.trainer or not self.trainer.is_global_zero:
            return

        loader = self._get_single_val_loader()

        # If your loader doesn't have __len__ (iterable-style), just do nothing
        if not hasattr(loader, "__len__"):
            self.images_to_visualize = {0: []}
            return

        n_batches = len(loader)
        if n_batches == 0:
            self.images_to_visualize = {0: []}
            return

        if self.num_images_to_show < 0:
            chosen = list(range(n_batches))
        else:
            k = min(int(self.num_images_to_show), n_batches)
            rng = random.Random(seed)  # doesn't touch global RNG
            chosen = rng.sample(range(n_batches), k=k)
            chosen.sort()

        self.images_to_visualize = {0: chosen}

    # ---------------- Matching / labels ----------------
    def _match_anchors_atss(
            self,
            anchors_per_level: List[Tensor],
            targets: List[Dict[str, Tensor]],
            spacing: Optional[Tensor] = None,
    ) -> List[ImageInstancesData]:
        anchors = torch.cat(anchors_per_level, dim=0)  # [A, 6]
        A = anchors.size(0)
        device = anchors.device
        num_anchors_per_level = [a.size(0) for a in anchors_per_level]

        instance_data_list: List[ImageInstancesData] = []
        for targets_per_image in targets:
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                matched_idx = torch.full((A,), -1, dtype=torch.long, device=device)
            else:
                _, matched_idx = self.matcher.compute_matches(
                    boxes=gt_boxes,
                    anchors=anchors,  # same tensor for all images
                    num_anchors_per_level=num_anchors_per_level,
                    num_anchors_per_loc=1,
                    spacing=spacing,
                )

            # generate instance for image
            image_instance_data = ImageInstancesData.from_targets(
                anchors=anchors,
                matched_idx=matched_idx,
                targets=targets_per_image,
            )
            instance_data_list.append(image_instance_data)

        return instance_data_list

    def _get_gt_instance_data_list(self, outputs, targets):
        features = outputs["features"]

        anchors_per_level = generate_3d_anchors(
            outputs["shape"],
            [f.shape for f in features],
            anchor_sizes=self.anchors_sizes,
            device=outputs["device"],
        )

        instance_data_list = self._match_anchors_atss(
            anchors_per_level, targets, spacing=outputs["spacing"]
        )
        return instance_data_list

    # ---------------- Losses ----------------
    def _compute_detection_loss(self, outputs, targets):
        instance_data_list = self._get_gt_instance_data_list(outputs, targets)

        cls_logits = outputs["cls_logits"]  # [B, A, C]
        B, A, C = cls_logits.shape
        device = cls_logits.device

        # [B, A] with class ids for positives, -1 for negatives
        all_gt_class_ids = torch.stack([x.gt_classes for x in instance_data_list], dim=0)

        # positives are anchors with matched GT
        pos_mask = all_gt_class_ids >= 0  # [B, A]

        # one-hot targets for focal (multi-label sigmoid formulation)
        gt_onehot = torch.zeros((B, A, C), device=device, dtype=cls_logits.dtype)
        if pos_mask.any():
            b_idx, a_idx = pos_mask.nonzero(as_tuple=True)  # [Npos]
            c_idx = all_gt_class_ids[b_idx, a_idx].long().clamp(0, C - 1)
            gt_onehot[b_idx, a_idx, c_idx] = 1.0

        # focal loss per entry: [B, A, C] (expects reduction="none")
        loss_per_entry = self.losses["classification"](cls_logits, gt_onehot)

        # reduce over classes -> [B, A]
        loss_per_anchor = loss_per_entry.sum(dim=-1)

        # sum over all anchors in the batch
        loss_sum = loss_per_anchor.sum()

        # normalize by number of positives (RetinaNet-style); avoid div by 0
        num_pos = pos_mask.sum().to(loss_sum.dtype)
        loss_cls = loss_sum / torch.clamp(num_pos, min=1.0)

        return loss_cls, instance_data_list

    def _compute_segmentation_loss(
            self,
            outputs: Dict[str, Tensor],
            targets,
            instance_list: "InstanceList",
    ) -> Dict[str, Tensor]:
        """
        Computes instance segmentation loss.

        Expects:
          outputs["instance_logits"]: Tensor [N, 1(or C), W, H, D] (example)
          self.instance_segmentation_head.stride: int or (sx, sy, sz)
          self.segmentation_loss: callable(pred_logits, gt_masks) -> Tensor
          instance_list.get_gt_mask(targets, dtype) -> Tensor aligned with instance_logits
        """

        instance_logits = outputs["instance_logits"]
        semantic_logits = outputs["semantic_logits"]
        semantic_gt_mask = torch.stack([t["semantic_mask"] for t in targets], dim=0)
        stride = getattr(self.instance_segmentation_head, "stride", 1)

        # upsample if needed (supports int or tuple stride)
        needs_upsample = (stride > 1) if isinstance(stride, int) else any(s > 1 for s in stride)
        if needs_upsample:
            instance_logits = aligned_trilinear(instance_logits, stride)
            semantic_logits = aligned_trilinear(semantic_logits, stride)

        semantic_loss = self.losses["semantic_segmentation"](semantic_logits, semantic_gt_mask)
        n_insts = len(instance_list)
        if n_insts == 0:
            instance_loss = instance_logits.sum() * 0.0
        else:
            gt_masks = instance_list.get_gt_mask(targets, dtype=instance_logits.dtype)
            instance_loss = self.losses["instance_segmentation"](instance_logits, gt_masks)

        return {
            "instance_segmentation": instance_loss,
            "semantic_segmentation": semantic_loss,
        }

    def compute_loss(self, outputs, targets):
        classification_loss, instance_data_list = self._compute_detection_loss(outputs, targets)

        instance_list = InstanceList(instance_data_list, self.max_mask_to_train)

        outputs["instance_logits"] = self.instance_segmentation_head(
            outputs["f_mask"],
            outputs["controller_logits"],
            instance_list,
        )

        segmentation_losses = self._compute_segmentation_loss(
            outputs=outputs,
            targets=targets,
            instance_list=instance_list,
        )

        return {
            "classification": classification_loss,
            "instance_segmentation": segmentation_losses["instance_segmentation"],
            "semantic_segmentation": 0.5 * segmentation_losses["semantic_segmentation"],
        }

    # ---------------- Inference: detections + seg ----------------
    @torch.no_grad()
    def _compute_detections(
            self,
            outputs: Dict[str, Tensor],
    ) -> Tuple[List[Dict[str, Tensor]], List[ImageInstancesData]]:
        """
        outputs must contain:
          - outputs["shape"] : (N, C, W, H, D) or similar
          - outputs["features"] : list of feature tensors per level
          - outputs["cls_logits"] : [N, A_total, num_classes] (A_total = sum A_l)
        Returns:
          detections: List[dict] length N (per-image)
          per_image_instances: List[ImageInstancesData] length N
        """
        # ---- unpack ----
        N = int(outputs["shape"][0])
        features = outputs["features"]
        cls_logits = outputs["cls_logits"]  # [N, A_total, C]
        device = cls_logits.device
        offsets = outputs["offsets"] if outputs["offsets"] is not None else torch.zeros(N, 3, device=device)

        score_thresh = float(self.inference_hyperparams["score_thresh"])
        topk_candidates = int(self.inference_hyperparams["topk_candidates"])

        # ---- anchors per level + flat anchors ----
        anchors_per_level = generate_3d_anchors(
            outputs["shape"],
            [f.shape for f in features],
            self.anchors_sizes,
            device=device,
        )  # list[L] of [A_l, 6]

        num_anchors_per_level = [a.shape[0] for a in anchors_per_level]
        level_offsets = torch.as_tensor(
            np.cumsum([0] + num_anchors_per_level[:-1]),
            device=device,
            dtype=torch.long,
        )  # [L]

        anchors_cat = torch.cat(anchors_per_level, dim=0)  # [A_total, 6]

        # ---- split logits by level (cheap view) ----
        logits_by_level = list(cls_logits.split(num_anchors_per_level, dim=1))  # L items of [N, A_l, C]

        detections: List[Dict[str, Tensor]] = []
        per_image_instances: List[ImageInstancesData] = []

        # ---- per image ----
        for i in range(N):
            img_scores_all: List[Tensor] = []
            img_classes_all: List[Tensor] = []
            img_keep_idxs_all: List[Tensor] = []

            # ---- per level ----
            for l, logits_l in enumerate(logits_by_level):
                logits_l = logits_l[i]  # [A_l, C]
                A_l, C = logits_l.shape

                # flatten so each anchor-class pair is a candidate
                scores = torch.sigmoid(logits_l).reshape(-1)  # [A_l*C]

                keep = scores > score_thresh
                if not torch.any(keep):
                    continue

                keep_idx = torch.where(keep)[0]  # indices in [0, A_l*C)
                keep_scores = scores[keep_idx]  # [K]

                # top-k per level
                k = min(topk_candidates, keep_idx.numel())
                if k < keep_idx.numel():
                    keep_scores, order = torch.topk(keep_scores, k, largest=True)
                    keep_idx = keep_idx[order]

                # decode (anchor_idx, class_idx) from flattened index
                anchor_idx = torch.div(keep_idx, C, rounding_mode="floor")  # [k] in [0..A_l-1]
                class_idx = keep_idx.remainder(C)  # [k]

                # map anchor indices to global (flattened across levels)
                global_anchor_idx = anchor_idx + level_offsets[l]  # [k] in [0..A_total-1]

                img_scores_all.append(keep_scores)
                img_classes_all.append(class_idx.to(torch.long))
                img_keep_idxs_all.append(global_anchor_idx.to(torch.long))

            if len(img_keep_idxs_all) == 0:
                # consistent empty outputs
                empty_scores = torch.empty((0,), device=device, dtype=cls_logits.dtype)
                empty_classes = torch.empty((0,), device=device, dtype=torch.long)
                empty_keep = torch.empty((0,), device=device, dtype=torch.long)

                inst = ImageInstancesData.from_keep(anchors=anchors_cat, keep_idxs=empty_keep)
                per_image_instances.append(inst)
                detections.append(
                    {
                        "anchor_centers": inst.anchor_centers[empty_keep],
                        "anchor_strides": inst.anchor_strides[empty_keep],
                        "classes": empty_classes,
                        "scores": empty_scores,
                        "keep_idxs": empty_keep,
                        "offset": offsets[i],
                    }
                )
                continue

            # concat across levels for this image
            keep_idxs = torch.cat(img_keep_idxs_all, dim=0)  # [K_total]
            scores = torch.cat(img_scores_all, dim=0)  # [K_total]
            classes = torch.cat(img_classes_all, dim=0)  # [K_total]

            # build per-image instance data (stores centers/strides for all anchors + keep indices)
            inst = ImageInstancesData.from_keep(anchors=anchors_cat, keep_idxs=keep_idxs)
            per_image_instances.append(inst)

            # centers/strides for selected candidates (K_total,3)
            det_centers = inst.anchor_centers[keep_idxs]
            det_strides = inst.anchor_strides[keep_idxs]

            detections.append(
                {
                    "anchor_centers": det_centers,
                    "anchor_strides": det_strides,
                    "classes": classes,
                    "scores": scores,
                    "keep_idxs": keep_idxs,
                    "offset": offsets[i],
                }
            )

        return detections, per_image_instances

    @torch.no_grad()
    def compute_detections_and_segmentations(self, inputs: Tensor) -> List[Dict[str, List[Tensor]]]:
        """
        Returns: length = batch_size (B)
          Each element is a dict with keys mapping to a list length = n_patches (P).
          Each patch-list element is a Tensor (possibly empty) for that patch.

        Expected inputs: [B, P, Ch, ph, pw, pd]
        """
        batch_size, n_patches = inputs.shape[:2]

        # Flatten patches -> [B*P, Ch, ph, pw, pd]
        flat_inputs = inputs.reshape(batch_size * n_patches, *inputs.shape[-4:])

        outputs = self.forward(flat_inputs)

        # IMPORTANT: this must return per-(flattened)-patch detections in same order as flat_inputs
        # detections: List[Dict[str, Tensor]] length = B*P
        # per_patch_instances: whatever your InstanceList expects, aligned with detections
        detections, per_patch_instances = self._compute_detections(outputs)

        inst_list = InstanceList(per_patch_instances, max_samples=-1)

        inst_logits = self.instance_segmentation_head(
            outputs["f_mask"],
            outputs["controller_logits"],
            inst_list,
        )

        stride = getattr(self.instance_segmentation_head, "stride", 1)
        needs_upsample = (stride > 1) if isinstance(stride, int) else any(s > 1 for s in stride)
        if needs_upsample:
            inst_logits = aligned_trilinear(inst_logits, stride)

        # Map instance logits back to each flattened patch index i in [0, B*P)
        img_idx = inst_list.get_image_indices()  # [M] indices into detections list (0..B*P-1)
        per_patch_logits: List[Tensor] = []
        for i in range(len(detections)):
            sel = (img_idx == i)
            per_patch_logits.append(inst_logits[sel])  # [Ki,1,ph,pw,pd] (Ki may be 0)

        mask_thresh = float(self.inference_hyperparams["mask_thresh"])
        nms_thresh = float(self.inference_hyperparams["nms_thresh"])

        detseg_per_patch: List[Dict[str, Tensor]] = []
        for i, det in enumerate(detections):
            logits_i = per_patch_logits[i]  # [K,1,ph,pw,pd] (logits)

            # boxes computed from thresholded mask (on logits)
            bboxes = get_onehot_instance_mask_boxes(torch.sigmoid(logits_i) >= mask_thresh)  # [K,6]

            out = {
                "anchor_centers": det["anchor_centers"],
                "anchor_strides": det["anchor_strides"],
                "classes": det["classes"],
                "scores": det["scores"],
                "onehot_logits": logits_i,
                "bboxes": bboxes,
                "offset": det["offset"],
            }

            if out["scores"].numel() > 1:
                keep = batched_nms(out["bboxes"], out["scores"], nms_thresh, metric="iom")
                for k in ("anchor_centers", "anchor_strides", "classes", "scores", "onehot_logits", "bboxes"):
                    out[k] = out[k][keep]

            detseg_per_patch.append(out)

        # -------- Unflatten: [B*P] -> per-image list of per-patch outputs --------
        # We return per image a dict of lists (each list has length P)
        per_image: List[Dict[str, List[Tensor]]] = []
        for b in range(batch_size):
            per_image.append({
                "anchor_centers": [],
                "anchor_strides": [],
                "classes": [],
                "scores": [],
                "onehot_logits": [],
                "bboxes": [],
                "offset": [],
            })

        for flat_i, out in enumerate(detseg_per_patch):
            b = flat_i // n_patches
            p = flat_i % n_patches
            # keep patch order stable (append in order)
            per_image[b]["anchor_centers"].append(out["anchor_centers"])
            per_image[b]["anchor_strides"].append(out["anchor_strides"])
            per_image[b]["classes"].append(out["classes"])
            per_image[b]["scores"].append(out["scores"])
            per_image[b]["onehot_logits"].append(out["onehot_logits"])
            per_image[b]["bboxes"].append(out["bboxes"])
            per_image[b]["offset"].append(out["offset"])

        return per_image

    @torch.no_grad()
    def postprocess(self, det_and_seg: List[Dict[str, List[Tensor]]]) -> List[Dict[str, List[Tensor]]]:
        mask_thresh = float(self.inference_hyperparams.get("mask_thresh", 0.5))
        output = []
        for per_img_det_and_seg in det_and_seg:
            per_img_output = {"anchor_centers": [], "anchor_strides": [], "classes": [], "scores": [], "offset": [],
                              "onehot_logits": [], "bboxes": []}
            onehot_mask_per_patch = per_img_det_and_seg["onehot_logits"]
            for i in range(len(onehot_mask_per_patch)):
                # 1) binarize + postproc
                onehot = onehot_mask_per_patch[i].squeeze(1) >= mask_thresh
                K = onehot.shape[0]
                if K > 0:
                    onehot_pp = self.postproc_transform(onehot)

                    # 2) remove empty instances
                    keep = onehot_pp.flatten(1).any(dim=1) # [K]
                    keep_idx = torch.where(keep)[0]

                    for key in ["anchor_centers", "anchor_strides", "classes", "scores", "offset"]:
                        per_img_output[key].append(per_img_det_and_seg[key][i][keep_idx])

                    logits_pp = (per_img_det_and_seg["onehot_logits"][i] * onehot_pp)[keep_idx]
                    bboxes_pp = get_onehot_instance_mask_boxes(logits_pp)
                    per_img_output["onehot_logits"].append(logits_pp)
                    per_img_output["bboxes"].append(bboxes_pp)
                else:
                    for key in ["anchor_centers", "anchor_strides", "classes", "scores", "offset", "onehot_logits", "bboxes"]:
                        per_img_output[key].append(per_img_det_and_seg[key][i])

            output.append(per_img_output)
        return output

    # ---------------- Forward ----------------
    def forward(self, inputs: MetaTensor) -> Any:
        # using chunks to reduce GPU memory usage
        chunk_size = int(self.forward_chunk_size)
        b_size = inputs.shape[0]
        shape = inputs.shape
        device = inputs.device
        offsets = None
        input_meta = inputs.meta
        spacing = input_meta["pixdim"][1:4]
        inputs = inputs.as_tensor()

        if "location" in input_meta:
            offsets = input_meta["location"]
            if isinstance(offsets, (list, tuple, np.ndarray)):
                offsets = torch.as_tensor(offsets, device=inputs.device)
            if offsets.ndim == 2 and offsets.shape[0] == 3:
                offsets = offsets.T  # [total_patches,3]
            offsets = offsets.to(device=inputs.device)

        chunked_outputs = []
        for i in range(0, b_size, chunk_size):
            input_chunk = inputs[i: i + chunk_size]

            f_mask, head_features = self.backbone(input_chunk)
            cls_logits = self.classification_head(head_features)
            controller_logits = self.controller_head(head_features)
            semantic_logits = self.semantic_head(f_mask)

            chunked_outputs.append({
                "cls_logits": cls_logits,
                "controller_logits": controller_logits,
                "f_mask": f_mask,
                "features": head_features,
                "semantic_logits": semantic_logits,
            })

        f_mask = torch.cat([c["f_mask"] for c in chunked_outputs], dim=0)
        semantic_logits = torch.cat([c["semantic_logits"] for c in chunked_outputs], dim=0)
        f_mask = MetaTensor(f_mask, meta=input_meta)
        semantic_logits = MetaTensor(semantic_logits, meta=input_meta)

        outputs = {
            "cls_logits": torch.cat([c["cls_logits"] for c in chunked_outputs], dim=0),
            "controller_logits": torch.cat([c["controller_logits"] for c in chunked_outputs], dim=0),
            "f_mask": f_mask,
            "features": [
                torch.cat([c["features"][j] for c in chunked_outputs], dim=0)
                for j in range(len(self.anchors_sizes))
            ],
            "semantic_logits": semantic_logits,
            "offsets": offsets,
            "shape": shape,
            "spacing": spacing,
            "device": device,
        }
        return outputs

    def on_train_epoch_start(self):
        dl = self.trainer.train_dataloader
        sampler = getattr(dl, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.trainer.current_epoch)

    # ---------------- Train / Val ----------------
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        inputs = batch["inputs"]
        targets = batch["targets"]
        bs = inputs.shape[0] if hasattr(inputs, "shape") else len(targets)

        outputs = self.forward(inputs)
        losses = self.compute_loss(outputs, targets)  # dict[str, Tensor]

        # sum losses
        loss = torch.stack([v for v in losses.values()]).sum()

        # build log dict
        log_step = {f"Train-Loss/{k}": v for k, v in losses.items()}
        log_step["Train-Loss/overall"] = loss

        self.log_dict(
            log_step,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=bs,
        )

        return loss

    def on_validation_start(self) -> None:
        # Choose once per fit unless user changed params.
        if not getattr(self, "images_to_visualize", None):
            self._assign_images_to_visualize()

    def validation_step(self, batch: dict, batch_idx: int) -> Any:
        targets = batch["targets"]  # list[dict]
        preds = self.predict_step(batch, batch_idx)

        metric_dict = self.metrics["validation"]

        # -------- metrics (fast path) --------
        for det, tgt in zip(preds, targets):
            scores = det["scores"]  # [K]

            # --- masks ---
            pred_onehot = _squeeze_onehot(det["onehot_instance_mask"])  # [K,H,W,D]
            gt_onehot = _squeeze_onehot(tgt["onehot"])  # [G,H,W,D]

            pairwise_mask_iou = mask_intersection_over_union(
                pred_onehot, gt_onehot, max_chunk_size=32
            )
            metric_dict["mask_cfm"].update(pairwise_mask_iou)
            metric_dict["mask_ap"].update(pairwise_mask_iou, scores)

            # --- boxes ---
            pred_boxes = _ensure_boxes_2d(det["bboxes"], "pred_boxes")
            gt_boxes = _ensure_boxes_2d(tgt["boxes"], "gt_boxes")

            pairwise_box_iou = box_intersection_over_union(pred_boxes, gt_boxes)  # [K,G]
            metric_dict["box_cfm"].update(pairwise_box_iou)
            metric_dict["box_ap"].update(pairwise_box_iou, scores)

        # -------- visualization gate --------
        vis_indices = set(self.images_to_visualize.get(0, []))
        do_vis = self.trainer.is_global_zero and (batch_idx in vis_indices)
        if not do_vis:
            return preds

        # Prefer full-volume input for plotting (no patch stitching)
        inputs_full = batch.get("inputs_orig", batch.get("inputs", None))
        cases = batch.get("case", None)

        # If inputs_full is missing, skip gracefully (still return preds)
        if inputs_full is None:
            return preds

        for i, (det, tgt) in enumerate(zip(preds, targets)):
            semantic_gt = tgt["semantic_mask"]
            img_to_show = torch.cat([inputs_full[i], semantic_gt], dim=0)

            scores = det["scores"]

            gt_onehot = _squeeze_onehot(tgt["onehot"])
            pred_onehot = _squeeze_onehot(det["onehot_instance_mask"])

            y_true = onehot_to_instance_mask(gt_onehot)
            y_pred = det["instance_mask"]

            true_ids = torch.unique(y_true)
            true_ids = true_ids[true_ids > 0]
            pred_ids = torch.unique(y_pred)
            pred_ids = pred_ids[pred_ids > 0]

            pairwise_mask_iou = mask_intersection_over_union(pred_onehot, gt_onehot)
            stats = get_stats(pairwise_mask_iou, y_true_ids=true_ids, y_pred_ids=pred_ids, scores=scores)

            title = str(cases[i]) if cases is not None else f"idx={i}"

            figs = self.instance_head_visualizer.plot(
                inputs=img_to_show,
                y_pred=y_pred,
                y_true=y_true,
                stats=[stats],
                title=title,
                add_info_text=True,
                boxes_true=tgt.get("boxes", None),
                boxes_pred=det.get("bboxes", None),
                boxes_scores=det.get("scores", None),
            )

            for t, fig in figs.items():
                self._log_figure(f"Images-{title}/{t}", fig)
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass

        return preds

    # ---------------- Figure logging ----------------
    def _log_figure(self, name: str, fig, *, close: bool = True) -> None:
        """
        Logs a matplotlib figure to the active experiment logger (e.g. TensorBoard).
        Only rank0 should create/submit figures to avoid duplicates.
        """
        if not getattr(self.trainer, "is_global_zero", True):
            return
        if self.logger is None:
            return
        exp = getattr(self.logger, "experiment", None)
        if exp is None:
            return

        # tensorboard SummaryWriter has add_figure
        add_figure = getattr(exp, "add_figure", None)
        if add_figure is None:
            return

        add_figure(name, fig, global_step=self.current_epoch)

        if close:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass

    # ---------------- scalar logging ----------------
    def _log_scalar(self, name: str, value: Tensor | float, *, sync_dist: bool = True) -> None:
        """
        Single place to control defaults.
        IMPORTANT: logger=True forced here.
        """
        self.log(
            name,
            _to_float(value),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,  # <-- forced TRUE
            batch_size=1,
            sync_dist=sync_dist,
        )

    @torch.no_grad()
    def _log_cfm_series(
            self,
            prefix: str,
            cfm: Tensor,  # [T, 3] -> TP,FP,FN
            iou_thresholds: Iterable[float],
            ap_per_thr: Optional[Tensor] = None,  # [T] optional
    ) -> None:
        precision = compute_precision(cfm)
        recall = compute_recall(cfm)
        f1 = compute_fi(cfm, i=1)
        f2 = compute_fi(cfm, i=2)

        for i, th in enumerate(iou_thresholds):
            th_str = f"{float(th):.2f}"

            tp, fp, fn = cfm[i][0], cfm[i][1], cfm[i][2]
            self._log_scalar(f"{prefix}/TP@{th_str}", tp)
            self._log_scalar(f"{prefix}/FP@{th_str}", fp)
            self._log_scalar(f"{prefix}/FN@{th_str}", fn)

            self._log_scalar(f"{prefix}/Precision@{th_str}", precision[i])
            self._log_scalar(f"{prefix}/Recall@{th_str}", recall[i])
            self._log_scalar(f"{prefix}/F1@{th_str}", f1[i])
            self._log_scalar(f"{prefix}/F2@{th_str}", f2[i])

            if ap_per_thr is not None:
                # if ap_per_thr isn't 1D, best-effort take scalar per threshold
                ap_i = ap_per_thr[i]
                ap_i = ap_i.mean() if isinstance(ap_i, Tensor) and ap_i.numel() > 1 else ap_i
                self._log_scalar(f"{prefix}/AP@{th_str}", ap_i)

    def on_validation_epoch_end(self) -> None:
        """
        Single validation dataloader: log BOTH mask and box metrics.
        Assumes self.metrics["validation"]["0"] contains:
          - mask_cfm, mask_ap
          - box_cfm, box_ap
        with .compute(), .plot() and .reset().
        """
        metric_dict = self.metrics["validation"]

        # ---------------- Masks ----------------
        mask_cfm: Tensor = metric_dict["mask_cfm"].compute()     # [T,3]
        mask_ap: Tensor = metric_dict["mask_ap"].compute()       # [T] or [T,...]
        mask_fig, _ = metric_dict["mask_ap"].plot()

        self._log_figure("Validation/Masks/PR-curve", mask_fig, close=True)
        self._log_scalar("Validation/Masks/mAP", _safe_mean(mask_ap))

        mask_thresholds = metric_dict["mask_cfm"].iou_thresholds
        self._log_cfm_series(
            prefix="Validation/Masks-IoU",
            cfm=mask_cfm,
            iou_thresholds=mask_thresholds,
            ap_per_thr=mask_ap if mask_ap.numel() else None,
        )

        # ---------------- Boxes ----------------
        box_cfm: Tensor = metric_dict["box_cfm"].compute()       # [T,3]
        box_ap: Tensor = metric_dict["box_ap"].compute()         # [T] or [T,...]
        box_fig, _ = metric_dict["box_ap"].plot()

        self._log_figure("Validation/Boxes/PR-curve", box_fig, close=True)
        self._log_scalar("Validation/Boxes/mAP", _safe_mean(box_ap))

        box_thresholds = metric_dict["box_cfm"].iou_thresholds
        self._log_cfm_series(
            prefix="Validation/Boxes-IoU",
            cfm=box_cfm,
            iou_thresholds=box_thresholds,
            ap_per_thr=box_ap if box_ap.numel() else None,
        )

        # ---------------- reset once ----------------
        metric_dict.reset()

    def predict_step(self, batch: dict, batch_idx, dataloader_idx=0) -> Any:
        inputs = batch["inputs"]
        output_shape = batch["inputs_orig"].shape[-3:]
        det = self.compute_detections_and_segmentations(inputs)
        # det = self.postprocess(det)

        # group instances in overlapping area
        mask_thresh = float(self.inference_hyperparams["mask_thresh"])
        group_thresh = float(self.inference_hyperparams["group_thresh"])
        detseg_per_image = [aggregate_per_patch_detections(x, group_thresh, mask_thresh, output_shape) for x in det]

        return detseg_per_image

    # ---------------- Optimizers ----------------
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
            Expects cfg to contain:
              cfg.optim.optimizer: Hydra config that instantiates torch.optim.Optimizer
              cfg.optim.scheduler: (optional) Hydra config that instantiates a LR scheduler
              cfg.optim.scheduler_cfg: (optional) Lightning scheduler wrapper params

            Returns:
              - optimizer
              - OR dict with optimizer + lr_scheduler
        """
        if not hasattr(self, "cfg"):
            raise AttributeError("Expected self.cfg to exist (store your Hydra cfg on the module).")

        opt_cfg = self.cfg.optim  # <-- Fix 2 expects scheduler here, not under optimizer

        if "optimizer" not in opt_cfg or opt_cfg.optimizer is None:
            raise ValueError("cfg.optim.optimizer must be provided.")

        # --- Optimizer ---
        optimizer = instantiate(opt_cfg.optimizer, params=self.parameters())

        # --- Scheduler config location (Fix 2) ---
        scheduler_cfg_node = opt_cfg.get("scheduler", None)

        # Backward-compatible fallback (in case an old config still nests it)
        if scheduler_cfg_node is None and opt_cfg.optimizer is not None:
            scheduler_cfg_node = opt_cfg.optimizer.get("scheduler", None)

        # No scheduler
        if scheduler_cfg_node is None:
            return optimizer

        # --- Scheduler ---
        scheduler = instantiate(scheduler_cfg_node, optimizer=optimizer)

        # Default Lightning wrapper
        sched_wrap: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

        # Optional wrapper settings (Fix 2: cfg.optim.scheduler_cfg)
        user_cfg = opt_cfg.get("scheduler_cfg", None)
        if user_cfg is not None:
            user_wrap = OmegaConf.to_container(user_cfg, resolve=True)
            if not isinstance(user_wrap, dict):
                raise ValueError("cfg.optim.scheduler_cfg must be a mapping/dict.")
            sched_wrap.update(user_wrap)

        # ReduceLROnPlateau requires monitor
        if scheduler.__class__.__name__ == "ReduceLROnPlateau":
            if "monitor" not in sched_wrap:
                raise ValueError(
                    "ReduceLROnPlateau requires cfg.optim.scheduler_cfg.monitor, e.g. "
                    "monitor: 'Validation/Masks/mAP'"
                )

        return {"optimizer": optimizer, "lr_scheduler": sched_wrap}