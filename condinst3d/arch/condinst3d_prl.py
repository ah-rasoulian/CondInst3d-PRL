from typing import Any, Dict, List, Tuple, Optional, Literal, Sequence, Union
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from monai.data import MetaTensor
from monai.transforms import Compose
import pytorch_lightning as pl
import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import random
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from condinst3d.arch.heads import ClassificationHead, ControllerHead, DynamicMaskHead
from condinst3d.evaluator.metrics import AveragePrecision, DetectionConfusionMatrix
from condinst3d.utils.detection import ImageInstancesData, InstanceList
from condinst3d.utils.anchors import AnisotropicATSSMatcher, generate_3d_anchors
from condinst3d.arch.backbone.abstract import AbstractBackbone
from condinst3d.utils.spatial import aligned_trilinear

# from deepnets.utils.detection import (
#     batched_nms, relabel_sequential, get_unique_labels, instance_mask_to_onehot,
#     onehot_to_instance_mask, priority_based_onehot_to_instance_mask,
#     generate_3d_anchors, get_instance_segmentation_centerness,
#     onehot_instance_mask_get_axial_eccentricity, batched_lbe
# )
# from deepnets.visualization.list_instance_boxseg_visualizer import ListInstanceBoxSegSliceVisualizer
# from deepnets.losses import *
# from deepnets.utils.evaluation import (
#     mask_intersection_over_union, get_stats,
#     compute_precision, compute_recall, compute_fi
# )
# from deepnets.utils.basic import aligned_trilinear, filter_dictionary_of_tensors, patch_tensor_concat
# from deepnets.utils.postprocessing import (
#     OnehotInstanceMaskPixelsAxialThreshold,
#     OnehotInstanceMaskEccentricityAxialThreshold, ConnectDisconnectedSlices,
#     AxialKeepLargestConnectedComponent
# )


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

        # -------------------- loss functions ---------------------
        self.losses = {
            "classification": instantiate(cfg.losses.classification),
            "segmentation": instantiate(cfg.losses.segmentation),
        }

        # -------------------- postproc params --------------------
        self.postproc_transform = None
        if "postproc" in cfg and cfg.postproc is not None:
            transforms = [instantiate(t) for t in cfg.postproc.transforms]
            self.postproc_transform = Compose(transforms)

        # -------------------- inference/matching hyperparams --------------------
        self.inference_hyperparams = {}
        self.set_inference_params(cfg.inference)

        # -------------------- metrics --------------------
        self.metrics = nn.ModuleDict({
            "validation": self._generate_metrics(cfg.evaluation),
            "test": self._generate_metrics(cfg.evaluation),
        })

        # # -------------------- visualization --------------------
        # img_channels = modalities + list(mcfg.visualization.extra_img_channels)
        # self.instance_head_visualizer = instantiate(
        #     cfg.visualization.instance_head_visualizer,
        #     img_channels=img_channels,
        #     channel_seg_under_image=freqmap_index,
        # )
        # self.images_to_visualize: Dict[int, List[int]] = {}
        # self.num_images_to_show = int(mcfg.visualization.num_images_to_show)

    def _generate_metrics(self, eval_cfg: Dict):
        metrics = torchmetrics.MetricCollection({
                "mask_cfm": DetectionConfusionMatrix(iou_thresholds=eval_cfg.iou_list),
                "mask_ap": AveragePrecision(iou_thresholds=eval_cfg.iou_list, interpolation=eval_cfg.ap_n_interp),
                "box_cfm": DetectionConfusionMatrix(iou_thresholds=eval_cfg.iou_list),
                "box_ap": AveragePrecision(iou_thresholds=eval_cfg.iou_list, interpolation=eval_cfg.ap_n_interp),
        })
        return metrics

    def set_inference_params(self, infer_cfg: Dict):
        params = {
            "mask_threshold": float(infer_cfg.mask_threshold),
            "score_thresh": float(infer_cfg.score_thresh),
            "nms_thresh": float(infer_cfg.nms_thresh),
            "topk_candidates": int(infer_cfg.topk_candidates),
            "detections_per_img": int(infer_cfg.detections_per_img),
        }
        self.inference_hyperparams = params

    # ---------------- Viz helpers ----------------
    def log_figure(self, name, figure):
        if self.logger is not None:
            self.logger.experiment.add_figure(name, figure, self.current_epoch)

    def assign_images_to_visualize(self, seed=42):
        if self.trainer.is_global_zero:
            val_loader = self.trainer.val_dataloaders
            if not isinstance(val_loader, list):
                val_loader = [val_loader]

            for i, loader in enumerate(val_loader):
                if self.num_images_to_show >= 0:
                    num_images_to_show = min(self.num_images_to_show, len(loader))
                    random.seed(seed)
                    self.images_to_visualize[i] = random.sample(range(len(loader)), k=num_images_to_show)
                else:
                    self.images_to_visualize[i] = list(range(len(loader)))

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


    # ---------------- Losses ----------------
    def _compute_detection_loss(self, outputs, targets):
        features = outputs["features"]

        anchors_per_level = generate_3d_anchors(
            outputs["shape"],
            [f.shape for f in features],
            anchor_sizes=self.anchors_sizes,
            device=outputs["device"],
        )

        instance_data_list = self._match_anchors_atss(anchors_per_level, targets)

        cls_logits = outputs["cls_logits"]  # [B, A, C]

        all_gt_class_ids = torch.stack([x.gt_classes for x in instance_data_list])  # [B, A]
        foreground_mask = all_gt_class_ids >= 0  # [B, A]

        num_foreground = int(foreground_mask.sum().item())

        gt_onehot = torch.zeros_like(cls_logits)

        if num_foreground > 0:
            b_idx, a_idx = foreground_mask.nonzero(as_tuple=True)  # both [Nfg]
            c_idx = all_gt_class_ids[b_idx, a_idx].long()  # [Nfg]
            gt_onehot[b_idx, a_idx, c_idx] = 1.0

        loss_cls = self.losses["classification"](cls_logits, gt_onehot)  # [B, A, C] if reduction="none"
        loss_cls = loss_cls.sum() / max(1, num_foreground)

        return loss_cls, instance_data_list

    def _compute_segmentation_loss(
            self,
            outputs: Dict[str, Tensor],
            targets,
            instance_list: "InstanceList",
    ) -> Tensor:
        """
        Computes instance segmentation loss.

        Expects:
          outputs["instance_logits"]: Tensor [N, 1(or C), W, H, D] (example)
          self.instance_segmentation_head.stride: int or (sx, sy, sz)
          self.segmentation_loss: callable(pred_logits, gt_masks) -> Tensor
          instance_list.get_gt_mask(targets, dtype) -> Tensor aligned with instance_logits
        """

        instance_logits = outputs["instance_logits"]
        stride = getattr(self.instance_segmentation_head, "stride", 1)

        # upsample if needed (supports int or tuple stride)
        needs_upsample = (stride > 1) if isinstance(stride, int) else any(s > 1 for s in stride)
        if needs_upsample:
            instance_logits = aligned_trilinear(instance_logits, stride)

        n_insts = len(instance_list)
        if n_insts == 0:
            return instance_logits.sum() * 0.0

        gt_masks = instance_list.get_gt_mask(targets, dtype=instance_logits.dtype)
        loss = self.losses["segmentation"](instance_logits, gt_masks)
        if loss.ndim > 0:
            loss = loss.sum() / max(1, n_insts)
        return loss

    def compute_loss(self, outputs, targets):
        classification_loss, instance_data_list = self._compute_detection_loss(outputs, targets)

        instance_list = InstanceList(instance_data_list, self.max_mask_to_train)

        outputs["instance_logits"] = self.instance_segmentation_head(
            outputs["f_mask"],
            outputs["controller_logits"],
            instance_list,
        )

        instance_segmentation_loss = self._compute_segmentation_loss(
            outputs=outputs,
            targets=targets,
            instance_list=instance_list,
        )

        return {
            "classification": classification_loss,
            "instance_segmentation": instance_segmentation_loss,
        }

    # ---------------- Inference: detections + seg ----------------
    def compute_detections(
        self,
        image_shape: Sequence[int],
        outputs,
    ) -> Tuple[List[Dict[str, Tensor]], List[ImageInstancesData]]:
        N, _, w, h, d = image_shape
        features = outputs["features"]

        anchors_all_levels = generate_3d_anchors(
            image_shape,
            [f.shape for f in features],
            self.anchors_sizes,
            device=self.device
        )
        num_anchors_per_level = [len(x) for x in anchors_all_levels]
        num_anchors_cumsum = [0] + np.cumsum(num_anchors_per_level).tolist()

        split_cls_logits = list(outputs["cls_logits"].split(num_anchors_per_level, dim=1))
        detections: List[Dict[str, Tensor]] = []
        instance_list = []

        for index in range(N):
            logits_per_image = [cl[index] for cl in split_cls_logits]

            image_scores, image_classes = [], []
            anchors_per_image_all_levels, anchor_idxs_all_levels = [], []

            for l, (logits_per_level, anchors_per_level) in enumerate(zip(logits_per_image, anchors_all_levels)):
                num_classes = logits_per_level.shape[-1]

                scores_per_level = torch.sqrt(torch.sigmoid(logits_per_level)).flatten()
                keep_idxs = scores_per_level > self.inference_hyperparams["score_thresh"]
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                num_topk = min(self.inference_hyperparams["topk_candidates"], topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                anchor_idxs_adjusted = anchor_idxs + num_anchors_cumsum[l]
                classes_per_level = topk_idxs % num_classes

                image_scores.append(scores_per_level)
                image_classes.append(classes_per_level)
                anchors_per_image_all_levels.append(anchors_per_level)
                anchor_idxs_all_levels.append(anchor_idxs_adjusted)

            image_scores = torch.cat(image_scores, dim=0)
            image_classes = torch.cat(image_classes, dim=0)
            anchors_per_image_all_levels = torch.cat(anchors_per_image_all_levels, dim=0)
            anchor_idxs_all_levels = torch.cat(anchor_idxs_all_levels, dim=0)

            anchor_centers = (anchors_per_image_all_levels[:, :3] + anchors_per_image_all_levels[:, 3:]) / 2
            anchor_strides = anchors_per_image_all_levels[:, 3:] - anchors_per_image_all_levels[:, :3]

            instance_data = ImageInstancesData(
                anchors_centers=anchor_centers,
                anchors_strides=anchor_strides,
                keep_idxs=anchor_idxs_all_levels,
            )
            instance_list.append(instance_data)

            detections.append({
                "anchor_centers": instance_data.get_pos_points(),
                "anchor_strides": instance_data.get_pos_strides(),
                "classes": image_classes,
                "scores": image_scores,
            })

        return detections, instance_list

    def compute_detections_and_segmentations(self, inputs):
        output_shape = self.get_out_shape(None)
        out_h, out_w, out_d = output_shape
        h, w, d = inputs.shape[-3:]
        n_patches = 0

        if h < out_h or w < out_w or d < out_d:
            if inputs.ndim == 6:
                batch_size, n_patches = inputs.shape[:2]
            else:
                batch_size = 1
                n_patches = inputs.shape[0]
            inputs = inputs.view(batch_size * n_patches, *inputs.shape[-4:])

        outputs = self.forward(inputs)
        detections, instance_data_list = self.compute_detections(inputs.shape, outputs)

        instance_list = InstanceList(instance_data_list)
        instance_logits = self.instance_segmentation_head(outputs["f_mask"], outputs["controller_logits"], instance_list)
        instance_logits = torch.sigmoid(instance_logits)

        instance_logits_list = []
        im_inds = instance_list.get_image_indices()
        for i in range(len(detections)):
            inds = torch.where(im_inds == i)
            instance_logits_list.append(instance_logits[inds])

        detections_and_segmentations = []
        for i in range(len(detections)):
            cntrness, bboxes = get_instance_segmentation_centerness(
                instance_logits_list[i] >= self.mask_threshold,
                detections[i]["anchor_centers"],
                True
            )
            centers = (bboxes[:, 3:] + bboxes[:, :3]) / 2
            strides = bboxes[:, 3:] - bboxes[:, :3]
            detections_and_segmentations.append(
                detections[i] | {
                    "centers": centers,
                    "strides": strides,
                    "cntrness": cntrness,
                    "onehot_mask": instance_logits_list[i],
                }
            )

        if n_patches > 0:
            for i in range(len(detections_and_segmentations)):
                scores = detections_and_segmentations[i]["scores"]
                if len(scores) > 1:
                    keep = batched_nms(
                        detections_and_segmentations[i]["onehot_mask"] >= self.mask_threshold,
                        scores,
                        self.nms_thresh,
                        metric="iom"
                    )
                    for key in list(detections_and_segmentations[i].keys()):
                        detections_and_segmentations[i][key] = detections_and_segmentations[i][key][keep]

            aggregated = []
            for b in range(batch_size):
                start, end = b * n_patches, (b + 1) * n_patches
                det_and_seg_list = {key: [] for key in detections_and_segmentations[0].keys()}

                offsets = inputs.meta["location"].transpose(1, 0)[start:end]
                for det, offset in zip(detections_and_segmentations[start:end], offsets):
                    for key, val in det.items():
                        if key in ["anchor_centers", "centers"]:
                            det_and_seg_list[key].append(val + torch.tensor(offset, device=val.device))
                        else:
                            det_and_seg_list[key].append(val)

                det_and_seg = {
                    key: torch.cat(val, dim=0)
                    for key, val in det_and_seg_list.items()
                    if key != "onehot_mask"
                }
                onehot_mask = patch_tensor_concat(
                    like_shape=(len(det_and_seg["scores"]), 1, out_h, out_w, out_d),
                    patches=det_and_seg_list["onehot_mask"],
                )
                aggregated.append(det_and_seg | {"onehot_mask": onehot_mask})

            detections_and_segmentations = [batched_lbe(d, self.batched_lbe_method) for d in aggregated]

            patch_semantic_logits = outputs["semantic_logits"]
            semantic_logits = patch_tensor_concat(
                like_shape=(batch_size, int(self.cfg.model.semantic_num_classes), out_h, out_w, out_d),
                patches=patch_semantic_logits.view(batch_size, n_patches, int(self.cfg.model.semantic_num_classes),
                                                   *patch_semantic_logits.shape[-3:])
            )
        else:
            semantic_logits = outputs["semantic_logits"]
            batch_size = semantic_logits.shape[0]

        semantic_mask = torch.argmax(torch.softmax(semantic_logits, dim=1), dim=1, keepdim=True)
        semantic_mask = semantic_mask.to(dtype=torch.float) / float(self.cfg.model.semantic_divisor)

        for i, per_img in enumerate(detections_and_segmentations):
            per_img.update({"semantic_mask": semantic_mask[i]})
        return detections_and_segmentations

    def postprocess(self, detection_and_segmentation):
        filter_keys = ["anchor_centers", "anchor_strides", "classes", "scores", "centers", "strides", "cntrness"]

        for i, per_img in enumerate(detection_and_segmentation):
            scores = per_img["scores"]
            onehot_logits = per_img["onehot_mask"]
            onehot_mask = (onehot_logits >= self.inference_hyperparams["mask_threshold"]).to(dtype=torch.bool, copy=False)
            semantic_mask = per_img["semantic_mask"]

            if len(scores) > 0:
                onehot_mask = self.postproc_transform(onehot_mask)
                pp_keep = torch.any(onehot_mask, dim=(1, 2, 3, 4))
                onehot_mask = onehot_mask[pp_keep]
                scores = scores[pp_keep]
                per_img = filter_dictionary_of_tensors(per_img, filter_keys, pp_keep)

                if onehot_mask.shape[0] > 0:
                    down_sampled_mask = F.max_pool3d(onehot_mask.float(), 2, 2) >= 0.5
                    down_sampled_mask_filled = torch.stack([fill_convex_hull_4d(x) for x in down_sampled_mask], dim=0)

                    down_sampled_semantic_mask = F.max_pool3d(semantic_mask, 2, 2)
                    semantic_consensus = torch.any(
                        down_sampled_mask_filled & (down_sampled_semantic_mask > float(self.cfg.model.semantic_consensus_threshold)),
                        dim=(1, 2, 3, 4)
                    )
                    scores.add_(semantic_consensus.float() * self.semantic_consensus_score)

                    nms_keep = batched_nms(down_sampled_mask_filled, scores, threshold=self.nms_thresh, metric="iom")
                    onehot_mask = onehot_mask[nms_keep]
                    scores = scores[nms_keep]
                    per_img = filter_dictionary_of_tensors(per_img, filter_keys, nms_keep)

            if self.stride > 1:
                onehot_mask = F.interpolate(onehot_mask.float(), scale_factor=self.stride, mode="trilinear") >= 0.5

            instance_mask = priority_based_onehot_to_instance_mask(onehot_mask=onehot_mask, scores=scores)

            remained_instances = list(get_unique_labels(instance_mask, is_onehot=False, discard=0))
            remained_instances = [x - 1 for x in remained_instances]
            per_img = filter_dictionary_of_tensors(per_img, filter_keys, remained_instances)

            instance_mask = relabel_sequential(instance_mask, exclude_background=True)
            mutually_exclusive_onehot_mask = instance_mask_to_onehot(instance_mask)

            per_img["onehot_mask"] = mutually_exclusive_onehot_mask
            per_img["instance_mask"] = instance_mask

            detection_and_segmentation[i] = per_img

        return detection_and_segmentation

    def get_out_shape(self, batch):
        return torch.size(1, 1, 192, 192, 192)

    # ---------------- Forward ----------------
    def forward(self, inputs) -> Any:
        # using chunks to reduce GPU memory usage
        chunk_size = int(self.forward_chunk_size)
        b_size = inputs.shape[0]

        is_meta_tensor = isinstance(inputs, MetaTensor)
        if is_meta_tensor:
            input_meta = inputs.meta
            inputs = inputs.as_tensor()

        chunked_outputs = []
        for i in range(0, b_size, chunk_size):
            input_chunk = inputs[i: i + chunk_size]

            f_mask, head_features = self.backbone(input_chunk)
            cls_logits = self.classification_head(head_features)
            controller_logits = self.controller_head(head_features)

            chunked_outputs.append({
                "cls_logits": cls_logits,
                "controller_logits": controller_logits,
                "f_mask": f_mask,
                "features": head_features,
            })

        f_mask = torch.cat([c["f_mask"] for c in chunked_outputs], dim=0)

        if is_meta_tensor:
            f_mask = MetaTensor(f_mask, meta=input_meta)

        outputs = {
            "cls_logits": torch.cat([c["cls_logits"] for c in chunked_outputs], dim=0),
            "controller_logits": torch.cat([c["controller_logits"] for c in chunked_outputs], dim=0),
            "f_mask": f_mask,
            "features": [
                torch.cat([c["features"][j] for c in chunked_outputs], dim=0)
                for j in range(len(self.anchors_sizes))
            ],
            "shape": inputs.shape,
            "device": inputs.device,
        }
        return outputs

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
        if len(self.images_to_visualize) == 0:
            self.assign_images_to_visualize()

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        targets = batch["targets"]
        detections_and_segmentations = self.predict_step(batch, batch_idx)

        image_to_show = None
        for image_index, det_per_image in enumerate(detections_and_segmentations):
            pred_onehot_mask = det_per_image["onehot_mask"]
            scores = det_per_image["scores"]
            semantic_mask = det_per_image["semantic_mask"]
            gt_masks = targets[image_index]["onehot"]

            pairwise_mask_iou = mask_intersection_over_union(pred_onehot_mask.squeeze(1), gt_masks)
            self.metrics["validation"][str(dataloader_idx)]["mask_cfm"].update(pairwise_mask_iou)
            self.metrics["validation"][str(dataloader_idx)]["mask_ap"].update(pairwise_mask_iou, scores)

            if self.trainer.is_global_zero and batch_idx in self.images_to_visualize.get(dataloader_idx, []):
                target_instance_mask = onehot_to_instance_mask(gt_masks)
                prediction_instance_mask = det_per_image["instance_mask"]
                stats = get_stats(pairwise_mask_iou, scores=scores)

                _, eccentricity = onehot_instance_mask_get_axial_eccentricity(det_per_image["onehot_mask"].unsqueeze(1))
                for ecc, st in zip(eccentricity, stats["y_pred"].values()):
                    st.update({"min-eccentricity": min(ecc)})

                if image_to_show is None:
                    current_input = batch["current_image"]
                    out_h, out_w, out_d = self.get_out_shape(batch)
                    h, w, d = current_input.shape[-3:]
                    if h < out_h or w < out_w or d < out_d:
                        if current_input.ndim == 6:
                            batch_size, n_patches, ch = current_input.shape[:3]
                        else:
                            batch_size = 1
                            n_patches, ch = current_input.shape[:2]
                        current_input = patch_tensor_concat(
                            like_shape=(batch_size, ch, out_h, out_w, out_d),
                            patches=current_input
                        )
                    image_to_show = torch.cat([
                        current_input,
                        batch["ct2f_stx"],
                        semantic_mask.unsqueeze(1),
                        batch.get("cprl_previous", torch.zeros_like(batch["ct2f_stx"])),
                    ], dim=1)

                title = "{}/{}/{}/{}".format(
                    batch["trial"][image_index],
                    batch["site"][image_index],
                    batch["subject"][image_index],
                    batch["timepoint"][image_index]
                )

                figs = self.instance_head_visualizer.plot(
                    inputs=image_to_show[image_index],
                    y_pred=prediction_instance_mask,
                    y_true=target_instance_mask,
                    stats=[stats],
                    title=title,
                    add_info_text=True,
                    boxes_true=None,
                    boxes_pred=None,
                    boxes_scores=None
                )
                for t, fig in figs.items():
                    fig_name = "Images- {}- {}/{} from {} - {}".format(
                        group,
                        title.replace("/", "-"),
                        batch["n_conf_lesions"][image_index] if "n_conf_lesions" in batch else "?",
                        batch["n_lesions"][image_index] if "n_lesions" in batch else "?",
                        t,
                    )
                    self.log_figure(fig_name, fig)
                    plt.close(fig)

        return detections_and_segmentations

    def on_validation_epoch_end(self) -> None:
        val_loaders = self.trainer.val_dataloaders
        if isinstance(val_loaders, DataLoader):
            val_loaders = [val_loaders]

        for dataloader_idx, _ in enumerate(val_loaders):
            didx = str(dataloader_idx)
            group = "cross_sectional" if didx == "0" else "longitudinal"
            metric_dict = self.metrics["validation"][didx]

            per_iou_cfm = metric_dict["mask_cfm"].compute()
            per_iou_ap = metric_dict["mask_ap"].compute()
            pr_fig, _ = metric_dict["mask_ap"].plot()

            precision = compute_precision(per_iou_cfm)
            recall = compute_recall(per_iou_cfm)
            f1 = compute_fi(per_iou_cfm, i=1)
            f2 = compute_fi(per_iou_cfm, i=2)

            self.log_figure(f"Validation-Masks/{group}/Precision-Recall curve", pr_fig)
            self.log(f"Validation-Masks/{group}/Mean AP", torch.mean(per_iou_ap),
                     on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

            for i, th in enumerate(metric_dict["mask_cfm"].iou_thresholds):
                self.log(f"Validation-Masks-IoU/{group}/TP at iou={th:.2f}", per_iou_cfm[i][0],
                         on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"Validation-Masks-IoU/{group}/FP at iou={th:.2f}", per_iou_cfm[i][1],
                         on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"Validation-Masks-IoU/{group}/FN at iou={th:.2f}", per_iou_cfm[i][2],
                         on_step=False, on_epoch=True, sync_dist=True)

                self.log(f"Validation-Masks-IoU/{group}/Precision at iou={th:.2f}", precision[i],
                         on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"Validation-Masks-IoU/{group}/Recall at iou={th:.2f}", recall[i],
                         on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"Validation-Masks-IoU/{group}/F1-score at iou={th:.2f}", f1[i],
                         on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"Validation-Masks-IoU/{group}/F2-score at iou={th:.2f}", f2[i],
                         on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"Validation-Masks-IoU/{group}/Average-Precision at iou={th:.2f}", per_iou_ap[i],
                         on_step=False, on_epoch=True, sync_dist=True)

            metric_dict.reset()

    def predict_step(self, batch: dict, batch_idx, dataloader_idx=0) -> Any:
        inputs = batch["inputs"]
        det = self.compute_detections_and_segmentations(inputs)
        det = self.postprocess(det)
        return det

        # ---------------- Optimizers ----------------
        def configure_optimizers(self) -> OptimizerLRScheduler:
            """
            Fully configurable with Hydra:
              cfg.optim.pretrain / cfg.optim.finetune / cfg.optim.slicewise
            """
            stage = str(self.train_stage)
            optim_cfg = self.cfg.optim.get(stage, None)
            if optim_cfg is None:
                raise ValueError(f"Missing cfg.optim.{stage} in config")

            # Optional: stage can freeze/unfreeze in cfg too
            if stage == "slicewise" and bool(self.cfg.optim.slicewise.freeze_all_then_unfreeze):
                for p in self.parameters():
                    p.requires_grad = False

                # unfreeze list of modules by config attribute names
                for name in list(self.cfg.optim.slicewise.unfreeze_modules):
                    module = self._resolve_attr(name)
                    for p in module.parameters():
                        p.requires_grad = True

                params = filter(lambda p: p.requires_grad, self.parameters())
            else:
                params = self.parameters()

            optim = instantiate(optim_cfg.optimizer, params=params)

            if "scheduler" in optim_cfg and optim_cfg.scheduler is not None:
                sched = instantiate(optim_cfg.scheduler, optimizer=optim)
                sched_dict = {"scheduler": sched, "interval": str(optim_cfg.get("interval", "epoch"))}
                return [optim], [sched_dict]
            return optim

        def _resolve_attr(self, dotted: str):
            """
            Resolve dotted attribute like 'backbone.encoder.specific_modality_inp'
            """
            obj = self
            for part in dotted.split("."):
                obj = getattr(obj, part)
            return obj