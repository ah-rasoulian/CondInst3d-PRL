from typing import Dict, List, Any, Literal, Optional, Tuple, Iterable
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_class
import hydra
import torch
from torch import Tensor
import torch.nn as nn
import torchmetrics
import random
import os
from monai.data import MetaTensor
import numpy as np

from monai.transforms import Transform, Compose, KeepLargestConnectedComponent, RemoveSmallObjects
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.nn import BCEWithLogitsLoss

from condinst3d.utils.info import extract_input_metadata
from condinst3d.utils.spatial import aligned_trilinear
from condinst3d.arch.backbone.abstract import AbstractBackbone
from condinst3d.arch.heads.classification import ClassificationHead
from condinst3d.arch.heads.controller import ControllerHead
from condinst3d.arch.heads.dynamic_mask import DynamicMaskHead
from condinst3d.utils.anchors import AnisotropicATSSMatcher, generate_3d_anchors
from condinst3d.utils.detection import (ImageInstancesData, InstanceList, get_onehot_instance_mask_boxes,
                                        priority_based_onehot_to_instance_mask, instance_mask_to_onehot,
                                        onehot_to_instance_mask)
from condinst3d.io.transforms.remove_small_bbox import RemoveSmallBBox
from condinst3d.visualization.utils import get_stats
from condinst3d.utils.bached_ag import merge_patch_prediction, batched_nms, merge_semantic_logits
from condinst3d.evaluator.iou import mask_intersection_over_union, box_intersection_over_union
from condinst3d.evaluator.metrics.cfm_based import compute_recall, compute_fi, compute_precision
from condinst3d.utils.mask import build_gt_cluster_ids, connected_components
from condinst3d.evaluator.metrics import (AveragePrecision, GlobalConfluentInstanceRecall, DetectionConfusionMatrix,
                                          SemanticDice)
from condinst3d.visualization.list_instance_boxseg_visualizer import ListInstanceBoxSegSliceVisualizer
from condinst3d.utils.lr import instantiate_scheduler


def _squeeze_onehot_channel(x: Tensor) -> Tensor:
    if x.ndim == 5 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 4:
        return x
    raise ValueError(f"Unexpected onehot mask shape: {tuple(x.shape)}")

class CondInst3d(pl.LightningModule):
    """
    A 3D segmentation / instance-segmentation network.

    Modes
    -----
    semantic:
        Train only semantic segmentation on top of the backbone.
        Instance separation can later be obtained with connected components.

    instance:
        Train detection/controller + dynamic mask heads for
        CondInst-style instance separation.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        self._init_task_mode(cfg)
        self._init_model(cfg)
        self._init_training_components(cfg)
        self._init_metrics(cfg)
        self._init_visualization(cfg)

        if self.is_instance:
            self.load_backbone()

    def _init_task_mode(self, cfg: DictConfig) -> None:
        self.task_mode: str = str(cfg.task_mode).lower()
        if self.task_mode not in {"semantic", "instance"}:
            raise ValueError(
                f"cfg.task_mode must be 'semantic' or 'instance', got {cfg.task_mode!r}."
            )

    @property
    def is_semantic(self) -> bool:
        return self.task_mode == "semantic"

    @property
    def is_instance(self) -> bool:
        return self.task_mode == "instance"

    def _init_model(self, cfg: DictConfig) -> None:
        self.forward_chunk_size: int = int(cfg.forward_chunk_size)
        self.max_mask_to_train: int = int(cfg.max_mask_to_train)

        self.backbone: AbstractBackbone = instantiate(
            cfg.backbone,
            enable_heads=self.is_instance,
        )

        # selected detection head strides now come from backbone
        self.head_level_strides = None if self.is_semantic else torch.tensor(
            self.backbone.head_strides, dtype=torch.float32
        )

        self._init_heads(cfg)

    def _init_heads(self, cfg: DictConfig) -> None:
        if self.is_semantic:
            self.classification_head = None
            self.instance_segmentation_head = None
            self.controller_head = None
            return

        # instance mode only
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
            max_batch_size=cfg.heads.instance_segmentation.max_batch_size,
        )

        self.controller_head = ControllerHead(
            in_channels=self.backbone.heads_dim,
            num_params=self.instance_segmentation_head.num_params,
        )

    def _init_training_components(self, cfg: DictConfig) -> None:
        # matcher / anchors only needed for instance mode
        if self.is_instance:
            self.matcher = AnisotropicATSSMatcher(
                num_candidates=cfg.matcher.num_candidates,
                center_in_gt=cfg.matcher.center_in_gt,
            )
            self.anchors_sizes = cfg.matcher.anchors_sizes
        else:
            self.matcher = None
            self.anchors_sizes = None

        self.losses = {
            "semantic_segmentation": instantiate(cfg.losses.semantic_segmentation),
        }

        if self.is_instance:
            self.losses["classification"] = instantiate(cfg.losses.classification)
            self.losses["instance_segmentation"] = instantiate(cfg.losses.instance_segmentation)

        self.inference_hyperparams = cfg.inference

    @property
    def postproc_transform(self) -> Transform:
        return Compose([
            KeepLargestConnectedComponent(num_components=1, applied_labels=1, connectivity=1),
            RemoveSmallObjects(min_size=16, connectivity=1),
            RemoveSmallBBox(min_size=(5, 5, 2), threshold=0.5),
        ])

    def _generate_metrics(self, eval_cfg: Dict):
        metrics = torchmetrics.MetricCollection({
            "cfm": DetectionConfusionMatrix(iou_thresholds=eval_cfg.iou_list),
            "ap": AveragePrecision(iou_thresholds=eval_cfg.iou_list, interpolation=eval_cfg.ap_n_interp),
            "gcir": GlobalConfluentInstanceRecall(iou_thresholds=0.1),
            "semantic_dice": SemanticDice(),
        })
        return metrics

    def _init_metrics(self, cfg):
        self.metrics = nn.ModuleDict({
            "validation": self._generate_metrics(cfg.evaluation),
            "test": self._generate_metrics(cfg.evaluation),
        })

    def _init_visualization(self, cfg: DictConfig) -> None:
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
        self.images_to_visualize: Dict[str, List[int]] = {"train": [], "val": []}
        self.num_train_images_to_show = int(cfg.visualization.num_train_images_to_show)
        self.num_val_images_to_show = int(cfg.visualization.num_val_images_to_show)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if not hasattr(self, "cfg"):
            raise AttributeError("Expected self.cfg to exist.")

        opt_cfg = self.cfg.optim
        if opt_cfg is None or opt_cfg.get("optimizer", None) is None:
            raise ValueError("cfg.optim.optimizer must be provided.")

        freeze_backbone = bool(opt_cfg.get("freeze_backbone", False))
        backbone_lr_mult = float(opt_cfg.get("backbone_lr_mult", 1.0))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # base optimizer kwargs without params
        optimizer_cfg = OmegaConf.to_container(opt_cfg.optimizer, resolve=True)
        optimizer_target = optimizer_cfg.pop("_target_")
        optimizer_cfg.pop("params", None)

        optimizer_cls = hydra.utils.get_class(optimizer_target)

        if freeze_backbone:
            param_groups = [
                {
                    "params": [p for n, p in self.named_parameters() if
                               p.requires_grad and not n.startswith("backbone.")],
                }
            ]
        elif backbone_lr_mult != 1.0:
            base_lr = float(optimizer_cfg["lr"])
            param_groups = [
                {
                    "params": [p for p in self.backbone.parameters() if p.requires_grad],
                    "lr": base_lr * backbone_lr_mult,
                },
                {
                    "params": [
                        p for n, p in self.named_parameters()
                        if p.requires_grad and not n.startswith("backbone.")
                    ],
                    "lr": base_lr,
                },
            ]
        else:
            param_groups = [{"params": [p for p in self.parameters() if p.requires_grad]}]

        optimizer = optimizer_cls(param_groups, **optimizer_cfg)

        # optional scheduler
        sched_node = opt_cfg.get("scheduler", None)
        if sched_node is None:
            return optimizer

        scheduler = instantiate_scheduler(sched_node, optimizer)

        sched_wrap: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "name": "lr",
        }

        user_cfg = opt_cfg.get("scheduler_cfg", None)
        if user_cfg is not None:
            user_wrap = OmegaConf.to_container(user_cfg, resolve=True)
            if not isinstance(user_wrap, dict):
                raise ValueError("cfg.optim.scheduler_cfg must be a mapping/dict.")
            sched_wrap.update(user_wrap)

        if scheduler.__class__.__name__ == "ReduceLROnPlateau" and "monitor" not in sched_wrap:
            raise ValueError(
                "ReduceLROnPlateau requires cfg.optim.scheduler_cfg.monitor."
            )

        return {"optimizer": optimizer, "lr_scheduler": sched_wrap}

    def load_backbone(self) -> None:
        """
        Load backbone weights from a checkpoint for instance finetuning.

        Expected config:
            cfg.finetune.backbone_ckpt: path/to/semantic_checkpoint.ckpt
        """

        finetune_cfg = self.cfg.get("finetune", None)
        if finetune_cfg is None:
            return

        ckpt_path = finetune_cfg.get("backbone_ckpt", None)
        if ckpt_path is None:
            return

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Backbone checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" not in ckpt:
            raise RuntimeError("Checkpoint does not contain a Lightning state_dict.")

        state_dict = ckpt["state_dict"]

        backbone_state = {
            k.replace("backbone.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }

        if len(backbone_state) == 0:
            raise RuntimeError("No backbone weights found in checkpoint.")

        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)

        if missing:
            print(f"[Backbone load] Missing keys: {missing}")

        if unexpected:
            print(f"[Backbone load] Unexpected keys: {unexpected}")

        n = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Backbone load] {n / 1e6:.2f}M parameters loaded from: {ckpt_path}")

    def _assign_images_to_visualize(
            self,
            split: Literal["train", "val"],
            seed: int = 42,
    ) -> None:
        """
        Pick a fixed set of batch indices to visualize for a given split.
        Stores them in self.images_to_visualize[split].
        """
        if self.trainer is None or not self.trainer.is_global_zero:
            return

        if split == "train":
            loaders = self.trainer.train_dataloader
            num_to_show = self.num_train_images_to_show
        elif split == "val":
            loaders = self.trainer.val_dataloaders
            num_to_show = self.num_val_images_to_show
        else:
            raise ValueError(f"Unknown split: {split}")

        loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders

        if loader is None or not hasattr(loader, "__len__"):
            self.images_to_visualize[split] = []
            return

        n_batches = len(loader)
        if n_batches <= 0:
            self.images_to_visualize[split] = []
            return

        if num_to_show < 0:
            chosen = list(range(n_batches))
        else:
            k = min(int(num_to_show), n_batches)
            rng = random.Random(seed)
            chosen = sorted(rng.sample(range(n_batches), k=k))

        self.images_to_visualize[split] = chosen

    def _should_visualize_split(self, split: str, batch_idx: int) -> bool:
        if self.trainer is None or not self.trainer.is_global_zero:
            return False

        if split == "train":
            every_n_epochs = 5
            if self.current_epoch == 0 or self.current_epoch % every_n_epochs != 0:
                return False

        return batch_idx in self.images_to_visualize.get(split, [])

    def _visualize_batch(
            self,
            *,
            inputs: Tensor,  # expected [B, C, H, W, D]
            preds: list[dict],
            targets: list[dict],
            cases=None,
            prefix: str = "validation",
    ) -> None:
        """
        Pure visualization logic. Caller is responsible for gating.

        Assumes preds already contain final per-image outputs with keys like:
          - onehot_masks
          - scores
          - bboxes

        and targets contain:
          - semantic_mask
          - onehot
          - boxes
        """
        if inputs.ndim != 5:
            raise ValueError(
                f"_visualize_batch expects inputs of shape [B,C,H,W,D], got {tuple(inputs.shape)}"
            )

        for i, (det, tgt) in enumerate(zip(preds, targets)):
            x_i = inputs[i].detach()
            semantic_gt = tgt["semantic_mask"].detach()

            img_to_show = torch.cat([x_i, semantic_gt], dim=0)

            pred_onehot = _squeeze_onehot_channel(det["onehot_masks"])
            gt_onehot = _squeeze_onehot_channel(tgt["onehot"])

            y_pred = onehot_to_instance_mask(pred_onehot)
            y_true = onehot_to_instance_mask(gt_onehot)

            pred_ids = torch.unique(y_pred)
            pred_ids = pred_ids[pred_ids > 0]

            true_ids = torch.unique(y_true)
            true_ids = true_ids[true_ids > 0]

            scores = det.get("scores", None)
            pairwise_mask_iou = mask_intersection_over_union(pred_onehot, gt_onehot)

            stats = get_stats(
                pairwise_mask_iou,
                y_true_ids=true_ids,
                y_pred_ids=pred_ids,
                scores=scores,
            )

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

            for tag, fig in figs.items():
                self._log_figure(f"{prefix}-Images-{title}/{tag}", fig)

    def _log_figure(self, name: str, fig, *, close: bool = True) -> None:
        """
        Log a matplotlib figure to the active experiment logger.
        Only rank 0 should log figures.
        """
        if self.trainer is None or not getattr(self.trainer, "is_global_zero", True):
            return
        if self.logger is None:
            return

        exp = getattr(self.logger, "experiment", None)
        if exp is None:
            return

        add_figure = getattr(exp, "add_figure", None)
        if add_figure is None:
            return

        add_figure(name, fig, global_step=self.global_step)

        if close:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass

    def _log_scalar(
            self,
            name: str,
            value: Tensor | float,
            *,
            sync_dist: bool = True,
    ) -> None:
        """
        Centralized scalar logging helper.
        """
        self.log(
            name,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=sync_dist,
        )

    @torch.no_grad()
    def _log_cfm_series(
            self,
            prefix: str,
            cfm: Tensor,  # [T, 3] -> TP, FP, FN
            iou_thresholds: Iterable[float],
            ap_per_thr: Optional[Tensor] = None,
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

            if ap_per_thr is not None and ap_per_thr.numel() > 0:
                ap_i = ap_per_thr[i]
                if isinstance(ap_i, Tensor) and ap_i.numel() > 1:
                    ap_i = ap_i.mean()
                self._log_scalar(f"{prefix}/AP@{th_str}", ap_i)

    def on_train_start(self) -> None:
        self._assign_images_to_visualize("train", seed=42)

    def on_validation_start(self) -> None:
        if not self.images_to_visualize["val"]:
            self._assign_images_to_visualize("val", seed=42)

    def _forward_chunk(
            self,
            x_chunk: Tensor,
    ) -> Dict[str, Any]:
        """
        Forward a single chunk through the backbone and task-specific heads.
        """
        backbone_out = self.backbone(x_chunk)
        semantic_logits = self.backbone.forward_logits(backbone_out.semantic_output)

        out = {
            "semantic_output": backbone_out.semantic_output,
            "semantic_logits": semantic_logits,
            "decoder_outputs": backbone_out.decoder_outputs,
        }

        if self.is_instance:
            out["features"] = backbone_out.heads
            out["cls_logits"] = self.classification_head(backbone_out.heads)
            out["controller_logits"] = self.controller_head(backbone_out.heads)

        return out

    def _aggregate_chunk_outputs(
            self,
            chunked_outputs: List[Dict[str, Any]],
            metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Aggregate outputs from all chunks back into a single output dict.
        """
        semantic_output = torch.cat(
            [c["semantic_output"] for c in chunked_outputs],
            dim=0,
        )
        semantic_logits = torch.cat(
            [c["semantic_logits"] for c in chunked_outputs],
            dim=0,
        )

        input_meta = metadata.get("input_meta", {})
        outputs: Dict[str, Any] = {
            "semantic_output": MetaTensor(semantic_output, meta=input_meta),
            "semantic_logits": MetaTensor(semantic_logits, meta=input_meta),
            "decoder_outputs": [
                torch.cat([c["decoder_outputs"][j] for c in chunked_outputs], dim=0)
                for j in range(self.backbone.num_decoder_levels)
            ],
            "offsets": metadata.get("offsets", None),
            "input_shape": metadata["input_shape"],
            "spacing": metadata.get("spacing", None),
            "device": metadata["device"],
        }

        if self.is_instance:
            num_head_levels = len(self.backbone.head_indices)

            outputs["features"] = [
                torch.cat([c["features"][j] for c in chunked_outputs], dim=0)
                for j in range(num_head_levels)
            ]
            outputs["cls_logits"] = torch.cat(
                [c["cls_logits"] for c in chunked_outputs],
                dim=0,
            )
            outputs["controller_logits"] = torch.cat(
                [c["controller_logits"] for c in chunked_outputs],
                dim=0,
            )

        return outputs

    def forward(self, inputs: MetaTensor | Tensor) -> Dict[str, Any]:
        """
        Forward pass with chunking to reduce GPU memory usage.
        """
        metadata = extract_input_metadata(inputs)

        x = inputs.as_tensor() if isinstance(inputs, MetaTensor) else inputs

        chunk_size = int(self.forward_chunk_size)
        batch_size = int(x.shape[0])

        chunked_outputs: List[Dict[str, Any]] = []
        for start in range(0, batch_size, chunk_size):
            end = start + chunk_size
            x_chunk = x[start:end]
            chunked_outputs.append(self._forward_chunk(x_chunk))

        return self._aggregate_chunk_outputs(chunked_outputs, metadata)

    def _maybe_upsample_logits(
            self,
            logits: Tensor,
            stride: int | Tuple[int, int, int] | List[int],
    ) -> Tensor:
        """
        Upsample logits to input/grid resolution if stride > 1.
        """
        if isinstance(stride, int):
            needs_upsample = stride > 1
        else:
            needs_upsample = any(int(s) > 1 for s in stride)

        if not needs_upsample:
            return logits

        return aligned_trilinear(logits, stride)

    def _compute_semantic_loss(
            self,
            outputs: Dict[str, Any],
            targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Semantic-only loss branch.
        """
        semantic_logits = outputs["semantic_logits"]
        semantic_logits = self._maybe_upsample_logits(
            semantic_logits,
            self.backbone.semantic_stride,
        )

        semantic_gt = torch.stack([t["semantic_mask"] for t in targets], dim=0)
        semantic_gt = semantic_gt.to(device=semantic_logits.device, dtype=semantic_logits.dtype)

        loss_semantic = self.losses["semantic_segmentation"](semantic_logits, semantic_gt)
        return {"semantic_segmentation": loss_semantic}

    def _match_anchors_atss(
            self,
            anchors_per_level: List[Tensor],
            targets: List[Dict[str, Tensor]],
            spacing: Optional[Tensor] = None,
    ) -> List["ImageInstancesData"]:
        """
        Match anchors to GT boxes independently for each image.
        """
        anchors = torch.cat(anchors_per_level, dim=0)  # [A, 6]
        level_strides = self.expand_level_strides_to_anchors(anchors_per_level).to(anchors.device)

        num_anchors_per_level = [a.shape[0] for a in anchors_per_level]
        num_total_anchors = anchors.shape[0]
        device = anchors.device

        instance_data_list: List[ImageInstancesData] = []
        for targets_per_image in targets:
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                matched_idx = torch.full(
                    (num_total_anchors,),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
            else:
                _, matched_idx = self.matcher.compute_matches(
                    boxes=gt_boxes,
                    anchors=anchors,
                    num_anchors_per_level=num_anchors_per_level,
                    num_anchors_per_loc=1,
                    spacing=spacing,
                )

            instance_data = ImageInstancesData.from_targets(
                anchors=anchors,
                level_strides=level_strides,
                matched_idx=matched_idx,
                targets=targets_per_image,
            )
            instance_data_list.append(instance_data)

        return instance_data_list

    def _get_gt_instance_data_list(
            self,
            outputs: Dict[str, Any],
            targets: List[Dict[str, Tensor]],
    ) -> List["ImageInstancesData"]:
        """
        Build anchor matches / GT assignment for the current batch.
        """
        features = outputs["features"]

        anchors_per_level = generate_3d_anchors(
            outputs["input_shape"],
            [f.shape for f in features],
            anchor_sizes=self.anchors_sizes,
            device=outputs["device"],
        )

        return self._match_anchors_atss(
            anchors_per_level=anchors_per_level,
            targets=targets,
            spacing=outputs.get("spacing", None),
        )

    def _select_hard_negative_mask(
            self,
            loss_per_anchor: Tensor,
            pos_mask: Tensor,
            *,
            max_neg_per_pos: int = 3,
            min_negatives: int = 32,
    ) -> Tensor:
        """
        Select hard negatives based on highest per-anchor classification loss.

        Parameters
        ----------
        loss_per_anchor:
            Tensor of shape [B, A], already reduced over class dimension.

        pos_mask:
            Bool tensor of shape [B, A].

        max_neg_per_pos:
            Keep at most this many negatives per positive.

        min_negatives:
            When there are zero positives in an image, still keep at least this many
            hardest negatives if available.
        """
        B, A = loss_per_anchor.shape
        neg_mask = ~pos_mask
        selected_neg_mask = torch.zeros_like(neg_mask)

        for b in range(B):
            neg_idx = torch.nonzero(neg_mask[b], as_tuple=False).flatten()
            if neg_idx.numel() == 0:
                continue

            num_pos = int(pos_mask[b].sum().item())
            if num_pos > 0:
                k = min(neg_idx.numel(), max_neg_per_pos * num_pos)
            else:
                k = min(neg_idx.numel(), int(min_negatives))

            if k <= 0:
                continue

            neg_losses = loss_per_anchor[b, neg_idx]
            hard_order = torch.argsort(neg_losses, descending=True)
            chosen_neg_idx = neg_idx[hard_order[:k]]
            selected_neg_mask[b, chosen_neg_idx] = True

        return selected_neg_mask

    def _compute_instance_classification_loss(
            self,
            outputs: Dict[str, Any],
            targets: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, "InstanceList"]:
        """
        Instance classification loss with hard negative sampling.

        Returns
        -------
        classification_loss:
            Scalar tensor.

        instance_list:
            Filtered instance list used later by the dynamic mask head.
        """
        instance_data_list = self._get_gt_instance_data_list(outputs, targets)
        instance_list = InstanceList(instance_data_list, self.max_mask_to_train)

        cls_logits = outputs["cls_logits"]  # [B, A, C]
        B, A, C = cls_logits.shape
        device = cls_logits.device
        dtype = cls_logits.dtype

        gt_class_ids = torch.stack([x.gt_classes for x in instance_data_list], dim=0)  # [B, A]
        pos_mask = gt_class_ids >= 0
        neg_mask = ~pos_mask

        gt_onehot = torch.zeros((B, A, C), device=device, dtype=dtype)
        if pos_mask.any():
            b_idx, a_idx = pos_mask.nonzero(as_tuple=True)
            c_idx = gt_class_ids[b_idx, a_idx].long().clamp_(0, C - 1)
            gt_onehot[b_idx, a_idx, c_idx] = 1.0

        # expected reduction="none" -> [B, A, C]
        loss_per_entry = self.losses["classification"](cls_logits, gt_onehot)
        loss_per_anchor = loss_per_entry.sum(dim=-1)  # [B, A]

        hard_neg_ratio = int(self.cfg.losses.classification_mining.get("hard_negative_ratio", 3))
        hard_neg_min = int(self.cfg.losses.classification_mining.get("hard_negative_min", 32))

        selected_neg_mask = self._select_hard_negative_mask(
            loss_per_anchor=loss_per_anchor,
            pos_mask=pos_mask,
            max_neg_per_pos=hard_neg_ratio,
            min_negatives=hard_neg_min,
        )

        selected_mask = pos_mask | selected_neg_mask
        selected_loss = loss_per_anchor[selected_mask]

        if selected_loss.numel() == 0:
            loss_cls = cls_logits.sum() * 0.0
        else:
            normalizer = pos_mask.sum().to(dtype)
            if normalizer.item() == 0:
                normalizer = selected_mask.sum().to(dtype).clamp(min=1.0)
            else:
                normalizer = normalizer.clamp(min=1.0)

            loss_cls = selected_loss.sum() / normalizer

        return loss_cls, instance_list

    def _compute_instance_mask_loss(
            self,
            outputs: Dict[str, Any],
            targets: List[Dict[str, Tensor]],
            instance_list: InstanceList,
    ) -> Tensor:
        """
        Dynamic instance mask loss branch.
        """
        instance_logits = self.instance_segmentation_head(
            outputs["semantic_output"],
            outputs["controller_logits"],
            instance_list,
        )

        stride = getattr(self.instance_segmentation_head, "stride", 1)
        instance_logits = self._maybe_upsample_logits(instance_logits, stride)

        if len(instance_list) == 0:
            return (
                    instance_logits.sum() * 0.0
                    + outputs["controller_logits"].sum() * 0.0
            )

        gt_masks = instance_list.get_gt_mask(targets, dtype=instance_logits.dtype)
        return self.losses["instance_segmentation"](instance_logits, gt_masks)

    def compute_loss(
            self,
            outputs: Dict[str, Any],
            targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Dispatch loss computation by task mode.

        semantic mode:
            semantic loss only

        instance mode:
            classification + instance mask loss only
        """
        if self.is_semantic:
            return self._compute_semantic_loss(outputs, targets)

        loss_cls, instance_list = self._compute_instance_classification_loss(outputs, targets)
        loss_mask = self._compute_instance_mask_loss(outputs, targets, instance_list)

        return {
            "classification": loss_cls,
            "instance_segmentation": loss_mask,
        }

    def on_train_epoch_start(self):
        dataloader = self.trainer.train_dataloader

        # handle list or single loader
        loader = dataloader[0] if isinstance(dataloader, (list, tuple)) else dataloader

        sampler = getattr(loader, "sampler", None)

        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        inputs = batch["inputs"]
        targets = batch["targets"]
        cases = batch.get("case", None)

        bs = inputs.shape[0] if hasattr(inputs, "shape") else len(targets)

        outputs = self.forward(inputs)
        losses = self.compute_loss(outputs, targets)  # dict[str, Tensor]

        if not isinstance(losses, dict) or len(losses) == 0:
            raise RuntimeError("compute_loss must return a non-empty dict[str, Tensor].")

        loss = torch.stack([v for v in losses.values()]).sum()

        log_dict = {f"Train-Loss/{k}": v for k, v in losses.items()}
        log_dict["Train-Loss/overall"] = loss

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=bs,
        )

        # visualization
        if self._should_visualize_split("train", batch_idx):
            inputs = inputs[0:1, ...]
            targets = targets[0:1]
            cases = cases[0:1]
            with torch.no_grad():
                batch['inputs'] = inputs
                preds = self.predict_step(batch, batch_idx)

            self._visualize_batch(
                inputs=inputs,
                preds=preds,
                targets=targets,
                cases=cases,
                prefix="train",
            )

        return loss

    def expand_level_strides_to_anchors(self, anchors_per_level: List[torch.Tensor]) -> torch.Tensor:
        assert len(anchors_per_level) == self.head_level_strides.shape[0]
        per_level = []
        for l, a in enumerate(anchors_per_level):
            per_level.append(self.head_level_strides[l].view(1, 3).repeat(a.shape[0], 1))  # [Al,3]
        return torch.cat(per_level, dim=0)  # [A,3]

    @torch.no_grad()
    def _compute_instance_detections(
            self,
            outputs: Dict[str, Tensor],
    ) -> Tuple[List[Dict[str, Tensor]], List[ImageInstancesData]]:
        """
        Compute per-sample instance detection candidates from classification logits.

        Expects
        -------
        outputs["input_shape"]: tuple
        outputs["features"]: list[Tensor]
        outputs["cls_logits"]: Tensor [N, A_total, C]
        outputs["offsets"]: Tensor [N, 3] or None
        """
        cls_logits = outputs["cls_logits"]  # [N, A_total, C]
        features = outputs["features"]
        device = cls_logits.device
        N = int(cls_logits.shape[0])

        offsets = outputs.get("offsets", None)
        if offsets is None:
            offsets = torch.zeros((N, 3), device=device, dtype=torch.long)
        else:
            offsets = offsets.to(device=device)

        score_thresh = float(self.inference_hyperparams["score_thresh"])
        topk_candidates = int(self.inference_hyperparams["topk_candidates"])

        anchors_per_level = generate_3d_anchors(
            outputs["input_shape"],
            [f.shape for f in features],
            anchor_sizes=self.anchors_sizes,
            device=device,
        )

        num_anchors_per_level = [a.shape[0] for a in anchors_per_level]
        level_offsets = torch.as_tensor(
            np.cumsum([0] + num_anchors_per_level[:-1]),
            device=device,
            dtype=torch.long,
        )

        anchors_cat = torch.cat(anchors_per_level, dim=0)
        level_strides = self.expand_level_strides_to_anchors(anchors_per_level).to(device)

        logits_by_level = list(cls_logits.split(num_anchors_per_level, dim=1))

        detections: List[Dict[str, Tensor]] = []
        per_image_instances: List[ImageInstancesData] = []

        for i in range(N):
            img_scores_all: List[Tensor] = []
            img_classes_all: List[Tensor] = []
            img_keep_idxs_all: List[Tensor] = []

            for l, logits_l in enumerate(logits_by_level):
                logits_l = logits_l[i]  # [A_l, C]
                A_l, C = logits_l.shape

                scores_flat = torch.sigmoid(logits_l).reshape(-1)
                keep = scores_flat > score_thresh
                if not keep.any():
                    continue

                keep_idx = torch.where(keep)[0]
                keep_scores = scores_flat[keep_idx]

                k = min(topk_candidates, keep_idx.numel())
                if k < keep_idx.numel():
                    keep_scores, order = torch.topk(keep_scores, k, largest=True)
                    keep_idx = keep_idx[order]

                anchor_idx = torch.div(keep_idx, C, rounding_mode="floor")
                class_idx = keep_idx.remainder(C)
                global_anchor_idx = anchor_idx + level_offsets[l]

                img_scores_all.append(keep_scores)
                img_classes_all.append(class_idx.long())
                img_keep_idxs_all.append(global_anchor_idx.long())

            if len(img_keep_idxs_all) == 0:
                empty_scores = torch.empty((0,), device=device, dtype=cls_logits.dtype)
                empty_classes = torch.empty((0,), device=device, dtype=torch.long)
                empty_keep = torch.empty((0,), device=device, dtype=torch.long)

                inst = ImageInstancesData.from_keep(
                    anchors=anchors_cat,
                    level_strides=level_strides,
                    keep_idxs=empty_keep,
                )
                per_image_instances.append(inst)
                detections.append({
                    "anchor_centers": torch.empty((0, 3), device=device, dtype=anchors_cat.dtype),
                    "anchor_strides": torch.empty((0, 3), device=device, dtype=level_strides.dtype),
                    "classes": empty_classes,
                    "scores": empty_scores,
                    "keep_idxs": empty_keep,
                    "offsets": torch.empty((0, 3), device=device, dtype=offsets.dtype),
                })
                continue

            keep_idxs = torch.cat(img_keep_idxs_all, dim=0)
            scores = torch.cat(img_scores_all, dim=0)
            classes = torch.cat(img_classes_all, dim=0)

            inst = ImageInstancesData.from_keep(
                anchors=anchors_cat,
                level_strides=level_strides,
                keep_idxs=keep_idxs,
            )
            per_image_instances.append(inst)

            det_centers = inst.anchor_centers[keep_idxs]
            det_strides = inst.anchor_strides[keep_idxs]
            off_i = offsets[i].view(1, 3).expand(det_centers.shape[0], 3)

            detections.append({
                "anchor_centers": det_centers,
                "anchor_strides": det_strides,
                "classes": classes,
                "scores": scores,
                "keep_idxs": keep_idxs,
                "offsets": off_i,
            })

        return detections, per_image_instances

    @torch.no_grad()
    def _decode_instance_segmentations(
            self,
            outputs: Dict[str, Tensor],
            detections: List[Dict[str, Tensor]],
            per_image_instances: List["ImageInstancesData"],
    ) -> List[Dict[str, Tensor]]:
        """
        Decode dynamic masks for the selected instance detections.
        """
        inst_list = InstanceList(per_image_instances, max_samples=-1)

        instance_logits = self.instance_segmentation_head(
            outputs["semantic_output"],
            outputs["controller_logits"],
            inst_list,
        )

        stride = getattr(self.instance_segmentation_head, "stride", 1)
        if isinstance(stride, int):
            needs_upsample = stride > 1
        else:
            needs_upsample = any(int(s) > 1 for s in stride)

        if needs_upsample:
            instance_logits = aligned_trilinear(instance_logits, stride)

        img_idx = inst_list.get_image_indices()

        decoded: List[Dict[str, Tensor]] = []
        for flat_i, det in enumerate(detections):
            sel = (img_idx == flat_i)
            det_out = dict(det)
            det_out["onehot_logits"] = instance_logits[sel]
            decoded.append(det_out)

        return decoded

    @torch.no_grad()
    def _predict_instance_full(
            self,
            inputs: Tensor,
    ) -> List[Dict[str, Tensor]]:
        """
        Instance inference for full images.
        inputs: [B, C, H, W, D]
        """
        outputs = self.forward(inputs)
        detections, per_image_instances = self._compute_instance_detections(outputs)
        decoded = self._decode_instance_segmentations(outputs, detections, per_image_instances)

        for det in decoded:
            det["onehot_prob"] = torch.sigmoid(det.pop("onehot_logits"))

        decoded = self.postprocess(decoded)
        decoded = self.create_instance_mask(decoded)
        return decoded

    @torch.no_grad()
    def _predict_instance_patches(
            self,
            inputs: Tensor,
            output_shape: Tuple[int, int, int],
    ) -> List[Dict[str, Tensor]]:
        """
        Instance inference for patch inputs.
        inputs: [B, P, C, H, W, D]
        """
        B, P = inputs.shape[:2]

        flat_inputs = inputs.reshape(B * P, *inputs.shape[-4:])
        outputs = self.forward(flat_inputs)

        detections, per_patch_instances = self._compute_instance_detections(outputs)
        decoded = self._decode_instance_segmentations(outputs, detections, per_patch_instances)

        per_image: List[Dict[str, List[Tensor]]] = [
            {
                "anchor_centers": [],
                "anchor_strides": [],
                "classes": [],
                "scores": [],
                "onehot_logits": [],
                "offsets": [],
            }
            for _ in range(B)
        ]

        for flat_i, det in enumerate(decoded):
            b = flat_i // P
            per_image[b]["anchor_centers"].append(det["anchor_centers"])
            per_image[b]["anchor_strides"].append(det["anchor_strides"])
            per_image[b]["classes"].append(det["classes"])
            per_image[b]["scores"].append(det["scores"])
            per_image[b]["onehot_logits"].append(det["onehot_logits"])
            per_image[b]["offsets"].append(det["offsets"])

        merged = self.merge_prediction(per_image, output_shape=output_shape)
        merged = self.postprocess(merged)
        merged = self.create_instance_mask(merged)
        return merged

    @torch.no_grad()
    def _semantic_logits_to_candidates(
            self,
            semantic_logits: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Convert one semantic logit map into raw instance candidates using connected components.

        Returns a dict compatible with postprocess(), with:
          - onehot_prob: [K, H, W, D]
          - scores: [K]
          - anchor_centers: [K, 3]
          - anchor_strides: [K, 3]
          - classes: [K]
        """
        if semantic_logits.ndim == 4:
            if semantic_logits.shape[0] != 1:
                raise ValueError(
                    f"Expected semantic logits with one channel, got shape {tuple(semantic_logits.shape)}"
                )
            semantic_logits = semantic_logits[0]

        device = semantic_logits.device
        prob = torch.sigmoid(semantic_logits)  # [H, W, D]
        mask_thresh = float(self.inference_hyperparams.get("mask_thresh", 0.5))
        binary = prob >= mask_thresh

        empty_long = torch.empty((0,), device=device, dtype=torch.long)
        empty_float = torch.empty((0,), device=device, dtype=prob.dtype)
        empty_vec3 = torch.empty((0, 3), device=device, dtype=prob.dtype)
        empty_prob = torch.empty((0, *prob.shape), device=device, dtype=prob.dtype)

        if not binary.any():
            return {
                "anchor_centers": empty_vec3,
                "anchor_strides": empty_vec3,
                "classes": empty_long,
                "scores": empty_float,
                "onehot_prob": empty_prob,
            }

        cc = connected_components(binary)
        instance_ids = torch.unique(cc)
        instance_ids = instance_ids[instance_ids > 0]

        onehot_probs = []
        scores = []
        centers = []

        for inst_id in instance_ids:
            inst_mask = (cc == inst_id)
            if not inst_mask.any():
                continue

            inst_prob = prob * inst_mask
            score = prob[inst_mask].mean()
            center = inst_mask.nonzero(as_tuple=False).float().mean(dim=0)

            onehot_probs.append(inst_prob)
            scores.append(score)
            centers.append(center)

        if len(onehot_probs) == 0:
            return {
                "anchor_centers": empty_vec3,
                "anchor_strides": empty_vec3,
                "classes": empty_long,
                "scores": empty_float,
                "onehot_prob": empty_prob,
            }

        onehot_prob_t = torch.stack(onehot_probs, dim=0)  # [K,H,W,D]
        scores_t = torch.stack(scores, dim=0)  # [K]
        centers_t = torch.stack(centers, dim=0).to(dtype=prob.dtype)
        strides_t = torch.ones_like(centers_t)
        classes_t = torch.zeros((len(onehot_probs),), device=device, dtype=torch.long)

        return {
            "anchor_centers": centers_t,
            "anchor_strides": strides_t,
            "classes": classes_t,
            "scores": scores_t,
            "onehot_prob": onehot_prob_t,
        }

    @torch.no_grad()
    def _predict_semantic_full(
            self,
            inputs: Tensor,
    ) -> List[Dict[str, Tensor]]:
        """
        Semantic inference for full images.
        inputs: [B, C, H, W, D]
        """
        outputs = self.forward(inputs)
        semantic_logits = outputs["semantic_logits"]

        stride = self.backbone.semantic_stride
        semantic_logits = self._maybe_upsample_logits(semantic_logits, stride)

        preds = []
        for i in range(semantic_logits.shape[0]):
            preds.append(self._semantic_logits_to_candidates(semantic_logits[i]))

        preds = self.postprocess(preds)
        preds = self.create_instance_mask(preds)
        return preds

    @torch.no_grad()
    def _predict_semantic_patches(
            self,
            inputs: Tensor,
            output_shape: Tuple[int, int, int],
    ) -> List[Dict[str, Tensor]]:
        """
        Semantic inference for patch inputs.
        inputs: [B, P, C, H, W, D]
        """
        B, P = inputs.shape[:2]

        flat_inputs = inputs.reshape(B * P, *inputs.shape[-4:])
        outputs = self.forward(flat_inputs)

        semantic_logits = outputs["semantic_logits"]
        semantic_logits = self._maybe_upsample_logits(
            semantic_logits,
            self.backbone.semantic_stride,
        )

        offsets = outputs.get("offsets", None)
        if offsets is None:
            raise RuntimeError("Patch inference requires offsets in outputs['offsets'].")

        per_image_logits = [[] for _ in range(B)]
        per_image_offsets = [[] for _ in range(B)]

        for flat_i in range(B * P):
            b = flat_i // P
            per_image_logits[b].append(semantic_logits[flat_i])
            per_image_offsets[b].append(offsets[flat_i])

        preds = []
        for b in range(B):
            merged_logits = merge_semantic_logits(
                patch_logits=per_image_logits[b],
                patch_offsets=per_image_offsets[b],
                output_shape=output_shape,
            )
            preds.append(self._semantic_logits_to_candidates(merged_logits))

        preds = self.postprocess(preds)
        preds = self.create_instance_mask(preds)
        return preds

    @torch.no_grad()
    def merge_prediction(
            self,
            det_and_seg: List[Dict[str, List[Tensor]]],
            output_shape: Tuple[int, int, int],
    ) -> List[Dict[str, Tensor]]:
        mask_thresh = float(self.inference_hyperparams.get("mask_thresh", 0.5))
        nms_thresh = float(self.inference_hyperparams.get("nms_thresh", 0.75))
        group_thresh = float(self.inference_hyperparams.get("group_thresh", 0.35))

        aggregated = []
        for per_img_det in det_and_seg:
            merged_img = {
                "anchor_centers": [],
                "anchor_strides": [],
                "classes": [],
                "scores": [],
                "onehot_logits": [],
                "offsets": [],
            }

            num_parts = len(per_img_det["scores"])
            for p in range(num_parts):
                scores_p = per_img_det["scores"][p]
                centers_p = per_img_det["anchor_centers"][p]
                strides_p = per_img_det["anchor_strides"][p]
                classes_p = per_img_det["classes"][p]
                logits_p = per_img_det["onehot_logits"][p]
                offsets_p = per_img_det["offsets"][p]

                if scores_p.numel() > 1:
                    probs_p = torch.sigmoid(logits_p)
                    masks_p = probs_p >= mask_thresh
                    keep = batched_nms(masks_p, scores_p, nms_thresh, metric="iom")

                    scores_p = scores_p[keep]
                    centers_p = centers_p[keep]
                    strides_p = strides_p[keep]
                    classes_p = classes_p[keep]
                    logits_p = logits_p[keep]
                    offsets_p = offsets_p[keep]

                merged_img["anchor_centers"].append(centers_p)
                merged_img["anchor_strides"].append(strides_p)
                merged_img["classes"].append(classes_p)
                merged_img["scores"].append(scores_p)
                merged_img["onehot_logits"].append(logits_p)
                merged_img["offsets"].append(offsets_p)

            for key in merged_img:
                merged_img[key] = torch.cat(merged_img[key], dim=0)

            aggregated.append(merged_img)

        return [
            merge_patch_prediction(
                per_img,
                output_shape=output_shape,
                mask_thresh=mask_thresh,
                group_iom_thresh=group_thresh,
            )
            for per_img in aggregated
        ]

    @torch.no_grad()
    def postprocess(self, det_and_seg: List[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
        """
        Unified postprocessing for both semantic and instance branches.

        Expected per-image keys:
          - anchor_centers: [K,3]
          - anchor_strides: [K,3]
          - classes: [K]
          - scores: [K]
          - onehot_prob: [K,H,W,D]/[K,1,H,W,D]

        Returns per-image dict with:
          - anchor_centers
          - anchor_strides
          - classes
          - scores
          - onehot_masks
        """
        mask_thresh = float(self.inference_hyperparams.get("mask_thresh", 0.5))
        nms_thresh = float(self.inference_hyperparams.get("nms_thresh", 0.75))

        output = []
        for per_img in det_and_seg:
            anchor_centers = per_img["anchor_centers"]
            anchor_strides = per_img["anchor_strides"]
            classes = per_img["classes"]
            scores = per_img["scores"]
            onehot_prob = _squeeze_onehot_channel(per_img["onehot_prob"])

            onehot_masks = onehot_prob >= mask_thresh

            if onehot_masks.numel() == 0:
                output.append({
                    "anchor_centers": anchor_centers,
                    "anchor_strides": anchor_strides,
                    "classes": classes,
                    "scores": scores,
                    "onehot_masks": onehot_masks,
                })
                continue

            K = onehot_masks.shape[0]

            if K > 0:
                onehot_masks_pp = []
                for k in range(K):
                    onehot_masks_pp.append(self.postproc_transform(onehot_masks[k].unsqueeze(0))[0])
                onehot_masks_pp = torch.stack(onehot_masks_pp, dim=0)
            else:
                onehot_masks_pp = onehot_masks

            keep_nonempty = onehot_masks_pp.flatten(1).any(dim=1)

            anchor_centers = anchor_centers[keep_nonempty]
            anchor_strides = anchor_strides[keep_nonempty]
            classes = classes[keep_nonempty]
            scores = scores[keep_nonempty]
            onehot_masks = onehot_masks_pp[keep_nonempty]

            if scores.numel() > 1:
                bboxes = get_onehot_instance_mask_boxes(onehot_masks.unsqueeze(1)).float()
                bboxes[:, 2] -= 0.5
                bboxes[:, 5] += 0.5

                keep_nms = batched_nms(bboxes, scores, threshold=nms_thresh, metric="iom")

                anchor_centers = anchor_centers[keep_nms]
                anchor_strides = anchor_strides[keep_nms]
                classes = classes[keep_nms]
                scores = scores[keep_nms]
                onehot_masks = onehot_masks[keep_nms]

            output.append({
                "anchor_centers": anchor_centers,
                "anchor_strides": anchor_strides,
                "classes": classes,
                "scores": scores,
                "onehot_masks": onehot_masks,
            })

        return output

    @torch.no_grad()
    def create_instance_mask(self, det_and_seg: List[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
        """
        Create final instance masks from postprocessed onehot masks and filter
        metadata to only the instances that remain after priority-based overlap
        resolution.

        Expected per-image keys:
          - anchor_centers
          - anchor_strides
          - classes
          - scores
          - onehot_masks
        """
        outputs = []

        for per_img in det_and_seg:
            onehot = per_img["onehot_masks"]
            scores = per_img["scores"]

            if onehot.numel() == 0:
                spatial_shape = onehot.shape[1:]
                if len(spatial_shape) != 3:
                    raise ValueError("Empty onehot_masks must have shape [0, H, W, D].")
                instance_mask = torch.zeros(
                    spatial_shape,
                    device=onehot.device,
                    dtype=torch.long,
                )
                bboxes = torch.empty((0, 6), device=onehot.device, dtype=torch.float32)

                out = dict(per_img)
                out["instance_mask"] = instance_mask
                out["bboxes"] = bboxes
                outputs.append(out)
                continue

            instance_mask = priority_based_onehot_to_instance_mask(
                onehot_mask=onehot,
                scores=scores,
            )

            remained_onehot, remained_ids = instance_mask_to_onehot(
                instance_mask,
                return_instance_ids=True,
            )

            if remained_onehot.ndim == 5 and remained_onehot.shape[1] == 1:
                remained_onehot = remained_onehot[:, 0]
            if remained_onehot.ndim == 3:
                remained_onehot = remained_onehot.unsqueeze(0)

            out = dict(per_img)

            if len(remained_onehot) != len(onehot):
                priority_keep = torch.isin(
                    torch.arange(1, len(onehot) + 1, device=remained_ids.device),
                    remained_ids,
                )
                for key in ["anchor_centers", "anchor_strides", "classes", "scores"]:
                    out[key] = out[key][priority_keep]

            out["onehot_masks"] = remained_onehot
            out["instance_mask"] = instance_mask
            out["bboxes"] = get_onehot_instance_mask_boxes(remained_onehot.unsqueeze(1))
            outputs.append(out)

        return outputs

    @torch.no_grad()
    def predict_step(self, batch: dict, batch_idx, dataloader_idx=0) -> Any:
        """
        Supports:
          full image input:  [B, C, H, W, D]
          patch image input: [B, P, C, H, W, D]
        """
        inputs = batch["inputs"]

        if inputs.ndim == 5:
            is_patch_input = False
            output_shape = tuple(inputs.shape[-3:])
        elif inputs.ndim == 6:
            is_patch_input = True
            output_shape = tuple(batch.get("inputs_orig", inputs[:, 0]).shape[-3:])
        else:
            raise ValueError(
                f"Expected inputs.ndim in {{5, 6}}, got shape {tuple(inputs.shape)}"
            )

        if self.is_semantic:
            if is_patch_input:
                return self._predict_semantic_patches(inputs, output_shape=output_shape)
            return self._predict_semantic_full(inputs)

        if is_patch_input:
            return self._predict_instance_patches(inputs, output_shape=output_shape)
        return self._predict_instance_full(inputs)

    def validation_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        targets = batch["targets"]  # list[dict]
        preds = self.predict_step(batch, batch_idx)

        metric_dict = self.metrics["validation"]

        # -------- metrics (fast path) --------
        for i, (det, tgt) in enumerate(zip(preds, targets)):
            scores = det["scores"]  # [K]

            # --- masks ---
            pred_onehot = _squeeze_onehot_channel(det["onehot_masks"])  # [K,H,W,D]
            gt_onehot = _squeeze_onehot_channel(tgt["onehot"])  # [G,H,W,D]

            # --- semantic dice ---
            if pred_onehot.shape[0] == 0:
                semantic_pred = torch.zeros(
                    gt_onehot.shape[1:],
                    device=gt_onehot.device,
                    dtype=torch.bool,
                )
            else:
                semantic_pred = pred_onehot.any(dim=0)

            if gt_onehot.shape[0] == 0:
                semantic_gt = torch.zeros(
                    pred_onehot.shape[1:],
                    device=pred_onehot.device,
                    dtype=torch.bool,
                )
            else:
                semantic_gt = gt_onehot.any(dim=0)

            metric_dict["semantic_dice"].update(semantic_pred, semantic_gt)

            pairwise_mask_iou = mask_intersection_over_union(
                pred_onehot, gt_onehot, max_chunk_size=32
            )
            metric_dict["cfm"].update(pairwise_mask_iou, scores)
            metric_dict["ap"].update(pairwise_mask_iou, scores)

            gt_cluster_ids = build_gt_cluster_ids(
                semantic_mask=tgt["semantic_mask"][0],
                instance_mask=batch["instance_mask"][i][0],
                connectivity=26,
                min_instances_in_cluster=2,
            )
            metric_dict["gcir"].update(pairwise_mask_iou, scores, gt_cluster_ids)

        # -------- visualization gate --------
        if self._should_visualize_split("val", batch_idx):
            inputs = batch.get("inputs_orig", batch.get("inputs", None))
            cases = batch.get("case", None)

            if inputs is not None:
                self._visualize_batch(
                    inputs=inputs,
                    preds=preds,
                    targets=targets,
                    cases=cases,
                    prefix="validation",
                )

        return preds

    def on_validation_epoch_end(self) -> None:
        metric_dict = self.metrics["validation"]

        mask_cfm = metric_dict["cfm"].compute()
        mask_ap = metric_dict["ap"].compute()
        mask_gcir = metric_dict["gcir"].compute()
        semantic_dice = metric_dict["semantic_dice"].compute()

        self._log_scalar("Validation/mAP", mask_ap.mean())
        self._log_scalar("Validation/GCIR", mask_gcir)
        self._log_scalar("Validation/Semantic-Dice", semantic_dice)

        self._log_cfm_series(
            prefix="Validation-Masks-IoU",
            cfm=mask_cfm,
            iou_thresholds=metric_dict["cfm"].iou_thresholds,
            ap_per_thr=mask_ap if isinstance(mask_ap, Tensor) and mask_ap.numel() > 0 else None,
        )


        if self.trainer is not None and self.trainer.is_global_zero:
            mask_plot_fn = getattr(metric_dict["ap"], "plot", None)
            if callable(mask_plot_fn):
                mask_fig, _ = mask_plot_fn()
                self._log_figure("Validation/PR-curve", mask_fig, close=True)

        metric_dict.reset()

    def test_step(self, batch: dict, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        pass
