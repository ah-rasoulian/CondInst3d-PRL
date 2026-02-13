from typing import Any, Dict, List, Optional, Iterable
import random
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from condinst3d.evaluator.iou import mask_intersection_over_union, box_intersection_over_union
from condinst3d.utils.detection import onehot_to_instance_mask
from condinst3d.visualization.utils import get_stats
from condinst3d.evaluator.metrics.cfm_based import compute_precision, compute_fi, compute_recall
from condinst3d.evaluator.metrics import AveragePrecision, DetectionConfusionMatrix
from condinst3d.visualization.list_instance_boxseg_visualizer import ListInstanceBoxSegSliceVisualizer
import torchmetrics

def _to_float(x: Tensor | float) -> Tensor | float:
    # keep tensors as tensors (Lightning likes tensors), but avoid accidental MetaTensor issues
    return x


def _safe_mean(x: Tensor) -> Tensor:
    # some AP implementations return shape [T] or [T, ...]
    return x.mean() if x.numel() else x.new_tensor(0.0)


def _squeeze_onehot(onehot: torch.Tensor) -> torch.Tensor:
    # Normalize to [N,H,W,D]
    if onehot.ndim == 5 and onehot.shape[1] == 1:
        return onehot.squeeze(1)
    return onehot

def _ensure_boxes_2d(boxes: torch.Tensor, name: str) -> torch.Tensor:
    # Accept empty [0,6], [6], or [G,6]
    if boxes.numel() == 0:
        return boxes.reshape(0, 6)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    assert boxes.ndim == 2 and boxes.shape[1] == 6, f"{name} must be [N,6], got {tuple(boxes.shape)}"
    return boxes


class ModelEvaluator(pl.LightningModule):
    def __init__(
            self,
            target_key,
            pred_key,
    ):
        super().__init__()
        self.pred_key = pred_key
        self.target_key = target_key

        iou_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
        ap_n_interp = 11
        # -------------------- metrics --------------------
        self.metrics = nn.ModuleDict({
            "test": torchmetrics.MetricCollection({
                "mask_cfm": DetectionConfusionMatrix(iou_thresholds=iou_list),
                "mask_ap": AveragePrecision(iou_thresholds=iou_list, interpolation=ap_n_interp),
                "box_cfm": DetectionConfusionMatrix(iou_thresholds=iou_list),
                "box_ap": AveragePrecision(iou_thresholds=iou_list, interpolation=ap_n_interp),
            })
        })

        # -------------------- visualization --------------------
        img_channels = ["t1w", "t2w", "flr", "freqmap"]
        self.instance_head_visualizer = ListInstanceBoxSegSliceVisualizer(
            crop_size=(96, 96),
            pred_seg_is_binary=False,
            draw_boxes=False,
            show_slices=(-1, 0, 1),
            figsize=2.0,
            img_channels=img_channels + ["semantic_mask"],
            channel_seg_under_image=3,
        )
        self.images_to_visualize: Dict[int, List[int]] = {}
        self.num_images_to_show = int(16)

    def _get_single_test_loader(self) -> DataLoader:
        """You said you only have one val dataloader."""
        test_loader = self.trainer.test_dataloaders
        if isinstance(test_loader, list):
            # take the first; you said only 1 exists
            return test_loader[0]
        return test_loader

    def _assign_images_to_visualize(self, seed: int = 42) -> None:
        """
        Pick a fixed set of batch indices to visualize from the (single) val dataloader.
        Stores them in self.images_to_visualize[0] as a sorted list[int].
        """
        if not self.trainer or not self.trainer.is_global_zero:
            return

        loader = self._get_single_test_loader()

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

    def on_test_start(self) -> None:
        # Choose once per fit unless user changed params.
        if not getattr(self, "images_to_visualize", None):
            self._assign_images_to_visualize()

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        targets = batch[self.target_key]  # list[dict]
        pred = batch[self.pred_key]
        has_mask = "onehot" in pred[0]

        metric_dict = self.metrics["test"]

        # -------- metrics (fast path) --------
        for det, tgt in zip(pred, targets):
            scores = det["scores"]  # [K]

            # --- boxes ---
            pred_boxes = _ensure_boxes_2d(det["boxes"], "pred_boxes")
            gt_boxes = _ensure_boxes_2d(tgt["boxes"], "gt_boxes")

            pairwise_box_iou = box_intersection_over_union(pred_boxes, gt_boxes)  # [K,G]
            metric_dict["box_cfm"].update(pairwise_box_iou)
            metric_dict["box_ap"].update(pairwise_box_iou, scores)

            if has_mask:
            # --- masks ---
                pred_onehot = _squeeze_onehot(det["onehot"])  # [K,H,W,D]
                gt_onehot = _squeeze_onehot(tgt["onehot"])  # [G,H,W,D]

                pairwise_mask_iou = mask_intersection_over_union(
                    pred_onehot, gt_onehot, max_chunk_size=32
                )
                metric_dict["mask_cfm"].update(pairwise_mask_iou)
                metric_dict["mask_ap"].update(pairwise_mask_iou, scores)

        if has_mask:
            # -------- visualization gate --------
            vis_indices = set(self.images_to_visualize.get(0, []))
            do_vis = self.trainer.is_global_zero and (batch_idx in vis_indices)
            if not do_vis:
                return

            # Prefer full-volume input for plotting (no patch stitching)
            inputs_full = batch.get("inputs")
            cases = batch.get("case", None)

            for i, (det, tgt) in enumerate(zip(pred, targets)):
                semantic_gt = tgt["semantic_mask"]
                img_to_show = torch.cat([inputs_full[i], semantic_gt], dim=0)

                scores = det["scores"]

                gt_onehot = _squeeze_onehot(tgt["onehot"])
                pred_onehot = _squeeze_onehot(det["onehot"])

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

        return

    def on_test_epoch_end(self) -> None:
        """
                Single validation dataloader: log BOTH mask and box metrics.
                Assumes self.metrics["validation"]["0"] contains:
                  - mask_cfm, mask_ap
                  - box_cfm, box_ap
                with .compute(), .plot() and .reset().
                """
        metric_dict = self.metrics["test"]

        # ---------------- Boxes ----------------
        box_cfm: Tensor = metric_dict["box_cfm"].compute()  # [T,3]
        box_ap: Tensor = metric_dict["box_ap"].compute()  # [T] or [T,...]
        box_fig, _ = metric_dict["box_ap"].plot()

        self._log_figure("Boxes/PR-curve", box_fig, close=True)
        self._log_scalar("Boxes/mAP", _safe_mean(box_ap))

        box_thresholds = metric_dict["box_cfm"].iou_thresholds
        self._log_cfm_series(
            prefix="Boxes-IoU",
            cfm=box_cfm,
            iou_thresholds=box_thresholds,
            ap_per_thr=box_ap if box_ap.numel() else None,
        )

        # ---------------- Masks ----------------
        if len(metric_dict["mask_ap"].scores) > 0:
            mask_cfm: Tensor = metric_dict["mask_cfm"].compute()  # [T,3]
            mask_ap: Tensor = metric_dict["mask_ap"].compute()  # [T] or [T,...]
            mask_fig, _ = metric_dict["mask_ap"].plot()

            self._log_figure("Masks/PR-curve", mask_fig, close=True)
            self._log_scalar("Masks/mAP", _safe_mean(mask_ap))

            mask_thresholds = metric_dict["mask_cfm"].iou_thresholds
            self._log_cfm_series(
                prefix="Masks-IoU",
                cfm=mask_cfm,
                iou_thresholds=mask_thresholds,
                ap_per_thr=mask_ap if mask_ap.numel() else None,
            )

        # ---------------- reset once ----------------
        metric_dict.reset()

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