from typing import Any, Dict, List, Optional, Iterable
import random

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import torchmetrics
import matplotlib.pyplot as plt

from condinst3d.evaluator.iou import mask_intersection_over_union
from condinst3d.utils.detection import onehot_to_instance_mask
from condinst3d.visualization.utils import get_stats
from condinst3d.evaluator.metrics.cfm_based import (
    compute_precision,
    compute_fi,
    compute_recall,
)
from condinst3d.evaluator.metrics import (
    AveragePrecision,
    DetectionConfusionMatrix,
    GlobalConfluentInstanceRecall,
    DetectionFROC,
    SemanticDice,
)
from condinst3d.utils.mask import build_gt_cluster_ids
from condinst3d.visualization.list_instance_boxseg_visualizer import (
    ListInstanceBoxSegSliceVisualizer,
)


@torch.no_grad()
def plot_froc_curve(
    curve: Dict[str, torch.Tensor],
    fppi_max: float = 8.0,
    title: str = "FROC",
):
    x = curve["fppi"].detach().cpu()
    y = curve["sensitivity"].detach().cpu()

    fig = plt.figure(figsize=(7, 6))
    plt.plot(x, y, linewidth=2)
    plt.xlim(0, fppi_max)
    plt.ylim(0, 1.0)
    plt.xlabel("False Positives per Image (FPPI)")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def _to_float(x: Tensor | float) -> Tensor | float:
    return x


def _safe_mean(x: Tensor) -> Tensor:
    return x.mean() if x.numel() else x.new_tensor(0.0)


def _squeeze_onehot(onehot: torch.Tensor) -> torch.Tensor:
    if onehot.ndim == 5 and onehot.shape[1] == 1:
        return onehot.squeeze(1)
    return onehot


class ModelEvaluator(pl.LightningModule):
    def __init__(
        self,
        target_key: str,
        pred_key: str,
        iou_list: Optional[Iterable[float]] = None,
    ):
        super().__init__()
        self.pred_key = pred_key
        self.target_key = target_key

        if iou_list is None:
            iou_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        self.iou_list = [float(x) for x in iou_list]

        self.metrics = nn.ModuleDict({
            "test": torchmetrics.MetricCollection({
                "cfm": DetectionConfusionMatrix(iou_thresholds=self.iou_list),
                "ap": AveragePrecision(iou_thresholds=self.iou_list),
                "gcir": GlobalConfluentInstanceRecall(iou_thresholds=self.iou_list),
                "froc": DetectionFROC(
                    iou_thr=0.10,
                    n_thresholds=200,
                    score_min=0.0,
                    score_max=1.0,
                ),
                "semantic_dice": SemanticDice(),
            })
        })

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
        test_loader = self.trainer.test_dataloaders
        if isinstance(test_loader, list):
            return test_loader[0]
        return test_loader

    def _assign_images_to_visualize(self, seed: int = 42) -> None:
        if not self.trainer or not self.trainer.is_global_zero:
            return

        loader = self._get_single_test_loader()
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
            rng = random.Random(seed)
            chosen = sorted(rng.sample(range(n_batches), k=k))

        self.images_to_visualize = {0: chosen}

    def on_test_start(self) -> None:
        if not getattr(self, "images_to_visualize", None):
            self._assign_images_to_visualize()

    def test_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        targets = batch[self.target_key]
        preds = batch[self.pred_key]

        metric_dict = self.metrics["test"]

        for i, (det, tgt) in enumerate(zip(preds, targets)):
            scores = det["scores"]

            pred_onehot = _squeeze_onehot(det["onehot"])
            gt_onehot = _squeeze_onehot(tgt["onehot"])

            # semantic dice
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
                pred_onehot,
                gt_onehot,
                max_chunk_size=32,
            )
            metric_dict["cfm"].update(pairwise_mask_iou, scores)
            metric_dict["ap"].update(pairwise_mask_iou, scores)

            gt_cluster_ids = build_gt_cluster_ids(
                semantic_mask=tgt["semantic_mask"][0],
                instance_mask=tgt["instance_mask"][0],
                connectivity=26,
                min_instances_in_cluster=2,
            )
            metric_dict["gcir"].update(pairwise_mask_iou, scores, gt_cluster_ids)
            metric_dict["froc"].update(pairwise_mask_iou, scores)

        vis_indices = set(self.images_to_visualize.get(0, []))
        do_vis = self.trainer.is_global_zero and (batch_idx in vis_indices)
        if not do_vis:
            return

        inputs_full = batch.get("inputs")
        cases = batch.get("case", None)

        for i, (det, tgt) in enumerate(zip(preds, targets)):
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

            for t, fig in figs.items():
                self._log_figure(f"Images-{title}/{t}", fig, close=True)

        return

    def on_test_epoch_end(self) -> None:
        metric_dict = self.metrics["test"]

        mask_cfm = metric_dict["cfm"].compute()
        mask_ap = metric_dict["ap"].compute()
        mask_gcir = metric_dict["gcir"].compute()
        semantic_dice = metric_dict["semantic_dice"].compute()

        self._log_scalar("Metric:Average-Precision/mAP", _safe_mean(mask_ap))
        self._log_scalar("Metric:Global-Confluent-Instance-Recall/mGCIR", _safe_mean(mask_gcir))
        self._log_scalar("Metric:Semantic-Mask/Dice", semantic_dice)

        self._log_cfm_series(
            cfm=mask_cfm,
            iou_thresholds=metric_dict["cfm"].iou_thresholds,
            ap_per_thr=mask_ap if mask_ap.numel() else None,
            gcir_per_thr=mask_gcir if mask_gcir.numel() else None,
        )

        if self.trainer is not None and self.trainer.is_global_zero:
            mask_plot_fn = getattr(metric_dict["ap"], "plot", None)
            if callable(mask_plot_fn):
                mask_fig, _ = mask_plot_fn()
                self._log_figure("Metric:Figures/PR-curve", mask_fig, close=True)

        froc = metric_dict["froc"]
        curve = froc.compute()
        auc = froc.auc(fppi_max=8.0)
        ops = froc.sensitivity_at_fppi((0.5, 1, 2, 4, 8))

        self._log_scalar("Metric:FROC/FROC_AUC@8", auc)
        for fp, sens in ops.items():
            self._log_scalar(f"Metric:FROC/Sens@FPPI{fp:g}", sens)

        if self.trainer is not None and self.trainer.is_global_zero:
            fig = plot_froc_curve(curve, fppi_max=8.0, title="Masks FROC @ IoU=0.10")
            self._log_figure("Metric:Figures/FROC", fig, close=True)

        metric_dict.reset()

    def _log_figure(self, name: str, fig, *, close: bool = True) -> None:
        if not getattr(self.trainer, "is_global_zero", True):
            return
        if self.logger is None:
            return
        exp = getattr(self.logger, "experiment", None)
        if exp is None:
            return

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

    def _log_scalar(self, name: str, value: Tensor | float, *, sync_dist: bool = True) -> None:
        self.log(
            name,
            _to_float(value),
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
        cfm: Tensor,
        iou_thresholds: Iterable[float],
        ap_per_thr: Optional[Tensor] = None,
        gcir_per_thr: Optional[Tensor] = None,
    ) -> None:
        precision = compute_precision(cfm)
        recall = compute_recall(cfm)
        f1 = compute_fi(cfm, i=1)
        f2 = compute_fi(cfm, i=2)

        for i, th in enumerate(iou_thresholds):
            th_str = f"{float(th):.2f}"

            tp, fp, fn = cfm[i][0], cfm[i][1], cfm[i][2]
            self._log_scalar(f"Metric:True-Positives/TP@{th_str}", tp)
            self._log_scalar(f"Metric:False-Positives/FP@{th_str}", fp)
            self._log_scalar(f"Metric:False-Negatives/FN@{th_str}", fn)

            self._log_scalar(f"Metric:Precision/Precision@{th_str}", precision[i])
            self._log_scalar(f"Metric:Recall/Recall@{th_str}", recall[i])
            self._log_scalar(f"Metric:F1-score/F1@{th_str}", f1[i])
            self._log_scalar(f"Metric:F2-score/F2@{th_str}", f2[i])

            if ap_per_thr is not None:
                ap_i = ap_per_thr[i]
                if isinstance(ap_i, Tensor) and ap_i.numel() > 1:
                    ap_i = ap_i.mean()
                self._log_scalar(f"Metric:Average-Precision/AP@{th_str}", ap_i)

            if gcir_per_thr is not None:
                gcir_i = gcir_per_thr[i]
                if isinstance(gcir_i, Tensor) and gcir_i.numel() > 1:
                    gcir_i = gcir_i.mean()
                self._log_scalar(f"Metric:Global-Confluent-Instance-Recall/GCIR@{th_str}", gcir_i)
