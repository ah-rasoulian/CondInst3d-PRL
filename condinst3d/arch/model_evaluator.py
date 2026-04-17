from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Any
from pathlib import Path
import pandas as pd
import random
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import torchmetrics
import torch.distributed as dist
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
    title: str = "FROC",
    fppi_limit: float | None = None,
):
    x = curve["fppi"].detach().cpu()
    y = curve["sensitivity"].detach().cpu()

    fig = plt.figure(figsize=(7, 6))
    plt.plot(x, y, linewidth=2)

    if x.numel() > 0:
        x_max_data = float(x.max().item())
    else:
        x_max_data = 1.0

    if fppi_limit is None:
        x_max_plot = max(0.25, 1.05 * x_max_data)
    else:
        x_max_plot = min(float(fppi_limit), max(0.25, 1.05 * x_max_data))

    plt.xlim(0, x_max_plot)
    plt.ylim(0, 1.0)
    plt.xlabel("False Positives per Image (FPPI)")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def _safe_mean(x: Tensor) -> Tensor:
    return x.mean() if x.numel() else x.new_tensor(0.0)


def _to_log_value(x: Tensor | float | int) -> Tensor | float:
    if isinstance(x, Tensor):
        return x.float() if not x.is_floating_point() else x
    return float(x)


def _squeeze_onehot(onehot: Tensor) -> Tensor:
    if onehot.ndim == 5 and onehot.shape[1] == 1:
        return onehot.squeeze(1)
    return onehot


def _safe_empty_iou(pred_onehot: Tensor, gt_onehot: Tensor) -> Tensor:
    return pred_onehot.new_zeros(
        (pred_onehot.shape[0], gt_onehot.shape[0]),
        dtype=torch.float32,
    )


def _gt_size_bin_masks(gt_onehot: Tensor) -> Dict[str, Tensor]:
    if gt_onehot.shape[0] == 0:
        empty = torch.zeros((0,), dtype=torch.bool, device=gt_onehot.device)
        return {"small": empty, "medium": empty, "large": empty}

    gt_sizes = gt_onehot.sum(dim=(1, 2, 3)).long()
    return {
        "small": gt_sizes <= 57,
        "medium": (gt_sizes >= 58) & (gt_sizes <= 141),
        "large": gt_sizes >= 142,
    }


def _tensor_to_float(x: Tensor | float | int) -> float:
    if isinstance(x, Tensor):
        if x.numel() == 0:
            return 0.0
        return float(x.detach().float().item())
    return float(x)


def _dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


class ModelEvaluator(pl.LightningModule):
    def __init__(
        self,
        target_key: str,
        pred_key: str,
        iou_list: Optional[Iterable[float]] = None,
        log_froc_figure: bool = False,
        froc_plot_limit: float | None = None,
        per_case_csv_name: str = "per_case_metrics.csv",
        primary_case_iou: float = 0.10,
        save_per_case_csv: bool = True,
    ):
        super().__init__()

        self.pred_key = pred_key
        self.target_key = target_key
        self.log_froc_figure = bool(log_froc_figure)
        self.froc_plot_limit = froc_plot_limit
        self.per_case_csv_name = str(per_case_csv_name)
        self.primary_case_iou = float(primary_case_iou)
        self.save_per_case_csv = bool(save_per_case_csv)

        if iou_list is None:
            iou_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        self.iou_list = [float(x) for x in iou_list]

        self.primary_iou_idx = min(
            range(len(self.iou_list)),
            key=lambda i: abs(self.iou_list[i] - self.primary_case_iou),
        )
        self.primary_iou_value = self.iou_list[self.primary_iou_idx]

        self.metrics = nn.ModuleDict({
            "all": torchmetrics.MetricCollection({
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
            }),
            "small": torchmetrics.MetricCollection({
                "cfm": DetectionConfusionMatrix(iou_thresholds=self.iou_list),
            }),
            "medium": torchmetrics.MetricCollection({
                "cfm": DetectionConfusionMatrix(iou_thresholds=self.iou_list),
            }),
            "large": torchmetrics.MetricCollection({
                "cfm": DetectionConfusionMatrix(iou_thresholds=self.iou_list),
            }),
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
        self.num_images_to_show = 16
        self._per_case_rows: List[Dict[str, Any]] = []

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
        self._per_case_rows = []

    @torch.no_grad()
    def _build_case_data(
        self,
        det: dict,
        tgt: dict,
        batch: dict | None = None,
    ) -> Dict[str, Any]:
        scores = det["scores"]
        pred_onehot = _squeeze_onehot(det["onehot"])
        gt_onehot = _squeeze_onehot(tgt["onehot"])

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

        if pred_onehot.shape[0] == 0 or gt_onehot.shape[0] == 0:
            pairwise_mask_iou = _safe_empty_iou(pred_onehot, gt_onehot)
        else:
            pairwise_mask_iou = mask_intersection_over_union(
                pred_onehot,
                gt_onehot,
                max_chunk_size=32,
            )

        n_scores = int(scores.shape[0])
        n_pred_masks = int(pred_onehot.shape[0])
        n_gt = int(gt_onehot.shape[0])

        if n_scores != n_pred_masks:
            case_info = batch.get("case", "unknown") if batch is not None else "unknown"
            raise RuntimeError(
                f"Prediction mismatch: scores={n_scores}, "
                f"pred_onehot={n_pred_masks}, gt={n_gt}, case={case_info}"
            )

        gt_cluster_ids = build_gt_cluster_ids(
            semantic_mask=tgt["semantic_mask"][0],
            instance_mask=tgt["instance_mask"][0],
            connectivity=26,
            min_instances_in_cluster=2,
        )

        return {
            "scores": scores,
            "pred_onehot": pred_onehot,
            "gt_onehot": gt_onehot,
            "semantic_pred": semantic_pred,
            "semantic_gt": semantic_gt,
            "pairwise_mask_iou": pairwise_mask_iou,
            "gt_cluster_ids": gt_cluster_ids,
        }

    @torch.no_grad()
    def _make_single_case_metrics(self, device: torch.device):
        return {
            "cfm": DetectionConfusionMatrix(iou_thresholds=self.iou_list).to(device),
            "ap": AveragePrecision(iou_thresholds=self.iou_list).to(device),
            "gcir": GlobalConfluentInstanceRecall(iou_thresholds=self.iou_list).to(device),
            "froc": DetectionFROC(
                iou_thr=0.10,
                n_thresholds=200,
                score_min=0.0,
                score_max=1.0,
            ).to(device),
            "semantic_dice": SemanticDice().to(device),
        }

    @torch.no_grad()
    def _update_all_metrics(self, case_data: Dict[str, Any]) -> None:
        metric_all = self.metrics["all"]
        metric_all["semantic_dice"].update(case_data["semantic_pred"], case_data["semantic_gt"])
        metric_all["cfm"].update(case_data["pairwise_mask_iou"], case_data["scores"])
        metric_all["ap"].update(case_data["pairwise_mask_iou"], case_data["scores"])
        metric_all["gcir"].update(
            case_data["pairwise_mask_iou"],
            case_data["scores"],
            case_data["gt_cluster_ids"],
        )
        metric_all["froc"].update(case_data["pairwise_mask_iou"], case_data["scores"])

    @torch.no_grad()
    def _update_size_bin_metrics(self, case_data: Dict[str, Any]) -> None:
        gt_bin_masks = _gt_size_bin_masks(case_data["gt_onehot"])
        pairwise_mask_iou = case_data["pairwise_mask_iou"]
        scores = case_data["scores"]

        for bin_name in ("small", "medium", "large"):
            gt_mask = gt_bin_masks[bin_name]
            if gt_mask.numel() == 0 or not gt_mask.any():
                continue
            self.metrics[bin_name]["cfm"].update(pairwise_mask_iou[:, gt_mask], scores)

    @torch.no_grad()
    def _compute_case_row(
        self,
        case_name: str,
        case_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        pairwise_mask_iou = case_data["pairwise_mask_iou"]
        scores = case_data["scores"]
        gt_onehot = case_data["gt_onehot"]
        pred_onehot = case_data["pred_onehot"]
        gt_cluster_ids = case_data["gt_cluster_ids"]
        semantic_pred = case_data["semantic_pred"]
        semantic_gt = case_data["semantic_gt"]

        local = self._make_single_case_metrics(pairwise_mask_iou.device)
        local["cfm"].update(pairwise_mask_iou, scores)
        local["ap"].update(pairwise_mask_iou, scores)
        local["gcir"].update(pairwise_mask_iou, scores, gt_cluster_ids)
        local["froc"].update(pairwise_mask_iou, scores)
        local["semantic_dice"].update(semantic_pred, semantic_gt)

        cfm = local["cfm"].compute()
        ap = local["ap"].compute()
        gcir = local["gcir"].compute()
        semantic_dice = local["semantic_dice"].compute()
        froc_auc = local["froc"].auc(fppi_max=8.0)
        froc_ops = local["froc"].sensitivity_at_fppi((0.5, 1, 2, 4, 8))

        precision = compute_precision(cfm)
        recall = compute_recall(cfm)
        f1 = compute_fi(cfm, i=1)
        f2 = compute_fi(cfm, i=2)

        if gt_onehot.shape[0] > 0:
            gt_sizes = gt_onehot.sum(dim=(1, 2, 3)).long()
        else:
            gt_sizes = torch.empty(0, device=gt_onehot.device, dtype=torch.long)

        gt_bin_masks = _gt_size_bin_masks(gt_onehot)

        confluent_mask = gt_cluster_ids >= 0
        n_confluent_gt = int(confluent_mask.sum().item()) if gt_cluster_ids.numel() > 0 else 0
        confluent_cluster_ids = gt_cluster_ids[confluent_mask]
        n_confluent_clusters = (
            int(torch.unique(confluent_cluster_ids).numel())
            if confluent_cluster_ids.numel() > 0
            else 0
        )

        row: Dict[str, Any] = {
            "case": case_name,
            "n_gt": int(gt_onehot.shape[0]),
            "n_pred": int(pred_onehot.shape[0]),
            "n_confluent_gt": n_confluent_gt,
            "n_confluent_clusters": n_confluent_clusters,
            "gt_total_vox": int(gt_sizes.sum().item()) if gt_sizes.numel() > 0 else 0,
            "gt_mean_size_vox": float(gt_sizes.float().mean().item()) if gt_sizes.numel() > 0 else 0.0,
            "gt_median_size_vox": float(gt_sizes.float().median().item()) if gt_sizes.numel() > 0 else 0.0,
            "n_small_gt": int(gt_bin_masks["small"].sum().item()) if gt_bin_masks["small"].numel() > 0 else 0,
            "n_medium_gt": int(gt_bin_masks["medium"].sum().item()) if gt_bin_masks["medium"].numel() > 0 else 0,
            "n_large_gt": int(gt_bin_masks["large"].sum().item()) if gt_bin_masks["large"].numel() > 0 else 0,
            "semantic_dice": _tensor_to_float(semantic_dice),
            "ap_mean": _tensor_to_float(_safe_mean(ap)),
            f"ap@{self.primary_iou_value:.2f}": _tensor_to_float(ap[self.primary_iou_idx]),
            "gcir_mean": _tensor_to_float(_safe_mean(gcir)),
            f"gcir@{self.primary_iou_value:.2f}": _tensor_to_float(gcir[self.primary_iou_idx]),
            f"tp@{self.primary_iou_value:.2f}": _tensor_to_float(cfm[self.primary_iou_idx][0]),
            f"fp@{self.primary_iou_value:.2f}": _tensor_to_float(cfm[self.primary_iou_idx][1]),
            f"fn@{self.primary_iou_value:.2f}": _tensor_to_float(cfm[self.primary_iou_idx][2]),
            f"precision@{self.primary_iou_value:.2f}": _tensor_to_float(precision[self.primary_iou_idx]),
            f"recall@{self.primary_iou_value:.2f}": _tensor_to_float(recall[self.primary_iou_idx]),
            f"f1@{self.primary_iou_value:.2f}": _tensor_to_float(f1[self.primary_iou_idx]),
            f"f2@{self.primary_iou_value:.2f}": _tensor_to_float(f2[self.primary_iou_idx]),
            "froc_auc@8": _tensor_to_float(froc_auc),
            "sens@fppi0.5": _tensor_to_float(froc_ops[0.5]),
            "sens@fppi1": _tensor_to_float(froc_ops[1.0]),
            "sens@fppi2": _tensor_to_float(froc_ops[2.0]),
            "sens@fppi4": _tensor_to_float(froc_ops[4.0]),
            "sens@fppi8": _tensor_to_float(froc_ops[8.0]),
        }
        return row

    @torch.no_grad()
    def _gather_per_case_rows(self) -> List[Dict[str, Any]]:
        local_rows = [r for r in self._per_case_rows if isinstance(r, dict)]

        if not _dist_available_and_initialized():
            return local_rows

        gathered_rows = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_rows, local_rows)

        merged: List[Dict[str, Any]] = []
        for rows in gathered_rows:
            if rows is None:
                continue
            merged.extend([r for r in rows if isinstance(r, dict)])
        return merged

    def _resolve_output_dir(self) -> Path:
        if self.logger is not None:
            log_dir = getattr(self.logger, "log_dir", None)
            if log_dir:
                return Path(log_dir)

            save_dir = getattr(self.logger, "save_dir", None)
            name = getattr(self.logger, "name", None)
            version = getattr(self.logger, "version", None)
            if save_dir is not None and name is not None and version is not None:
                return Path(save_dir) / str(name) / str(version)

        return Path(getattr(self.trainer, "default_root_dir", "."))

    torch.no_grad()

    def _save_per_case_csv(self) -> None:
        if not self.save_per_case_csv:
            return

        rows = self._gather_per_case_rows()

        if not getattr(self.trainer, "is_global_zero", True):
            return

        rows = [r for r in rows if isinstance(r, dict)]
        if not rows:
            print("[ModelEvaluator] no per-case rows to save")
            return

        rows = sorted(rows, key=lambda x: str(x.get("case", "")))

        output_dir = self._resolve_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / self.per_case_csv_name

        df = pd.DataFrame(rows)

        # sort columns but keep "case" first
        cols = sorted(df.columns)
        if "case" in cols:
            cols.remove("case")
            cols = ["case"] + cols

        df = df[cols]

        df.to_csv(csv_path, index=False)

        print(f"[ModelEvaluator] saved {len(df)} per-case rows to: {csv_path}")

    @torch.no_grad()
    def _maybe_visualize_batch(
            self,
            batch: dict,
            batch_idx: int,
            preds: List[dict],
            targets: List[dict],
            case_data_list: List[Dict[str, Any]],
    ) -> None:
        vis_indices = set(self.images_to_visualize.get(0, []))
        do_vis = self.trainer.is_global_zero and (batch_idx in vis_indices)
        if not do_vis:
            return

        inputs_full = batch.get("inputs")
        cases = batch.get("case", None)
        if inputs_full is None:
            return

        for i, (det, tgt, case_data) in enumerate(zip(preds, targets, case_data_list)):
            semantic_gt = tgt["semantic_mask"]
            img_to_show = torch.cat([inputs_full[i], semantic_gt], dim=0)

            gt_onehot = case_data["gt_onehot"]
            y_true = onehot_to_instance_mask(gt_onehot)
            y_pred = det["instance_mask"]

            true_ids = torch.unique(y_true)
            true_ids = true_ids[true_ids > 0]
            pred_ids = torch.unique(y_pred)
            pred_ids = pred_ids[pred_ids > 0]

            stats = get_stats(
                case_data["pairwise_mask_iou"],
                y_true_ids=true_ids,
                y_pred_ids=pred_ids,
                scores=case_data["scores"],
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

    def test_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        targets = batch[self.target_key]
        preds = batch[self.pred_key]
        cases = batch.get("case", None)

        case_data_list: List[Dict[str, Any]] = []

        for i, (det, tgt) in enumerate(zip(preds, targets)):
            case_data = self._build_case_data(det=det, tgt=tgt, batch=batch)
            case_data_list.append(case_data)

            self._update_all_metrics(case_data)
            self._update_size_bin_metrics(case_data)

            case_name = str(cases[i]) if cases is not None else f"batch{batch_idx}_idx{i}"
            self._per_case_rows.append(
                self._compute_case_row(case_name=case_name, case_data=case_data)
            )

        self._maybe_visualize_batch(
            batch=batch,
            batch_idx=batch_idx,
            preds=preds,
            targets=targets,
            case_data_list=case_data_list,
        )
        return None

    def on_test_epoch_end(self) -> None:
        self._log_all_lesion_metrics()
        self._log_size_bin_recalls_only()
        self._save_per_case_csv()

        for metric_dict in self.metrics.values():
            metric_dict.reset()

        self._per_case_rows = []

    # -------------------------------------------------------------------------
    # logging
    # -------------------------------------------------------------------------
    def _log_all_lesion_metrics(self) -> None:
        metric_all = self.metrics["all"]

        mask_cfm = metric_all["cfm"].compute()
        mask_ap = metric_all["ap"].compute()
        mask_gcir = metric_all["gcir"].compute()
        semantic_dice = metric_all["semantic_dice"].compute()

        self._log_scalar("Metric:All-Lesions/mAP", _safe_mean(mask_ap))
        self._log_scalar("Metric:All-Lesions/mGCIR", _safe_mean(mask_gcir))
        self._log_scalar("Metric:All-Lesions/Semantic-Dice", semantic_dice)

        self._log_cfm_series(
            prefix="Metric:All-Lesions",
            cfm=mask_cfm,
            iou_thresholds=metric_all["cfm"].iou_thresholds,
            ap_per_thr=mask_ap if mask_ap.numel() else None,
            gcir_per_thr=mask_gcir if mask_gcir.numel() else None,
        )

        if self.trainer is not None and self.trainer.is_global_zero:
            mask_plot_fn = getattr(metric_all["ap"], "plot", None)
            if callable(mask_plot_fn):
                try:
                    mask_fig, _ = mask_plot_fn()
                    self._log_figure(
                        "Metric:All-Lesions/Figures/PR-curve",
                        mask_fig,
                        close=True,
                    )
                except Exception:
                    pass

        froc = metric_all["froc"]
        curve = froc.compute()
        auc = froc.auc(fppi_max=8.0)
        ops = froc.sensitivity_at_fppi((0.5, 1, 2, 4, 8))

        self._log_scalar("Metric:All-Lesions/FROC_AUC@8", auc)
        for fp, sens in ops.items():
            self._log_scalar(f"Metric:All-Lesions/Sens@FPPI{fp:g}", sens)

        if self.log_froc_figure and self.trainer is not None and self.trainer.is_global_zero:
            fig = plot_froc_curve(
                curve=curve,
                title="All Lesions FROC @ IoU=0.10",
                fppi_limit=self.froc_plot_limit,
            )
            self._log_figure("Metric:All-Lesions/Figures/FROC", fig, close=True)

    def _log_size_bin_recalls_only(self) -> None:
        for group_name, display_name in (
                ("small", "Small-Lesions"),
                ("medium", "Medium-Lesions"),
                ("large", "Large-Lesions"),
        ):
            cfm = self.metrics[group_name]["cfm"].compute()
            recall = compute_recall(cfm)
            iou_thresholds = self.metrics[group_name]["cfm"].iou_thresholds

            for i, th in enumerate(iou_thresholds):
                th_str = f"{float(th):.2f}"
                self._log_scalar(
                    f"Metric:{display_name}/Recall@{th_str}",
                    recall[i],
                )

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
                plt.close(fig)
            except Exception:
                pass

    def _log_scalar(
            self,
            name: str,
            value: Tensor | float | int,
            *,
            sync_dist: bool = True,
    ) -> None:
        self.log(
            name,
            _to_log_value(value),
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
            self._log_scalar(f"{prefix}/TP@{th_str}", tp)
            self._log_scalar(f"{prefix}/FP@{th_str}", fp)
            self._log_scalar(f"{prefix}/FN@{th_str}", fn)

            self._log_scalar(f"{prefix}/Precision@{th_str}", precision[i])
            self._log_scalar(f"{prefix}/Recall@{th_str}", recall[i])
            self._log_scalar(f"{prefix}/F1@{th_str}", f1[i])
            self._log_scalar(f"{prefix}/F2@{th_str}", f2[i])

            if ap_per_thr is not None:
                self._log_scalar(f"{prefix}/AP@{th_str}", ap_per_thr[i])

            if gcir_per_thr is not None:
                self._log_scalar(f"{prefix}/GCIR@{th_str}", gcir_per_thr[i])
