from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable

import fire
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar

from condinst3d.io.datamodule.nndet_result import nnDetectionResult
from condinst3d.io.datamodule.nnunet_result import nnUNetResult
from condinst3d.io.datamodule.condinst_result import CondInstResult
from condinst3d.arch.model_evaluator import ModelEvaluator


def setup_logger(
    save_dir: str = "results",
    name: str = "tb",
    version: Optional[str] = None,
    iou_thresholds: Optional[Iterable[float]] = None,
    add_custom_scalars: bool = True,
) -> TensorBoardLogger:
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=version,
        default_hp_metric=False,
    )

    if add_custom_scalars and iou_thresholds:
        iou_list = [float(x) for x in iou_thresholds]

        layout = {
            "IoU-Based": {
                "TP": ["Multiline", [f"Metric:True-Positives/TP@{th:.2f}" for th in iou_list]],
                "FP": ["Multiline", [f"Metric:True-Positives/FP@{th:.2f}" for th in iou_list]],
                "FN": ["Multiline", [f"Metric:False-Negatives/FN@{th:.2f}" for th in iou_list]],
                "Precision": ["Multiline", [f"Metric:Precision/Precision@{th:.2f}" for th in iou_list]],
                "Recall": ["Multiline", [f"Metric:Recall/Recall@{th:.2f}" for th in iou_list]],
                "F1": ["Multiline", [f"Metric:F1-score/F1@{th:.2f}" for th in iou_list]],
                "F2": ["Multiline", [f"Metric:F2-score/F1@{th:.2f}" for th in iou_list]],
                "AP": ["Multiline", [f"Metric:Average-Precision/AP@{th:.2f}" for th in iou_list]],
            },
        }

        logger.experiment.add_custom_scalars(layout)

    return logger


def infer_experiment_name_from_pred_root(pred_root: str) -> str:
    """
    Example:
      outputs/dynunet_instance_full/2026-03-28_23-18-53/tb/version_0/predictions
    -> dynunet_instance_full
    """
    p = Path(pred_root).expanduser().resolve()
    parts = p.parts

    if "tb" in parts:
        tb_idx = parts.index("tb")
        if tb_idx >= 2:
            return parts[tb_idx - 2]

    return p.parent.name if p.parent.name else p.name


class Runner:
    def test(
        self,
        pred_root: str,
        split_root: str = "/scratch/04/public_datasets/nnDet/Task101_PRL/raw_splitted",
        n_modalities: int = 4,
        devices: int | list[int] = 1,
        num_workers: int = 32,
        results_dir: str = "results",
        version: Optional[str] = None,
    ) -> None:
        dm = CondInstResult(
            split_root=split_root,
            pred_root=pred_root,
            n_modalities=n_modalities,
            n_workers=num_workers,
        )

        model = ModelEvaluator(target_key="targets", pred_key="preds")
        # model.num_images_to_show = -1

        experiment_name = infer_experiment_name_from_pred_root(pred_root)

        logger = setup_logger(
            save_dir=results_dir,
            name=experiment_name,
            version=version,
            iou_thresholds=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
            add_custom_scalars=True,
        )

        pbar = TQDMProgressBar(
            refresh_rate=1,
            leave=True,
        )

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=devices,
            logger=logger,
            callbacks=[pbar],
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_progress_bar=True,
            log_every_n_steps=1,
            inference_mode=True,
            deterministic=True,
        )

        trainer.test(model=model, datamodule=dm)

    def test_nnunet(
        self,
        pred_root: str,
        split_root: str = "/scratch/04/public_datasets/nnDet/Task101_PRL/raw_splitted",
        n_modalities: int = 4,
        devices: int | list[int] = 1,
        num_workers: int = 32,
        results_dir: str = "results",
        version: Optional[str] = None,
    ) -> None:
        dm = nnUNetResult(
            split_root=split_root,
            pred_root=pred_root,
            n_modalities=n_modalities,
            n_workers=num_workers,
        )

        model = ModelEvaluator(target_key="targets", pred_key="preds")
        # model.num_images_to_show = -1

        experiment_name = infer_experiment_name_from_pred_root(pred_root)

        logger = setup_logger(
            save_dir=results_dir,
            name=experiment_name,
            version=version,
            iou_thresholds=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
            add_custom_scalars=True,
        )

        pbar = TQDMProgressBar(
            refresh_rate=1,
            leave=True,
        )

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=devices,
            logger=logger,
            callbacks=[pbar],
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_progress_bar=True,
            log_every_n_steps=1,
            inference_mode=True,
            deterministic=True,
        )

        trainer.test(model=model, datamodule=dm)

    def test_nndet(
        self,
        pred_root: str,
        split_root: str = "/scratch/04/public_datasets/nnDet/Task101_PRL/raw_splitted",
        n_modalities: int = 4,
        devices: int | list[int] = 1,
        num_workers: int = 32,
        results_dir: str = "results",
        version: Optional[str] = None,
        score_threshold: float = 0.1,
        iou_threshold: float = 0.1,
    ) -> None:
        dm = nnDetectionResult(
            split_root=split_root,
            pred_root=pred_root,
            n_modalities=n_modalities,
            n_workers=num_workers,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )

        model = ModelEvaluator(target_key="targets", pred_key="preds")
        # model.num_images_to_show = -1

        experiment_name = infer_experiment_name_from_pred_root(pred_root)

        logger = setup_logger(
            save_dir=results_dir,
            name=experiment_name,
            version=version,
            iou_thresholds=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
            add_custom_scalars=True,
        )

        pbar = TQDMProgressBar(
            refresh_rate=1,
            leave=True,
        )

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=devices,
            logger=logger,
            callbacks=[pbar],
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_progress_bar=True,
            log_every_n_steps=1,
            inference_mode=True,
            deterministic=True,
        )

        trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    fire.Fire(Runner)
