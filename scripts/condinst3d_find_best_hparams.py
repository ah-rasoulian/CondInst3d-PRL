from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable
import torch
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from condinst3d.io.datamodule.prl import PRLDataModule
from condinst3d.arch.condinst3d_prl import CondInst3dPRL


def setup_logger(
    save_dir: str = "logs",
    name: str = "tb",
    version: Optional[str] = None,
    iou_thresholds: Optional[Iterable[float]] = None,
    add_custom_scalars: bool = True,
) -> TensorBoardLogger:
    """
    Create a default TensorBoardLogger with optional custom scalar layouts.

    Args:
        save_dir: Root directory for logs
        name: Experiment name
        version: Logger version (None = auto)
        iou_thresholds: IoU thresholds for custom scalar layout
                        (e.g. [0.5, 0.4, 0.3])
        add_custom_scalars: Whether to register TensorBoard custom scalars

    Returns:
        TensorBoardLogger
    """

    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=version,
        default_hp_metric=False,
    )

    if add_custom_scalars and iou_thresholds:
        iou_list = [float(x) for x in iou_thresholds]

        layout = {
            "Masks-IoU": {
                "TP": ["Multiline", [f"Masks-IoU/TP@{th:.2f}" for th in iou_list]],
                "FP": ["Multiline", [f"Masks-IoU/FP@{th:.2f}" for th in iou_list]],
                "FN": ["Multiline", [f"Masks-IoU/FN@{th:.2f}" for th in iou_list]],
                "Precision": ["Multiline", [f"Masks-IoU/Precision@{th:.2f}" for th in iou_list]],
                "Recall": ["Multiline", [f"Masks-IoU/Recall@{th:.2f}" for th in iou_list]],
                "F1": ["Multiline", [f"Masks-IoU/F1@{th:.2f}" for th in iou_list]],
                "AP": ["Multiline", [f"Masks-IoU/AP@{th:.2f}" for th in iou_list]],
            },
            "Boxes-IoU": {
                "TP": ["Multiline", [f"Boxes-IoU/TP@{th:.2f}" for th in iou_list]],
                "FP": ["Multiline", [f"Boxes-IoU/FP@{th:.2f}" for th in iou_list]],
                "FN": ["Multiline", [f"Boxes-IoU/FN@{th:.2f}" for th in iou_list]],
                "Precision": ["Multiline", [f"Boxes-IoU/Precision@{th:.2f}" for th in iou_list]],
                "Recall": ["Multiline", [f"Boxes-IoU/Recall@{th:.2f}" for th in iou_list]],
                "F1": ["Multiline", [f"Boxes-IoU/F1@{th:.2f}" for th in iou_list]],
                "AP": ["Multiline", [f"Boxes-IoU/AP@{th:.2f}" for th in iou_list]],
            },
        }

        logger.experiment.add_custom_scalars(layout)

    return logger


# -------------------- main --------------------
@hydra.main(config_path="../condinst3d/conf", config_name="condinst3d-prl", version_base=None)
def main(cfg: DictConfig) -> None:
    # instantiate model/datamodule from config
    ckpt_path = "/scratch/01/ahrasoulian/projects/CondInst3d-PRL/scripts/outputs/2026-02-07/08-04-57/tb/version_0/checkpoints/last.ckpt"
    model = CondInst3dPRL.load_from_checkpoint(ckpt_path, cfg=cfg.model)

    model.inference_hyperparams = {
        "mask_thresh": 0.4,
        "score_thresh": 0.4,
        "nms_thresh": 0.5,
        "group_thresh": 0.5,
        "topk_candidates": 15,
    }
    dm = PRLDataModule(cfg.data)

    logger = setup_logger()

    trainer = pl.Trainer(
        accelerator="gpu",
        precision="bf16-mixed",
        devices=[0],
        logger=logger,

        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=True,
        log_every_n_steps=1,

        inference_mode=True,
        deterministic=True,
    )
    trainer.validate(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
