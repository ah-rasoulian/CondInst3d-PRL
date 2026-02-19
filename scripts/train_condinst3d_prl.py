# scripts/train.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from condinst3d.io.datamodule.prl import PRLDataModule
from condinst3d.arch.condinst3d_prl import CondInst3dPRL
try:
    import resource  # Unix only
except Exception:
    resource = None


# -------------------- utils --------------------
def _maybe_set_rlimit_nofile(limit: int) -> None:
    if resource is None:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(limit, hard), hard))
    except Exception:
        pass


def _seed_everything(seed: int, deterministic: bool = False) -> None:
    pl.seed_everything(seed, workers=True)
    # torch matmul perf/precision
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # optional MONAI determinism
    if deterministic:
        try:
            from monai.utils import set_determinism
            set_determinism(seed=seed)
        except Exception:
            pass


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


# -------------------- logger --------------------
def setup_logger(cfg: DictConfig) -> TensorBoardLogger:
    """
    cfg.train.logger:
      _target_: pytorch_lightning.loggers.TensorBoardLogger
      save_dir: ${hydra:runtime.output_dir}
      name: tb
      version: null
      default_hp_metric: false

    cfg.train.logger_layout (optional):
      iou_thresholds: [0.50, 0.40, 0.30, 0.20, 0.10, 0.01]
      add_custom_scalars: true
    """
    logger: TensorBoardLogger = instantiate(cfg.train.logger)

    layout_cfg = getattr(cfg.train, "logger_layout", None)
    if layout_cfg and getattr(layout_cfg, "add_custom_scalars", False):
        iou_list = [float(x) for x in getattr(layout_cfg, "iou_thresholds", [])]
        if iou_list:
            # These names match the logging you asked for:
            # Validation/Masks-IoU/Recall@0.50 etc. OR your custom names.
            # Adjust here if your naming differs.
            layout = {
                "Validation-Masks": {
                    "TP": ["Multiline", [f"Validation/Masks-IoU/TP@{th:.2f}" for th in iou_list]],
                    "FP": ["Multiline", [f"Validation/Masks-IoU/FP@{th:.2f}" for th in iou_list]],
                    "FN": ["Multiline", [f"Validation/Masks-IoU/FN@{th:.2f}" for th in iou_list]],
                    "Precision": ["Multiline", [f"Validation/Masks-IoU/Precision@{th:.2f}" for th in iou_list]],
                    "Recall": ["Multiline", [f"Validation/Masks-IoU/Recall@{th:.2f}" for th in iou_list]],
                    "F1": ["Multiline", [f"Validation/Masks-IoU/F1@{th:.2f}" for th in iou_list]],
                    "AP": ["Multiline", [f"Validation/Masks-IoU/AP@{th:.2f}" for th in iou_list]],
                },
                "Validation-Boxes": {
                    "TP": ["Multiline", [f"Validation/Boxes-IoU/TP@{th:.2f}" for th in iou_list]],
                    "FP": ["Multiline", [f"Validation/Boxes-IoU/FP@{th:.2f}" for th in iou_list]],
                    "FN": ["Multiline", [f"Validation/Boxes-IoU/FN@{th:.2f}" for th in iou_list]],
                    "Precision": ["Multiline", [f"Validation/Boxes-IoU/Precision@{th:.2f}" for th in iou_list]],
                    "Recall": ["Multiline", [f"Validation/Boxes-IoU/Recall@{th:.2f}" for th in iou_list]],
                    "F1": ["Multiline", [f"Validation/Boxes-IoU/F1@{th:.2f}" for th in iou_list]],
                    "AP": ["Multiline", [f"Validation/Boxes-IoU/AP@{th:.2f}" for th in iou_list]],
                },
            }
            logger.experiment.add_custom_scalars(layout)

    return logger


# -------------------- callbacks --------------------
def setup_callbacks(cfg: DictConfig) -> List[Any]:
    """
    Either define callbacks explicitly in Hydra:
      train.callbacks:
        - _target_: pytorch_lightning.callbacks.ModelCheckpoint
          monitor: Validation/Masks-IoU/Recall@0.01
          mode: max
          save_top_k: 5
          filename: "best_recall_{epoch}-{step}"
        - _target_: pytorch_lightning.callbacks.LearningRateMonitor
          logging_interval: step

    Or rely on defaults here.
    """
    callbacks_cfg = getattr(cfg.train, "callbacks", None)
    if callbacks_cfg:
        return [instantiate(c) for c in _as_list(callbacks_cfg)]

    # sane defaults
    ckpt = ModelCheckpoint(
        monitor=str(getattr(cfg.train, "monitor", "Validation/Masks-IoU/Recall@0.01")),
        mode=str(getattr(cfg.train, "monitor_mode", "max")),
        save_top_k=int(getattr(cfg.train, "save_top_k", 5)),
        filename=str(getattr(cfg.train, "ckpt_filename", "best_{epoch}-{step}")),
        save_last=bool(getattr(cfg.train, "save_last", True)),
        every_n_epochs=int(getattr(cfg.train, "ckpt_every_n_epochs", 1)),
    )
    lr_monitor = LearningRateMonitor(logging_interval=str(getattr(cfg.train, "lr_logging_interval", "step")))
    return [ckpt, lr_monitor]


# -------------------- trainer/strategy --------------------
def setup_strategy(cfg: DictConfig):
    strategy_cfg = getattr(cfg.train, "strategy", None)

    # If user provided a hydra-instantiable strategy, use it
    if strategy_cfg and "_target_" in strategy_cfg:
        return instantiate(strategy_cfg)

    # else: default DDP strategy
    return DDPStrategy(
        process_group_backend=str(getattr(cfg.train, "ddp_backend", "nccl")),
        find_unused_parameters=bool(getattr(cfg.train, "find_unused_parameters", True)),
    )


def setup_trainer(cfg: DictConfig, logger, callbacks) -> Trainer:
    """
    cfg.train.trainer may include anything Trainer accepts, e.g.
      train.trainer:
        accelerator: gpu
        devices: [0,1,2,3]
        precision: bf16-mixed
        max_epochs: 60
        gradient_clip_val: 1.0
        gradient_clip_algorithm: norm
        accumulate_grad_batches: 1
        log_every_n_steps: 1
        check_val_every_n_epoch: 1
        num_sanity_val_steps: 2
        limit_train_batches: 1.0
        limit_val_batches: 1.0
        enable_progress_bar: true
        deterministic: false
    """
    trainer_kwargs: Dict[str, Any] = {}
    if getattr(cfg.train, "trainer", None) is not None:
        # Convert to primitive dict so Trainer gets plain python types
        trainer_kwargs = OmegaConf.to_container(cfg.train.trainer, resolve=True)  # type: ignore[assignment]

    # ensure required/overridden
    trainer_kwargs["logger"] = logger
    trainer_kwargs["callbacks"] = callbacks
    trainer_kwargs["strategy"] = setup_strategy(cfg)

    # defaults if not set by config
    trainer_kwargs.setdefault("accelerator", "gpu")
    trainer_kwargs.setdefault("log_every_n_steps", 1)
    trainer_kwargs.setdefault("enable_checkpointing", True)

    return pl.Trainer(**trainer_kwargs)


# -------------------- main --------------------
@hydra.main(config_path="../condinst3d/conf", config_name="condinst3d-prl-best", version_base=None)
def main(cfg: DictConfig) -> None:
    # global setup from cfg.train
    _maybe_set_rlimit_nofile(int(getattr(cfg.train, "rlimit_nofile", 1048576)))
    _seed_everything(
        seed=int(getattr(cfg.train, "seed", 1000)),
        deterministic=bool(getattr(cfg.train, "deterministic", False)),
    )

    # instantiate model/datamodule from config
    model = CondInst3dPRL(cfg.model)
    dm = PRLDataModule(cfg.data)

    logger = setup_logger(cfg)
    callbacks = setup_callbacks(cfg)
    trainer = setup_trainer(cfg, logger=logger, callbacks=callbacks)

    # resume
    resume: Optional[str] = getattr(cfg.train, "resume", None)
    ckpt_path = None
    if resume:
        ckpt_path = str(Path(resume).expanduser())

    trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
