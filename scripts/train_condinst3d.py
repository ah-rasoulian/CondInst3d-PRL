# scripts/train.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import fire
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import os
from datetime import datetime
from omegaconf import open_dict

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from condinst3d.io.datamodule.prl import PRLDataModule
from condinst3d.arch.condinst3d import CondInst3d

try:
    import resource  # Unix only
except Exception:
    resource = None


# -------------------- utils --------------------
def _maybe_set_rlimit_nofile(limit: int) -> None:
    if resource is None:
        return
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(limit, hard), hard))
    except Exception:
        pass


def _seed_everything(seed: int, deterministic: bool = False) -> None:
    pl.seed_everything(seed, workers=True)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

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


def _normalize_overrides(overrides: Optional[Sequence[str] | str]) -> List[str]:
    if overrides is None:
        return []
    if isinstance(overrides, str):
        overrides = overrides.strip()
        return overrides.split() if overrides else []
    return [str(x) for x in overrides]


def _get_config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "condinst3d" / "conf"


def _load_cfg(config_name: str, overrides: Optional[Sequence[str] | str] = None) -> DictConfig:
    config_dir = _get_config_dir()
    if not config_dir.exists():
        raise FileNotFoundError(f"Hydra config directory not found: {config_dir}")

    hydra_overrides = _normalize_overrides(overrides)

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=hydra_overrides)

    return cfg


# -------------------- logger --------------------
def setup_logger(cfg: DictConfig) -> TensorBoardLogger:
    logger: TensorBoardLogger = instantiate(cfg.train.logger)

    layout_cfg = getattr(cfg.train, "logger_layout", None)
    if layout_cfg and getattr(layout_cfg, "add_custom_scalars", False):
        iou_list = [float(x) for x in getattr(layout_cfg, "iou_thresholds", [])]
        if iou_list:
            layout = {
                "Validation-Masks": {
                    "TP": ["Multiline", [f"Validation-Masks-IoU/TP@{th:.2f}" for th in iou_list]],
                    "FP": ["Multiline", [f"Validation-Masks-IoU/FP@{th:.2f}" for th in iou_list]],
                    "FN": ["Multiline", [f"Validation-Masks-IoU/FN@{th:.2f}" for th in iou_list]],
                    "Precision": ["Multiline", [f"Validation-Masks-IoU/Precision@{th:.2f}" for th in iou_list]],
                    "Recall": ["Multiline", [f"Validation-Masks-IoU/Recall@{th:.2f}" for th in iou_list]],
                    "F1": ["Multiline", [f"Validation-Masks-IoU/F1@{th:.2f}" for th in iou_list]],
                    "AP": ["Multiline", [f"Validation-Masks-IoU/AP@{th:.2f}" for th in iou_list]],
                },
            }
            logger.experiment.add_custom_scalars(layout)

    return logger


# -------------------- callbacks --------------------
def setup_callbacks(cfg: DictConfig) -> List[Any]:
    callbacks_cfg = getattr(cfg.train, "callbacks", None)
    if callbacks_cfg:
        return [instantiate(c) for c in _as_list(callbacks_cfg)]

    best_ckpt = ModelCheckpoint(
        monitor=str(getattr(cfg.train, "monitor", "Validation/mAP")),
        mode=str(getattr(cfg.train, "monitor_mode", "max")),
        save_top_k=int(getattr(cfg.train, "save_top_k", 3)),
        filename=str(getattr(cfg.train, "ckpt_filename", "best_{epoch}-{step}")),
        save_last=False,
    )

    last_ckpt = ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        filename="step_{epoch}-{step}",
        every_n_train_steps=1000,
        save_on_train_epoch_end=True,
    )

    lr_monitor = LearningRateMonitor(
        logging_interval=str(getattr(cfg.train, "lr_logging_interval", "step"))
    )
    return [best_ckpt, last_ckpt, lr_monitor]


# -------------------- trainer/strategy --------------------
def setup_strategy(cfg: DictConfig):
    strategy_cfg = getattr(cfg.train, "strategy", None)

    if strategy_cfg and "_target_" in strategy_cfg:
        return instantiate(strategy_cfg)

    return DDPStrategy(
        process_group_backend=str(getattr(cfg.train, "ddp_backend", "nccl")),
        find_unused_parameters=bool(getattr(cfg.train, "find_unused_parameters", False)),
    )


def setup_trainer(cfg: DictConfig, logger, callbacks) -> Trainer:
    trainer_kwargs: Dict[str, Any] = {}
    if getattr(cfg.train, "trainer", None) is not None:
        trainer_kwargs = OmegaConf.to_container(cfg.train.trainer, resolve=True)  # type: ignore[assignment]

    trainer_kwargs["logger"] = logger
    trainer_kwargs["callbacks"] = callbacks
    trainer_kwargs["strategy"] = setup_strategy(cfg)

    trainer_kwargs.setdefault("accelerator", "gpu")
    trainer_kwargs.setdefault("log_every_n_steps", 1)
    trainer_kwargs.setdefault("enable_checkpointing", True)

    return pl.Trainer(**trainer_kwargs)


# -------------------- runner --------------------
class Runner:
    def _build(
        self,
        config_name: str,
        overrides: Optional[Sequence[str] | str] = None,
        print_config: bool = False,
    ):
        cfg = _load_cfg(config_name=config_name, overrides=overrides)

        if print_config:
            print(OmegaConf.to_yaml(cfg, resolve=True))

        _maybe_set_rlimit_nofile(int(getattr(cfg.train, "rlimit_nofile", 1048576)))
        _seed_everything(
            seed=int(getattr(cfg.train, "seed", 1000)),
            deterministic=bool(getattr(cfg.train, "deterministic", False)),
        )

        model = CondInst3d(cfg.model)
        dm = PRLDataModule(cfg.data)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_root = str(getattr(cfg.train, "output_root", "outputs"))
        run_dir = os.path.join(output_root, config_name, timestamp)

        with open_dict(cfg):
            cfg.train.logger.save_dir = run_dir

        logger = setup_logger(cfg)
        callbacks = setup_callbacks(cfg)
        trainer = setup_trainer(cfg, logger=logger, callbacks=callbacks)

        resume: Optional[str] = getattr(cfg.train, "resume", None)
        ckpt_path = str(Path(resume).expanduser()) if resume else None

        return cfg, model, dm, trainer, ckpt_path

    def fit(
        self,
        config_name: str,
        overrides: Optional[Sequence[str] | str] = None,
        print_config: bool = False,
    ) -> None:
        """
        Train the model.

        Example:
            python scripts/train.py fit --config_name=condinst3d
            python scripts/train.py fit --config_name=condinst3d --overrides="train.trainer.devices=[0,1]"
        """
        _, model, dm, trainer, ckpt_path = self._build(
            config_name=config_name,
            overrides=overrides,
            print_config=print_config,
        )
        trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_path)

    def validate(
        self,
        config_name: str,
        overrides: Optional[Sequence[str] | str] = None,
        print_config: bool = False,
        ckpt_path: Optional[str] = None,
    ) -> None:
        """
        Run validation.

        Example:
            python scripts/train.py validate --config_name=condinst3d
            python scripts/train.py validate --config_name=condinst3d --ckpt_path=/path/to/best.ckpt
        """
        cfg, model, dm, trainer, resume_ckpt_path = self._build(
            config_name=config_name,
            overrides=overrides,
            print_config=print_config,
        )

        final_ckpt_path = ckpt_path
        if final_ckpt_path is None:
            final_ckpt_path = resume_ckpt_path

        trainer.validate(model=model, datamodule=dm, ckpt_path=final_ckpt_path)


if __name__ == "__main__":
    fire.Fire(Runner)
