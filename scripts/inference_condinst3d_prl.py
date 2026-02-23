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
from pytorch_lightning.callbacks import TQDMProgressBar
from condinst3d.io.datamodule.prl import PRLDataModule
from condinst3d.arch.condinst3d_prl import CondInst3dPRL
import resource

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


# -------------------- main --------------------
@hydra.main(config_path="../condinst3d/conf", config_name="condinst3d-prl-final", version_base=None)
def main(cfg: DictConfig) -> None:
    # global setup from cfg.train
    _maybe_set_rlimit_nofile(int(getattr(cfg.train, "rlimit_nofile", 1048576)))
    _seed_everything(
        seed=int(getattr(cfg.train, "seed", 1000)),
        deterministic=bool(getattr(cfg.train, "deterministic", False)),
    )

    # instantiate model/datamodule from config
    model = CondInst3dPRL(cfg.model)
    ckpt_path = "/scratch/01/ahrasoulian/projects/CondInst3d-PRL/scripts/outputs/condinst3d-prl-final/2026-02-21_00-58-25/tb/version_0/checkpoints/best_epoch=309-step=29450.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.inference_hyperparams = {
        "mask_thresh": 0.4688665405825742,
        "score_thresh": 0.40114189214259616,
        "nms_thresh": 0.4364592296874598,
        "group_thresh": 0.2669170250188043,
        "topk_candidates": 22,
    }

    dm = PRLDataModule(cfg.data)
    dm.patch_overlap = 0.5311165433269576
    dm.setup('test')

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )

    trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
