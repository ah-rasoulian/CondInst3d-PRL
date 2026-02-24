from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable
import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from condinst3d.io.datamodule.prl import PRLDataModule
from condinst3d.arch.condinst3d_prl import CondInst3dPRL
import optuna
import re
import json
from pytorch_lightning.callbacks import TQDMProgressBar


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


def get_json_path(checkpoint_path: str, suffix=""):
    optimum_param_dir = os.path.dirname(checkpoint_path).replace("checkpoint", "hyperparam")
    os.makedirs(optimum_param_dir, exist_ok=True)

    pattern = r"version_([\d-]+).*?epoch=(\d+)"
    match = re.search(pattern, checkpoint_path)

    if match is None:
        base = os.path.splitext(os.path.basename(checkpoint_path))[0]
        version_name = f"validation_{base}{suffix}.json"
    else:
        version_name = f"validation_version_{match.group(1)}-epoch={match.group(2)}{suffix}.json"

    optimum_param_path = os.path.join(optimum_param_dir, version_name)
    i = 1
    while os.path.isfile(optimum_param_path):
        root, ext = os.path.splitext(version_name)
        optimum_param_path = os.path.join(optimum_param_dir, f"{root}_{i}{ext}")
        i += 1
    return optimum_param_path


class HyperParamOptimizer:
    def __init__(self, cfg: DictConfig, ckpt_path: str):
        self.cfg = cfg
        self.dm = PRLDataModule(cfg.data)
        self.ckpt_path = ckpt_path

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = CondInst3dPRL(cfg=cfg.model)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.num_images_to_show = 0
        self.model = model

        # Make validate fast + deterministic-ish
        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=[0, 1, 2, 3],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
        )

        # self.metric_key = "Validation/Masks/mAP"  # change if needed
        self.metric_key = "Validation/Masks-IoU/AP@0.10"

    def objective_function(self, trial: optuna.Trial):
        # IMPORTANT: reseed so every trial sees the same validation randomness/order
        # (still better: remove randomness from val transforms entirely)
        seed = int(getattr(self.cfg.train, "seed", 1000))
        pl.seed_everything(seed, workers=True)

        params = {
            "mask_thresh": trial.suggest_float("mask_thresh", 0.4, 0.6),
            "score_thresh": trial.suggest_float("score_thresh", 0.05, 0.4, step=0.05),
            "nms_thresh": trial.suggest_float("nms_thresh", 0.1, 0.6, step=0.05),
            "group_thresh": trial.suggest_float("group_thresh", 0.05, 0.35, step=0.05),
            "topk_candidates": trial.suggest_int("topk_candidates", 5, 30, step=5),
        }
        self.model.inference_hyperparams = params

        with torch.inference_mode():
            val_metrics = self.trainer.validate(self.model, datamodule=self.dm, verbose=False)

        if not val_metrics or self.metric_key not in val_metrics[0]:
            raise KeyError(
                f"Metric '{self.metric_key}' not found. Got keys: {list(val_metrics[0].keys()) if val_metrics else val_metrics}"
            )

        metric = float(val_metrics[0][self.metric_key])
        return metric

    def find_optimum(self, n_trials: int):
        sampler = optuna.samplers.TPESampler(seed=int(getattr(self.cfg.train, "seed", 1000)))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self.objective_function, n_trials=n_trials)

        optimum_param_path = get_json_path(self.ckpt_path, suffix="det-params")
        optimum_params = dict(study.best_params)
        optimum_params["best_val"] = float(study.best_value)

        with open(optimum_param_path, "w") as f:
            json.dump(optimum_params, f, indent=4)

        return optimum_params


# -------------------- main --------------------
@hydra.main(config_path="../condinst3d/conf", config_name="condinst3d-prl-final", version_base=None)
def main(cfg: DictConfig) -> None:
    # global setup from cfg.train
    _maybe_set_rlimit_nofile(int(getattr(cfg.train, "rlimit_nofile", 1048576)))
    _seed_everything(
        seed=int(getattr(cfg.train, "seed", 1000)),
        deterministic=bool(getattr(cfg.train, "deterministic", False)),
    )

    ckpt_path = "/scratch/01/ahrasoulian/projects/CondInst3d-PRL/scripts/outputs/condinst3d-prl-final/2026-02-21_00-58-25/tb/version_0/checkpoints/best_epoch=309-step=29450.ckpt"
    optimum_finder = HyperParamOptimizer(cfg=cfg, ckpt_path=ckpt_path)

    best_params = optimum_finder.find_optimum(20)
    print(f"Best val inference result:")
    print(best_params)


if __name__ == "__main__":
    main()
