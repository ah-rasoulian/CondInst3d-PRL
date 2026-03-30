# scripts/optimize_inference.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json
import os
import re
import gc
import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from omegaconf import open_dict

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy

import optuna

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


def _maybe_disable_visualization(model: CondInst3d) -> None:
    """
    Best-effort disabling of expensive visualization during hyperparameter search.
    Adjust/remove fields if your module uses different names.
    """
    for attr in [
        "num_images_to_show",
        "num_train_images_to_show",
        "num_val_images_to_show",
    ]:
        if hasattr(model, attr):
            try:
                setattr(model, attr, 0)
            except Exception:
                pass


def _load_checkpoint_weights(model: CondInst3d, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[warn] Missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys when loading checkpoint: {len(unexpected)}")


def get_json_path(checkpoint_path: str, suffix: str = "") -> str:
    """
    Save hyperparameter search results near the checkpoint directory.

    Example:
      .../tb/version_0/checkpoints/best_epoch=49-step=9500.ckpt
    ->
      .../tb/version_0/hyperparams/validation_best_epoch=49-step=9500_det-params.json
    """
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    ckpt_dir = ckpt_path.parent
    out_dir = ckpt_dir.parent / "hyperparams"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = ckpt_path.stem
    suffix = f"_{suffix}" if suffix else ""
    out_path = out_dir / f"validation_{base}{suffix}.json"

    i = 1
    while out_path.exists():
        out_path = out_dir / f"validation_{base}{suffix}_{i}.json"
        i += 1

    return str(out_path)


# -------------------- optimizer --------------------
class HyperParamOptimizer:
    def __init__(
        self,
        cfg: DictConfig,
        ckpt_path: str,
        metric_key: str = "Validation/mAP",
        devices: Optional[Sequence[int] | int] = None,
        precision: str = "bf16-mixed",
    ):
        self.cfg = cfg
        self.ckpt_path = str(Path(ckpt_path).expanduser())
        self.metric_key = metric_key
        self.precision = precision

        self.dm = PRLDataModule(cfg.data)

        self.model = CondInst3d(cfg.model)
        _load_checkpoint_weights(self.model, self.ckpt_path)
        _maybe_disable_visualization(self.model)

        trainer_devices = devices
        if trainer_devices is None:
            trainer_devices = getattr(getattr(cfg.train, "trainer", {}), "devices", 1)

        self.trainer = self._build_trainer(trainer_devices)

    def _build_trainer(self, devices: Optional[Sequence[int] | int] = None) -> Trainer:
        trainer_kwargs: Dict[str, Any] = {
            "precision": self.precision,
            "accelerator": "gpu",
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": True,
            "callbacks": [TQDMProgressBar(refresh_rate=20)],
        }

        if devices is not None:
            trainer_kwargs["devices"] = devices

        # Only use DDP if we are actually on multiple devices.
        use_ddp = False
        if isinstance(devices, int):
            use_ddp = devices > 1
        elif isinstance(devices, (list, tuple)):
            use_ddp = len(devices) > 1

        if use_ddp:
            trainer_kwargs["strategy"] = DDPStrategy(
                process_group_backend=str(getattr(self.cfg.train, "ddp_backend", "nccl")),
                find_unused_parameters=bool(getattr(self.cfg.train, "find_unused_parameters", False)),
            )

        return pl.Trainer(**trainer_kwargs)

    def _set_inference_params(self, params: Dict[str, Any]) -> None:
        """
        Best-effort update for whichever inference hyperparameter container your
        current model uses.
        """
        if hasattr(self.model, "inference_hyperparams"):
            self.model.inference_hyperparams = params
            return

        # fallback: merge into cfg.model.inference if it exists
        if hasattr(self.model, "cfg"):
            try:
                with open_dict(self.model.cfg):
                    if "inference" not in self.model.cfg:
                        self.model.cfg.inference = {}
                    for k, v in params.items():
                        self.model.cfg.inference[k] = v
                return
            except Exception:
                pass

        raise AttributeError(
            "Could not find where to set inference hyperparameters. "
            "Expected model.inference_hyperparams or model.cfg.inference."
        )

    def objective_function(self, trial: optuna.Trial) -> float:
        seed = int(getattr(self.cfg.train, "seed", 1000))
        _seed_everything(
            seed=seed,
            deterministic=bool(getattr(self.cfg.train, "deterministic", False)),
        )

        params = {
            "mask_thresh": trial.suggest_float("mask_thresh", 0.4, 0.6, step=0.05),
            "nms_thresh": trial.suggest_float("nms_thresh", 0.2, 0.6, step=0.05),
        }
        if self.cfg.model.task_mode == "instance":
            params["score_thresh"] = trial.suggest_float("score_thresh", 0.1, 0.6, step=0.05)
            if self.cfg.data.image_mode == "full":
                params["topk_candidates"] = trial.suggest_int("topk_candidates", 20, 60, step=5)
            else:
                params["topk_candidates"] = trial.suggest_int("topk_candidates", 5, 25, step=5)
                params["group_thresh"] = trial.suggest_float("group_thresh", 0.4, 0.8, step=0.05)

        self._set_inference_params(params)

        try:
            with torch.inference_mode():
                val_metrics = self.trainer.validate(
                    model=self.model,
                    datamodule=self.dm,
                    verbose=False,
                )

            if not val_metrics:
                raise RuntimeError("trainer.validate() returned no metrics.")

            metric_dict = val_metrics[0]
            if self.metric_key not in metric_dict:
                raise KeyError(
                    f"Metric '{self.metric_key}' not found. "
                    f"Available keys: {list(metric_dict.keys())}"
                )

            return float(metric_dict[self.metric_key])

        except torch.OutOfMemoryError as e:
            print(f"[OOM] Trial failed for params={params}: {e}")
            trial.set_user_attr("oom", True)

            # cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            raise optuna.exceptions.TrialPruned("Pruned due to CUDA OOM")

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def find_optimum(self, n_trials: int = 25, study_name: Optional[str] = None) -> Dict[str, Any]:
        sampler = optuna.samplers.TPESampler(seed=int(getattr(self.cfg.train, "seed", 1000)))
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=study_name,
        )
        study.optimize(self.objective_function, n_trials=n_trials)

        optimum_param_path = get_json_path(self.ckpt_path, suffix="det-params")
        optimum_params = dict(study.best_params)
        optimum_params["best_val"] = float(study.best_value)
        optimum_params["metric_key"] = self.metric_key
        optimum_params["checkpoint"] = self.ckpt_path

        with open(optimum_param_path, "w") as f:
            json.dump(optimum_params, f, indent=4)

        print(f"Saved best params to: {optimum_param_path}")
        return optimum_params


# -------------------- runner --------------------
class Runner:
    def _build_cfg(
        self,
        config_name: str,
        overrides: Optional[Sequence[str] | str] = None,
        print_config: bool = False,
    ) -> DictConfig:
        cfg = _load_cfg(config_name=config_name, overrides=overrides)

        if print_config:
            print(OmegaConf.to_yaml(cfg, resolve=True))

        _maybe_set_rlimit_nofile(int(getattr(cfg.train, "rlimit_nofile", 1048576)))
        _seed_everything(
            seed=int(getattr(cfg.train, "seed", 1000)),
            deterministic=bool(getattr(cfg.train, "deterministic", False)),
        )
        return cfg

    def optimize(
        self,
        config_name: str,
        ckpt_path: str,
        precision: str = "bf16-mixed",
        overrides: Optional[Sequence[str] | str] = None,
        print_config: bool = False,
        n_trials: int = 25,
        metric_key: str = "Validation/mAP",
        devices: Optional[Sequence[int] | int] = None,
    ) -> None:
        """
        Search for best inference hyperparameters on validation set.

        Example:
            python scripts/condinst3d_optimize_inference.py optimize \\
                --config_name=condinst3d \\
                --ckpt_path=/path/to/best.ckpt

            python scripts/condinst3d_optimize_inference.py optimize \\
                --config_name=condinst3d \\
                --ckpt_path=/path/to/best.ckpt \\
                --overrides="data.batch_size=1" \\
                --n_trials=50

            python scripts/condinst3d_optimize_inference.py optimize \\
                --config_name=condinst3d \\
                --ckpt_path=/path/to/best.ckpt \\
                --devices=[0,1]
        """
        cfg = self._build_cfg(
            config_name=config_name,
            overrides=overrides,
            print_config=print_config,
        )

        optimizer = HyperParamOptimizer(
            cfg=cfg,
            ckpt_path=ckpt_path,
            metric_key=metric_key,
            devices=devices,
            precision=precision,
        )
        best_params = optimizer.find_optimum(n_trials=n_trials)

        print("Best validation inference result:")
        print(json.dumps(best_params, indent=4))


if __name__ == "__main__":
    fire.Fire(Runner)
