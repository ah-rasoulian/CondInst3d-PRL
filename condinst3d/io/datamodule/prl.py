from typing import Any, Optional
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import torch
from monai.data import Dataset
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from omegaconf import DictConfig, OmegaConf
import pickle
import json
import os
import math
from monai.transforms import (
    Compose,
    LoadImaged, Lambdad, EnsureChannelFirstd, EnsureTyped, ConcatItemsd, DeleteItemsd, CropForegroundd,
    RandWeightedCropd, GridPatchd,
    RandRotated, RandZoomd, RandFlipd, OneOf, RandGaussianNoised,
    RandScaleIntensityd, RandShiftIntensityd, RandAdjustContrastd, CopyItemsd,
    CenterSpatialCropd, SpatialPadd,
)
import numpy as np
from condinst3d.io.transforms import (
    MaskedPercentileNormalizeIntensityd, InstanceMaskToDetd,
    MakeBalancedInstanceWeightMapd, SymmetricGridPad
)
from condinst3d.io.sampler import DistributedWeightedSampler
from condinst3d.io.collate import multi_instance_collate
from functools import partial


def build_cases(cases, modalities, image_directory, label_directory):
    df_list = []
    for case in cases:
        case_dict = {"case": case}
        for i, m in enumerate(modalities):
            case_dict[m] = os.path.join(image_directory, f"{case}_{i:04d}.nii.gz")
        case_dict["instance_mask"] = os.path.join(label_directory, f"{case}.nii.gz")
        case_dict["brain_mask"] = os.path.join(image_directory, "brainmask", f"{case}_brainmask.nii.gz")
        with open(os.path.join(label_directory, f"{case}.json"), "r") as f:
            info = json.load(f)
        case_dict["info"] = info
        df_list.append(case_dict)
    return df_list


class PRLDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, fold: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        root_path = cfg.root_path
        fold = fold if fold is not None else cfg.fold
        modalities = cfg.modalities

        all_cases = [c for c in os.listdir(os.path.join(root_path, "raw")) if c.startswith("case")]
        with open(os.path.join(root_path, "preprocessed", "splits_final.pkl"), "rb") as f:
            splits = pickle.load(f)

        train_cases, val_cases = splits[fold]["train"], splits[fold]["val"]
        test_cases = [c for c in all_cases if c not in train_cases and c not in val_cases]

        splitted_path = os.path.join(root_path, "raw_splitted")
        train_df = build_cases(
            train_cases, modalities,
            os.path.join(splitted_path, "imagesTr"),
            os.path.join(splitted_path, "labelsTr"),
        )
        val_df = build_cases(
            val_cases, modalities,
            os.path.join(splitted_path, "imagesTr"),
            os.path.join(splitted_path, "labelsTr"),
        )
        test_df = build_cases(
            test_cases, modalities,
            os.path.join(splitted_path, "imagesTs"),
            os.path.join(splitted_path, "labelsTs"),
        )

        self.cases = {"train": train_df, "val": val_df, "test": test_df}
        self.datasets = {}

        self.modalities = list(modalities)
        self.num_workers = cfg.num_workers
        self.batch_size = cfg.batch_size

        self.patch_size = tuple(cfg.patch_size)
        self.patch_overlap = float(cfg.patch_overlap)

        self.spatial_augmentation_prob = float(cfg.spatial_augmentation_prob)
        self.intensity_augmentation_prob = float(cfg.intensity_augmentation_prob)

        self.image_mode = str(getattr(cfg, "image_mode", "patch")).lower()
        self.full_image_size = tuple(getattr(cfg, "full_image_size", [192, 240, 72]))

        valid_modes = {"patch", "full"}
        if self.image_mode not in valid_modes:
            raise ValueError(f"image_mode must be one of {valid_modes}, got {self.image_mode!r}")

        self.collate_fn = partial(
            multi_instance_collate,
            collate_keys=[
                "inputs", "inputs_orig", "instance_mask", "semantic_mask",
                "boxes", "classes", "onehot"
            ],
            target_keys={
                "targets": {
                    "boxes": "boxes",
                    "classes": "classes",
                    "onehot": "onehot",
                    "semantic_mask": "semantic_mask",
                }
            },
        )

        # sampling weights for train cases
        train_counts = []
        for item in train_df:
            n_inst = len(item["info"].get("instances", {}).keys())
            train_counts.append(n_inst)

        empty_weight = float(getattr(cfg, "empty_case_weight", 0.2))
        power = float(getattr(cfg, "instance_weight_power", 0.5))
        max_case_weight = float(getattr(cfg, "max_case_weight", 3.0))
        eps = 1e-6

        self.train_weights = []
        for n in train_counts:
            if n == 0:
                w = empty_weight
            else:
                w = min((n + eps) ** power, max_case_weight)
            self.train_weights.append(float(w))

        self.train_num_samples = int(getattr(cfg, "train_num_samples", len(train_df)))
    # --------------------------------------------------
    # transforms
    # --------------------------------------------------
    def _get_load_transforms(self):
        return [
            LoadImaged(keys=self.modalities),
            LoadImaged(keys=["instance_mask"], dtype=torch.int32),
            LoadImaged(keys=["brain_mask"], dtype=torch.uint8),

            EnsureChannelFirstd(keys=self.modalities + ["instance_mask", "brain_mask"]),

            CopyItemsd(keys=["instance_mask"], times=1, names=["semantic_mask"]),
            Lambdad(keys=["semantic_mask"], func=lambda m: m > 0),

            CropForegroundd(
                keys=self.modalities + ["instance_mask", "semantic_mask", "brain_mask"],
                source_key="brain_mask",
                allow_smaller=True,
                margin=4,
            ),

            MaskedPercentileNormalizeIntensityd(
                keys=self.modalities,
                mask_key="brain_mask",
                percentiles=(1.0, 99.0),
                channel_wise=False,
                z_clamp=(-6, 6),
            ),

            ConcatItemsd(keys=self.modalities, name="inputs", dim=0),
            DeleteItemsd(keys=self.modalities + ["brain_mask"]),

            EnsureTyped(keys=["inputs"], dtype=torch.float32, track_meta=True),
            EnsureTyped(keys=["instance_mask", "semantic_mask"], dtype=torch.int32, track_meta=False),
        ]

    def _get_augmentation_transforms(self):
        keys_all = ["inputs", "instance_mask", "semantic_mask"]

        return [
            RandRotated(
                keys=keys_all,
                range_x=0.0,
                range_y=0.0,
                range_z=np.pi / 12,
                prob=self.spatial_augmentation_prob,
                keep_size=True,
                mode=["bilinear", "nearest", "nearest"],
            ),
            RandZoomd(
                keys=keys_all,
                prob=self.spatial_augmentation_prob,
                min_zoom=(0.9, 0.9, 1.0),
                max_zoom=(1.1, 1.1, 1.0),
                mode=["bilinear", "nearest", "nearest"],
                keep_size=True,
            ),
            RandFlipd(
                keys=keys_all,
                prob=self.spatial_augmentation_prob,
                spatial_axis=[0, 1],
            ),
            OneOf([
                RandScaleIntensityd(
                    keys=["inputs"],
                    prob=self.intensity_augmentation_prob,
                    factors=0.1,
                ),
                RandShiftIntensityd(
                    keys=["inputs"],
                    prob=self.intensity_augmentation_prob,
                    offsets=0.03,
                ),
                RandAdjustContrastd(
                    keys=["inputs"],
                    prob=self.intensity_augmentation_prob,
                    gamma=(0.9, 1.1),
                ),
                RandGaussianNoised(
                    keys=["inputs"],
                    prob=self.intensity_augmentation_prob,
                    std=0.005,
                ),
            ]),
        ]

    def _get_center_patches_transforms(self):
        return [
            MakeBalancedInstanceWeightMapd(instance_key="instance_mask", out_key="inst_wmap"),

            RandWeightedCropd(
                keys=["inputs", "instance_mask", "semantic_mask"],
                w_key="inst_wmap",
                spatial_size=self.patch_size,
                num_samples=1,
            ),

            SpatialPadd(
                keys=["inputs", "instance_mask", "semantic_mask"],
                spatial_size=self.patch_size,
                method="symmetric",
                mode="constant",
                value=0,
            )
        ]

    def _get_grid_patches_transforms(self):
        return [
            CopyItemsd(keys=["inputs"], times=1, names=["inputs_orig"]),

            SymmetricGridPad(
                keys=["inputs", "inputs_orig", "instance_mask", "semantic_mask"],
                patch_size=self.patch_size,
                overlap=self.patch_overlap,
            ),

            GridPatchd(
                keys=["inputs"],
                patch_size=self.patch_size,
                overlap=self.patch_overlap,
                pad_mode=None,
            ),
        ]

    def _get_full_image_transforms(self):
        """
        Produce a single full-size cropped/padded image.
        Final size: [192, 240, 72] by default.
        """
        return [
            CopyItemsd(keys=["inputs"], times=1, names=["inputs_orig"]),

            SpatialPadd(
                keys=["inputs", "inputs_orig", "instance_mask", "semantic_mask"],
                spatial_size=self.full_image_size,
                method="symmetric",
                mode="constant",
                value=0,
            ),

            CenterSpatialCropd(
                keys=["inputs", "inputs_orig", "instance_mask", "semantic_mask"],
                roi_size=self.full_image_size,
            ),
        ]

    def _get_det_transforms(self):
        return [
            InstanceMaskToDetd(
                instance_key="instance_mask",
                default_class=0,
            )
        ]

    def _build_split_transform(
            self,
            *,
            split: str,
            image_mode: str,
            with_augmentation: bool,
    ):
        tfm = []
        tfm += self._get_load_transforms()

        if image_mode == "patch":
            if split == "train":
                tfm += self._get_center_patches_transforms()
                if with_augmentation:
                    tfm += self._get_augmentation_transforms()
            else:
                tfm += self._get_grid_patches_transforms()

        elif image_mode == "full":
            tfm += self._get_full_image_transforms()
            if split == "train" and with_augmentation:
                tfm += self._get_augmentation_transforms()
        else:
            raise ValueError(f"Unknown image_mode={image_mode!r}")

        tfm += self._get_det_transforms()
        return Compose(tfm)

    # --------------------------------------------------
    # setup
    # --------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.datasets["train"] = Dataset(
                data=self.cases["train"],
                transform=self._build_split_transform(
                    split="train",
                    image_mode=self.image_mode,
                    with_augmentation=True,
                ),
            )

            self.datasets["val"] = Dataset(
                data=self.cases["val"],
                transform=self._build_split_transform(
                    split="val",
                    image_mode=self.image_mode,
                    with_augmentation=False,
                ),
            )

        if stage in (None, "validate") and "val" not in self.datasets:
            self.datasets["val"] = Dataset(
                data=self.cases["val"],
                transform=self._build_split_transform(
                    split="val",
                    image_mode=self.image_mode,
                    with_augmentation=False,
                ),
            )

        if stage in (None, "test"):
            self.datasets["test"] = Dataset(
                data=self.cases["test"],
                transform=self._build_split_transform(
                    split="test",
                    image_mode=self.image_mode,
                    with_augmentation=False,
                ),
            )

        if stage in (None, "predict"):
            self.datasets["predict"] = Dataset(
                data=self.cases["test"],
                transform=self._build_split_transform(
                    split="predict",
                    image_mode=self.image_mode,
                    with_augmentation=False,
                ),
            )

    # --------------------------------------------------
    # device transfer
    # --------------------------------------------------
    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        def move_iterable_to_device(iterable):
            if isinstance(iterable, list):
                for i, v in enumerate(iterable):
                    if isinstance(v, torch.Tensor):
                        iterable[i] = v.to(device)
                    elif isinstance(v, (list, dict)):
                        move_iterable_to_device(v)
            elif isinstance(iterable, dict):
                for k, v in iterable.items():
                    if isinstance(v, torch.Tensor):
                        iterable[k] = v.to(device)
                    elif isinstance(v, (list, dict)):
                        move_iterable_to_device(v)

        if isinstance(device, str):
            device = torch.device(device)

        if isinstance(batch, tuple):
            if len(batch) > 0 and isinstance(batch[0], dict):
                move_iterable_to_device(batch[0])
        else:
            move_iterable_to_device(batch)

        return batch

    # --------------------------------------------------
    # dataloaders
    # --------------------------------------------------
    def _get_dataloader(self, subset: str):
        dataset = self.datasets[subset]

        sampler = None
        shuffle = False

        if subset == "train":
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()

                # IMPORTANT:
                # train_num_samples should be GLOBAL target if you want to preserve
                # your old epoch size; divide by world_size for per-rank samples.
                global_num_samples = getattr(self, "train_num_samples", len(self.cases["train"]))
                per_rank_num_samples = math.ceil(global_num_samples / world_size)

                sampler = DistributedWeightedSampler(
                    weights=self.train_weights,
                    num_samples=per_rank_num_samples,
                    replacement=True,
                    num_replicas=world_size,
                    rank=rank,
                    seed=getattr(self, "seed", 0),
                )
            else:
                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(self.train_weights, dtype=torch.double),
                    num_samples=getattr(self, "train_num_samples", len(self.cases["train"])),
                    replacement=True,
                )
        else:
            shuffle = False
            sampler = None

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size[subset],
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers[subset],
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers[subset] > 0),
            drop_last=(subset == "train"),
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._get_dataloader("train")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader("val")

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader("test")

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader("predict")