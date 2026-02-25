from typing import Any
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_lightning as pl
import torch
from monai.data import Dataset
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from omegaconf import DictConfig, OmegaConf
import pickle
import json
import os
from monai.transforms import (
    Compose,
    LoadImaged, Lambdad, EnsureChannelFirstd, EnsureTyped, ConcatItemsd, DeleteItemsd, CropForegroundd,
    RandWeightedCropd, RandSpatialCropd, GridPatchd,
    RandRotated, RandZoomd, RandFlipd, OneOf, RandRicianNoised, RandGaussianNoised,
    RandScaleIntensityd, RandShiftIntensityd, RandAdjustContrastd, RandGaussianSharpend, CopyItemsd,
)
import numpy as np
from condinst3d.io.transforms import (MaskedPercentileNormalizeIntensityd, InstanceMaskToDetd,
                                      MakeBalancedInstanceWeightMapd, SpatialPadWithMind, SymmetricGridPad)
from condinst3d.io.collate import multi_instance_collate
from functools import partial


def build_cases(cases, modalities, image_directory, label_directory):
    df_list = []
    for case in cases:
        case_dict = {'case': case}
        for i, m in enumerate(modalities):
            case_dict[m] = os.path.join(image_directory, f"{case}_{i:04d}.nii.gz")
        case_dict['instance_mask'] = os.path.join(label_directory, f"{case}.nii.gz")
        case_dict['brain_mask'] = os.path.join(image_directory, "brainmask", f"{case}_brainmask.nii.gz")
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

        all_cases = [c for c in os.listdir(os.path.join(root_path, 'raw')) if c.startswith('case')]
        with open(os.path.join(root_path, 'preprocessed', 'splits_final.pkl'), "rb") as f:
            splits = pickle.load(f)

        train_cases, val_cases = splits[fold]['train'], splits[fold]['val']
        test_cases = [c for c in all_cases if c not in train_cases and c not in val_cases]

        splitted_path = os.path.join(root_path, 'raw_splitted')
        train_df = build_cases(train_cases, modalities, os.path.join(splitted_path, 'imagesTr'), os.path.join(splitted_path, 'labelsTr'))
        val_df = build_cases(val_cases, modalities, os.path.join(splitted_path, 'imagesTr'), os.path.join(splitted_path, 'labelsTr'))
        test_df = build_cases(test_cases, modalities, os.path.join(splitted_path, 'imagesTs'), os.path.join(splitted_path, 'labelsTs'))

        self.cases = {'train': train_df, 'val': val_df, 'test': test_df}
        self.modalities = modalities
        self.num_workers = cfg.num_workers
        self.batch_size = cfg.batch_size
        self.patch_size = cfg.patch_size
        self.patches_per_subject = cfg.patches_per_subject
        self.patch_overlap = cfg.patch_overlap
        self.spatial_augmentation_prob = cfg.spatial_augmentation_prob
        self.intensity_augmentation_prob = cfg.intensity_augmentation_prob
        self.collate_fn = partial(
            multi_instance_collate,
            collate_keys=['inputs', 'inputs_orig', 'instance_mask', 'semantic_mask', 'boxes', 'classes', 'onehot'],
            target_keys={"targets":
                             {"boxes": "boxes",
                              "classes": "classes",
                              "onehot": "onehot",
                              "semantic_mask": "semantic_mask"}
                         }
        )
        self.datasets = {}

        # --- compute instance counts & sampling weights for train cases ---
        train_counts = []
        for item in train_df:
            # info format: {"instances": {0:0, 1:0, ...}} → number of instances = number of keys
            n_inst = len(item["info"].get("instances", {}).keys())
            train_counts.append(n_inst)

        # weight design:
        # - keep some probability for N=0 (so model learns true negatives)
        # - oversample larger-N cases smoothly
        empty_weight = getattr(cfg, "empty_case_weight", 0.2)  # keep negatives but not too often
        power = getattr(cfg, "instance_weight_power", 0.7)  # smooth oversampling
        eps = 1e-6

        self.train_weights = []
        for n in train_counts:
            if n == 0:
                w = empty_weight
            else:
                w = float((n + eps) ** power)
            self.train_weights.append(w)

    def _get_load_transforms(self):
        return [
            # load .nii/.nii.gz images
            LoadImaged(keys=self.modalities),
            LoadImaged(keys=['instance_mask'], dtype=torch.int32),
            LoadImaged(keys=['brain_mask'], dtype=torch.uint8),

            # make each modality channel-first: [C=1, H, W, D]
            EnsureChannelFirstd(keys=self.modalities + ["instance_mask", "brain_mask"]),

            # add semantic mask
            CopyItemsd(keys=['instance_mask'], times=1, names=['semantic_mask']),
            Lambdad(keys=['semantic_mask'], func=lambda m: m > 0),

            # crop with the brain mask
            CropForegroundd(
                keys=self.modalities + ["instance_mask", "semantic_mask", "brain_mask"],
                source_key="brain_mask",
                allow_smaller=True,
                margin=4,
            ),

            # normalize input intensities
            MaskedPercentileNormalizeIntensityd(
                keys=[x for x in self.modalities if x != "freqmap"],
                mask_key="brain_mask",
                percentiles=(0.5, 99.5),
                channel_wise=False,
            ),
            MaskedPercentileNormalizeIntensityd(
                keys=["freqmap"],
                mask_key="brain_mask",
                percentiles=(2, 98),
                channel_wise=False,
            ),

            # concat modalities -> "inputs" with shape [C, H, W, D]
            ConcatItemsd(keys=self.modalities, name="inputs", dim=0),

            # delete individual mods and brain_mask
            DeleteItemsd(keys=self.modalities + ["brain_mask"]),

            # make torch tensors + meta, put on correct dtype
            EnsureTyped(keys=["inputs"], dtype=torch.float32, track_meta=True),
            EnsureTyped(keys=["instance_mask", "semantic_mask"], dtype=torch.int32, track_meta=False),
        ]

    def _get_augmentation_transorms(self):
        keys_all = ["inputs", "instance_mask", "semantic_mask"]

        return [
            # ---------------------------
            # spatial augmentation (XY only)
            # ---------------------------

            # rotate only in the XY plane (no XZ/YZ rotations)
            RandRotated(
                keys=keys_all,
                range_x=0.0,  # rotation about x (affects YZ) -> disable
                range_y=0.0,  # rotation about y (affects XZ) -> disable
                range_z=np.pi / 12,  # rotation about z (in-plane XY) -> keep
                prob=self.spatial_augmentation_prob,
                keep_size=False,
                mode=["bilinear", "nearest", "nearest"],
            ),

            # zoom only in XY (keep z zoom = 1.0)
            RandZoomd(
                keys=keys_all,
                prob=self.spatial_augmentation_prob,
                min_zoom=(0.9, 0.9, 1.0),
                max_zoom=(1.1, 1.1, 1.0),
                mode=["bilinear", "nearest", "nearest"],
                keep_size=False,
            ),

            # flips only along X and Y and Z
            RandFlipd(
                keys=keys_all,
                prob=self.spatial_augmentation_prob,
                spatial_axis=[0, 1],
            ),

            # ---------------------------
            # intensity augmentation
            # ---------------------------
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
            # build balanced weight map from instance ids (recommended)
            MakeBalancedInstanceWeightMapd(instance_key="instance_mask", out_key="inst_wmap"),

            RandWeightedCropd(
                keys=["inputs", "instance_mask", "semantic_mask"],
                w_key="inst_wmap",
                spatial_size=np.array(self.patch_size) * 1.5,
                num_samples=self.patches_per_subject,
            ),

            RandSpatialCropd(
                keys=["inputs", "instance_mask", "semantic_mask"],
                roi_size=self.patch_size,
                random_center=True,
            ),

            # pad inputs with background-like value in normalized space; masks with 0
            SpatialPadWithMind(
                keys=["inputs", "instance_mask", "semantic_mask"],
                mask_keys=["instance_mask", "semantic_mask"],
                spatial_size=self.patch_size,
            ),
        ]

    def _get_grid_patches_transforms(self):
        return [
            CopyItemsd(keys=["inputs"], times=1, names=["inputs_orig"]),

            # Symmetric, min-value padding so first/last patches are equally affected
            SymmetricGridPad(
                keys=["inputs", "inputs_orig", "instance_mask", "semantic_mask"],
                patch_size=self.patch_size,
                overlap=self.patch_overlap,
            ),

            GridPatchd(
                keys=["inputs"],
                patch_size=self.patch_size,
                overlap=self.patch_overlap,
                pad_mode=None,  # preferred: no extra padding needed now
            ),
        ]

    def _get_det_transforms(self):
        return [
            # create boxes, class and onehot tensors
            InstanceMaskToDetd(
                instance_key="instance_mask",
                default_class=0,
            )
        ]

    def setup(self, stage):
        if stage == 'fit':
            train_transform = self._get_load_transforms()
            train_transform += self._get_augmentation_transorms()
            train_transform += self._get_center_patches_transforms()
            train_transform += self._get_det_transforms()
            self.datasets['train'] = Dataset(data=self.cases['train'], transform=Compose(train_transform))

        if stage in ["fit", "validate"]:
            val_transform = self._get_load_transforms()
            val_transform += self._get_grid_patches_transforms()
            val_transform += self._get_det_transforms()
            self.datasets['val'] = Dataset(data=self.cases['val'], transform=Compose(val_transform))

        # if stage in ["test", "predict"]:
            test_transform = self._get_load_transforms()
            test_transform += self._get_grid_patches_transforms()
            test_transform += self._get_det_transforms()
            # self.datasets[stage] = Dataset(data=self.cases['test'], transform=Compose(test_transform))
            self.datasets['test'] = Dataset(data=self.cases['test'], transform=Compose(test_transform))

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        def move_iterable_to_device(iterable):
            if isinstance(iterable, list):
                for i, v in enumerate(iterable):
                    if isinstance(v, torch.Tensor):
                        iterable[i] = v.to(device)
                    elif isinstance(v, list) or isinstance(v, dict):
                        move_iterable_to_device(v)
                    else:
                        continue
            if isinstance(iterable, dict):
                for k, v in iterable.items():
                    if isinstance(v, torch.Tensor):
                        iterable[k] = v.to(device)
                    elif isinstance(v, list) or isinstance(v, dict):
                        move_iterable_to_device(v)
                    else:
                        continue

        if isinstance(device, str):
            device = torch.device(device)

        if isinstance(batch, tuple):
            if len(batch[0]) == 2:
                for v in batch[0].values():
                    move_iterable_to_device(v)
            else:
                move_iterable_to_device(batch[0])
        else:
            move_iterable_to_device(batch)
        return batch

    def _get_dataloader(self, subset):
        dataset = self.datasets[subset]
        shuffle = (subset == "train")

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size[subset],
            shuffle=shuffle,
            num_workers=self.num_workers[subset],
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers[subset] > 0),
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._get_dataloader(subset='train')

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # return self._get_dataloader(subset='val')
        return self._get_dataloader(subset='test')

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader(subset='test')

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader(subset='predict')
