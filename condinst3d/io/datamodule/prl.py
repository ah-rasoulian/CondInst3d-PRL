from typing import Any, Literal
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from monai.data import PersistentDataset, Dataset, GDSDataset
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from omegaconf import DictConfig, OmegaConf
import pickle
import json
import os
from monai.transforms import (
    Compose,
    LoadImaged, Lambdad, EnsureChannelFirstd, EnsureTyped, ConcatItemsd, DeleteItemsd,
    Orientationd, NormalizeIntensityd, CenterSpatialCropd,
    RandCropByLabelClassesd, RandSpatialCropd, SpatialPadd, GridPatchd,
    RandRotated, RandZoomd, RandFlipd, OneOf, RandRicianNoised, RandGaussianNoised,
    RandScaleIntensityd, RandShiftIntensityd, RandAdjustContrastd, RandGaussianSharpend, CopyItemsd,
)
import numpy as np
from condinst3d.io.transforms import LoadJSONd, InstanceMaskToOneHotd, OneHotToBoxesd
from condinst3d.io.collate import multi_instance_collate
from condinst3d.io.sampler import DistributedWeightedSampler
from functools import partial


def build_cases(cases, modalities, image_directory, label_directory):
    df_list = []
    for case in cases:
        case_dict = {'case': case}
        for i, m in enumerate(modalities):
            case_dict[m] = os.path.join(image_directory, f"{case}_{i:04d}.nii.gz")
        case_dict['instance_mask'] = os.path.join(label_directory, f"{case}.nii.gz")
        case_dict['info'] = os.path.join(label_directory, f"{case}.json")
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
        self.augmentation_prob = cfg.augmentation_prob
        self.collate_fn = partial(
            multi_instance_collate,
            collate_keys=['inputs', 'inputs_orig', 'instance_mask', 'boxes', 'classes', 'onehot'],
            target_keys={"targets": {"boxes": "boxes", "classes": "classes", "onehot": "onehot"}}
        )
        self.datasets = {}

        # --- compute instance counts & sampling weights for train cases ---
        train_counts = []
        for item in train_df:
            with open(item["info"], "r") as f:
                info = json.load(f)

            # your format: {"instances": {0:0, 1:0, ...}} → number of instances = number of keys
            n_inst = len(info.get("instances", {}).keys())
            train_counts.append(n_inst)

        # weight design (simple + works well):
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
            # load .nii/.nii.gz images and .json info
            LoadImaged(keys=self.modalities),
            LoadImaged(keys=['instance_mask'], dtype=torch.int16),
            LoadJSONd(keys=['info']),

            # make each modality channel-first: [C=1, H, W, D]
            EnsureChannelFirstd(keys=self.modalities + ["instance_mask"]),
            # make orientation consistent
            Orientationd(keys=self.modalities + ["instance_mask"], axcodes="RAS", labels=None),
            # concat modalities -> "inputs" with shape [C, H, W, D]
            ConcatItemsd(keys=self.modalities, name="inputs", dim=0),
            # delete individual mods
            DeleteItemsd(keys=self.modalities),
            # center cropping background
            CenterSpatialCropd(keys=["inputs", "instance_mask"], roi_size=(192, 240, 72)),
            # make torch tensors + meta, put on correct dtype
            EnsureTyped(keys=["inputs"], dtype=torch.float32, track_meta=True),
            EnsureTyped(keys=["instance_mask"], dtype=torch.int16, track_meta=False),
            # normalize modality intensities
            NormalizeIntensityd(keys=["inputs"], channel_wise=True),
        ]

    def _get_augmentation_transorms(self):
        return [
            # spatial augmentation
            RandRotated(
                keys=['inputs', 'instance_mask'],
                range_x=np.pi / 4, range_y=np.pi / 4, range_z=np.pi / 4,
                prob=self.augmentation_prob,
                keep_size=False,
                mode=["bilinear", "nearest"],
                lazy=True,
            ),
            RandZoomd(
                keys=['inputs', 'instance_mask'],
                prob=self.augmentation_prob,
                min_zoom=0.75, max_zoom=1.25,
                mode=["area", "nearest-exact"],
                keep_size=False,
                lazy=True,
            ),
            RandFlipd(
                keys=['inputs', 'instance_mask'],
                prob=self.augmentation_prob,
                spatial_axis=[0, 1, 2],
                lazy=True,
            ),
            # intensity augmentation
            RandScaleIntensityd(
                keys=['inputs'],
                prob=self.augmentation_prob,
                factors=0.25,
            ),
            RandShiftIntensityd(
                keys=['inputs'],
                prob=self.augmentation_prob,
                offsets=0.1,
            ),
            RandAdjustContrastd(
                keys=['inputs'],
                prob=self.augmentation_prob,
                gamma=(0.7, 1.5),
            ),
            RandGaussianSharpend(
                keys=['inputs'],
                prob=self.augmentation_prob,
            ),
            OneOf([
                RandRicianNoised(
                    keys=['inputs'],
                    prob=self.augmentation_prob,
                    channel_wise=True,
                    relative=True,
                ),
                RandGaussianNoised(
                    keys=['inputs'],
                    prob=self.augmentation_prob,
                    std=0.01,
                )
            ]),

        ]

    def _get_center_patches_transforms(self):
        return [
            RandCropByLabelClassesd(
                keys=["inputs", "instance_mask"],
                label_key="instance_mask",
                image_key="inputs",
                image_threshold=0.1,
                spatial_size=np.array(self.patch_size) * 1.5,
                num_classes=64,
                num_samples=self.patches_per_subject,
                allow_smaller=True,
                warn=False,
                lazy=True,
            ),
            RandSpatialCropd(
                keys=['inputs', 'instance_mask'],
                roi_size=self.patch_size,
                random_center=True,
                lazy=True,
            ),
            SpatialPadd(
                keys=['inputs', 'instance_mask'],
                spatial_size=self.patch_size,
                mode='constant',
                lazy=True,
            )
        ]

    def _get_grid_patches_transforms(self):
        return [
            CopyItemsd(keys=['inputs'], times=1, names=['inputs_orig']),

            GridPatchd(
                keys=['inputs'],
                patch_size=self.patch_size,
                overlap=self.patch_overlap,
                pad_mode='constant'
            ),
        ]

    def _get_det_transforms(self):
        return [
            # create a onehot mask for instance mask
            InstanceMaskToOneHotd(keys=['instance_mask'], out_key='onehot', include_background=False),
            # create boxes and class tensors
            OneHotToBoxesd(
                onehot_key="onehot",
                boxes_key="boxes",
                classes_key="classes",
                default_class=0,
                max_instances=-1,
            ),
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

        if stage in ["test", "predict"]:
            test_transform = self._get_load_transforms()
            test_transform += self._get_grid_patches_transforms()
            test_transform += self._get_det_transforms()
            self.datasets[stage] = Dataset(data=self.cases['test'], transform=Compose(test_transform))

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
        sampler = None
        shuffle = (subset == "train")

        if subset == "train" and hasattr(self, "train_weights"):
            sampler = DistributedWeightedSampler(
                self.train_weights,
                num_samples=len(self.train_weights),  # one "epoch"
                replacement=True,
                seed=42,
            )
            shuffle = False

        return DataLoader(
            dataset=self.datasets[subset],
            batch_size=self.batch_size[subset],
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=self.num_workers[subset],
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers[subset] > 0),
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._get_dataloader(subset='train')

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader(subset='val')

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader(subset='test')

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader(subset='predict')
