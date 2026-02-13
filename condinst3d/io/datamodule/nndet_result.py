from typing import Any
import pytorch_lightning as pl
import os
from monai.data import Dataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
import torch
import numpy as np
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ConcatItemsd, DeleteItemsd, Lambdad, CopyItemsd
from condinst3d.io.transforms import (LoadInfod, InstanceMaskToDetd, SemanticToInstanced, InstanceScoresFromSoftmaxd,
                                      FilterAndUnpackPredsd, MatchSegBoxesToPredScoresd)
from functools import partial
from condinst3d.io.collate import multi_instance_collate


class nnDetectionResult(pl.LightningDataModule):
    def __init__(
            self,
            split_root,
            pred_root,
            n_modalities,
            n_workers = 0,
            score_threshold = 0.5,
            iou_threshold = 0.5,
    ):
        super().__init__()

        test_images_dir = os.path.join(split_root, 'imagesTs')
        test_labels_dir = os.path.join(split_root, 'labelsTs')
        case_names = [f.removesuffix(".nii.gz") for f in os.listdir(test_labels_dir) if f.endswith("nii.gz")]
        cases = []
        for case_name in case_names:
            case_dict = {
                "case": case_name,
                "instance_mask": os.path.join(test_labels_dir, f"{case_name}.nii.gz"),
                "instance_mask_info": os.path.join(test_labels_dir, f"{case_name}.json"),
                "pred_boxes": os.path.join(pred_root, f"{case_name}_boxes.pkl"),
                "pred_seg": os.path.join(pred_root, f"{case_name}_seg.pkl"),
            }
            for i in range(n_modalities):
                case_dict[f"modality-{i}"] = os.path.join(test_images_dir, f"{case_name}_{i:04d}.nii.gz")
            cases.append(case_dict)

        self.cases = cases
        self.dataset = None
        self.modalities = [f"modality-{i}" for i in range(n_modalities)]
        self.n_workers = n_workers
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        self.collate_fn = partial(
            multi_instance_collate,
            collate_keys=['inputs', 'instance_mask', 'semantic_mask', 'gt_onehot', 'gt_boxes', 'gt_classes',
                          'pred_instance_mask', 'pred_seg', 'pred_onehot', 'pred_boxes', 'pred_classes', 'pred_scores',
                          'pred_boxes_f', 'pred_classes_f', 'pred_scores_f'],
            target_keys={
                "targets": {
                    "instance_mask": "instance_mask",
                    "semantic_mask": "semantic_mask",
                    "gt_onehot": "onehot",
                    "gt_boxes": "boxes",
                    "gt_classes": "classes"
                },
                "pred_seg": {
                    "pred_instance_mask": "instance_mask",
                    "pred_seg": "semantic_mask",
                    "pred_onehot": "onehot",
                    "pred_boxes": "boxes",
                    "pred_classes": "classes",
                    "pred_scores": "scores",
                },
                "pred_box": {
                    "pred_boxes_f": "boxes",
                    "pred_scores_f": "scores",
                    "pred_classes_f": "classes",
                }
            }
        )

    def _get_load_transforms(self):
        return [
            LoadImaged(keys=self.modalities),
            LoadImaged(keys=["instance_mask"]),
            LoadInfod(keys=["pred_boxes", "pred_seg"]),
            FilterAndUnpackPredsd(
                keys="pred_boxes",
                score_threshold=self.score_threshold,
                out_boxes_key="pred_boxes_f",
                out_scores_key="pred_scores_f",
                out_labels_key="pred_classes_f"
            ),
            DeleteItemsd(["pred_boxes"]),
            Lambdad(keys="pred_seg", func=lambda x: np.transpose(x['pred_seg'], (2, 1, 0))),

            CopyItemsd(keys=["instance_mask"], times=1, names=["semantic_mask"]),
            Lambdad(keys=["semantic_mask"], func=lambda x: (x > 0).float()),

            EnsureChannelFirstd(keys=self.modalities + ["instance_mask", "semantic_mask", "pred_seg"], channel_dim='no_channel'),
            ConcatItemsd(keys=self.modalities, name="inputs", dim=0),
            DeleteItemsd(keys=self.modalities),
        ]

    def _get_det_transforms(self):
        return [
            SemanticToInstanced(
                keys="pred_seg",
                out_key="pred_instance_mask",
                connectivity=3,  # 3D: 1=6-neigh, 2=18, 3=26s
            ),
            # create boxes, class and onehot tensors
            InstanceMaskToDetd(
                instance_key="instance_mask",
                onehot_key="gt_onehot",
                boxes_key="gt_boxes",
                classes_key="gt_classes",
                default_class=0,
            ),
            InstanceMaskToDetd(
                instance_key="pred_instance_mask",
                onehot_key="pred_onehot",
                boxes_key="pred_boxes",
                classes_key="pred_classes",
                default_class=0,
            ),
            MatchSegBoxesToPredScoresd(
                seg_boxes_key="pred_boxes",
                pred_boxes_key="pred_boxes_f",  # from FilterAndUnpackPredsd (xyzxyz)
                pred_scores_key="pred_scores_f",
                out_scores_key="pred_scores",
                iou_threshold=self.iou_threshold,
                default_score=self.score_threshold,
            )
        ]

    def setup(self, stage):
        t = self._get_load_transforms()
        t += self._get_det_transforms()

        self.dataset = Dataset(data=self.cases, transform=Compose(t))

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


    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.n_workers,
            collate_fn=self.collate_fn
        )

