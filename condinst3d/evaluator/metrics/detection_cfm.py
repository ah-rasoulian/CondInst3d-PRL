from typing import Sequence, Any
import torch
from torch import Tensor
from torchmetrics import Metric
import matplotlib.pyplot as plt
import numpy as np


class DetectionConfusionMatrix(Metric):
    def __init__(
            self,
            iou_thresholds: float | Sequence[float] = 0.1,
            dist_sync_on_step: bool = False,
    ):
        """
        a class for computing confusion matrix per IoU for an instance segmentation or object detection task
        the format of confusion matrix is: [TP, FP, FN]

        :param iou_thresholds: thresholds for a detection to be considered as TP
        :param dist_sync_on_step: whether to sync multiple devices at steps
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        iou_thresholds = [iou_thresholds] if isinstance(iou_thresholds, float) else iou_thresholds
        self.iou_thresholds = torch.tensor(iou_thresholds)
        self.add_state(f"cfm", default=torch.zeros(len(self.iou_thresholds), 3), dist_reduce_fx="sum")

    def update(self, iou_matrix: Tensor) -> None:
        """
        Updates the confusion matrix given the IoU matrix.

        :param iou_matrix: tensor of shape [N, M] - pairwise IoU between N pred instances and M target instances
        """
        assert iou_matrix.dim() == 2
        iou_matrix = iou_matrix.to(self.device)
        self.iou_thresholds = self.iou_thresholds.to(self.device)

        if iou_matrix.size(1) == 0:
            # we have predictions but no ground truth
            tp = torch.zeros_like(self.iou_thresholds)
            fp = torch.ones_like(tp) * iou_matrix.size(0)
            fn = torch.zeros_like(tp)
        else:
            # Match closest gt to each prediction and get it's iou
            iou_max, assigned_gt = iou_matrix.max(dim=1)
            # Binary classification: 1 if IoU exceeds threshold, else 0; Broadcast over all thresholds.
            tp_mask = (iou_max >= self.iou_thresholds.view(-1, 1))

            tp = torch.zeros_like(self.iou_thresholds)
            fp = torch.zeros_like(self.iou_thresholds)
            fn = torch.zeros_like(self.iou_thresholds)
            for i, th in enumerate(self.iou_thresholds):
                assigned_gt_filtered = assigned_gt[tp_mask[i]]
                unique_gt, counts = torch.unique(assigned_gt_filtered, return_counts=True)

                # True Positives: one per GT
                tp[i] = len(unique_gt)
                fp[i] = iou_matrix.size(0) - tp[i]
                fn[i] = iou_matrix.size(1) - tp[i]

        cfm = torch.stack([tp, fp, fn], dim=1)

        self.cfm += cfm

    def compute(self) -> Any:
        return self.cfm
