from typing import Sequence, Any
import torch
from torch import Tensor
from torchmetrics import Metric
import matplotlib.pyplot as plt
import numpy as np

class AveragePrecision(Metric):
    def __init__(
            self,
            iou_thresholds: float | Sequence[float] = 0.1,
            interpolation: None | int = None,
            dist_sync_on_step: bool = False,
    ):
        """
        a class for computing average precision per IoU for an instance segmentation or object detection task

        :param iou_thresholds: thresholds for a detection to be considered as TP
        :param interpolation: n-point interpolation on precision/recall curve. None for every-point interpolation.
        :param dist_sync_on_step: whether to sync multiple devices at steps
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        iou_thresholds = [iou_thresholds] if isinstance(iou_thresholds, float) else iou_thresholds
        self.interpolation = interpolation
        self.iou_thresholds = torch.tensor(iou_thresholds)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("iou_max", default=[], dist_reduce_fx="cat")
        self.add_state("num_fg", default=torch.tensor(0), dist_reduce_fx="sum")

        self.precision_curve = None
        self.recall_curve = None

    def update(self, iou_matrix: Tensor, scores: Tensor) -> None :
        assert iou_matrix.dim() == 2
        iou_matrix = iou_matrix.to(self.device)
        scores = scores.to(self.device)
        self.iou_thresholds = self.iou_thresholds.to(self.device)

        if iou_matrix.size(1) == 0:
            # we have predictions but no ground truth
            iou_max = iou_matrix.new_zeros(size=(iou_matrix.size(0),))
        else:
            # Match closest gt to each prediction and get it's iou
            iou_max, assigned_gt = iou_matrix.max(dim=1)
            # Only keep the best detection for each ground-truth and consider others as FP
            # TODO: Make it faster
            for det_idx in range(iou_matrix.size(0)):
                best_iou = torch.max(iou_max[assigned_gt == assigned_gt[det_idx]])
                if iou_max[det_idx] < best_iou:
                    iou_max[det_idx] = 0
        self.iou_max.append(iou_max)
        self.scores.append(scores)
        self.num_fg += iou_matrix.shape[1]

    def compute(self) -> Any:
        self.iou_thresholds = self.iou_thresholds.to(self.device)
        if isinstance(self.iou_max, list):
            iou_max = torch.cat(self.iou_max)
        else:
            iou_max = self.iou_max
        if isinstance(self.scores, list):
            scores = torch.cat(self.scores)
        else:
            scores = self.scores

        # Sort predictions by scores in descending order
        sorted_indices = scores.argsort(descending=True)
        sorted_iou = iou_max[sorted_indices]

        # Binary classification: 1 if IoU exceeds threshold, else 0; Broadcast over all thresholds.
        tp_mask = sorted_iou >= self.iou_thresholds.view(-1, 1)

        tp_mask_cum = torch.cumsum(tp_mask, dim=1)
        index_tensor = torch.arange(1, tp_mask_cum.size(1) + 1, dtype=torch.float, device=self.device)
        precision = tp_mask_cum / index_tensor.unsqueeze(0)
        recall = tp_mask_cum / (self.num_fg + 1e-6)

        # Prepare tensor to store average precision for each IoU threshold
        precisions_curve, recall_curve = [], []

        if self.interpolation is not None:
            recall_levels = torch.linspace(0, 1, self.interpolation, device=self.device)
            precision_interp = torch.stack([
                torch.tensor([torch.max(precision[i][recall[i] >= r]) if torch.any(recall[i] >= r)
                              else 0. for r in recall_levels], device=self.device).float()
                for i in range(len(self.iou_thresholds))
            ])
            average_precisions = precision_interp.mean(dim=1)
            self.precision_curve = precision_interp
            self.recall_curve = recall_levels.unsqueeze(0).repeat(len(self.iou_thresholds), 1)
        else:  # every point interpolation
            average_precisions = torch.trapz(precision, recall, dim=1)
            self.precision_curve = precisions_curve
            self.recall_curve = recall_curve

        return average_precisions

    def plot(self) -> Any:
        if self.precision_curve is None or self.recall_curve is None:
            self.compute()

        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(10, 6))

        # Generate a color map with m different colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.iou_thresholds)))

        self.recall_curve = self.recall_curve.cpu().numpy()
        self.precision_curve = self.precision_curve.cpu().numpy()
        # Plot each row with a different color
        for i, iou in enumerate(self.iou_thresholds):
            ax.plot(self.recall_curve[i], self.precision_curve[i], color=colors[i], label=f"{iou:.2f}")

        # Add labels and title
        ax.set_xlabel('Recall')
        ax.set_xlim([0, 1])
        ax.set_ylabel('Precision')
        ax.set_ylim([0, 1])
        ax.set_title(f'Precision/Recall curve with {self.interpolation} interpolation.')
        ax.legend()
        ax.grid(True)

        return fig, ax
