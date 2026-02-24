from torch import Tensor
import torch


def compute_precision(confusion_matrix: Tensor, eps: float = 1e-8) -> Tensor:
    assert confusion_matrix.dim() == 2 and confusion_matrix.size(1) == 3
    tp = confusion_matrix[:, 0].to(torch.float32)
    fp = confusion_matrix[:, 1].to(torch.float32)
    return tp / (tp + fp + eps)


def compute_recall(confusion_matrix: Tensor, eps: float = 1e-8) -> Tensor:
    assert confusion_matrix.dim() == 2 and confusion_matrix.size(1) == 3
    tp = confusion_matrix[:, 0].to(torch.float32)
    fn = confusion_matrix[:, 2].to(torch.float32)
    return tp / (tp + fn + eps)


def compute_fi(confusion_matrix: Tensor, i: float = 1.0, eps: float = 1e-8) -> Tensor:
    precision = compute_precision(confusion_matrix, eps=eps)
    recall = compute_recall(confusion_matrix, eps=eps)
    beta2 = i ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)

