from torch import Tensor

def compute_precision(confusion_matrix: Tensor):
    assert confusion_matrix.dim() == 2 and confusion_matrix.size(1) == 3
    return confusion_matrix[:, 0] / (confusion_matrix[:, 0] + confusion_matrix[:, 1])


def compute_recall(confusion_matrix: Tensor):
    assert confusion_matrix.dim() == 2 and confusion_matrix.size(1) == 3
    return confusion_matrix[:, 0] / (confusion_matrix[:, 0] + confusion_matrix[:, 2])


def compute_fi(confusion_matrix: Tensor, i=1):
    precision = compute_precision(confusion_matrix)
    recall = compute_recall(confusion_matrix)
    return (1 + i ** 2) * precision * recall / (i ** 2 * precision + recall)
