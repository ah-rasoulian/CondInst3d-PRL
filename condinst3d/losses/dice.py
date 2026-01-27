import torch


def dice_coefficient(x, target, fg_weight=1.0):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)

    weight_mask = torch.ones_like(target)
    weight_mask[target == 1] = fg_weight

    intersection = (weight_mask * x * target).sum(dim=1)
    union = (weight_mask * (x ** 2.0)).sum(dim=1) + (weight_mask * (target ** 2.0)).sum(dim=1)
    loss = 1. - (2 * intersection + eps) / (union + eps)
    return loss
