import torch.nn as nn
from functools import partial


def get_norm_layer(name):
    if name == "GN":
        norm = partial(nn.GroupNorm, 32)
    elif name == "BN":
        norm = nn.BatchNorm3d
    elif name == "IN":
        norm = nn.InstanceNorm3d
    else:
        raise NotImplementedError(f"{name} norm layer is not implemented!")
    return norm
