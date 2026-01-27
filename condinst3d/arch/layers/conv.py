from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from .norm import get_norm_layer


def conv_with_xavier_uniform(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        gain: float = 1.,
        bias_init_value=0.,
        custom_bias: Optional[Tensor] = None,
        norm: Optional[str] = None,
        activation: bool = False,
):
    layers = []
    conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    nn.init.xavier_uniform_(conv.weight, gain)
    if bias:
        if custom_bias is not None:
            assert custom_bias.shape == conv.bias.shape
            with torch.no_grad():
                conv.bias.copy_(custom_bias)
        else:
            nn.init.constant_(conv.bias, val=bias_init_value)
    layers.append(conv)
    if norm is not None:
        norm_layer = get_norm_layer(norm)
        layers.append(norm_layer(out_channels))
    if activation:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)
