from monai.networks.blocks import ResBlock, SEBlock
from condinst3d.arch.layers.conv import conv_with_xavier_uniform
import torch
import torch.nn as nn
from typing import Sequence
from torch import Tensor
import math


class ClassificationHead(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            prior_probability: float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = SEBlock(
            spatial_dims=3,
            in_channels=in_channels,
            n_chns_1=in_channels // 2,
            n_chns_2=in_channels // 2,
            n_chns_3=in_channels,
            r=4,
        )
        self.conv2 = ResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            norm=("INSTANCE", {"affine": True}),
            kernel_size=3,
        )
        self.conv3 = ResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            norm=("INSTANCE", {"affine": True}),
            kernel_size=3,
        )
        bias_value = -math.log((1 - prior_probability) / prior_probability)
        self.cls_logits = conv_with_xavier_uniform(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1, bias_init_value=bias_value,
        )

    def forward(self, features: Sequence[Tensor]):
        all_logits = []
        for feats in features:
            cls_feat = self.conv1(feats)
            cls_feat = self.conv2(cls_feat)
            cls_feat = self.conv3(cls_feat)
            logits = self.cls_logits(cls_feat)

            # Permute classification output from (N, K, W, H, D) to (N, WHD, K).
            N, _, W, H, D = logits.shape
            logits = logits.permute(0, 2, 3, 4, 1)
            logits = logits.reshape(N, -1, self.num_classes)  # Size=(N, WHD, K)

            all_logits.append(logits)

        return torch.cat(all_logits, dim=1)
