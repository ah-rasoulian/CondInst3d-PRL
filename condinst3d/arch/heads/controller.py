from monai.networks.blocks import ResBlock, SEBlock
from condinst3d.arch.layers.conv import conv_with_xavier_uniform
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from torch import Tensor


class ControllerHead(nn.Module):
    def __init__(
            self,
            in_channels,
            num_params,
    ):
        super().__init__()
        self.num_params = num_params
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
        self.param_pred = conv_with_xavier_uniform(
            in_channels, num_params, kernel_size=3, stride=1, padding=1,
        )

    def forward(self, features: Sequence[Tensor]):
        all_params = []
        for feats in features:
            param_feat = self.conv1(feats)
            param_feat = self.conv2(param_feat)
            param_feat = self.conv3(param_feat)
            params_logits = self.param_pred(param_feat)

            # Permute classification output from (N, K, W, H, D) to (N, WHD, K).
            N, _, W, H, D = params_logits.shape
            params_logits = params_logits.permute(0, 2, 3, 4, 1)
            params_logits = params_logits.reshape(N, -1, self.num_params)  # Size=(N, WHD, K)

            all_params.append(params_logits)

        return torch.cat(all_params, dim=1)
