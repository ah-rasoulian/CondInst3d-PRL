from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .unetr import ResidualUNetEncoder, ResidualUnetRDecoder
from .unetr_base import UNetRBackboneBase


class ResidualUNetEncoderSpecificMod(ResidualUNetEncoder):
    def __init__(
        self,
        in_channels,
        filters,
        kernels,
        strides,
        specific_modality_index,
    ):
        super().__init__(in_channels, filters, kernels, strides)
        self.specific_modality_index = specific_modality_index

        self.specific_modality_inp = super()._get_input_block(1)
        self.inp_skip_fusion = nn.Sequential(
            nn.Conv3d(2 * filters[0], filters[0], kernel_size=3, padding="same"),
            nn.InstanceNorm3d(filters[0]),
            nn.LeakyReLU(),
        )

    def forward(self, inputs: Tensor):
        # standard path
        inp = self.input_block(inputs)

        downs = []
        down_i = inp
        for blk in self.down_samples:
            down_i = blk(down_i)
            downs.append(down_i)

        bottleneck = self.bottleneck(downs[-1])

        # specific modality skip fusion at input resolution
        specific = self.specific_modality_inp(
            inputs[:, self.specific_modality_index, ...].unsqueeze(1)
        )
        fused_inp = self.inp_skip_fusion(torch.cat([inp, specific], dim=1))

        return [fused_inp, *downs, bottleneck]


class UNetRSpecificMod(UNetRBackboneBase):
    def __init__(
        self,
        specific_modality_index,
        in_channels,
        filters,
        kernels,
        strides,
        head_start_index,
        heads_dim,
        out_channels,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters,
            kernels=kernels,
            strides=strides,
            head_start_index=head_start_index,
            heads_dim=heads_dim,
        )

        self.encoder = ResidualUNetEncoderSpecificMod(
            in_channels, filters, kernels, strides, specific_modality_index
        )
        self.decoder = ResidualUnetRDecoder(out_channels, filters, kernels, strides)
