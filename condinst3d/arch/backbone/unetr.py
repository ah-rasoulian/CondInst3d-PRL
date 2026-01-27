from __future__ import annotations

from typing import List
import torch.nn as nn
from torch import Tensor
from monai.networks.blocks import UnetResBlock, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

from .unetr_base import UNetRBackboneBase


class UNetR(UNetRBackboneBase):
    def __init__(
        self,
        in_channels,
        filters,
        kernels,
        strides,
        head_start_index,
        heads_dim,
        out_channels,
    ):
        # build encoder/decoder config + heads in base
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters,
            kernels=kernels,
            strides=strides,
            head_start_index=head_start_index,
            heads_dim=heads_dim,
        )

        # modules
        self.encoder = ResidualUNetEncoder(in_channels, filters, kernels, strides)
        self.decoder = ResidualUnetRDecoder(out_channels, filters, kernels, strides)


class ResidualUNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        kernels,
        strides,
    ):
        super().__init__()
        self.filters = filters
        self.kernels = kernels
        self.strides = strides

        self.input_block = self._get_input_block(in_channels)
        self.down_samples = self._get_downsample_blocks()
        self.bottleneck = self._get_bottleneck_block()

    def _get_input_block(self, in_channels: int) -> nn.Module:
        return UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=self.filters[0],
            kernel_size=self.kernels[0],
            stride=self.strides[0],
            norm_name=("INSTANCE", {"affine": True}),
        )

    def _get_downsample_blocks(self) -> nn.ModuleList:
        modules = []
        for i in range(len(self.filters[:-2])):
            in_ch = self.filters[i]
            out_ch = self.filters[i + 1]
            ksize = self.kernels[i + 1]
            stride = self.strides[i + 1]
            modules.append(
                UnetResBlock(
                    spatial_dims=3,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=ksize,
                    stride=stride,
                    norm_name=("INSTANCE", {"affine": True}),
                )
            )
        return nn.ModuleList(modules)

    def _get_bottleneck_block(self) -> nn.Module:
        return UnetResBlock(
            spatial_dims=3,
            in_channels=self.filters[-2],
            out_channels=self.filters[-1],
            kernel_size=self.kernels[-1],
            stride=self.strides[-1],
            norm_name=("INSTANCE", {"affine": True}),
        )

    def forward(self, inputs: Tensor):
        inp = self.input_block(inputs)

        downs = []
        down_i = inp
        for blk in self.down_samples:
            down_i = blk(down_i)
            downs.append(down_i)

        bottleneck = self.bottleneck(downs[-1])
        return [inp, *downs, bottleneck]


class ResidualUnetRDecoder(nn.Module):
    def __init__(
        self,
        out_channels,
        filters,
        kernels,
        strides,
    ):
        super().__init__()
        self.filters = filters
        self.kernels = kernels
        self.strides = strides

        self.skips = self._get_skip_blocks()
        self.up_samples = self._get_upsample_blocks()
        self.output_block = self._get_output_block(out_channels)

    def _get_skip_blocks(self) -> nn.ModuleList:
        modules = []
        for ch in self.filters[:-1]:
            modules.append(
                UnetrBasicBlock(
                    spatial_dims=3,
                    in_channels=ch,
                    out_channels=ch,
                    kernel_size=3,
                    stride=1,
                    norm_name=("INSTANCE", {"affine": True}),
                    res_block=True,
                )
            )
        return nn.ModuleList(modules)

    def _get_upsample_blocks(self) -> nn.ModuleList:
        modules = []
        for i in range(1, len(self.filters)):
            modules.append(
                UnetrUpBlock(
                    spatial_dims=3,
                    in_channels=self.filters[i],
                    out_channels=self.filters[i - 1],
                    kernel_size=self.kernels[i],
                    upsample_kernel_size=self.strides[i],
                    norm_name=("INSTANCE", {"affine": True}),
                    res_block=True,
                )
            )
        return nn.ModuleList(modules)

    def _get_output_block(self, out_channels) -> nn.Module:
        return UnetOutBlock(
            spatial_dims=3,
            in_channels=self.filters[0],
            out_channels=out_channels,
        )

    def forward(self, encoder_feats: List[Tensor]):
        skips = [blk(feat) for blk, feat in zip(self.skips, encoder_feats[:-1])]

        ups = []
        up_j = encoder_feats[-1]
        for j in range(len(self.up_samples) - 1, -1, -1):
            up_j = self.up_samples[j](up_j, skips[j])
            ups.append(up_j)

        output = self.output_block(ups[-1])
        return [*ups, output]
