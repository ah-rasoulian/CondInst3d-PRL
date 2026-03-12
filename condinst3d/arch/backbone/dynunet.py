from __future__ import annotations

from typing import List, Sequence

import torch.nn as nn
from torch import Tensor
from monai.networks.nets import DynUNet

from .abstract import AbstractBackbone


class DynUNetBackbone(AbstractBackbone):
    """
    DynUNet backbone adapter for CondInst-style models.

    This backbone:
      - runs a MONAI DynUNet semantic model
      - returns all decoder outputs
      - returns the final semantic feature map before semantic logits
      - maps only decoder outputs in [head_start, head_end] to heads_dim via 1x1 convs

    Expected decoder_outputs order:
      deep -> shallow
      decoder_outputs[-1] is the highest-resolution decoder feature map

    semantic_output:
      highest-resolution decoder feature map before DynUNet's final output_block
    """

    def __init__(
        self,
        *,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size,
        strides,
        upsample_kernel_size,
        filters,
        heads_dim: int,
        decoder_strides: Sequence[Sequence[int]],
        head_start: int,
        head_end: int = -1,
        norm_name=("INSTANCE", {"affine": True}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
        trans_bias: bool = False,
        dropout=None,
        enable_heads: bool = True,
    ):
        super().__init__(enable_heads=enable_heads)

        self._in_channels = int(in_channels)
        self._out_channels = int(filters[0])  # semantic feature channels before output_block
        self._heads_dim = int(heads_dim)
        self._decoder_feature_channels = list(filters[:-1])[::-1]  # deep -> shallow
        self._decoder_strides = [tuple(s) for s in decoder_strides]
        self._semantic_stride = tuple(self._decoder_strides[-1])
        self._head_start = int(head_start)
        self._head_end = int(head_end)
        self._filters = list(filters)
        self._spatial_dims = int(spatial_dims)

        if len(self._decoder_feature_channels) != len(self._decoder_strides):
            raise ValueError(
                "decoder_feature_channels and decoder_strides must have the same length. "
                f"Got {len(self._decoder_feature_channels)} and {len(self._decoder_strides)}."
            )

        self.model = DynUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=filters,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            deep_supervision=deep_supervision,
            deep_supr_num=deep_supr_num,
            res_block=res_block,
            trans_bias=trans_bias,
        )

        self._init_head_mappings()

    # ------------------------------------------------------------------
    # Required metadata
    # ------------------------------------------------------------------

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        """
        Channels of semantic_output, not semantic logits.
        """
        return self._out_channels

    @property
    def heads_dim(self) -> int:
        return self._heads_dim

    @property
    def decoder_feature_channels(self) -> Sequence[int]:
        return self._decoder_feature_channels

    @property
    def decoder_strides(self) -> Sequence[Sequence[int]]:
        return self._decoder_strides

    @property
    def semantic_stride(self) -> Sequence[int]:
        return self._semantic_stride

    @property
    def head_start(self) -> int:
        return self._head_start

    @property
    def head_end(self) -> int:
        return self._head_end

    @property
    def filters(self):
        return self._filters

    # ------------------------------------------------------------------
    # Internal forward utilities
    # ------------------------------------------------------------------

    def _forward_encoder(self, x: Tensor) -> tuple[List[Tensor], Tensor]:
        """
        Returns:
            skips: feature maps used as skip connections
            bottleneck: deepest feature map
        """
        x = self.model.input_block(x)
        skips = [x]

        for down in self.model.downsamples:
            x = down(x)
            skips.append(x)

        bottleneck = self.model.bottleneck(x)
        return skips, bottleneck

    def _forward_decoder(self, skips: List[Tensor], bottleneck: Tensor) -> List[Tensor]:
        """
        Returns decoder outputs in deep -> shallow order.
        """
        x = bottleneck
        decoder_outputs: List[Tensor] = []

        # DynUNet upsamples from deepest to shallowest skip
        for up, skip in zip(self.model.upsamples, reversed(skips)):
            x = up(x, skip)
            decoder_outputs.append(x)

        return decoder_outputs

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------

    def forward_features(self, x: Tensor) -> tuple[Tensor, List[Tensor]]:
        skips, bottleneck = self._forward_encoder(x)
        decoder_outputs = self._forward_decoder(skips, bottleneck)

        # final high-res feature map before semantic logits
        semantic_output = decoder_outputs[-1]
        return semantic_output, decoder_outputs

    def forward_logits(self, semantic_output: Tensor) -> Tensor:
        """
        Optional helper if you want semantic logits from the DynUNet head.
        """
        return self.model.output_block(semantic_output)
