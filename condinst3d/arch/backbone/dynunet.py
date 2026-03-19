from __future__ import annotations

from typing import List, Sequence

from torch import Tensor
from monai.networks.nets import DynUNet

from .abstract import AbstractBackbone


class DynUNetBackbone(AbstractBackbone):
    """
    DynUNet backbone adapter for CondInst-style models.

    Public convention:
      - decoder_outputs are ordered shallow -> deep
      - decoder_outputs[0] is the highest-resolution decoder feature map
      - decoder_outputs[-1] is the lowest-resolution decoder feature map

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
        head_indices: Sequence[int],
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
        self._out_channels = int(filters[0])
        self._heads_dim = int(heads_dim)

        # Public convention: shallow -> deep
        self._decoder_feature_channels = list(filters[:-1])
        self._decoder_strides = [tuple(s) for s in decoder_strides]
        self._semantic_stride = tuple(self._decoder_strides[0])

        self._head_indices = list(head_indices)

        self._filters = list(filters)
        self._spatial_dims = int(spatial_dims)

        if len(self._decoder_feature_channels) != len(self._decoder_strides):
            raise ValueError(
                "decoder_feature_channels and decoder_strides must have the same length. "
                f"Got {len(self._decoder_feature_channels)} and {len(self._decoder_strides)}."
            )

        if len(self._decoder_strides) > 1:
            stride_products = []
            for s in self._decoder_strides:
                prod = 1
                for v in s:
                    prod *= int(v)
                stride_products.append(prod)

            if stride_products != sorted(stride_products):
                raise ValueError(
                    "decoder_strides must be provided in shallow -> deep order "
                    "(smallest stride to largest stride). "
                    f"Got: {self._decoder_strides}"
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
        """
        Channels of raw decoder outputs in shallow -> deep order.
        """
        return self._decoder_feature_channels

    @property
    def decoder_strides(self) -> Sequence[Sequence[int]]:
        """
        Strides of raw decoder outputs in shallow -> deep order.
        """
        return self._decoder_strides

    @property
    def semantic_stride(self) -> Sequence[int]:
        return self._semantic_stride

    @property
    def head_indices(self) -> Sequence[int]:
        return self._head_indices

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
        Returns decoder outputs in shallow -> deep order.
        """
        x = bottleneck
        decoder_outputs_deep_to_shallow: List[Tensor] = []

        for up, skip in zip(self.model.upsamples, reversed(skips)):
            x = up(x, skip)
            decoder_outputs_deep_to_shallow.append(x)

        return decoder_outputs_deep_to_shallow[::-1]

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------

    def forward_features(self, x: Tensor) -> tuple[Tensor, List[Tensor]]:
        skips, bottleneck = self._forward_encoder(x)
        decoder_outputs = self._forward_decoder(skips, bottleneck)

        if len(decoder_outputs) != len(self._decoder_strides):
            raise RuntimeError(
                f"Number of decoder outputs ({len(decoder_outputs)}) does not match "
                f"number of decoder strides ({len(self._decoder_strides)})."
            )

        semantic_output = decoder_outputs[0]
        return semantic_output, decoder_outputs

    def forward_logits(self, semantic_output: Tensor) -> Tensor:
        """
        Optional helper if you want semantic logits from the DynUNet head.
        """
        return self.model.output_block(semantic_output)
