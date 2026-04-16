from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock

from .dynunet import DynUNetBackbone


class DynUNetBackboneSpecificMod(DynUNetBackbone):
    """
    DynUNet backbone with shallow fusion of a specific modality.

    - All modalities are processed in the main encoder path.
    - One selected modality is processed separately.
    - The two streams are fused at the first encoder stage.

    Public decoder convention is inherited from DynUNetBackbone:
      - decoder_outputs are ordered shallow -> deep
      - decoder_outputs[0] is the highest-resolution decoder feature map
      - decoder_outputs[-1] is the lowest-resolution decoder feature map
    """

    def __init__(
        self,
        *,
        specific_modality_index: int,
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
        if spatial_dims != 3:
            raise ValueError("DynUNetBackboneSpecificMod supports only 3D inputs.")

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=filters,
            heads_dim=heads_dim,
            decoder_strides=decoder_strides,
            head_indices=head_indices,
            norm_name=norm_name,
            act_name=act_name,
            deep_supervision=deep_supervision,
            deep_supr_num=deep_supr_num,
            res_block=res_block,
            trans_bias=trans_bias,
            dropout=dropout,
            enable_heads=enable_heads,
        )

        self.specific_modality_index = int(specific_modality_index)

        if not (0 <= self.specific_modality_index < in_channels):
            raise ValueError(
                f"specific_modality_index={self.specific_modality_index} "
                f"is out of range for in_channels={in_channels}"
            )

        self.specific_input_block = self._build_input_block(
            in_channels=1,
            out_channels=filters[0],
            kernel_size=kernel_size[0],
            stride=strides[0],
            norm_name=norm_name,
            act_name=act_name,
            dropout=dropout,
            res_block=res_block,
            trans_bias=trans_bias,
        )

        self.input_fusion = nn.Sequential(
            nn.Conv3d(
                in_channels=2 * filters[0],
                out_channels=filters[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(filters[0]),
            nn.LeakyReLU(inplace=True),
        )

    @staticmethod
    def _build_input_block(
        *,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        norm_name,
        act_name,
        dropout,
        res_block: bool,
        trans_bias: bool,
    ) -> nn.Module:
        """
        Builds a DynUNet-compatible input block for the specific modality path.
        """

        block_cls = UnetResBlock if res_block else UnetBasicBlock

        return block_cls(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_name=norm_name,
            act_name=act_name,
            dropout=dropout,
        )

    def _forward_encoder(self, x: Tensor) -> tuple[List[Tensor], Tensor]:
        """
        Encoder forward with specific-modality shallow fusion.

        Returns
        -------
        skips:
            Encoder skip features in shallow -> deep traversal order of the encoder.
        bottleneck:
            Deepest encoder feature map.
        """
        main = self.model.input_block(x)

        x_spec = x[:, self.specific_modality_index:self.specific_modality_index + 1]
        spec = self.specific_input_block(x_spec)

        x = self.input_fusion(torch.cat([main, spec], dim=1))

        skips = [x]

        for down in self.model.downsamples:
            x = down(x)
            skips.append(x)

        bottleneck = self.model.bottleneck(x)
        return skips, bottleneck
