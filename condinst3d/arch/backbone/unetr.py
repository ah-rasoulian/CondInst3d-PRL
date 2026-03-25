from __future__ import annotations

from typing import Sequence

from torch import Tensor
from monai.networks.nets import UNETR

from .abstract import AbstractBackbone


class UNETRBackbone(AbstractBackbone):
    """
    UNETR backbone adapter for CondInst-style models.

    Public convention:
      - decoder_outputs are ordered shallow -> deep
      - decoder_outputs[0] is the highest-resolution decoder feature map
      - decoder_outputs[-1] is the lowest-resolution decoder feature map

    Notes
    -----
    - semantic_output is the final feature map BEFORE semantic logits.
    - head_indices are interpreted in shallow -> deep indexing space.
    - the built-in UNETR segmentation head is kept as self.model.out, but your
      CondInst model can ignore it and use its own semantic head.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        img_size,
        feature_size: int,
        heads_dim: int,
        head_indices: Sequence[int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        norm_name="instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
        enable_heads: bool = True,
    ):
        super().__init__(enable_heads=enable_heads)

        if spatial_dims != 3:
            raise ValueError(f"This wrapper currently expects spatial_dims=3, got {spatial_dims}.")

        self._in_channels = int(in_channels)
        self._out_channels = int(feature_size)  # channels of semantic_output before logits
        self._heads_dim = int(heads_dim)
        self._head_indices = list(head_indices)
        self._feature_size = int(feature_size)
        self._hidden_size = int(hidden_size)
        self._spatial_dims = int(spatial_dims)

        # Native UNETR decoder traversal is deep -> shallow:
        #   dec3 = feature_size * 8
        #   dec2 = feature_size * 4
        #   dec1 = feature_size * 2
        #   out  = feature_size
        #
        # Public contract is shallow -> deep, so reverse it:
        self._decoder_feature_channels = [
            feature_size,
            feature_size * 2,
            feature_size * 4,
            feature_size * 8,
        ]

        # UNETR uses patch size 16, and decoder upsamples by 2 each stage.
        # Native decoder strides are deep -> shallow: 8, 4, 2, 1
        # Public contract is shallow -> deep: 1, 2, 4, 8
        self._decoder_strides = [
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
        ]
        self._semantic_stride = self._decoder_strides[0]

        self.model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed=pos_embed,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
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
        """
        Kept for compatibility with older code patterns.
        """
        return [
            self._feature_size,
            self._feature_size * 2,
            self._feature_size * 4,
            self._feature_size * 8,
            self._hidden_size,
        ]

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------

    def forward_features(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        # MONAI UNETR forward pattern:
        # x_vit, hidden_states_out = vit(x)
        # enc1 from input
        # enc2, enc3, enc4 from hidden_states_out[3], [6], [9]
        # dec4 = proj_feat(x_vit)
        # dec3 = decoder5(dec4, enc4)
        # dec2 = decoder4(dec3, enc3)
        # dec1 = decoder3(dec2, enc2)
        # out  = decoder2(dec1, enc1)
        x_vit, hidden_states_out = self.model.vit(x)

        enc1 = self.model.encoder1(x)

        x2 = hidden_states_out[3]
        enc2 = self.model.encoder2(self.model.proj_feat(x2))

        x3 = hidden_states_out[6]
        enc3 = self.model.encoder3(self.model.proj_feat(x3))

        x4 = hidden_states_out[9]
        enc4 = self.model.encoder4(self.model.proj_feat(x4))

        dec4 = self.model.proj_feat(x_vit)
        dec3 = self.model.decoder5(dec4, enc4)
        dec2 = self.model.decoder4(dec3, enc3)
        dec1 = self.model.decoder3(dec2, enc2)
        out = self.model.decoder2(dec1, enc1)

        # Public contract: shallow -> deep
        decoder_outputs = [out, dec1, dec2, dec3]

        if len(decoder_outputs) != len(self.decoder_feature_channels):
            raise RuntimeError(
                f"Expected {len(self.decoder_feature_channels)} decoder outputs, "
                f"got {len(decoder_outputs)}."
            )

        semantic_output = decoder_outputs[0]
        return semantic_output, decoder_outputs

    def forward_logits(self, semantic_output: Tensor) -> Tensor:
        """
        Optional helper:
        apply MONAI's built-in final conv on top of semantic_output.
        """
        return self.model.out(semantic_output)
