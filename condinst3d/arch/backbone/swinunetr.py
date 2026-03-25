from __future__ import annotations

from typing import Sequence

from torch import Tensor
from monai.networks.nets import SwinUNETR

from .abstract import AbstractBackbone


class SwinUNETRBackbone(AbstractBackbone):
    """
    SwinUNETR backbone adapter for CondInst-style models.

    Public convention:
      - decoder_outputs are ordered shallow -> deep
      - decoder_outputs[0] is the highest-resolution decoder feature map
      - decoder_outputs[-1] is the lowest-resolution decoder feature map

    Notes
    -----
    - semantic_output is the final feature map BEFORE semantic logits.
    - head_indices are interpreted in shallow -> deep indexing space.
    - the built-in SwinUNETR segmentation head is kept as self.model.out, but
      your CondInst model can ignore it and use its own semantic head.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        feature_size: int,
        heads_dim: int,
        head_indices: Sequence[int],
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        norm_name="instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2: bool = False,
        enable_heads: bool = True,
    ):
        super().__init__(enable_heads=enable_heads)

        if spatial_dims != 3:
            raise ValueError(
                f"This wrapper currently expects spatial_dims=3, got {spatial_dims}."
            )

        self._in_channels = int(in_channels)
        self._out_channels = int(feature_size)  # channels of semantic_output before logits
        self._heads_dim = int(heads_dim)
        self._head_indices = list(head_indices)
        self._feature_size = int(feature_size)
        self._spatial_dims = int(spatial_dims)

        # Public convention: shallow -> deep
        #
        # Native decoder traversal is:
        #   dec3, dec2, dec1, dec0, out
        # with strides:
        #   16,   8,   4,   2,   1
        # and channels:
        #  8f,   4f,  2f,   f,   f
        #
        # For the public API we reverse that ordering:
        #   out, dec0, dec1, dec2, dec3
        self._decoder_feature_channels = [
            feature_size,
            feature_size,
            feature_size * 2,
            feature_size * 4,
            feature_size * 8,
        ]

        self._decoder_strides = [
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
        ]
        self._semantic_stride = self._decoder_strides[0]

        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            depths=depths,
            num_heads=num_heads,
            feature_size=feature_size,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=normalize,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=downsample,
            use_v2=use_v2,
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
            self._feature_size,
            self._feature_size * 2,
            self._feature_size * 4,
            self._feature_size * 8,
            self._feature_size * 16,
        ]

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------

    def forward_features(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        # MONAI SwinUNETR forward pattern:
        # hidden_states_out = swinViT(x, normalize)
        # enc0 = encoder1(x)
        # enc1 = encoder2(hidden_states_out[0])
        # enc2 = encoder3(hidden_states_out[1])
        # enc3 = encoder4(hidden_states_out[2])
        # dec4 = encoder10(hidden_states_out[4])
        # dec3 = decoder5(dec4, hidden_states_out[3])
        # dec2 = decoder4(dec3, enc3)
        # dec1 = decoder3(dec2, enc2)
        # dec0 = decoder2(dec1, enc1)
        # out  = decoder1(dec0, enc0)

        hidden_states_out = self.model.swinViT(x, self.model.normalize)

        enc0 = self.model.encoder1(x)
        enc1 = self.model.encoder2(hidden_states_out[0])
        enc2 = self.model.encoder3(hidden_states_out[1])
        enc3 = self.model.encoder4(hidden_states_out[2])

        dec4 = self.model.encoder10(hidden_states_out[4])
        dec3 = self.model.decoder5(dec4, hidden_states_out[3])
        dec2 = self.model.decoder4(dec3, enc3)
        dec1 = self.model.decoder3(dec2, enc2)
        dec0 = self.model.decoder2(dec1, enc1)
        out = self.model.decoder1(dec0, enc0)

        # Public contract: shallow -> deep
        decoder_outputs = [out, dec0, dec1, dec2, dec3]
        semantic_output = decoder_outputs[0]

        if len(decoder_outputs) != len(self.decoder_feature_channels):
            raise RuntimeError(
                f"Expected {len(self.decoder_feature_channels)} decoder outputs, "
                f"got {len(decoder_outputs)}."
            )

        return semantic_output, decoder_outputs

    def forward_logits(self, semantic_output: Tensor) -> Tensor:
        """
        Optional helper:
        apply MONAI's built-in final conv on top of semantic_output.
        """
        return self.model.out(semantic_output)
