from __future__ import annotations

from typing import Sequence

from torch import Tensor
from monai.networks.nets import SwinUNETR

from .abstract import AbstractBackbone


class SwinUNETRBackbone(AbstractBackbone):
    """
    SwinUNETR backbone adapter for CondInst-style models.

    Returned decoder_outputs are ordered:
        deep -> shallow

    and include the final semantic feature map as the last element:
        semantic_output == decoder_outputs[-1]

    Notes
    -----
    - semantic_output is the final feature map BEFORE semantic logits.
    - heads are created only from decoder outputs in [head_start, head_end].
    - the built-in SwinUNETR segmentation head is kept as self.model.out, but
      your CondInst model can ignore it and use its own semantic head.
    """

    def __init__(
        self,
        *,
        img_size,
        in_channels: int,
        out_channels: int,
        feature_size: int,
        heads_dim: int,
        head_start: int,
        head_end: int = -1,
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
    ):
        super().__init__()

        if spatial_dims != 3:
            raise ValueError(f"This wrapper currently expects spatial_dims=3, got {spatial_dims}.")

        self._in_channels = int(in_channels)
        self._out_channels = int(feature_size)  # semantic feature channels before logits
        self._heads_dim = int(heads_dim)
        self._head_start = int(head_start)
        self._head_end = int(head_end)
        self._feature_size = int(feature_size)
        self._spatial_dims = int(spatial_dims)

        # SwinUNETR decoder channels from deep -> shallow, following MONAI:
        # dec3 = 8 * feature_size
        # dec2 = 4 * feature_size
        # dec1 = 2 * feature_size
        # dec0 = 1 * feature_size
        # out  = 1 * feature_size
        self._decoder_feature_channels = [
            feature_size * 8,
            feature_size * 4,
            feature_size * 2,
            feature_size,
            feature_size,
        ]

        # SwinUNETR uses patch_size=2 and 4 hierarchical downsamplings in the transformer,
        # so the decoder feature strides are naturally:
        # 16, 8, 4, 2, 1 (deep -> shallow).
        self._decoder_strides = [
            (16, 16, 16),
            (8, 8, 8),
            (4, 4, 4),
            (2, 2, 2),
            (1, 1, 1),
        ]
        self._semantic_stride = (1, 1, 1)

        self.model = SwinUNETR(
            img_size=img_size,
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

        decoder_outputs = [dec3, dec2, dec1, dec0, out]
        semantic_output = out
        return semantic_output, decoder_outputs

    def forward_logits(self, semantic_output: Tensor) -> Tensor:
        """
        Optional helper:
        apply MONAI's built-in final conv on top of semantic_output.
        """
        return self.model.out(semantic_output)
