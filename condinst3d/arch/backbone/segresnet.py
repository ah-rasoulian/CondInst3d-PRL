from __future__ import annotations

from typing import List, Sequence

from torch import Tensor
from monai.networks.nets import SegResNet

from .abstract import AbstractBackbone


class SegResNetBackbone(AbstractBackbone):
    """
    SegResNet backbone adapter for CondInst-style models.

    Public convention:
      - decoder_outputs are ordered shallow -> deep
      - decoder_outputs[0] is the highest-resolution decoder feature map
      - decoder_outputs[-1] is the lowest-resolution decoder feature map

    Notes
    -----
    - semantic_output is the final shallow decoder feature BEFORE semantic logits.
    - head_indices are interpreted in shallow -> deep indexing space.
    - semantic logits are produced by MONAI's conv_final when use_conv_final=True.
    """

    def __init__(
        self,
        *,
        spatial_dims: int,
        in_channels: int,
        init_filters: int,
        heads_dim: int,
        head_indices: Sequence[int],
        semantic_out_channels: int = 1,
        blocks_down: Sequence[int] = (1, 2, 2, 4),
        blocks_up: Sequence[int] = (1, 1, 1),
        dropout_prob: float | None = None,
        act=("RELU", {"inplace": True}),
        norm=("GROUP", {"num_groups": 8}),
        upsample_mode: str = "deconv",
        use_conv_final: bool = True,
        enable_heads: bool = True,
    ):
        super().__init__(enable_heads=enable_heads)

        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")

        self._spatial_dims = int(spatial_dims)
        self._in_channels = int(in_channels)
        self._init_filters = int(init_filters)
        self._heads_dim = int(heads_dim)
        self._head_indices = list(head_indices)
        self._semantic_out_channels = int(semantic_out_channels)
        self._use_conv_final = bool(use_conv_final)

        self._blocks_down = tuple(blocks_down)
        self._blocks_up = tuple(blocks_up)

        n_up = len(self._blocks_up)

        # Public convention is shallow -> deep.
        self._decoder_feature_channels = [
            self._init_filters * (2 ** k) for k in range(0, n_up)
        ]

        self._decoder_strides = [
            tuple([2 ** k] * self._spatial_dims) for k in range(0, n_up)
        ]

        # semantic_output is the highest-resolution decoder feature
        # BEFORE semantic logits, so its channel count is init_filters.
        self._out_channels = self._init_filters
        self._semantic_stride = tuple(self._decoder_strides[0])

        self.model = SegResNet(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=self._semantic_out_channels,
            dropout_prob=dropout_prob,
            act=act,
            norm=norm,
            use_conv_final=use_conv_final,
            blocks_down=tuple(blocks_down),
            blocks_up=tuple(blocks_up),
            upsample_mode=upsample_mode,
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
        Channels of semantic_output (pre-logits feature map).
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
    def head_indices(self) -> Sequence[int]:
        return self._head_indices

    @property
    def filters(self):
        return [self._init_filters * (2 ** i) for i in range(len(self._blocks_down))]

    # ------------------------------------------------------------------
    # Internal forward helpers
    # ------------------------------------------------------------------

    def _forward_encoder(self, x: Tensor) -> tuple[Tensor, List[Tensor]]:
        """
        Mirrors MONAI SegResNet.encode(), but keeps it explicit here.

        Returns
        -------
        x:
            Deepest encoder feature.
        down_x:
            Encoder features in shallow -> deep order.
        """
        x = self.model.convInit(x)

        if getattr(self.model, "dropout_prob", None) is not None:
            x = self.model.dropout(x)

        down_x: List[Tensor] = []
        for down in self.model.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def _forward_decoder(self, x: Tensor, down_x: List[Tensor]) -> List[Tensor]:
        """
        Returns decoder outputs in shallow -> deep order.
        """
        down_x = list(down_x)
        down_x.reverse()

        decoder_outputs_deep_to_shallow: List[Tensor] = []
        for i, (up, upl) in enumerate(zip(self.model.up_samples, self.model.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
            decoder_outputs_deep_to_shallow.append(x)

        return decoder_outputs_deep_to_shallow[::-1]

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------

    def forward_features(self, x: Tensor) -> tuple[Tensor, List[Tensor]]:
        x, down_x = self._forward_encoder(x)
        decoder_outputs = self._forward_decoder(x, down_x)

        if len(decoder_outputs) != len(self.decoder_feature_channels):
            raise RuntimeError(
                f"Expected {len(self.decoder_feature_channels)} decoder outputs, "
                f"got {len(decoder_outputs)}."
            )

        semantic_output = decoder_outputs[0]
        return semantic_output, decoder_outputs

    def forward_logits(self, semantic_output: Tensor) -> Tensor:
        if not self._use_conv_final:
            raise RuntimeError(
                "forward_logits() was called but SegResNet was created with "
                "use_conv_final=False."
            )
        return self.model.conv_final(semantic_output)