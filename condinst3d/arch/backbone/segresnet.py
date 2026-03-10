from __future__ import annotations

from typing import List, Sequence

import torch.nn as nn
from torch import Tensor
from monai.networks.nets import SegResNet

from .abstract import AbstractBackbone


class SegResNetBackbone(AbstractBackbone):
    """
    SegResNet backbone adapter for CondInst-style models.

    Returned decoder_outputs are ordered:
        deep -> shallow
    and include the final semantic feature map as the last element:
        semantic_output == decoder_outputs[-1]

    Notes
    -----
    - semantic_output is the final feature map BEFORE semantic logits.
    - heads are created only from decoder outputs in [head_start, head_end].
    - This wrapper keeps the semantic head outside the backbone, matching your setup.
    """

    def __init__(
        self,
        *,
        spatial_dims: int,
        in_channels: int,
        init_filters: int,
        heads_dim: int,
        head_start: int,
        head_end: int = -1,
        blocks_down: Sequence[int] = (1, 2, 2, 4),
        blocks_up: Sequence[int] = (1, 1, 1),
        dropout_prob: float | None = None,
        act=("RELU", {"inplace": True}),
        norm=("GROUP", {"num_groups": 8}),
        upsample_mode: str = "deconv",
        use_conv_final: bool = False,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")

        self._spatial_dims = int(spatial_dims)
        self._in_channels = int(in_channels)
        self._init_filters = int(init_filters)
        self._heads_dim = int(heads_dim)
        self._head_start = int(head_start)
        self._head_end = int(head_end)

        self._blocks_down = tuple(blocks_down)
        self._blocks_up = tuple(blocks_up)

        # Decoder outputs in SegResNet have channels:
        # init_filters * 2^(n_up-1), ..., init_filters
        n_up = len(self._blocks_up)
        self._decoder_feature_channels = [
            self._init_filters * (2 ** k) for k in range(n_up - 1, -1, -1)
        ]

        # SegResNet downsamples by factor 2 per encoder stage after the first one,
        # and decoder outputs reverse that. So decoder strides are:
        # 2^(n_up-1), ..., 1   (same value on each spatial axis)
        self._decoder_strides = [
            tuple([2 ** k] * self._spatial_dims) for k in range(n_up - 1, -1, -1)
        ]

        # semantic_output is the final shallow decoder feature
        self._out_channels = self._init_filters
        self._semantic_stride = tuple([1] * self._spatial_dims)

        self.model = SegResNet(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=1,  # unused here because use_conv_final=False
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
        Channels of semantic_output.
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
        Returns decoder outputs in deep -> shallow order.

        MONAI reverses down_x and then uses down_x[i + 1] as skip connections
        during decoding. We do the same here, but also collect each decoder stage.
        """
        down_x = list(down_x)
        down_x.reverse()

        decoder_outputs: List[Tensor] = []
        for i, (up, upl) in enumerate(zip(self.model.up_samples, self.model.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
            decoder_outputs.append(x)

        return decoder_outputs

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

        semantic_output = decoder_outputs[-1]
        return semantic_output, decoder_outputs

    def forward_logits(self, semantic_output: Tensor) -> Tensor:
        """
        Optional helper:
        apply MONAI's built-in final conv on top of semantic_output.
        """
        return self.model.conv_final(semantic_output)
