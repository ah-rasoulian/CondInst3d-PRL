from __future__ import annotations

from typing import List
import torch.nn as nn
from torch import Tensor

from .abstract import AbstractBackbone


class UNetRBackboneBase(AbstractBackbone):
    """
    Shared logic for UNetR-like backbones:
      - stores config fields (in/out/strides/filters/head dims)
      - builds head mappings
      - forward(): encoder -> decoder -> output + heads
    Subclasses only need to create self.encoder.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        filters,
        kernels,
        strides,
        head_start_index: int,
        heads_dim: int,
        head_end_index: int = -1,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._heads_dim = heads_dim
        self._strides = strides
        self._head_start_index = head_start_index
        self._head_end_index = head_end_index
        self._filters = filters

        self.kernels = kernels  # optional to expose if you want

        # subclasses must set:
        #   self.encoder: nn.Module
        # this base sets:
        #   self.decoder: nn.Module

        self.head_mappings = self._build_head_mappings()

    # -------- interface / properties --------

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def heads_dim(self) -> int:
        return self._heads_dim

    @property
    def mask_stride(self):
        return self._strides[0]

    @property
    def head_start_index(self) -> int:
        return self._head_start_index

    @property
    def head_end_index(self) -> int:
        return self._head_end_index

    @property
    def filters(self):
        return self._filters

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -------- shared implementation --------

    def _build_head_mappings(self) -> nn.ModuleList:
        """
        Builds a mapping aligned with decoder_feats indexing used in forward().
        We create convs for encoder resolution levels >= head_start_index.
        """
        modules: List[nn.Module | None] = []
        for i in range(len(self.filters)):
            if i < self.head_start_index:
                conv = None
            elif self.head_end_index != -1 and i > self.head_end_index:
                conv = None
            else:
                conv = nn.Conv3d(
                    self.filters[i],
                    self.heads_dim,
                    kernel_size=1,
                    stride=1,
                    padding="same",
                )
            modules.append(conv)

        modules.reverse()
        return nn.ModuleList(modules)

    def forward(self, inputs: Tensor):
        encoder_feats = self.encoder(inputs)
        decoder_feats = self.decoder(encoder_feats)

        output = decoder_feats[-1]

        heads = []
        for i in range(len(self.filters)):
            head_map = self.head_mappings[i]
            if head_map is not None:
                heads.append(head_map(decoder_feats[i]))

        heads.reverse()
        return output, heads
