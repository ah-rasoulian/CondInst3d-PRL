from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

import torch.nn as nn
from torch import Tensor


@dataclass
class BackboneOutput:
    """
    Standard output of a semantic backbone.

    Attributes
    ----------
    semantic_output:
        Final high-resolution semantic feature map returned by the backbone.

    decoder_outputs:
        All raw decoder feature maps returned by the semantic model, in a fixed
        order defined by the backbone implementation.

    heads:
        Selected decoder outputs after passing through 1x1 conv head mappings,
        all mapped to heads_dim channels.
    """
    semantic_output: Tensor
    decoder_outputs: List[Tensor]
    heads: List[Tensor]


class AbstractBackbone(nn.Module, ABC):
    """
    Generic interface for semantic backbones used inside CondInst-style models.

    Each backbone must expose:
      - in_channels
      - out_channels: channels of final semantic_output
      - heads_dim: common channel dimension after head mapping
      - decoder_feature_channels: channels of all raw decoder outputs
      - decoder_strides: strides of all raw decoder outputs w.r.t. input
      - semantic_stride: stride of semantic_output w.r.t. input
      - head_start: first decoder level used for head mappings
      - head_end: last decoder level used for head mappings, inclusive
                  if -1, use until the last decoder level

    The backbone is responsible for:
      1) producing all raw decoder outputs
      2) producing the final semantic output
      3) projecting only selected decoder outputs to a common heads_dim
    """

    def __init__(self):
        super().__init__()
        self.head_mappings: nn.ModuleList | None = None

    # ------------------------------------------------------------------
    # Required metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def in_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """
        Number of channels of the final semantic_output tensor.
        """
        pass

    @property
    @abstractmethod
    def heads_dim(self) -> int:
        """
        Common channel dimension of mapped head features.
        """
        pass

    @property
    @abstractmethod
    def decoder_feature_channels(self) -> Sequence[int]:
        """
        Channels of all raw decoder outputs, in the same order as returned by
        forward_features().
        """
        pass

    @property
    @abstractmethod
    def decoder_strides(self) -> Sequence[Sequence[int]]:
        """
        Spatial strides of all raw decoder outputs w.r.t. the input image,
        in the same order as decoder_feature_channels / decoder_outputs.
        """
        pass

    @property
    @abstractmethod
    def semantic_stride(self) -> Sequence[int]:
        """
        Spatial stride of semantic_output w.r.t. the input image.
        """
        pass

    @property
    @abstractmethod
    def head_start(self) -> int:
        """
        First decoder output index to use for head mappings.
        """
        pass

    @property
    @abstractmethod
    def head_end(self) -> int:
        """
        Last decoder output index to use for head mappings, inclusive.
        If -1, uses up to the final decoder output.
        """
        pass

    @property
    def mask_stride(self) -> Sequence[int]:
        """
        Alias kept for compatibility with existing CondInst code.
        """
        return self.semantic_stride

    @property
    def num_decoder_levels(self) -> int:
        return len(self.decoder_feature_channels)

    @property
    def resolved_head_end(self) -> int:
        return self.num_decoder_levels - 1 if self.head_end == -1 else self.head_end

    @property
    def head_indices(self) -> List[int]:
        end = self.resolved_head_end
        if not (0 <= self.head_start < self.num_decoder_levels):
            raise ValueError(
                f"head_start={self.head_start} is out of range for "
                f"{self.num_decoder_levels} decoder levels."
            )
        if not (0 <= end < self.num_decoder_levels):
            raise ValueError(
                f"head_end={self.head_end} resolves to {end}, which is out of range "
                f"for {self.num_decoder_levels} decoder levels."
            )
        if self.head_start > end:
            raise ValueError(
                f"head_start={self.head_start} must be <= resolved head_end={end}."
            )
        return list(range(self.head_start, end + 1))

    @property
    def head_feature_channels(self) -> List[int]:
        return [self.heads_dim] * len(self.head_indices)

    @property
    def head_strides(self) -> List[Sequence[int]]:
        return [self.decoder_strides[i] for i in self.head_indices]

    @property
    def selected_decoder_feature_channels(self) -> List[int]:
        return [self.decoder_feature_channels[i] for i in self.head_indices]

    # ------------------------------------------------------------------
    # Shared implementation
    # ------------------------------------------------------------------

    def _init_head_mappings(self) -> None:
        """
        Builds one 1x1 conv per selected decoder output so every selected
        decoder feature can be mapped to a common heads_dim.
        """
        self.head_mappings = nn.ModuleList([
            nn.Conv3d(
                in_channels=in_ch,
                out_channels=self.heads_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for in_ch in self.selected_decoder_feature_channels
        ])

    def map_decoder_outputs_to_heads(self, decoder_outputs: List[Tensor]) -> List[Tensor]:
        if self.head_mappings is None:
            raise RuntimeError(
                "head_mappings is not initialized. Call self._init_head_mappings() "
                "in the concrete backbone __init__ after metadata is available."
            )

        if len(decoder_outputs) != self.num_decoder_levels:
            raise ValueError(
                f"Expected {self.num_decoder_levels} decoder outputs, "
                f"got {len(decoder_outputs)}."
            )

        selected_decoder_outputs = [decoder_outputs[i] for i in self.head_indices]

        if len(selected_decoder_outputs) != len(self.head_mappings):
            raise ValueError(
                f"Expected {len(self.head_mappings)} selected decoder outputs, "
                f"got {len(selected_decoder_outputs)}."
            )

        return [
            head_map(decoder_out)
            for head_map, decoder_out in zip(self.head_mappings, selected_decoder_outputs)
        ]

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------

    @abstractmethod
    def forward_features(self, x: Tensor) -> tuple[Tensor, List[Tensor]]:
        """
        Returns
        -------
        semantic_output:
            Final semantic feature map of shape [B, out_channels, ...].

        decoder_outputs:
            List of all raw decoder outputs in a fixed order defined by the
            concrete backbone.
        """
        pass

    def forward(self, x: Tensor) -> BackboneOutput:
        semantic_output, decoder_outputs = self.forward_features(x)
        heads = self.map_decoder_outputs_to_heads(decoder_outputs)

        return BackboneOutput(
            semantic_output=semantic_output,
            decoder_outputs=decoder_outputs,
            heads=heads,
        )
