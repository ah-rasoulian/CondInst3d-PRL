from abc import ABC, abstractmethod
import torch.nn as nn
from torch import Tensor


class AbstractBackbone(nn.Module, ABC):
    """
    Interface for UNetR-style backbones that expose:
      - in_channels
      - out_channels
      - head_dim
      - strides
      - forward() -> (output, heads)
    """

    def __init__(self):
        super().__init__()

    # --------- Required properties ---------

    @property
    @abstractmethod
    def in_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def heads_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def mask_stride(self):
        pass

    @property
    @abstractmethod
    def filters(self):
        pass

    # --------- Required API ---------

    @abstractmethod
    def forward(self, x: Tensor):
        """
        Must return:
            output: Tensor
            heads: list[Tensor]
        """
        pass
