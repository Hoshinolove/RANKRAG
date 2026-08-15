from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class RankerModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scores and intermediate candidate representations."""
        raise NotImplementedError
