from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class CandidateInteraction(nn.Module, ABC):
    @abstractmethod
    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityInteraction(CandidateInteraction):
    """Extension point for set transformers, cross attention, or candidate GNNs."""

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        return representations
