from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from rankrag.models import RecommendationInstance


class DatasetAdapter(ABC):
    """Only dataset adapters know the source dataset schema."""

    @abstractmethod
    def iter_instances(self, limit: int | None = None) -> Iterator[RecommendationInstance]:
        raise NotImplementedError
