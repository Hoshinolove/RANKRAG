from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMRequest:
    query_id: str
    query_text: str
    candidates: list[dict[str, Any]]
    prompt_version: str


@dataclass(frozen=True)
class LLMResponse:
    ranking: list[dict[str, Any]]
    raw_response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    @property
    @abstractmethod
    def identity(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def rerank(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError
