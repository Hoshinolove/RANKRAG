from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightedCandidateScorer:
    semantic_weight: float = 0.5
    graph_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.semantic_weight < 0 or self.graph_weight < 0:
            raise ValueError("Score weights must be non-negative")
        if self.semantic_weight + self.graph_weight == 0:
            raise ValueError("At least one score weight must be positive")

    def score(self, semantic_score: float, graph_score: float) -> float:
        total = self.semantic_weight + self.graph_weight
        return (self.semantic_weight * semantic_score + self.graph_weight * graph_score) / total
