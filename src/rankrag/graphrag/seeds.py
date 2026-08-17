from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rankrag.models import Query


@dataclass(frozen=True)
class CandidateSeed:
    """Dataset-neutral candidate or graph node used as an expansion seed."""

    candidate_id: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    target_kind: str = "candidate"

    @property
    def target_id(self) -> str:
        return self.candidate_id


SemanticScoreLookup = Callable[[Sequence[str]], Mapping[str, float]]


class SeedProvider(ABC):
    """Turn a query and semantic hits into scored graph seeds."""

    @abstractmethod
    def get_seeds(
        self,
        query: Query,
        semantic_hits: Sequence[tuple[str, float]],
        semantic_score_lookup: SemanticScoreLookup,
    ) -> list[CandidateSeed]: ...


class DefaultSeedProvider(SeedProvider):
    """Backward-compatible semantic seeds plus optional query-provided seeds."""

    def __init__(
        self,
        semantic_seed_k: int = 20,
        use_query_weights: bool = False,
        merge_strategy: str = "first",
        include_query_candidates: bool = True,
        include_query_nodes: bool = False,
        node_seed_score: float = 1.0,
    ) -> None:
        self.semantic_seed_k = max(0, int(semantic_seed_k))
        self.use_query_weights = bool(use_query_weights)
        self.merge_strategy = str(merge_strategy)
        self.include_query_candidates = bool(include_query_candidates)
        self.include_query_nodes = bool(include_query_nodes)
        self.node_seed_score = float(np.clip(node_seed_score, 0.0, 1.0))
        if self.merge_strategy not in {"first", "max"}:
            raise ValueError("seed_provider.merge_strategy must be 'first' or 'max'")

    @staticmethod
    def _normalize(raw_score: float) -> float:
        return float(np.clip((float(raw_score) + 1.0) / 2.0, 0.0, 1.0))

    def get_seeds(
        self,
        query: Query,
        semantic_hits: Sequence[tuple[str, float]],
        semantic_score_lookup: SemanticScoreLookup,
    ) -> list[CandidateSeed]:
        seeds: dict[str, CandidateSeed] = {}
        for candidate_id, raw_score in semantic_hits[: self.semantic_seed_k]:
            seeds[candidate_id] = CandidateSeed(
                candidate_id=candidate_id,
                score=self._normalize(raw_score),
                source="semantic",
            )

        query_ids = list(dict.fromkeys(query.seed_candidate_ids)) if self.include_query_candidates else []
        raw_scores = semantic_score_lookup(query_ids) if query_ids else {}
        configured_weights = query.seed_candidate_weights
        weights_are_aligned = len(configured_weights) == len(query.seed_candidate_ids)
        query_weight_by_id: dict[str, float] = {}
        if self.use_query_weights and weights_are_aligned:
            for candidate_id, weight in zip(
                query.seed_candidate_ids,
                configured_weights,
                strict=True,
            ):
                query_weight_by_id[candidate_id] = max(
                    query_weight_by_id.get(candidate_id, 0.0),
                    float(np.clip(weight, 0.0, 1.0)),
                )

        for candidate_id in query_ids:
            if candidate_id not in raw_scores:
                continue
            score = query_weight_by_id.get(candidate_id, self._normalize(raw_scores[candidate_id]))
            seed = CandidateSeed(candidate_id, score, "query")
            previous = seeds.get(candidate_id)
            if previous is None:
                seeds[candidate_id] = seed
            elif self.merge_strategy == "max" and seed.score > previous.score:
                seeds[candidate_id] = CandidateSeed(
                    candidate_id,
                    seed.score,
                    "semantic+query",
                )

        node_seeds: list[CandidateSeed] = []
        if self.include_query_nodes:
            node_weights_are_aligned = len(query.seed_node_weights) == len(query.seed_node_ids)
            best_node_weights: dict[str, float] = {}
            for index, node_id in enumerate(query.seed_node_ids):
                weight = (
                    float(np.clip(query.seed_node_weights[index], 0.0, 1.0))
                    if node_weights_are_aligned
                    else self.node_seed_score
                )
                best_node_weights[node_id] = max(best_node_weights.get(node_id, 0.0), weight)
            node_seeds = [
                CandidateSeed(
                    candidate_id=node_id,
                    score=weight,
                    source="query_node",
                    target_kind="node",
                )
                for node_id, weight in best_node_weights.items()
            ]
        return [*seeds.values(), *node_seeds]


_SEED_PROVIDERS: dict[str, Callable[..., SeedProvider]] = {
    "default": DefaultSeedProvider,
    "weighted_query": DefaultSeedProvider,
    "graph_nodes": DefaultSeedProvider,
}


def create_seed_provider(config: dict[str, Any], semantic_seed_k: int) -> SeedProvider:
    provider_config = config.get("seed_provider", {})
    provider_type = str(provider_config.get("type", "default"))
    try:
        factory = _SEED_PROVIDERS[provider_type]
    except KeyError as exc:
        raise ValueError(f"Unknown seed provider: {provider_type}") from exc
    return factory(
        semantic_seed_k=int(provider_config.get("semantic_seed_k", semantic_seed_k)),
        use_query_weights=bool(provider_config.get("use_query_weights", provider_type == "weighted_query")),
        merge_strategy=str(provider_config.get("merge_strategy", "first")),
        include_query_candidates=bool(provider_config.get("include_query_candidates", provider_type != "graph_nodes")),
        include_query_nodes=bool(provider_config.get("include_query_nodes", provider_type == "graph_nodes")),
        node_seed_score=float(provider_config.get("node_seed_score", 1.0)),
    )
