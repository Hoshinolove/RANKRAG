from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecommendationInstance:
    query: Query
    candidates: list[Candidate]
    positive_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Node:
    node_id: str
    node_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedCandidate:
    candidate_id: str
    text: str
    semantic_score: float = 0.0
    graph_score: float = 0.0
    rag_score: float = 0.0
    evidence_nodes: list[dict[str, Any]] = field(default_factory=list)
    evidence_edges: list[dict[str, Any]] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)
    graph_features: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    neural_score: float | None = None
    neural_rank: int | None = None
    intermediate_representation: list[float] | None = None
    llm_score: float | None = None
    llm_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RankedCandidate":
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


@dataclass
class RankingResult:
    query_id: str
    query_text: str
    positive_ids: list[str]
    candidates: list[RankedCandidate]
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["schema_version"] = 1
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RankingResult":
        return cls(
            query_id=value["query_id"],
            query_text=value["query_text"],
            positive_ids=list(value.get("positive_ids", [])),
            candidates=[RankedCandidate.from_dict(item) for item in value["candidates"]],
            stage=value["stage"],
            metadata=dict(value.get("metadata", {})),
        )
