from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
from typing import Any


@dataclass(frozen=True)
class GraphAssociation:
    target_id: str
    relation: str = "associated_with"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class CandidateGraphIndex(ABC):
    """Dataset-neutral graph operations consumed by global GraphRAG."""

    backend: str

    @abstractmethod
    def nodes_for_candidates(self, candidate_ids: Sequence[str]) -> dict[str, list[str]]: ...

    @abstractmethod
    def candidates_for_nodes(
        self,
        node_ids: Sequence[str],
        per_node_limit: int,
    ) -> dict[str, list[str]]: ...

    @abstractmethod
    def neighbors(
        self,
        node_ids: Sequence[str],
        per_node_limit: int,
    ) -> dict[str, list[dict[str, Any]]]: ...

    @abstractmethod
    def node_metadata(self, node_ids: Iterable[str]) -> dict[str, dict[str, Any]]: ...

    @abstractmethod
    def candidate_node_id(self, candidate_id: str) -> str: ...

    @abstractmethod
    def candidate_id_from_node(self, node_id: str) -> str | None: ...

    def candidate_node_metadata(self, candidate_id: str) -> dict[str, Any]:
        return {
            "node_type": "candidate",
            "metadata": {"candidate_id": candidate_id},
            "include_corpus_metadata": True,
        }

    def candidate_association_relation(self, candidate_id: str, node_id: str) -> str:
        return "associated_with"

    def associations_for_candidates(
        self,
        candidate_ids: Sequence[str],
    ) -> dict[str, list[GraphAssociation]]:
        return {
            candidate_id: [
                GraphAssociation(
                    target_id=node_id,
                    relation=self.candidate_association_relation(candidate_id, node_id),
                )
                for node_id in node_ids
            ]
            for candidate_id, node_ids in self.nodes_for_candidates(candidate_ids).items()
        }

    def candidate_associations_for_nodes(
        self,
        node_ids: Sequence[str],
        per_node_limit: int,
    ) -> dict[str, list[GraphAssociation]]:
        return {
            node_id: [
                GraphAssociation(
                    target_id=candidate_id,
                    relation=self.candidate_association_relation(candidate_id, node_id),
                )
                for candidate_id in candidate_ids
            ]
            for node_id, candidate_ids in self.candidates_for_nodes(node_ids, per_node_limit).items()
        }

    @abstractmethod
    def close(self) -> None: ...


def _chunks(values: Sequence[str], size: int = 800):
    for start in range(0, len(values), size):
        yield values[start : start + size]


class SQLiteCandidateGraphIndex(CandidateGraphIndex):
    backend = "generic_sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)

    def nodes_for_candidates(self, candidate_ids: Sequence[str]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for chunk in _chunks(list(dict.fromkeys(candidate_ids))):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT candidate_id, node_id FROM candidate_nodes "
                f"WHERE candidate_id IN ({placeholders}) ORDER BY candidate_id, node_id",
                tuple(chunk),
            )
            for candidate_id, node_id in rows:
                result[str(candidate_id)].append(str(node_id))
        return dict(result)

    def candidates_for_nodes(self, node_ids: Sequence[str], per_node_limit: int) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for chunk in _chunks(list(dict.fromkeys(node_ids))):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT node_id, candidate_id FROM candidate_nodes "
                f"WHERE node_id IN ({placeholders}) ORDER BY node_id, candidate_id",
                tuple(chunk),
            )
            for node_id, candidate_id in rows:
                if len(result[str(node_id)]) < per_node_limit:
                    result[str(node_id)].append(str(candidate_id))
        return dict(result)

    def associations_for_candidates(
        self,
        candidate_ids: Sequence[str],
    ) -> dict[str, list[GraphAssociation]]:
        result: dict[str, list[GraphAssociation]] = defaultdict(list)
        for chunk in _chunks(list(dict.fromkeys(candidate_ids))):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT candidate_id, node_id, relation, weight, metadata_json FROM candidate_nodes "
                f"WHERE candidate_id IN ({placeholders}) "
                f"ORDER BY candidate_id, weight DESC, node_id, relation",
                tuple(chunk),
            )
            for candidate_id, node_id, relation, weight, metadata_json in rows:
                result[str(candidate_id)].append(
                    GraphAssociation(
                        target_id=str(node_id),
                        relation=str(relation),
                        weight=float(weight),
                        metadata=json.loads(metadata_json or "{}"),
                    )
                )
        return dict(result)

    def candidate_associations_for_nodes(
        self,
        node_ids: Sequence[str],
        per_node_limit: int,
    ) -> dict[str, list[GraphAssociation]]:
        result: dict[str, list[GraphAssociation]] = defaultdict(list)
        for chunk in _chunks(list(dict.fromkeys(node_ids))):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT node_id, candidate_id, relation, weight, metadata_json FROM candidate_nodes "
                f"WHERE node_id IN ({placeholders}) "
                f"ORDER BY node_id, weight DESC, candidate_id, relation",
                tuple(chunk),
            )
            for node_id, candidate_id, relation, weight, metadata_json in rows:
                node_id = str(node_id)
                if len(result[node_id]) < per_node_limit:
                    result[node_id].append(
                        GraphAssociation(
                            target_id=str(candidate_id),
                            relation=str(relation),
                            weight=float(weight),
                            metadata=json.loads(metadata_json or "{}"),
                        )
                    )
        return dict(result)

    def neighbors(self, node_ids: Sequence[str], per_node_limit: int) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in _chunks(list(dict.fromkeys(node_ids))):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT source_node_id, target_node_id, relation, weight, metadata_json "
                f"FROM graph_edges WHERE source_node_id IN ({placeholders}) "
                f"ORDER BY source_node_id, weight DESC, target_node_id, relation",
                tuple(chunk),
            )
            for source, target, relation, weight, metadata_json in rows:
                source = str(source)
                if len(result[source]) < per_node_limit:
                    result[source].append(
                        {
                            "target": str(target),
                            "relation": str(relation),
                            "weight": float(weight),
                            "metadata": json.loads(metadata_json or "{}"),
                        }
                    )
        return dict(result)

    def node_metadata(self, node_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        values = list(dict.fromkeys(node_ids))
        result: dict[str, dict[str, Any]] = {}
        for chunk in _chunks(values):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT node_id, node_type, text, metadata_json FROM nodes "
                f"WHERE node_id IN ({placeholders})",
                tuple(chunk),
            )
            for node_id, node_type, text, metadata_json in rows:
                metadata = json.loads(metadata_json or "{}")
                result[str(node_id)] = {"node_type": str(node_type), "text": str(text), **metadata}
        return result

    def candidate_node_id(self, candidate_id: str) -> str:
        return f"candidate::{candidate_id}"

    def candidate_id_from_node(self, node_id: str) -> str | None:
        return node_id.removeprefix("candidate::") if node_id.startswith("candidate::") else None

    def close(self) -> None:
        self.connection.close()


def create_candidate_graph_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE candidate_nodes (
            candidate_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL NOT NULL,
            metadata_json TEXT NOT NULL,
            PRIMARY KEY (candidate_id, node_id, relation)
        );
        CREATE TABLE graph_edges (
            source_node_id TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL NOT NULL,
            metadata_json TEXT NOT NULL,
            PRIMARY KEY (source_node_id, target_node_id, relation)
        );
        CREATE INDEX idx_candidate_nodes_candidate ON candidate_nodes(candidate_id);
        CREATE INDEX idx_candidate_nodes_node ON candidate_nodes(node_id);
        CREATE INDEX idx_graph_edges_source ON graph_edges(source_node_id);
        """
    )


def open_candidate_graph_index(path: str | Path, backend: str) -> CandidateGraphIndex:
    if backend == "generic_sqlite":
        return SQLiteCandidateGraphIndex(path)
    if backend == "hotpot_legacy_sqlite":
        from rankrag.graphrag.global_assets import GlobalGraphIndex

        return GlobalGraphIndex(path)
    raise ValueError(f"Unknown candidate graph backend: {backend}")
