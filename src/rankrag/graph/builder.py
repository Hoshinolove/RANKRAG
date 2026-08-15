from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
import re
from typing import Any

from rankrag.graph.store import GraphStore, NetworkXGraphStore
from rankrag.models import Edge, Node, RecommendationInstance


def _key(value: str) -> str:
    return " ".join(value.casefold().split())


class GraphBuilder(ABC):
    @abstractmethod
    def build(self, instance: RecommendationInstance) -> GraphStore:
        raise NotImplementedError


class KGExtractionIndex:
    """A compact title-keyed view over the existing paragraph KG extractions."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: dict[str, dict[str, Any]] = {}

    def load_for_titles(self, titles: set[str]) -> None:
        if not self.path or not self.path.exists() or not titles:
            return
        wanted = {_key(title) for title in titles}
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                entities = record.get("entities", [])
                if not entities:
                    continue
                title_key = _key(str(entities[0].get("entity_name", "")))
                if title_key in wanted:
                    self.records[title_key] = record

    def get(self, title: str) -> dict[str, Any] | None:
        return self.records.get(_key(title))


class HotpotQAGraphBuilder(GraphBuilder):
    """Dataset-specific graph construction; retrieval remains dataset agnostic."""

    _STOP_WORDS = {
        "about", "after", "again", "also", "been", "before", "being", "between", "could", "first",
        "from", "have", "into", "more", "other", "over", "such", "than", "that", "their", "there",
        "these", "they", "this", "those", "through", "under", "were", "which", "while", "with", "would",
    }

    def __init__(self, extraction_index: KGExtractionIndex | None = None, lexical_fallback: bool = True, max_fallback_terms: int = 10) -> None:
        self.extraction_index = extraction_index or KGExtractionIndex()
        self.lexical_fallback = lexical_fallback
        self.max_fallback_terms = max(1, max_fallback_terms)

    def _add_lexical_fallback(self, store: GraphStore, candidate_id: str, text: str) -> None:
        """Build a small deterministic token graph when no prepared KG record exists."""
        tokens = [token.casefold() for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", text)]
        frequencies: dict[str, int] = {}
        for token in tokens:
            if token not in self._STOP_WORDS:
                frequencies[token] = frequencies.get(token, 0) + 1
        terms = sorted(frequencies, key=lambda token: (-frequencies[token], token))[: self.max_fallback_terms]
        for term in terms:
            node_id = f"lexical::{term}"
            store.add_nodes([Node(node_id, "lexical_term", term, {"fallback": True})])
            store.add_edges([Edge(f"candidate::{candidate_id}", "contains_term", node_id, {"fallback": True})])

    def build(self, instance: RecommendationInstance) -> GraphStore:
        store = NetworkXGraphStore()
        for candidate in instance.candidates:
            candidate_node = f"candidate::{candidate.candidate_id}"
            store.add_nodes([Node(candidate_node, "candidate", candidate.text, {"candidate_id": candidate.candidate_id})])
            record = self.extraction_index.get(candidate.candidate_id)
            if not record:
                if self.lexical_fallback:
                    self._add_lexical_fallback(store, candidate.candidate_id, candidate.text)
                else:
                    entity_node = f"entity::{candidate.candidate_id}"
                    store.add_nodes([Node(entity_node, "entity", candidate.candidate_id, {"fallback": True})])
                    store.add_edges([Edge(candidate_node, "has_title", entity_node)])
                continue
            entity_names: set[str] = set()
            for entity in record.get("entities", []):
                name = str(entity.get("entity_name", "")).strip()
                if not name:
                    continue
                entity_names.add(name)
                node_id = f"entity::{name}"
                store.add_nodes([Node(node_id, str(entity.get("entity_type", "entity")), f"{name}. {entity.get('description', '')}".strip(), dict(entity))])
                store.add_edges([Edge(candidate_node, "mentions", node_id, {"source_record": record.get("id")})])
            for relation in record.get("relationships", []):
                source = str(relation.get("src_id", ""))
                target = str(relation.get("tgt_id", ""))
                if source == str(record.get("id")) and entity_names:
                    source = str(record.get("entities", [{}])[0].get("entity_name", ""))
                if source in entity_names and target in entity_names:
                    store.add_edges([Edge(f"entity::{source}", str(relation.get("keywords", "related_to")), f"entity::{target}", dict(relation))])
        return store
