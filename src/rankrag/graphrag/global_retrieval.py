from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

import numpy as np

from rankrag.data.paragraph_corpus import ParagraphCorpus
from rankrag.embedding import TextEmbedder
from rankrag.graphrag.global_assets import GlobalGraphIndex, SemanticParagraphIndex
from rankrag.graphrag.scorer import WeightedCandidateScorer
from rankrag.models import Query, RankedCandidate, RankingResult, RecommendationInstance


@dataclass(frozen=True)
class HybridCandidateConfig:
    semantic_top_k: int = 500
    seed_paragraph_k: int = 20
    graph_hops: int = 2
    max_graph_candidates: int = 1000
    max_entities_per_hop: int = 500
    max_relations_per_entity: int = 100
    max_paragraphs_per_entity: int = 100
    max_paths_per_candidate: int = 3


@dataclass(frozen=True)
class GlobalEvidencePath:
    nodes: tuple[str, ...]
    relations: tuple[str, ...]
    score: float


@dataclass
class GeneratedCandidate:
    paragraph_id: str
    semantic_score: float
    graph_score: float = 0.0
    paths: list[GlobalEvidencePath] = field(default_factory=list)


@dataclass
class CandidatePool:
    candidates: list[GeneratedCandidate]
    semantic_candidate_count: int
    graph_expansion_candidate_count: int
    retrieval_time_ms: float


class HybridCandidateGenerator:
    """Generate, merge, and deduplicate semantic and graph candidates.

    This component deliberately does not select the final GraphRAG Top-K.
    """

    def __init__(
        self,
        corpus: ParagraphCorpus,
        semantic_index: SemanticParagraphIndex,
        graph_index: GlobalGraphIndex,
        embedder: TextEmbedder,
        config: HybridCandidateConfig,
    ) -> None:
        self.corpus = corpus
        self.semantic_index = semantic_index
        self.graph_index = graph_index
        self.embedder = embedder
        self.config = config
        if embedder.dimension != semantic_index.dimension:
            raise ValueError("Query embedder dimension does not match offline corpus embeddings")

    def generate(self, query: Query) -> CandidatePool:
        started = time.perf_counter()
        query_vector = self.embedder.encode([query.text])[0]
        semantic_hits = self.semantic_index.search(query_vector, self.config.semantic_top_k)
        semantic_ids = [paragraph_id for paragraph_id, _ in semantic_hits]
        seed_ids = semantic_ids[: self.config.seed_paragraph_k]
        seed_entities = self.graph_index.entities_for_paragraphs(seed_ids)

        best_entity_paths: dict[str, GlobalEvidencePath] = {}
        frontier: list[str] = []
        for seed_id, raw_score in semantic_hits[: self.config.seed_paragraph_k]:
            seed_score = float(np.clip((raw_score + 1.0) / 2.0, 0.0, 1.0))
            for entity_id in seed_entities.get(seed_id, []):
                path = GlobalEvidencePath(
                    nodes=(f"paragraph::{seed_id}", f"entity::{entity_id}"),
                    relations=("mentions",),
                    score=seed_score,
                )
                previous = best_entity_paths.get(entity_id)
                if previous is None or path.score > previous.score:
                    best_entity_paths[entity_id] = path
                    frontier.append(entity_id)

        graph_paths: dict[str, list[GlobalEvidencePath]] = {}
        visited_entities: set[str] = set()
        for depth in range(self.config.graph_hops + 1):
            frontier = [entity for entity in dict.fromkeys(frontier) if entity not in visited_entities]
            frontier.sort(key=lambda entity: (-best_entity_paths[entity].score, entity))
            frontier = frontier[: self.config.max_entities_per_hop]
            if not frontier:
                break
            visited_entities.update(frontier)
            entity_paragraphs = self.graph_index.paragraphs_for_entities(
                frontier,
                self.config.max_paragraphs_per_entity,
            )
            for entity_id in frontier:
                entity_path = best_entity_paths[entity_id]
                for paragraph_id in entity_paragraphs.get(entity_id, []):
                    candidate_path = GlobalEvidencePath(
                        nodes=entity_path.nodes + (f"paragraph::{paragraph_id}",),
                        relations=entity_path.relations + ("mentions",),
                        score=entity_path.score * math.exp(-0.5 * (depth + 1)),
                    )
                    graph_paths.setdefault(paragraph_id, []).append(candidate_path)
            if depth >= self.config.graph_hops:
                break
            adjacency = self.graph_index.adjacent_entities(frontier, self.config.max_relations_per_entity)
            next_frontier: list[str] = []
            for source_id in frontier:
                source_path = best_entity_paths[source_id]
                for edge in adjacency.get(source_id, []):
                    target_id = edge["target"]
                    path = GlobalEvidencePath(
                        nodes=source_path.nodes + (f"entity::{target_id}",),
                        relations=source_path.relations + (edge["relation"],),
                        score=source_path.score * math.exp(-0.5),
                    )
                    previous = best_entity_paths.get(target_id)
                    if previous is None or path.score > previous.score:
                        best_entity_paths[target_id] = path
                        next_frontier.append(target_id)
            frontier = next_frontier

        for paths in graph_paths.values():
            paths.sort(key=lambda path: (-path.score, path.nodes))
            del paths[self.config.max_paths_per_candidate :]
        graph_ids = sorted(
            graph_paths,
            key=lambda paragraph_id: (-graph_paths[paragraph_id][0].score, paragraph_id),
        )[: self.config.max_graph_candidates]
        pool_ids = list(dict.fromkeys([*semantic_ids, *graph_ids]))
        raw_scores = self.semantic_index.scores(query_vector, pool_ids)
        candidates = [
            GeneratedCandidate(
                paragraph_id=paragraph_id,
                semantic_score=float(np.clip((raw_scores[paragraph_id] + 1.0) / 2.0, 0.0, 1.0)),
                graph_score=graph_paths[paragraph_id][0].score if paragraph_id in graph_paths else 0.0,
                paths=graph_paths.get(paragraph_id, []),
            )
            for paragraph_id in pool_ids
        ]
        graph_only = len(set(graph_ids) - set(semantic_ids))
        return CandidatePool(
            candidates=candidates,
            semantic_candidate_count=len(semantic_ids),
            graph_expansion_candidate_count=graph_only,
            retrieval_time_ms=(time.perf_counter() - started) * 1000.0,
        )


class HybridGraphRAGRetriever:
    """Score a generated global pool and preserve the GraphRAG JSONL contract."""

    def __init__(
        self,
        candidate_generator: HybridCandidateGenerator,
        scorer: WeightedCandidateScorer,
        top_k: int = 100,
    ) -> None:
        self.candidate_generator = candidate_generator
        self.scorer = scorer
        self.top_k = top_k

    def _serialize_evidence(
        self,
        paths: list[GlobalEvidencePath],
        entity_metadata: dict[str, dict[str, str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[str]]]:
        node_ids = list(dict.fromkeys(node for path in paths for node in path.nodes))
        evidence_nodes: list[dict[str, Any]] = []
        for node_id in node_ids:
            if node_id.startswith("paragraph::"):
                pid = node_id.removeprefix("paragraph::")
                record = self.candidate_generator.corpus.records[self.candidate_generator.corpus.id_to_row[pid]]
                evidence_nodes.append({"node_id": node_id, "node_type": "paragraph", "text": record.title, "metadata": {"paragraph_id": pid}})
            else:
                entity_id = node_id.removeprefix("entity::")
                metadata = entity_metadata.get(entity_id, {})
                evidence_nodes.append({"node_id": node_id, "node_type": metadata.get("entity_type", "entity"), "text": metadata.get("name", entity_id), "metadata": metadata})
        evidence_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for path in paths:
            for source, target, relation in zip(path.nodes[:-1], path.nodes[1:], path.relations, strict=True):
                key = (source, target, relation)
                if key not in seen_edges:
                    evidence_edges.append({"source": source, "relation": relation, "target": target, "metadata": {}})
                    seen_edges.add(key)
        return evidence_nodes, evidence_edges, [list(path.nodes) for path in paths]

    def rank(self, instance: RecommendationInstance) -> RankingResult:
        pool = self.candidate_generator.generate(instance.query)
        entity_ids = {
            node.removeprefix("entity::")
            for generated in pool.candidates
            for path in generated.paths
            for node in path.nodes
            if node.startswith("entity::")
        }
        entity_metadata = self.candidate_generator.graph_index.entity_metadata(entity_ids)
        ranked: list[RankedCandidate] = []
        for generated in pool.candidates:
            candidate = self.candidate_generator.corpus.candidate(generated.paragraph_id)
            evidence_nodes, evidence_edges, paths = self._serialize_evidence(generated.paths, entity_metadata)
            rag_score = self.scorer.score(generated.semantic_score, generated.graph_score)
            ranked.append(
                RankedCandidate(
                    candidate_id=candidate.candidate_id,
                    text=candidate.text,
                    semantic_score=generated.semantic_score,
                    graph_score=generated.graph_score,
                    rag_score=rag_score,
                    evidence_nodes=evidence_nodes,
                    evidence_edges=evidence_edges,
                    paths=paths,
                    graph_features={
                        "evidence_node_count": float(len(evidence_nodes)),
                        "evidence_edge_count": float(len(evidence_edges)),
                        "path_count": float(len(paths)),
                        "best_path_hops": float(len(paths[0]) - 1) if paths else 0.0,
                    },
                    metadata=candidate.metadata,
                )
            )
        ranked.sort(key=lambda candidate: (-candidate.rag_score, candidate.candidate_id))
        ranked = ranked[: min(self.top_k, len(ranked))]
        for rank, candidate in enumerate(ranked, start=1):
            candidate.rank = rank

        positives = set(instance.positive_ids)
        pool_ids = {candidate.paragraph_id for candidate in pool.candidates}
        ranked_ids = {candidate.candidate_id for candidate in ranked}
        pool_recall = len(pool_ids & positives) / len(positives) if positives else 0.0
        recall_at_100 = len(ranked_ids & positives) / len(positives) if positives else 0.0
        return RankingResult(
            query_id=instance.query.query_id,
            query_text=instance.query.text,
            positive_ids=instance.positive_ids,
            candidates=ranked,
            stage="graphrag",
            metadata={
                **instance.metadata,
                "candidate_pool_size": len(pool.candidates),
                "semantic_candidate_count": pool.semantic_candidate_count,
                "graph_expansion_candidate_count": pool.graph_expansion_candidate_count,
                "gold_recall_before_graphrag": pool_recall,
                "recall_at_100": recall_at_100,
                "retrieval_time_ms": pool.retrieval_time_ms,
            },
        )
