from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Sequence

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
    retrieval_time_ms: float = 0.0


@dataclass(frozen=True)
class GraphExpansion:
    graph_paths: dict[str, list[GlobalEvidencePath]]
    graph_ids: list[str]


@dataclass(frozen=True)
class BatchRetrievalTimings:
    query_embedding_time_ms: float
    semantic_search_time_ms: float
    graph_expansion_time_ms: float


def default_graph_workers() -> int:
    return min(16, os.cpu_count() or 1)


def _expand_graph(
    semantic_hits: list[tuple[str, float]],
    graph_index: GlobalGraphIndex,
    config: HybridCandidateConfig,
) -> GraphExpansion:
    seed_ids = [paragraph_id for paragraph_id, _ in semantic_hits[: config.seed_paragraph_k]]
    seed_entities = graph_index.entities_for_paragraphs(seed_ids)

    best_entity_paths: dict[str, GlobalEvidencePath] = {}
    frontier: list[str] = []
    for seed_id, raw_score in semantic_hits[: config.seed_paragraph_k]:
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
    for depth in range(config.graph_hops + 1):
        frontier = [entity for entity in dict.fromkeys(frontier) if entity not in visited_entities]
        frontier.sort(key=lambda entity: (-best_entity_paths[entity].score, entity))
        frontier = frontier[: config.max_entities_per_hop]
        if not frontier:
            break
        visited_entities.update(frontier)
        entity_paragraphs = graph_index.paragraphs_for_entities(frontier, config.max_paragraphs_per_entity)
        for entity_id in frontier:
            entity_path = best_entity_paths[entity_id]
            for paragraph_id in entity_paragraphs.get(entity_id, []):
                candidate_path = GlobalEvidencePath(
                    nodes=entity_path.nodes + (f"paragraph::{paragraph_id}",),
                    relations=entity_path.relations + ("mentions",),
                    score=entity_path.score * math.exp(-0.5 * (depth + 1)),
                )
                graph_paths.setdefault(paragraph_id, []).append(candidate_path)
        if depth >= config.graph_hops:
            break
        adjacency = graph_index.adjacent_entities(frontier, config.max_relations_per_entity)
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
        del paths[config.max_paths_per_candidate :]
    graph_ids = sorted(
        graph_paths,
        key=lambda paragraph_id: (-graph_paths[paragraph_id][0].score, paragraph_id),
    )[: config.max_graph_candidates]
    return GraphExpansion(graph_paths=graph_paths, graph_ids=graph_ids)


_WORKER_GRAPH_INDEX: GlobalGraphIndex | None = None
_WORKER_GRAPH_CONFIG: HybridCandidateConfig | None = None


def _close_worker_graph_index() -> None:
    global _WORKER_GRAPH_INDEX
    if _WORKER_GRAPH_INDEX is not None:
        _WORKER_GRAPH_INDEX.close()
        _WORKER_GRAPH_INDEX = None


def _initialize_graph_worker(graph_path: str, config: HybridCandidateConfig) -> None:
    global _WORKER_GRAPH_INDEX, _WORKER_GRAPH_CONFIG
    _WORKER_GRAPH_INDEX = GlobalGraphIndex(graph_path)
    _WORKER_GRAPH_CONFIG = config
    atexit.register(_close_worker_graph_index)


def _expand_graph_worker(semantic_hits: list[tuple[str, float]]) -> GraphExpansion:
    if _WORKER_GRAPH_INDEX is None or _WORKER_GRAPH_CONFIG is None:
        raise RuntimeError("Graph expansion worker was not initialized")
    return _expand_graph(semantic_hits, _WORKER_GRAPH_INDEX, _WORKER_GRAPH_CONFIG)


def create_graph_expansion_executor(
    graph_path: str | Path,
    config: HybridCandidateConfig,
    graph_workers: int | None = None,
) -> ProcessPoolExecutor | None:
    worker_count = default_graph_workers() if graph_workers is None else int(graph_workers)
    if worker_count <= 1:
        return None
    return ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_graph_worker,
        initargs=(str(Path(graph_path).resolve()), config),
    )


class HybridCandidateGenerator:
    """Generate, merge, and deduplicate semantic and graph candidates."""

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

    def _assemble_pool(
        self,
        query_vector: np.ndarray,
        semantic_hits: list[tuple[str, float]],
        expansion: GraphExpansion,
    ) -> CandidatePool:
        semantic_ids = [paragraph_id for paragraph_id, _ in semantic_hits]
        pool_ids = list(dict.fromkeys([*semantic_ids, *expansion.graph_ids]))
        raw_scores = self.semantic_index.scores(query_vector, pool_ids)
        candidates = [
            GeneratedCandidate(
                paragraph_id=paragraph_id,
                semantic_score=float(np.clip((raw_scores[paragraph_id] + 1.0) / 2.0, 0.0, 1.0)),
                graph_score=(
                    expansion.graph_paths[paragraph_id][0].score
                    if paragraph_id in expansion.graph_paths
                    else 0.0
                ),
                paths=expansion.graph_paths.get(paragraph_id, []),
            )
            for paragraph_id in pool_ids
        ]
        graph_only = len(set(expansion.graph_ids) - set(semantic_ids))
        return CandidatePool(
            candidates=candidates,
            semantic_candidate_count=len(semantic_ids),
            graph_expansion_candidate_count=graph_only,
        )

    def generate_batch(
        self,
        queries: Sequence[Query],
        graph_executor: ProcessPoolExecutor | None = None,
    ) -> tuple[list[CandidatePool], BatchRetrievalTimings]:
        if not queries:
            return [], BatchRetrievalTimings(0.0, 0.0, 0.0)

        started = time.perf_counter()
        query_vectors = np.asarray(self.embedder.encode([query.text for query in queries]), dtype=np.float32)
        if query_vectors.shape != (len(queries), self.semantic_index.dimension):
            raise ValueError(
                "Query embedder returned an unexpected shape: "
                f"expected {(len(queries), self.semantic_index.dimension)}, got {query_vectors.shape}"
            )
        embedding_time_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        semantic_hits_batch = self.semantic_index.search_batch(query_vectors, self.config.semantic_top_k)
        semantic_time_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        expansion_inputs = [hits[: self.config.seed_paragraph_k] for hits in semantic_hits_batch]
        if graph_executor is None:
            expansions = [
                _expand_graph(semantic_hits, self.graph_index, self.config)
                for semantic_hits in expansion_inputs
            ]
        else:
            expansions = list(graph_executor.map(_expand_graph_worker, expansion_inputs, chunksize=1))
        graph_time_ms = (time.perf_counter() - started) * 1000.0

        pools = [
            self._assemble_pool(query_vector, semantic_hits, expansion)
            for query_vector, semantic_hits, expansion in zip(
                query_vectors,
                semantic_hits_batch,
                expansions,
                strict=True,
            )
        ]
        retrieval_time_ms = (embedding_time_ms + semantic_time_ms + graph_time_ms) / len(queries)
        for pool in pools:
            pool.retrieval_time_ms = retrieval_time_ms
        return pools, BatchRetrievalTimings(embedding_time_ms, semantic_time_ms, graph_time_ms)

    def generate(self, query: Query) -> CandidatePool:
        pools, _ = self.generate_batch([query])
        return pools[0]


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
                evidence_nodes.append(
                    {
                        "node_id": node_id,
                        "node_type": "paragraph",
                        "text": record.title,
                        "metadata": {"paragraph_id": pid},
                    }
                )
            else:
                entity_id = node_id.removeprefix("entity::")
                metadata = entity_metadata.get(entity_id, {})
                evidence_nodes.append(
                    {
                        "node_id": node_id,
                        "node_type": metadata.get("entity_type", "entity"),
                        "text": metadata.get("name", entity_id),
                        "metadata": metadata,
                    }
                )
        evidence_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for path in paths:
            for source, target, relation in zip(path.nodes[:-1], path.nodes[1:], path.relations, strict=True):
                key = (source, target, relation)
                if key not in seen_edges:
                    evidence_edges.append(
                        {"source": source, "relation": relation, "target": target, "metadata": {}}
                    )
                    seen_edges.add(key)
        return evidence_nodes, evidence_edges, [list(path.nodes) for path in paths]

    def _rank_pool(
        self,
        instance: RecommendationInstance,
        pool: CandidatePool,
    ) -> tuple[RankingResult, float]:
        scored = [
            (generated, self.scorer.score(generated.semantic_score, generated.graph_score))
            for generated in pool.candidates
        ]
        scored.sort(key=lambda item: (-item[1], item[0].paragraph_id))
        selected = scored[: min(self.top_k, len(scored))]

        evidence_started = time.perf_counter()
        entity_ids = {
            node.removeprefix("entity::")
            for generated, _ in selected
            for path in generated.paths
            for node in path.nodes
            if node.startswith("entity::")
        }
        entity_metadata = self.candidate_generator.graph_index.entity_metadata(entity_ids)
        ranked: list[RankedCandidate] = []
        for generated, rag_score in selected:
            candidate = self.candidate_generator.corpus.candidate(generated.paragraph_id)
            evidence_nodes, evidence_edges, paths = self._serialize_evidence(
                generated.paths,
                entity_metadata,
            )
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
        evidence_time_ms = (time.perf_counter() - evidence_started) * 1000.0
        for rank, candidate in enumerate(ranked, start=1):
            candidate.rank = rank

        positives = set(instance.positive_ids)
        pool_ids = {candidate.paragraph_id for candidate in pool.candidates}
        ranked_ids = {candidate.candidate_id for candidate in ranked}
        pool_recall = len(pool_ids & positives) / len(positives) if positives else 0.0
        recall_at_100 = len(ranked_ids & positives) / len(positives) if positives else 0.0
        result = RankingResult(
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
            },
        )
        return result, evidence_time_ms

    def rank_batch(
        self,
        instances: Sequence[RecommendationInstance],
        graph_executor: ProcessPoolExecutor | None = None,
    ) -> list[RankingResult]:
        if not instances:
            return []
        batch_started = time.perf_counter()
        pools, timings = self.candidate_generator.generate_batch(
            [instance.query for instance in instances],
            graph_executor,
        )
        ranked_with_timings = [
            self._rank_pool(instance, pool)
            for instance, pool in zip(instances, pools, strict=True)
        ]
        total_per_query_ms = (time.perf_counter() - batch_started) * 1000.0 / len(instances)
        embedding_per_query_ms = timings.query_embedding_time_ms / len(instances)
        semantic_per_query_ms = timings.semantic_search_time_ms / len(instances)
        graph_per_query_ms = timings.graph_expansion_time_ms / len(instances)
        results: list[RankingResult] = []
        for result, evidence_time_ms in ranked_with_timings:
            result.metadata.update(
                {
                    "query_embedding_time_ms": embedding_per_query_ms,
                    "semantic_search_time_ms": semantic_per_query_ms,
                    "graph_expansion_time_ms": graph_per_query_ms,
                    "evidence_serialization_time_ms": evidence_time_ms,
                    "total_query_time_ms": total_per_query_ms,
                    "retrieval_time_ms": total_per_query_ms,
                }
            )
            results.append(result)
        return results

    def rank(self, instance: RecommendationInstance) -> RankingResult:
        return self.rank_batch([instance])[0]
