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

from rankrag.data.candidate_corpus import CandidateCorpus
from rankrag.embedding import TextEmbedder
from rankrag.graph.candidate_index import CandidateGraphIndex, GraphAssociation, open_candidate_graph_index
from rankrag.graphrag.global_assets import SemanticCandidateIndex
from rankrag.graphrag.scorer import WeightedCandidateScorer
from rankrag.graphrag.seeds import CandidateSeed, DefaultSeedProvider, SeedProvider
from rankrag.models import Query, RankedCandidate, RankingResult, RecommendationInstance


@dataclass(frozen=True)
class HybridCandidateConfig:
    semantic_top_k: int = 500
    seed_candidate_k: int = 20
    graph_hops: int = 2
    max_graph_candidates: int = 1000
    max_nodes_per_hop: int = 500
    max_neighbors_per_node: int = 100
    max_candidates_per_node: int = 100
    max_paths_per_candidate: int = 3
    graph_aggregation: str = "max"


@dataclass(frozen=True)
class GlobalEvidencePath:
    nodes: tuple[str, ...]
    relations: tuple[str, ...]
    score: float


@dataclass
class GeneratedCandidate:
    candidate_id: str
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
    seeds: list[CandidateSeed],
    graph_index: CandidateGraphIndex,
    config: HybridCandidateConfig,
) -> GraphExpansion:
    candidate_seeds = [seed for seed in seeds if seed.target_kind == "candidate"]
    node_seeds = [seed for seed in seeds if seed.target_kind == "node"]
    unknown_seed_kinds = {seed.target_kind for seed in seeds} - {"candidate", "node"}
    if unknown_seed_kinds:
        raise ValueError(f"Unknown graph seed target kinds: {sorted(unknown_seed_kinds)}")
    seed_ids = [seed.target_id for seed in candidate_seeds]
    association_method = getattr(graph_index, "associations_for_candidates", None)
    if association_method is not None:
        seed_nodes = association_method(seed_ids)
    else:
        relation_method = getattr(graph_index, "candidate_association_relation", None)
        seed_nodes = {
            candidate_id: [
                GraphAssociation(
                    node_id,
                    relation=(
                        relation_method(candidate_id, node_id)
                        if relation_method is not None
                        else "associated_with"
                    ),
                )
                for node_id in node_ids
            ]
            for candidate_id, node_ids in graph_index.nodes_for_candidates(seed_ids).items()
        }

    best_node_paths: dict[str, GlobalEvidencePath] = {}
    frontier: list[str] = []
    for seed in candidate_seeds:
        for association in seed_nodes.get(seed.target_id, []):
            node_id = association.target_id
            path = GlobalEvidencePath(
                nodes=(graph_index.candidate_node_id(seed.target_id), node_id),
                relations=(association.relation,),
                score=seed.score * max(0.0, association.weight),
            )
            previous = best_node_paths.get(node_id)
            if previous is None or path.score > previous.score:
                best_node_paths[node_id] = path
                frontier.append(node_id)
    for seed in node_seeds:
        path = GlobalEvidencePath(
            nodes=(seed.target_id,),
            relations=(),
            score=seed.score,
        )
        previous = best_node_paths.get(seed.target_id)
        if previous is None or path.score > previous.score:
            best_node_paths[seed.target_id] = path
            frontier.append(seed.target_id)

    graph_paths: dict[str, list[GlobalEvidencePath]] = {}
    visited_nodes: set[str] = set()
    for depth in range(config.graph_hops + 1):
        frontier = [node for node in dict.fromkeys(frontier) if node not in visited_nodes]
        frontier.sort(key=lambda node: (-best_node_paths[node].score, node))
        frontier = frontier[: config.max_nodes_per_hop]
        if not frontier:
            break
        visited_nodes.update(frontier)
        candidate_association_method = getattr(graph_index, "candidate_associations_for_nodes", None)
        if candidate_association_method is not None:
            node_candidates = candidate_association_method(frontier, config.max_candidates_per_node)
        else:
            relation_method = getattr(graph_index, "candidate_association_relation", None)
            node_candidates = {
                node_id: [
                    GraphAssociation(
                        candidate_id,
                        relation=(
                            relation_method(candidate_id, node_id)
                            if relation_method is not None
                            else "associated_with"
                        ),
                    )
                    for candidate_id in candidate_ids
                ]
                for node_id, candidate_ids in graph_index.candidates_for_nodes(
                    frontier,
                    config.max_candidates_per_node,
                ).items()
            }
        for node_id in frontier:
            node_path = best_node_paths[node_id]
            for association in node_candidates.get(node_id, []):
                candidate_id = association.target_id
                candidate_path = GlobalEvidencePath(
                    nodes=node_path.nodes + (graph_index.candidate_node_id(candidate_id),),
                    relations=node_path.relations + (association.relation,),
                    score=(
                        node_path.score
                        * math.exp(-0.5 * (depth + 1))
                        * max(0.0, association.weight)
                    ),
                )
                graph_paths.setdefault(candidate_id, []).append(candidate_path)
        if depth >= config.graph_hops:
            break
        adjacency = graph_index.neighbors(frontier, config.max_neighbors_per_node)
        next_frontier: list[str] = []
        for source_id in frontier:
            source_path = best_node_paths[source_id]
            for edge in adjacency.get(source_id, []):
                target_id = edge["target"]
                edge_weight = max(0.0, float(edge.get("weight", 1.0)))
                path = GlobalEvidencePath(
                    nodes=source_path.nodes + (target_id,),
                    relations=source_path.relations + (edge["relation"],),
                    score=source_path.score * math.exp(-0.5) * edge_weight,
                )
                previous = best_node_paths.get(target_id)
                if previous is None or path.score > previous.score:
                    best_node_paths[target_id] = path
                    next_frontier.append(target_id)
        frontier = next_frontier

    for paths in graph_paths.values():
        paths.sort(key=lambda path: (-path.score, path.nodes))
        del paths[config.max_paths_per_candidate :]
    graph_ids = sorted(
        graph_paths,
        key=lambda candidate_id: (-graph_paths[candidate_id][0].score, candidate_id),
    )[: config.max_graph_candidates]
    return GraphExpansion(graph_paths=graph_paths, graph_ids=graph_ids)


_WORKER_GRAPH_INDEX: CandidateGraphIndex | None = None
_WORKER_GRAPH_CONFIG: HybridCandidateConfig | None = None


def _close_worker_graph_index() -> None:
    global _WORKER_GRAPH_INDEX
    if _WORKER_GRAPH_INDEX is not None:
        _WORKER_GRAPH_INDEX.close()
        _WORKER_GRAPH_INDEX = None


def _initialize_graph_worker(graph_path: str, graph_backend: str, config: HybridCandidateConfig) -> None:
    global _WORKER_GRAPH_INDEX, _WORKER_GRAPH_CONFIG
    _WORKER_GRAPH_INDEX = open_candidate_graph_index(graph_path, graph_backend)
    _WORKER_GRAPH_CONFIG = config
    atexit.register(_close_worker_graph_index)


def _expand_graph_worker(seeds: list[CandidateSeed]) -> GraphExpansion:
    if _WORKER_GRAPH_INDEX is None or _WORKER_GRAPH_CONFIG is None:
        raise RuntimeError("Graph expansion worker was not initialized")
    return _expand_graph(seeds, _WORKER_GRAPH_INDEX, _WORKER_GRAPH_CONFIG)


def create_graph_expansion_executor(
    graph_path: str | Path,
    config: HybridCandidateConfig,
    graph_workers: int | None = None,
    graph_backend: str = "hotpot_legacy_sqlite",
) -> ProcessPoolExecutor | None:
    worker_count = default_graph_workers() if graph_workers is None else int(graph_workers)
    if worker_count <= 1:
        return None
    return ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_graph_worker,
        initargs=(str(Path(graph_path).resolve()), graph_backend, config),
    )


class HybridCandidateGenerator:
    """Generate, merge, and deduplicate semantic and graph candidates."""

    def __init__(
        self,
        corpus: CandidateCorpus,
        semantic_index: SemanticCandidateIndex,
        graph_index: CandidateGraphIndex,
        embedder: TextEmbedder,
        config: HybridCandidateConfig,
        seed_provider: SeedProvider | None = None,
    ) -> None:
        self.corpus = corpus
        self.semantic_index = semantic_index
        self.graph_index = graph_index
        self.embedder = embedder
        self.config = config
        self.seed_provider = seed_provider or DefaultSeedProvider(config.seed_candidate_k)
        if embedder.dimension != semantic_index.dimension:
            raise ValueError("Query embedder dimension does not match offline corpus embeddings")

    def _assemble_pool(
        self,
        query_vector: np.ndarray,
        semantic_hits: list[tuple[str, float]],
        expansion: GraphExpansion,
        excluded_ids: set[str] | None = None,
        allowed_ids: set[str] | None = None,
    ) -> CandidatePool:
        excluded_ids = excluded_ids or set()
        semantic_ids = [candidate_id for candidate_id, _ in semantic_hits]
        pool_ids = [
            candidate_id
            for candidate_id in dict.fromkeys([*semantic_ids, *expansion.graph_ids])
            if candidate_id not in excluded_ids and (allowed_ids is None or candidate_id in allowed_ids)
        ]
        raw_scores = self.semantic_index.scores(query_vector, pool_ids)
        candidates = [
            GeneratedCandidate(
                candidate_id=candidate_id,
                semantic_score=float(np.clip((raw_scores[candidate_id] + 1.0) / 2.0, 0.0, 1.0)),
                graph_score=self._aggregate_graph_score(expansion.graph_paths.get(candidate_id, [])),
                paths=expansion.graph_paths.get(candidate_id, []),
            )
            for candidate_id in pool_ids
        ]
        graph_only = len((set(pool_ids) & set(expansion.graph_ids)) - set(semantic_ids))
        return CandidatePool(
            candidates=candidates,
            semantic_candidate_count=len(semantic_ids),
            graph_expansion_candidate_count=graph_only,
        )

    def _aggregate_graph_score(self, paths: Sequence[GlobalEvidencePath]) -> float:
        if not paths:
            return 0.0
        scores = [float(np.clip(path.score, 0.0, 1.0)) for path in paths]
        if self.config.graph_aggregation == "max":
            return max(scores)
        if self.config.graph_aggregation == "sum":
            return min(1.0, sum(scores))
        if self.config.graph_aggregation == "noisy_or":
            complement = 1.0
            for score in scores:
                complement *= 1.0 - score
            return 1.0 - complement
        raise ValueError(f"Unknown graph aggregation: {self.config.graph_aggregation}")

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
        max_excluded = max((len(query.excluded_candidate_ids) for query in queries), default=0)
        search_k = min(len(self.corpus), self.config.semantic_top_k + max_excluded)
        raw_semantic_hits_batch = self.semantic_index.search_batch(query_vectors, search_k)
        semantic_hits_batch: list[list[tuple[str, float]]] = []
        for query, query_vector, raw_hits in zip(queries, query_vectors, raw_semantic_hits_batch, strict=True):
            excluded = set(query.excluded_candidate_ids)
            allowed = set(query.allowed_candidate_ids) if query.allowed_candidate_ids is not None else None
            if allowed is not None:
                eligible = sorted(allowed - excluded)
                allowed_scores = self.semantic_index.scores(query_vector, eligible)
                hits = sorted(allowed_scores.items(), key=lambda item: (-item[1], item[0]))
            else:
                hits = [(candidate_id, score) for candidate_id, score in raw_hits if candidate_id not in excluded]
            semantic_hits_batch.append(hits[: self.config.semantic_top_k])
        semantic_time_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        expansion_inputs: list[list[CandidateSeed]] = []
        for query, query_vector, hits in zip(queries, query_vectors, semantic_hits_batch, strict=True):
            def semantic_score_lookup(candidate_ids: Sequence[str]) -> dict[str, float]:
                valid_ids: list[str] = []
                for candidate_id in candidate_ids:
                    try:
                        self.corpus.row_for_id(candidate_id)
                    except KeyError:
                        continue
                    valid_ids.append(candidate_id)
                if not valid_ids:
                    return {}
                return self.semantic_index.scores(query_vector, valid_ids)

            seeds = self.seed_provider.get_seeds(query, hits, semantic_score_lookup)
            filtered_seeds: list[CandidateSeed] = []
            for seed in seeds:
                if seed.target_kind == "candidate":
                    try:
                        self.corpus.row_for_id(seed.target_id)
                    except KeyError:
                        continue
                elif seed.target_kind != "node":
                    raise ValueError(f"Unknown graph seed target kind: {seed.target_kind}")
                filtered_seeds.append(seed)
            expansion_inputs.append(filtered_seeds)
        if graph_executor is None:
            expansions = [
                _expand_graph(semantic_hits, self.graph_index, self.config)
                for semantic_hits in expansion_inputs
            ]
        else:
            expansions = list(graph_executor.map(_expand_graph_worker, expansion_inputs, chunksize=1))
        graph_time_ms = (time.perf_counter() - started) * 1000.0

        pools = [
            self._assemble_pool(
                query_vector,
                semantic_hits,
                expansion,
                excluded_ids=set(query.excluded_candidate_ids),
                allowed_ids=set(query.allowed_candidate_ids) if query.allowed_candidate_ids is not None else None,
            )
            for query, query_vector, semantic_hits, expansion in zip(
                queries,
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
        node_metadata: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[str]]]:
        node_ids = list(dict.fromkeys(node for path in paths for node in path.nodes))
        evidence_nodes: list[dict[str, Any]] = []
        for node_id in node_ids:
            candidate_id = self.candidate_generator.graph_index.candidate_id_from_node(node_id)
            if candidate_id is not None:
                candidate = self.candidate_generator.corpus.candidate(candidate_id)
                metadata_hook = getattr(
                    self.candidate_generator.graph_index,
                    "candidate_node_metadata",
                    None,
                )
                presentation = (
                    metadata_hook(candidate_id)
                    if metadata_hook is not None
                    else {
                        "node_type": "candidate",
                        "metadata": {"candidate_id": candidate_id},
                        "include_corpus_metadata": True,
                    }
                )
                metadata = dict(presentation.get("metadata", {}))
                if presentation.get("include_corpus_metadata", True):
                    metadata.update(candidate.metadata)
                evidence_nodes.append(
                    {
                        "node_id": node_id,
                        "node_type": presentation.get("node_type", "candidate"),
                        "text": str(candidate.metadata.get("title", candidate.text)),
                        "metadata": metadata,
                    }
                )
            else:
                metadata = node_metadata.get(node_id, {})
                evidence_nodes.append(
                    {
                        "node_id": node_id,
                        "node_type": metadata.get("node_type", metadata.get("entity_type", "graph_node")),
                        "text": metadata.get("text", metadata.get("name", node_id)),
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
        scored.sort(key=lambda item: (-item[1], item[0].candidate_id))
        selected = scored[: min(self.top_k, len(scored))]

        evidence_started = time.perf_counter()
        graph_node_ids = {
            node
            for generated, _ in selected
            for path in generated.paths
            for node in path.nodes
            if self.candidate_generator.graph_index.candidate_id_from_node(node) is None
        }
        node_metadata = self.candidate_generator.graph_index.node_metadata(graph_node_ids)
        ranked: list[RankedCandidate] = []
        for generated, rag_score in selected:
            candidate = self.candidate_generator.corpus.candidate(generated.candidate_id)
            evidence_nodes, evidence_edges, paths = self._serialize_evidence(
                generated.paths,
                node_metadata,
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
        pool_ids = {candidate.candidate_id for candidate in pool.candidates}
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
