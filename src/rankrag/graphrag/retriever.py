from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from rankrag.embedding import TextEmbedder
from rankrag.graph.builder import GraphBuilder
from rankrag.graphrag.evidence import serialize_path_evidence
from rankrag.graphrag.scorer import WeightedCandidateScorer
from rankrag.models import RankedCandidate, RankingResult, RecommendationInstance


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 100
    hops: int = 2
    seed_k: int = 8
    max_paths_per_candidate: int = 3


class GraphRAGRetriever:
    def __init__(
        self,
        embedder: TextEmbedder,
        graph_builder: GraphBuilder,
        scorer: WeightedCandidateScorer,
        config: RetrievalConfig,
    ) -> None:
        self.embedder = embedder
        self.graph_builder = graph_builder
        self.scorer = scorer
        self.config = config

    def rank(self, instance: RecommendationInstance) -> RankingResult:
        if not instance.candidates:
            return RankingResult(instance.query.query_id, instance.query.text, instance.positive_ids, [], "graphrag")
        graph = self.graph_builder.build(instance)
        query_vector = self.embedder.encode([instance.query.text])[0]
        candidate_vectors = self.embedder.encode([candidate.text for candidate in instance.candidates])
        semantic_scores = np.clip((candidate_vectors @ query_vector + 1.0) / 2.0, 0.0, 1.0)

        graph_node_ids = [node_id for node_id in graph.node_ids() if not node_id.startswith("candidate::")]
        if graph_node_ids:
            node_texts = [graph.node(node_id).text for node_id in graph_node_ids]  # type: ignore[union-attr]
            node_vectors = self.embedder.encode(node_texts)
            node_scores = np.clip((node_vectors @ query_vector + 1.0) / 2.0, 0.0, 1.0)
            order = np.argsort(-node_scores, kind="stable")[: min(self.config.seed_k, len(graph_node_ids))]
            seeds = [(graph_node_ids[int(index)], float(node_scores[int(index)])) for index in order]
        else:
            seeds = []

        ranked: list[RankedCandidate] = []
        for candidate, semantic_score in zip(instance.candidates, semantic_scores, strict=True):
            target = f"candidate::{candidate.candidate_id}"
            path_scores: list[tuple[float, list[str]]] = []
            for seed_id, seed_score in seeds:
                path = graph.shortest_path(seed_id, target, self.config.hops + 1)
                if path:
                    distance = len(path) - 1
                    path_scores.append((seed_score * math.exp(-max(0, distance - 1)), path))
            path_scores.sort(key=lambda item: item[0], reverse=True)
            selected_paths = [path for _, path in path_scores[: self.config.max_paths_per_candidate]]
            graph_score = path_scores[0][0] if path_scores else 0.0
            evidence_nodes, evidence_edges = serialize_path_evidence(graph, selected_paths)
            rag_score = self.scorer.score(float(semantic_score), graph_score)
            ranked.append(
                RankedCandidate(
                    candidate_id=candidate.candidate_id,
                    text=candidate.text,
                    semantic_score=float(semantic_score),
                    graph_score=float(graph_score),
                    rag_score=float(rag_score),
                    evidence_nodes=evidence_nodes,
                    evidence_edges=evidence_edges,
                    paths=selected_paths,
                    graph_features={
                        "evidence_node_count": float(len(evidence_nodes)),
                        "evidence_edge_count": float(len(evidence_edges)),
                        "path_count": float(len(selected_paths)),
                        "best_path_hops": float(len(selected_paths[0]) - 1) if selected_paths else 0.0,
                    },
                    metadata=candidate.metadata,
                )
            )
        ranked.sort(key=lambda item: (-item.rag_score, item.candidate_id))
        top_k = min(self.config.top_k, len(ranked))
        ranked = ranked[:top_k]
        for rank, candidate in enumerate(ranked, start=1):
            candidate.rank = rank
        return RankingResult(
            query_id=instance.query.query_id,
            query_text=instance.query.text,
            positive_ids=instance.positive_ids,
            candidates=ranked,
            stage="graphrag",
            metadata={"candidate_count": len(instance.candidates), **instance.metadata},
        )
