from __future__ import annotations

import math

import numpy as np

from rankrag.embedding import TextEmbedder
from rankrag.models import RankingResult


class RankerFeatureBuilder:
    NUMERIC_DIMENSION = 7

    def __init__(self, embedder: TextEmbedder) -> None:
        self.embedder = embedder

    @property
    def dimension(self) -> int:
        return self.embedder.dimension * 4 + self.NUMERIC_DIMENSION

    def build(
        self,
        result: RankingResult,
        candidate_embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        if not result.candidates:
            return np.empty((0, self.dimension), dtype=np.float32)
        query = self.embedder.encode([result.query_text])[0]
        if candidate_embeddings is None:
            candidate_texts = [
                f"{candidate.metadata.get('title', '')}\n{candidate.text}".strip()
                for candidate in result.candidates
            ]
            candidates = self.embedder.encode(candidate_texts)
        else:
            candidates = np.asarray(candidate_embeddings, dtype=np.float32)
            expected_shape = (len(result.candidates), self.embedder.dimension)
            if candidates.shape != expected_shape:
                raise ValueError(f"Expected candidate embeddings {expected_shape}, got {candidates.shape}")
        query_rows = np.repeat(query[None, :], len(candidates), axis=0)
        pair = np.concatenate([query_rows, candidates, query_rows * candidates, np.abs(query_rows - candidates)], axis=1)
        numeric = []
        for candidate in result.candidates:
            graph = candidate.graph_features
            numeric.append(
                [
                    candidate.semantic_score,
                    candidate.graph_score,
                    candidate.rag_score,
                    math.log1p(graph.get("evidence_node_count", 0.0)) / 5.0,
                    math.log1p(graph.get("evidence_edge_count", 0.0)) / 5.0,
                    math.log1p(graph.get("path_count", 0.0)) / 3.0,
                    graph.get("best_path_hops", 0.0) / 10.0,
                ]
            )
        return np.concatenate([pair, np.asarray(numeric, dtype=np.float32)], axis=1)
