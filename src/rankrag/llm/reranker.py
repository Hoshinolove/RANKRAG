from __future__ import annotations

from rankrag.llm.base import LLMProvider, LLMRequest
from rankrag.llm.cache import LLMResponseCache
from rankrag.models import RankingResult


class LLMReranker:
    def __init__(
        self,
        provider: LLMProvider,
        cache: LLMResponseCache,
        top_k: int = 10,
        prompt_version: str = "v1",
        max_text_chars: int = 4000,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.top_k = top_k
        self.prompt_version = prompt_version
        self.max_text_chars = max_text_chars

    def rank(self, result: RankingResult) -> RankingResult:
        candidate_payload = [
            {
                "candidate_id": candidate.candidate_id,
                "text": candidate.text[: self.max_text_chars],
                "rag_score": candidate.rag_score,
                "neural_score": candidate.neural_score,
                "graph_evidence": {
                    "nodes": candidate.evidence_nodes,
                    "edges": candidate.evidence_edges,
                    "paths": candidate.paths,
                },
            }
            for candidate in result.candidates
        ]
        request = LLMRequest(result.query_id, result.query_text, candidate_payload, self.prompt_version)
        cache_key = self.cache.key(request, self.provider.identity)
        response = self.cache.get(cache_key)
        cache_hit = response is not None
        if response is None:
            response = self.provider.rerank(request)
            self.cache.put(cache_key, response)

        candidates_by_id = {candidate.candidate_id: candidate for candidate in result.candidates}
        ordered = []
        seen: set[str] = set()
        for item in response.ranking:
            candidate_id = str(item.get("candidate_id", ""))
            if candidate_id not in candidates_by_id or candidate_id in seen:
                continue
            candidate = candidates_by_id[candidate_id]
            try:
                candidate.llm_score = float(item.get("score", len(result.candidates) - len(ordered)))
            except (TypeError, ValueError):
                candidate.llm_score = float(len(result.candidates) - len(ordered))
            candidate.metadata = {**candidate.metadata, "llm_reason": item.get("reason")}
            ordered.append(candidate)
            seen.add(candidate_id)
        ordered.extend(candidate for candidate in result.candidates if candidate.candidate_id not in seen)
        ordered = ordered[: min(self.top_k, len(ordered))]
        for rank, candidate in enumerate(ordered, start=1):
            candidate.llm_rank = rank
        metadata = {**result.metadata, "llm_cache_key": cache_key, "llm_cache_hit": cache_hit, "llm_provider": self.provider.identity}
        return RankingResult(result.query_id, result.query_text, result.positive_ids, ordered, "llm", metadata)
