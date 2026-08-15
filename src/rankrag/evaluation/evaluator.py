from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from rankrag.evaluation.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from rankrag.models import RankingResult


def evaluate_results(results: Iterable[RankingResult], ks: Sequence[int]) -> dict[str, float | int]:
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for result in results:
        ranked_ids = [candidate.candidate_id for candidate in result.candidates]
        positives = set(result.positive_ids)
        for k in ks:
            totals[f"Recall@{k}"] += recall_at_k(ranked_ids, positives, k)
            totals[f"NDCG@{k}"] += ndcg_at_k(ranked_ids, positives, k)
            totals[f"Hit@{k}"] += hit_at_k(ranked_ids, positives, k)
        totals["MRR"] += reciprocal_rank(ranked_ids, positives)
        count += 1
    metrics: dict[str, float | int] = {key: value / count if count else 0.0 for key, value in sorted(totals.items())}
    metrics["queries"] = count
    return metrics
