from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked_ids: Sequence[str], positive_ids: set[str], k: int) -> float:
    if not positive_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & positive_ids) / len(positive_ids)


def hit_at_k(ranked_ids: Sequence[str], positive_ids: set[str], k: int) -> float:
    return float(bool(set(ranked_ids[:k]) & positive_ids))


def reciprocal_rank(ranked_ids: Sequence[str], positive_ids: set[str]) -> float:
    for rank, candidate_id in enumerate(ranked_ids, start=1):
        if candidate_id in positive_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], positive_ids: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, candidate_id in enumerate(ranked_ids[:k], start=1) if candidate_id in positive_ids)
    ideal_count = min(k, len(positive_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0
