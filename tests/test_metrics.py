import pytest

from rankrag.evaluation.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank


def test_binary_ranking_metrics():
    ranked = ["negative", "positive-a", "positive-b"]
    positives = {"positive-a", "positive-b"}
    assert recall_at_k(ranked, positives, 2) == 0.5
    assert hit_at_k(ranked, positives, 1) == 0.0
    assert reciprocal_rank(ranked, positives) == 0.5
    assert ndcg_at_k(ranked, positives, 3) == pytest.approx((1 / 1.5849625 + 1 / 2) / (1 + 1 / 1.5849625))
