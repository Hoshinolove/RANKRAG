import pytest
import torch

from rankrag.ranker.loss import listwise_ranking_loss
from rankrag.ranker.set_transformer import CandidateSetTransformer
from rankrag.ranker.tensor_dataset import create_tensor_loader
from rankrag.ranker.trainer import RankingMetricAccumulator


def test_set_transformer_uses_batched_candidate_lists():
    torch.manual_seed(13)
    model = CandidateSetTransformer(input_dim=12, hidden_dim=16, num_heads=4, num_layers=2, feedforward_dim=32, dropout=0.0)
    features = torch.randn(3, 5, 12)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
    scores, representations = model(features, mask)
    assert scores.shape == (3, 5)
    assert representations.shape == (3, 5, 16)
    assert torch.all(scores[~mask] < -1e20)


def test_listwise_loss_supports_multiple_positive_candidates():
    scores = torch.tensor([[2.0, 1.0, 0.0, 9.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    loss = listwise_ranking_loss(scores, labels, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None
    assert scores.grad[0, 3] == 0


def test_multi_worker_reads_every_shard_exactly_once_per_epoch(tmp_path):
    shards = []
    for shard_id in range(8):
        path = tmp_path / f"shard-{shard_id}.pt"
        torch.save(
            {
                "features": torch.tensor([[[float(shard_id)]]]),
                "labels": torch.zeros(1, 1),
                "mask": torch.ones(1, 1, dtype=torch.bool),
            },
            path,
        )
        shards.append({"tensor": path.name, "count": 1})
    manifest = {
        "_manifest_dir": str(tmp_path),
        "splits": {"train": {"shards": shards}},
    }
    loader = create_tensor_loader(
        manifest,
        split="train",
        batch_size=1,
        shuffle=True,
        seed=13,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
    )

    observed = [int(batch["features"][0, 0, 0]) for batch in loader]

    assert sorted(observed) == list(range(8))
    assert len(observed) == len(set(observed)) == 8


def test_no_positive_query_counts_as_zero_in_validation_metrics():
    accumulator = RankingMetricAccumulator([1, 2])
    scores = torch.tensor([[3.0, 2.0], [3.0, 2.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)

    accumulator.update(scores, labels, mask)
    metrics = accumulator.compute()

    assert metrics["queries"] == 2
    assert metrics["Recall@1"] == pytest.approx(0.5)
    assert metrics["NDCG@1"] == pytest.approx(0.5)
    assert metrics["Hit@1"] == pytest.approx(0.5)
    assert metrics["MRR"] == pytest.approx(0.5)
