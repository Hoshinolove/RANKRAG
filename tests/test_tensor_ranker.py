import torch

from rankrag.ranker.loss import listwise_ranking_loss
from rankrag.ranker.set_transformer import CandidateSetTransformer


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
