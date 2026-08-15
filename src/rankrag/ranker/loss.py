from __future__ import annotations

import torch


def pointwise_bce_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum()
    negatives = labels.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).detach()
    return torch.nn.functional.binary_cross_entropy_with_logits(scores, labels, pos_weight=pos_weight)


def masked_pointwise_bce_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_scores = scores[mask]
    valid_labels = labels[mask]
    return pointwise_bce_loss(valid_scores, valid_labels)


def listwise_ranking_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross entropy between score softmax and uniform mass over positives."""
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    positive_count = (labels * mask).sum(dim=1, keepdim=True)
    valid_queries = positive_count.squeeze(1) > 0
    if not valid_queries.any():
        return scores.sum() * 0.0
    targets = labels / positive_count.clamp_min(1.0)
    log_probabilities = torch.log_softmax(masked_scores, dim=1)
    losses = -(targets * log_probabilities).sum(dim=1)
    return losses[valid_queries].mean()
