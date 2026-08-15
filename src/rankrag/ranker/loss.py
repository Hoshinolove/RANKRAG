from __future__ import annotations

import torch


def pointwise_bce_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum()
    negatives = labels.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).detach()
    return torch.nn.functional.binary_cross_entropy_with_logits(scores, labels, pos_weight=pos_weight)
