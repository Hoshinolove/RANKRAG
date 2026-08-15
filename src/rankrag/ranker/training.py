from __future__ import annotations

from pathlib import Path
import random
from typing import Callable, Iterable

import numpy as np
import torch

from rankrag.models import RankingResult
from rankrag.ranker.features import RankerFeatureBuilder
from rankrag.ranker.loss import pointwise_bce_loss
from rankrag.ranker.mlp import MLPRanker, save_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_mlp(
    model: MLPRanker,
    feature_builder: RankerFeatureBuilder,
    results_factory: Callable[[], Iterable[RankingResult]],
    checkpoint_path: str | Path,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    seed: int = 13,
) -> dict[str, float]:
    set_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    final_loss = 0.0
    updates = 0
    for _ in range(epochs):
        model.train()
        for result in results_factory():
            if not result.candidates:
                continue
            positive_ids = set(result.positive_ids)
            labels = torch.tensor(
                [1.0 if candidate.candidate_id in positive_ids else 0.0 for candidate in result.candidates],
                dtype=torch.float32,
                device=device,
            )
            if not labels.any():
                continue
            features = torch.from_numpy(feature_builder.build(result)).to(device)
            optimizer.zero_grad(set_to_none=True)
            scores, _ = model(features)
            loss = pointwise_bce_loss(scores, labels)
            loss.backward()
            optimizer.step()
            final_loss += float(loss.detach())
            updates += 1
    summary = {"mean_training_loss": final_loss / max(updates, 1), "updates": float(updates), "epochs": float(epochs)}
    save_checkpoint(model, checkpoint_path, summary)
    return summary
