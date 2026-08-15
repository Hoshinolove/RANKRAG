from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from rankrag.models import RankingResult
from rankrag.ranker.base import RankerModel
from rankrag.ranker.features import RankerFeatureBuilder
from rankrag.ranker.interaction import CandidateInteraction, IdentityInteraction


class MLPRanker(RankerModel):
    """Trainable pointwise baseline with an explicit interaction extension point."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.rag_feature_index = input_dim - RankerFeatureBuilder.NUMERIC_DIMENSION + 2

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encoder(features)
        learned_residual = self.head(representation).squeeze(-1)
        baseline = features[:, self.rag_feature_index]
        return baseline + learned_residual, representation


class NeuralReranker:
    def __init__(
        self,
        model: RankerModel,
        feature_builder: RankerFeatureBuilder,
        top_k: int = 20,
        interaction: CandidateInteraction | None = None,
        device: str = "cpu",
        representation_output_dim: int = 16,
    ) -> None:
        self.model = model.to(device)
        self.feature_builder = feature_builder
        self.top_k = top_k
        self.interaction = (interaction or IdentityInteraction()).to(device)
        self.device = device
        self.representation_output_dim = representation_output_dim

    def rank(self, result: RankingResult) -> RankingResult:
        if not result.candidates:
            return RankingResult(result.query_id, result.query_text, result.positive_ids, [], "neural", result.metadata)
        features = torch.from_numpy(self.feature_builder.build(result)).to(self.device)
        self.model.eval()
        self.interaction.eval()
        with torch.no_grad():
            scores, representations = self.model(features)
            representations = self.interaction(representations)
        for candidate, score, representation in zip(result.candidates, scores.cpu(), representations.cpu(), strict=True):
            candidate.neural_score = float(score)
            candidate.intermediate_representation = [float(value) for value in representation[: self.representation_output_dim]]
        candidates = sorted(result.candidates, key=lambda item: (-float(item.neural_score), item.candidate_id))
        candidates = candidates[: min(self.top_k, len(candidates))]
        for rank, candidate in enumerate(candidates, start=1):
            candidate.neural_rank = rank
        return RankingResult(result.query_id, result.query_text, result.positive_ids, candidates, "neural", result.metadata)


def save_checkpoint(model: MLPRanker, path: str | Path, metadata: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, destination)


def load_checkpoint(model: MLPRanker, path: str | Path, device: str = "cpu") -> dict:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return dict(checkpoint.get("metadata", {}))
