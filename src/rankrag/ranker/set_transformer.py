from __future__ import annotations

import torch
from torch import nn

from rankrag.ranker.base import RankerModel


NUMERIC_FEATURE_DIMENSION = 7


class CandidateSetTransformer(RankerModel):
    """Ranks a candidate list with candidate-to-candidate self-attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim))
        self.scoring_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected [B,K,D] features, got {tuple(features.shape)}")
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        representation = self.input_projection(features)
        representation = self.encoder(representation, src_key_padding_mask=~mask)
        scores = self.scoring_head(representation).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        return scores, representation


class PointwiseMLPSetRanker(RankerModel):
    """Pointwise MLP baseline with no candidate interaction."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.scoring_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.scoring_head.weight)
        nn.init.zeros_(self.scoring_head.bias)
        self.rag_feature_index = input_dim - NUMERIC_FEATURE_DIMENSION + 2

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encoder(features)
        scores = features[..., self.rag_feature_index] + self.scoring_head(representation).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        return scores, representation


def create_ranker_model(config: dict, input_dim: int) -> RankerModel:
    model_name = config.get("model", "set_transformer")
    if model_name == "set_transformer":
        return CandidateSetTransformer(
            input_dim=input_dim,
            hidden_dim=int(config.get("hidden_dim", 256)),
            num_heads=int(config.get("num_heads", 8)),
            num_layers=int(config.get("num_layers", 3)),
            feedforward_dim=int(config.get("feedforward_dim", 1024)),
            dropout=float(config.get("dropout", 0.1)),
        )
    if model_name == "mlp":
        return PointwiseMLPSetRanker(
            input_dim=input_dim,
            hidden_dim=int(config.get("hidden_dim", 256)),
            dropout=float(config.get("dropout", 0.1)),
        )
    raise ValueError(f"Unknown ranker model: {model_name}")
