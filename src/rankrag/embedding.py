from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import re
from typing import Sequence

import numpy as np


class TextEmbedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class HashingEmbedder(TextEmbedder):
    """Deterministic CPU baseline suitable for offline development and tests."""

    TOKEN_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)

    def __init__(self, dimension: int = 256, ngrams: int = 2) -> None:
        self._dimension = dimension
        self.ngrams = ngrams

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [token.lower() for token in self.TOKEN_PATTERN.findall(text)]
            features = tokens[:]
            if self.ngrams >= 2:
                features.extend(f"{left}::{right}" for left, right in zip(tokens, tokens[1:], strict=False))
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                matrix[row, value % self.dimension] += 1.0 if value & 1 else -1.0
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


class SentenceTransformerEmbedder(TextEmbedder):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the 'embeddings' extra to use sentence_transformer") from exc
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = int(self.model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


def create_embedder(config: dict) -> TextEmbedder:
    backend = config.get("backend", "hashing")
    if backend == "hashing":
        return HashingEmbedder(int(config.get("dimension", 256)), int(config.get("ngrams", 2)))
    if backend == "sentence_transformer":
        return SentenceTransformerEmbedder(config["model"], config.get("device", "cpu"))
    raise ValueError(f"Unknown embedding backend: {backend}")
