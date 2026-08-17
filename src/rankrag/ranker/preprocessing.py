from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rankrag.data.factory import create_candidate_corpus
from rankrag.embedding import create_embedder
from rankrag.io import iter_results, write_json
from rankrag.ranker.features import RankerFeatureBuilder


def _split_for_query(query_id: str, validation_fraction: float, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{query_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if value < validation_fraction else "train"


class _ShardWriter:
    def __init__(self, root: Path, split: str, shard_size: int, dtype: torch.dtype) -> None:
        self.root = root
        self.split = split
        self.shard_size = shard_size
        self.dtype = dtype
        self.features: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []
        self.metadata: list[dict[str, Any]] = []
        self.shards: list[dict[str, Any]] = []
        self.count = 0
        (root / split).mkdir(parents=True, exist_ok=True)

    def add(self, features: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, metadata: dict[str, Any]) -> None:
        self.features.append(features.to(self.dtype))
        self.labels.append(labels.to(torch.float32))
        self.masks.append(mask.to(torch.bool))
        self.metadata.append(metadata)
        self.count += 1
        if len(self.features) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.features:
            return
        index = len(self.shards)
        tensor_rel = Path(self.split) / f"shard-{index:05d}.pt"
        metadata_rel = Path(self.split) / f"shard-{index:05d}.jsonl"
        tensor_path = self.root / tensor_rel
        temporary = tensor_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "features": torch.stack(self.features),
                "labels": torch.stack(self.labels),
                "mask": torch.stack(self.masks),
            },
            temporary,
        )
        os.replace(temporary, tensor_path)
        with (self.root / metadata_rel).open("w", encoding="utf-8") as handle:
            for item in self.metadata:
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        count = len(self.features)
        self.shards.append({"tensor": tensor_rel.as_posix(), "metadata": metadata_rel.as_posix(), "count": count})
        self.features.clear()
        self.labels.clear()
        self.masks.clear()
        self.metadata.clear()


def prepare_ranker_dataset(config: dict[str, Any]) -> Path:
    dataset_config = config.get("ranker_dataset", {})
    graph_path = Path(dataset_config["graphrag_path"])
    output_dir = Path(dataset_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_k = int(dataset_config.get("candidate_k", 100))
    shard_size = int(dataset_config.get("shard_size", 512))
    split_strategy = str(dataset_config.get("split_strategy", "hash"))
    validation_fraction = float(dataset_config.get("validation_fraction", 0.1))
    if split_strategy == "hash" and not 0.0 < validation_fraction < 1.0:
        raise ValueError("ranker_dataset.validation_fraction must be between 0 and 1")
    if split_strategy not in {"hash", "source"}:
        raise ValueError("ranker_dataset.split_strategy must be hash or source")
    seed = int(dataset_config.get("split_seed", 13))
    storage_dtype_name = dataset_config.get("storage_dtype", "float16")
    storage_dtype = {"float16": torch.float16, "float32": torch.float32}[storage_dtype_name]
    feature_builder = RankerFeatureBuilder(create_embedder(config.get("embedding", {})))
    global_config = config.get("global_retrieval", {})
    corpus = None
    corpus_embeddings = None
    if global_config.get("enabled", False):
        asset_dir = Path(global_config.get("asset_dir", "outputs/global_retrieval"))
        corpus_path = Path(global_config.get("corpus_path", asset_dir / "corpus.jsonl"))
        embeddings_path = Path(global_config.get("embeddings_path", asset_dir / "paragraph_embeddings.npy"))
        corpus = create_candidate_corpus(config)
        corpus_embeddings = np.load(embeddings_path, mmap_mode="r")
        if corpus_embeddings.shape != (len(corpus), feature_builder.embedder.dimension):
            raise ValueError("Offline corpus embeddings do not match the ranker embedder")
    split_names = (
        [str(value) for value in dataset_config.get("source_splits", ["train", "validation", "test"])]
        if split_strategy == "source"
        else ["train", "validation"]
    )
    writers = {split: _ShardWriter(output_dir, split, shard_size, storage_dtype) for split in split_names}
    total = skipped_no_positive = truncated = 0
    for result in iter_results(graph_path):
        candidate_embeddings = None
        if corpus is not None and corpus_embeddings is not None:
            try:
                rows = [corpus.row_for_id(candidate.candidate_id) for candidate in result.candidates]
            except KeyError as exc:
                raise KeyError(f"GraphRAG candidate is missing from the global candidate corpus: {exc.args[0]}") from exc
            candidate_embeddings = np.asarray(corpus_embeddings[rows])
        raw_features = feature_builder.build(result, candidate_embeddings=candidate_embeddings)
        valid_count = min(candidate_k, len(result.candidates))
        if len(result.candidates) > candidate_k:
            truncated += 1
        features = np.zeros((candidate_k, feature_builder.dimension), dtype=np.float32)
        features[:valid_count] = raw_features[:valid_count]
        labels = np.zeros(candidate_k, dtype=np.float32)
        positive_ids = set(result.positive_ids)
        candidate_ids = [candidate.candidate_id for candidate in result.candidates[:valid_count]]
        for index, candidate_id in enumerate(candidate_ids):
            labels[index] = float(candidate_id in positive_ids)
        mask = np.zeros(candidate_k, dtype=np.bool_)
        mask[:valid_count] = True
        if not labels.any():
            skipped_no_positive += 1
        if split_strategy == "source":
            split = str(result.metadata.get("split", ""))
            if split not in writers:
                raise ValueError(
                    f"GraphRAG result {result.query_id!r} has unknown source split {split!r}; "
                    f"expected one of {sorted(writers)}"
                )
        else:
            split = _split_for_query(result.query_id, validation_fraction, seed)
        writers[split].add(
            torch.from_numpy(features),
            torch.from_numpy(labels),
            torch.from_numpy(mask),
            {"query_id": result.query_id, "candidate_ids": candidate_ids},
        )
        total += 1
        if total % 1000 == 0:
            counts = " ".join(f"{name}={writer.count}" for name, writer in writers.items())
            print(f"prepared={total} {counts}", flush=True)
    for writer in writers.values():
        writer.flush()
    manifest = {
        "schema_version": 1,
        "source_graphrag": str(graph_path),
        "candidate_k": candidate_k,
        "feature_dim": feature_builder.dimension,
        "feature_layout": "[query,candidate,query*candidate,abs(query-candidate),graph_features]",
        "storage_dtype": storage_dtype_name,
        "split_strategy": split_strategy,
        "embedding": config.get("embedding", {}),
        "statistics": {
            "total_queries": total,
            "queries_without_positive_in_top_k": skipped_no_positive,
            "queries_truncated_to_k": truncated,
        },
        "splits": {
            split: {"count": writer.count, "shards": writer.shards}
            for split, writer in writers.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    mmap = getattr(corpus_embeddings, "_mmap", None)
    if mmap is not None:
        mmap.close()
    return manifest_path
