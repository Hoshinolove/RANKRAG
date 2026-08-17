from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported ranker dataset schema: {manifest.get('schema_version')}")
    manifest["_manifest_dir"] = str(manifest_path.parent.resolve())
    return manifest


def resolve_shard_path(manifest: dict[str, Any], relative_path: str) -> Path:
    return Path(manifest["_manifest_dir"]) / relative_path


def load_split_query_ids(manifest: dict[str, Any], split: str) -> list[str]:
    """Load and validate the query order recorded by tensor shard sidecars."""
    if split not in manifest["splits"]:
        available = ", ".join(sorted(manifest["splits"]))
        raise ValueError(f"Unknown ranker split {split!r}; available splits: {available}")
    query_ids: list[str] = []
    for shard_info in manifest["splits"][split]["shards"]:
        metadata_path = resolve_shard_path(manifest, shard_info["metadata"])
        with metadata_path.open("r", encoding="utf-8") as handle:
            query_ids.extend(str(json.loads(line)["query_id"]) for line in handle if line.strip())
    expected_count = int(manifest["splits"][split]["count"])
    if len(query_ids) != expected_count:
        raise ValueError(
            f"Ranker split {split!r} declares {expected_count} queries but its sidecars contain {len(query_ids)}"
        )
    if len(set(query_ids)) != len(query_ids):
        raise ValueError(f"Ranker split {split!r} contains duplicate query IDs")
    return query_ids


class TensorShardBatchDataset(IterableDataset):
    """Loads one tensor shard at a time and yields already-batched query lists."""

    def __init__(
        self,
        manifest: dict[str, Any],
        split: str,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._iteration = 0
        self.shards = list(manifest["splits"][split]["shards"])

    def __len__(self) -> int:
        return sum(math.ceil(int(shard["count"]) / self.batch_size) for shard in self.shards)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        epoch = self._iteration
        self._iteration += 1

        # Every worker must start from the same epoch-level shard permutation.
        # Only after that common ordering is built may workers take disjoint slices.
        shard_rng = random.Random(self.seed + epoch * 1009)
        shards = self.shards[:]
        if self.shuffle:
            shard_rng.shuffle(shards)
        shards = shards[worker_id::worker_count]

        # Sample order is local to a worker and does not affect shard ownership.
        sample_rng = random.Random(self.seed + epoch * 1009 + worker_id * 1_000_003)
        for shard_info in shards:
            shard = torch.load(
                resolve_shard_path(self.manifest, shard_info["tensor"]),
                map_location="cpu",
                weights_only=True,
            )
            count = int(shard["features"].shape[0])
            indices = torch.randperm(
                count,
                generator=torch.Generator().manual_seed(sample_rng.randrange(2**31)),
            ) if self.shuffle else torch.arange(count)
            for start in range(0, count, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                yield {
                    "features": shard["features"][batch_indices],
                    "labels": shard["labels"][batch_indices],
                    "mask": shard["mask"][batch_indices],
                }


def create_tensor_loader(
    manifest: dict[str, Any],
    split: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    dataset = TensorShardBatchDataset(manifest, split, batch_size, shuffle, seed)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
