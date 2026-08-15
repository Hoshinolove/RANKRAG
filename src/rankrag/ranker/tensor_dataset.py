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
        rng = random.Random(self.seed + self._iteration * 1009 + worker_id)
        self._iteration += 1
        shards = self.shards[:]
        if self.shuffle:
            rng.shuffle(shards)
        shards = shards[worker_id::worker_count]
        for shard_info in shards:
            shard = torch.load(
                resolve_shard_path(self.manifest, shard_info["tensor"]),
                map_location="cpu",
                weights_only=True,
            )
            count = int(shard["features"].shape[0])
            indices = torch.randperm(count, generator=torch.Generator().manual_seed(rng.randrange(2**31))) if self.shuffle else torch.arange(count)
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
