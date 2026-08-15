from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch

from rankrag.models import RankingResult
from rankrag.ranker.set_transformer import create_ranker_model
from rankrag.ranker.tensor_dataset import load_manifest, resolve_shard_path


def _build_query_offsets(path: Path) -> dict[str, int]:
    offsets: dict[str, int] = {}
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            value = json.loads(line)
            offsets[str(value["query_id"])] = offset
    return offsets


def _read_result_at(handle, offset: int) -> RankingResult:
    handle.seek(offset)
    return RankingResult.from_dict(json.loads(handle.readline()))


def _checkpoint_path(config: dict[str, Any]) -> Path:
    ranker = config.get("ranker", {})
    if ranker.get("checkpoint"):
        return Path(ranker["checkpoint"])
    checkpoint_dir = Path(config.get("training", {})["checkpoint_dir"])
    best = checkpoint_dir / "best.pt"
    return best if best.exists() else checkpoint_dir / "last.pt"


def iter_tensor_rankings(config: dict[str, Any]) -> Iterator[RankingResult]:
    dataset_config = config["ranker_dataset"]
    manifest = load_manifest(dataset_config["manifest"])
    graph_path = Path(manifest["source_graphrag"])
    checkpoint_path = _checkpoint_path(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Ranker checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = dict(checkpoint.get("model_config", config.get("ranker", {})))
    feature_dim = int(checkpoint.get("feature_dim", manifest["feature_dim"]))
    if feature_dim != int(manifest["feature_dim"]):
        raise ValueError("Checkpoint feature dimension does not match ranker dataset")
    device = str(config.get("ranker", {}).get("device", "cuda"))
    model = create_ranker_model(model_config, feature_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    top_k = int(config.get("ranker", {}).get("top_k", 20))
    representation_dim = int(config.get("ranker", {}).get("representation_output_dim", 16))
    batch_size = int(config.get("training", {}).get("batch_size", 256))
    amp = bool(config.get("training", {}).get("amp", True)) and device.startswith("cuda")
    amp_dtype = torch.bfloat16 if config.get("training", {}).get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16
    offsets = _build_query_offsets(graph_path)
    with graph_path.open("rb") as graph_handle, torch.inference_mode():
        for split in ("train", "validation"):
            for shard_info in manifest["splits"][split]["shards"]:
                shard = torch.load(resolve_shard_path(manifest, shard_info["tensor"]), map_location="cpu", weights_only=True)
                metadata_path = resolve_shard_path(manifest, shard_info["metadata"])
                with metadata_path.open("r", encoding="utf-8") as metadata_handle:
                    metadata = [json.loads(line) for line in metadata_handle]
                for start in range(0, len(metadata), batch_size):
                    end = min(start + batch_size, len(metadata))
                    features = shard["features"][start:end].to(device)
                    mask = shard["mask"][start:end].to(device)
                    if not amp:
                        features = features.float()
                    context = torch.autocast("cuda", dtype=amp_dtype) if amp else torch.autocast("cpu", enabled=False)
                    with context:
                        scores, representations = model(features, mask)
                    scores = scores.float().cpu()
                    representations = representations.float().cpu()
                    for row, item in enumerate(metadata[start:end]):
                        query_id = str(item["query_id"])
                        if query_id not in offsets:
                            raise KeyError(f"Query {query_id} is missing from GraphRAG cache")
                        result = _read_result_at(graph_handle, offsets[query_id])
                        candidate_ids = list(item["candidate_ids"])
                        candidates_by_id = {candidate.candidate_id: candidate for candidate in result.candidates}
                        candidates = []
                        for candidate_index, candidate_id in enumerate(candidate_ids):
                            candidate = candidates_by_id[candidate_id]
                            candidate.neural_score = float(scores[row, candidate_index])
                            candidate.intermediate_representation = [
                                float(value) for value in representations[row, candidate_index, :representation_dim]
                            ]
                            candidates.append(candidate)
                        candidates.sort(key=lambda candidate: (-float(candidate.neural_score), candidate.candidate_id))
                        candidates = candidates[: min(top_k, len(candidates))]
                        for rank, candidate in enumerate(candidates, start=1):
                            candidate.neural_rank = rank
                        yield RankingResult(
                            query_id=result.query_id,
                            query_text=result.query_text,
                            positive_ids=result.positive_ids,
                            candidates=candidates,
                            stage="neural",
                            metadata={**result.metadata, "ranker_checkpoint": str(checkpoint_path), "ranker_dataset": dataset_config["manifest"]},
                        )
