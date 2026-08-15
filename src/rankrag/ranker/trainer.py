from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from rankrag.ranker.loss import listwise_ranking_loss, masked_pointwise_bce_loss
from rankrag.ranker.set_transformer import create_ranker_model
from rankrag.ranker.tensor_dataset import create_tensor_loader, load_manifest


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RankingMetricAccumulator:
    def __init__(self, ks: list[int]) -> None:
        self.ks = ks
        self.totals = {f"Recall@{k}": 0.0 for k in ks}
        self.totals.update({f"NDCG@{k}": 0.0 for k in ks})
        self.totals.update({f"Hit@{k}": 0.0 for k in ks})
        self.totals["MRR"] = 0.0
        self.count = 0

    def update(self, scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> None:
        scores = scores.detach().float().masked_fill(~mask, -1e30).cpu()
        labels = labels.detach().float().cpu()
        mask = mask.cpu()
        for row in range(scores.shape[0]):
            valid_count = int(mask[row].sum())
            positives = float(labels[row, :valid_count].sum())
            if not valid_count or not positives:
                continue
            order = torch.argsort(scores[row, :valid_count], descending=True)
            ranked_labels = labels[row, :valid_count][order]
            positive_positions = torch.nonzero(ranked_labels > 0, as_tuple=False)
            self.totals["MRR"] += 1.0 / (int(positive_positions[0]) + 1)
            for k in self.ks:
                cutoff = min(k, valid_count)
                top_labels = ranked_labels[:cutoff]
                hits = float(top_labels.sum())
                self.totals[f"Recall@{k}"] += hits / positives
                self.totals[f"Hit@{k}"] += float(hits > 0)
                discounts = 1.0 / torch.log2(torch.arange(2, cutoff + 2, dtype=torch.float32))
                dcg = float((top_labels * discounts).sum())
                ideal_count = min(cutoff, int(positives))
                idcg = float(discounts[:ideal_count].sum())
                self.totals[f"NDCG@{k}"] += dcg / idcg if idcg else 0.0
            self.count += 1

    def compute(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            key: value / self.count if self.count else 0.0
            for key, value in self.totals.items()
        }
        metrics["queries"] = self.count
        return metrics


def _autocast_context(device: str, enabled: bool, dtype_name: str):
    if not enabled or not device.startswith("cuda"):
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    model_config: dict[str, Any],
    feature_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model_config": model_config,
            "feature_dim": feature_dim,
        },
        temporary,
    )
    os.replace(temporary, path)


def train_tensor_ranker(config: dict[str, Any]) -> dict[str, Any]:
    dataset_config = config.get("ranker_dataset", {})
    manifest = load_manifest(dataset_config["manifest"])
    ranker_config = config.get("ranker", {})
    training = config.get("training", {})
    seed = int(training.get("seed", 13))
    set_global_seed(seed)
    device = str(ranker_config.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("ranker.device is cuda but CUDA is not available")
    model = create_ranker_model(ranker_config, int(manifest["feature_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    batch_size = int(training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 8))
    pin_memory = bool(training.get("pin_memory", True))
    persistent_workers = bool(training.get("persistent_workers", True))
    train_loader = create_tensor_loader(manifest, "train", batch_size, True, seed, num_workers, pin_memory, persistent_workers)
    validation_loader = create_tensor_loader(manifest, "validation", batch_size, False, seed, num_workers, pin_memory, persistent_workers)
    amp = bool(training.get("amp", True))
    amp_dtype = str(training.get("amp_dtype", "bfloat16"))
    scaler_enabled = amp and amp_dtype == "float16" and device.startswith("cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):  # PyTorch 2.0 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    loss_name = str(training.get("loss", "listwise"))
    epochs = int(training.get("epochs", 100))
    checkpoint_dir = Path(training["checkpoint_dir"])
    log_path = checkpoint_dir / "training_log.jsonl"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ndcg = -math.inf
    global_updates = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        update_count = 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            if not (amp and device.startswith("cuda")):
                features = features.float()
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp, amp_dtype):
                scores, _ = model(features, mask)
                loss = listwise_ranking_loss(scores, labels, mask) if loss_name == "listwise" else masked_pointwise_bce_loss(scores, labels, mask)
            scaler.scale(loss).backward()
            if float(training.get("max_grad_norm", 1.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("max_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            update_count += 1
            global_updates += 1

        model.eval()
        accumulator = RankingMetricAccumulator([5, 10])
        validation_loss = 0.0
        validation_batches = 0
        with torch.inference_mode():
            for batch in validation_loader:
                features = batch["features"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                if not (amp and device.startswith("cuda")):
                    features = features.float()
                with _autocast_context(device, amp, amp_dtype):
                    scores, _ = model(features, mask)
                    loss = listwise_ranking_loss(scores, labels, mask) if loss_name == "listwise" else masked_pointwise_bce_loss(scores, labels, mask)
                validation_loss += float(loss)
                validation_batches += 1
                accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        epoch_summary = {
            "epoch": epoch,
            "train_loss": loss_sum / max(update_count, 1),
            "validation_loss": validation_loss / max(validation_batches, 1),
            "updates_this_epoch": update_count,
            "global_updates": global_updates,
            "validation": metrics,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_summary, ensure_ascii=False) + "\n")
        print(json.dumps(epoch_summary, ensure_ascii=False), flush=True)
        _save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, metrics, ranker_config, int(manifest["feature_dim"]))
        ndcg10 = float(metrics.get("NDCG@10", 0.0))
        if ndcg10 > best_ndcg:
            best_ndcg = ndcg10
            _save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, metrics, ranker_config, int(manifest["feature_dim"]))
    return {"best_ndcg@10": best_ndcg, "global_updates": global_updates, "last_epoch": epochs, "checkpoint_dir": str(checkpoint_dir)}
