#!/usr/bin/env python3
"""Extract a paragraph KG with GLiNER entities and REBEL relations.

The script is designed for remote GPU workers. It streams the source JSON,
supports modulo sharding, and appends one valid JSON object per paragraph so a
worker can be restarted without repeating completed records.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rankrag.data.json_stream import iter_json_array  # noqa: E402


LOGGER = logging.getLogger("kg-extraction")
TRIPLET_MARKER = re.compile(r"<triplet>\s*(.*?)\s*<subj>\s*(.*?)\s*<obj>\s*(.*?)(?=<triplet>|$)", re.DOTALL)


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def paragraph_id(title: str, text: str) -> str:
    return hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()


def resolve_device(value: str) -> str:
    if value in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def load_models(config: dict[str, Any], device: str):
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError("Install requirements-kg.txt before running KG extraction") from exc
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-kg.txt before running KG extraction") from exc

    gliner_config = config.get("gliner", {})
    rebel_config = config.get("rebel", {})
    LOGGER.info("Loading GLiNER model: %s", gliner_config["model"])
    gliner = GLiNER.from_pretrained(gliner_config["model"])
    gliner.to(device)
    LOGGER.info("Loading REBEL model: %s", rebel_config["model"])
    tokenizer = AutoTokenizer.from_pretrained(rebel_config["model"])
    rebel = AutoModelForSeq2SeqLM.from_pretrained(rebel_config["model"])
    rebel.to(device)
    gliner.eval()
    rebel.eval()
    if config.get("dtype") == "float16" and device.startswith("cuda"):
        gliner.half()
        rebel.half()
    return gliner, tokenizer, rebel


def predict_entities(model: Any, texts: list[str], labels: list[str], threshold: float) -> list[list[dict[str, Any]]]:
    batch_method = getattr(model, "batch_predict_entities", None)
    if batch_method is not None:
        try:
            result = batch_method(texts, labels, threshold=threshold)
            return [list(items) for items in result]
        except (TypeError, AttributeError, RuntimeError):
            LOGGER.debug("GLiNER batch API unavailable; falling back to per-text inference", exc_info=True)
    return [list(model.predict_entities(text, labels, threshold=threshold)) for text in texts]


def predict_relations(
    tokenizer: Any,
    model: Any,
    texts: list[str],
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[list[dict[str, str]]]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generated = model.generate(**encoded, max_new_tokens=max_new_tokens, num_beams=1)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)
    results: list[list[dict[str, str]]] = []
    for output in decoded:
        triples = []
        for match in TRIPLET_MARKER.finditer(output):
            subject, relation, target = (part.strip() for part in match.groups())
            if subject and relation and target:
                triples.append({"subject": subject, "relation": relation, "object": target})
        results.append(triples)
    return results


def make_record(
    source: dict[str, Any],
    entities: list[dict[str, Any]],
    relations: list[dict[str, str]],
) -> dict[str, Any]:
    title = str(source.get("title", "")).strip()
    text = str(source.get("text", "")).strip()
    entity_map: dict[str, dict[str, Any]] = {}
    # Keeping the paragraph title first lets the existing GraphBuilder index
    # records by title without depending on model-specific entity ordering.
    entity_map[normalize(title)] = {
        "entity_name": title,
        "entity_type": "document",
        "description": text[:500],
    }
    for item in entities:
        name = str(item.get("text", "")).strip()
        if not name:
            continue
        entity_map.setdefault(
            normalize(name),
            {
                "entity_name": name,
                "entity_type": str(item.get("label", "entity")),
                "description": text[max(0, int(item.get("start", 0)) - 80) : int(item.get("end", 0)) + 80].strip(),
            },
        )
    clean_relations = []
    for relation in relations:
        subject = relation["subject"].strip()
        target = relation["object"].strip()
        if not subject or not target:
            continue
        entity_map.setdefault(normalize(subject), {"entity_name": subject, "entity_type": "relation_entity", "description": subject})
        entity_map.setdefault(normalize(target), {"entity_name": target, "entity_type": "relation_entity", "description": target})
        clean_relations.append(
            {
                "src_id": subject,
                "tgt_id": target,
                "description": relation["relation"],
                "keywords": relation["relation"],
            }
        )
    return {
        "id": paragraph_id(title, text),
        "title": title,
        "entities": list(entity_map.values()),
        "relationships": clean_relations,
    }


def iter_batches(records: Iterable[dict[str, Any]], batch_size: int):
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run(args: argparse.Namespace) -> None:
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    extraction_config = config.get("extraction", {})
    input_path = Path(extraction_config.get("input", "data/hotpot_train_paragraphs.json"))
    output_dir = Path(extraction_config.get("output_dir", "outputs/kg_train_gliner_rebel"))
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_index = args.shard_index
    num_shards = args.num_shards
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    filename = "kg_extractions.jsonl" if num_shards == 1 else f"kg_extractions.part-{shard_index:05d}.jsonl"
    output_path = output_dir / filename
    done_ids: set[str] = set()
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    done_ids.add(str(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    LOGGER.warning("Ignoring malformed completed line in %s", output_path)
        LOGGER.info("Resume: %d records already present in %s", len(done_ids), output_path)

    device = resolve_device(config.get("device", "auto"))
    gliner, tokenizer, rebel = load_models(config, device)
    gliner_config = config.get("gliner", {})
    rebel_config = config.get("rebel", {})
    labels = list(gliner_config.get("labels", ["person", "organization", "location", "product", "event", "work", "date"]))
    threshold = float(gliner_config.get("threshold", 0.35))
    batch_size = int(extraction_config.get("batch_size", 8))
    max_chars = int(extraction_config.get("max_chars", 6000))
    records = (item for index, item in enumerate(iter_json_array(input_path)) if index % num_shards == shard_index)
    processed = written = 0
    with output_path.open("a", encoding="utf-8", buffering=1) as output:
        for batch in iter_batches(records, batch_size):
            texts = [f"{item.get('title', '')}\n{str(item.get('text', ''))[:max_chars]}" for item in batch]
            with torch.inference_mode():
                entity_results = predict_entities(gliner, texts, labels, threshold)
                relation_results = predict_relations(
                    tokenizer,
                    rebel,
                    texts,
                    device,
                    int(rebel_config.get("max_input_tokens", 512)),
                    int(rebel_config.get("max_new_tokens", 128)),
                )
            for source, entities, relations in zip(batch, entity_results, relation_results, strict=True):
                record = make_record(source, entities, relations)
                processed += 1
                if record["id"] in done_ids:
                    continue
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                done_ids.add(record["id"])
                written += 1
            if processed % (batch_size * 10) == 0:
                LOGGER.info("shard=%d processed=%d written=%d", shard_index, processed, written)
    LOGGER.info("Completed shard=%d processed=%d written=%d output=%s", shard_index, processed, written, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract entities with GLiNER and relations with REBEL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
