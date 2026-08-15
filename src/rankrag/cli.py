from __future__ import annotations

import argparse
import json

from rankrag.io import load_config
from rankrag.pipeline.recommender import CascadePipeline
from rankrag.ranker.preprocessing import prepare_ranker_dataset
from rankrag.ranker.trainer import train_tensor_ranker


def pipeline_main() -> None:
    parser = argparse.ArgumentParser(description="Run the RankRAG cascade")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["all", "graphrag", "neural", "llm"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    pipeline = CascadePipeline(config, args.config)
    stages = ["graphrag", "neural", "llm"] if args.stage == "all" else [args.stage]
    for stage in stages:
        output_path = getattr(pipeline, f"{stage}_path")
        if output_path.exists() and not args.force:
            print(f"Reusing {stage}: {output_path}")
            continue
        if stage == "graphrag":
            pipeline.run_graphrag(args.limit)
        elif stage == "neural":
            pipeline.run_neural()
        else:
            pipeline.run_llm()
        print(f"Wrote {stage}: {output_path}")
    print(json.dumps(pipeline.evaluate(), ensure_ascii=False, indent=2))


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural ranker from precomputed tensor shards")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if not config.get("ranker_dataset", {}).get("manifest"):
        raise ValueError("ranker_dataset.manifest is required; run prepare_ranker_dataset.py first")
    print(json.dumps(train_tensor_ranker(config), ensure_ascii=False, indent=2))


def prepare_ranker_dataset_main() -> None:
    parser = argparse.ArgumentParser(description="Precompute ranker features into tensor shards")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    manifest = prepare_ranker_dataset(load_config(args.config))
    print(json.dumps({"manifest": str(manifest)}, ensure_ascii=False, indent=2))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cached cascade rankings")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["all", "graphrag", "neural", "llm"], default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    pipeline = CascadePipeline(config, args.config)
    metrics = pipeline.evaluate()
    if args.stage != "all":
        metrics = {args.stage: metrics.get(args.stage, {})}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
