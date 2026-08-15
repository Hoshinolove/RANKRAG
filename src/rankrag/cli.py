from __future__ import annotations

import argparse
import json
from pathlib import Path

from rankrag.embedding import create_embedder
from rankrag.evaluation.evaluator import evaluate_results
from rankrag.io import iter_results, load_config, write_json
from rankrag.pipeline.recommender import CascadePipeline
from rankrag.ranker.features import RankerFeatureBuilder
from rankrag.ranker.mlp import MLPRanker
from rankrag.ranker.training import set_seed, train_mlp


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
    parser = argparse.ArgumentParser(description="Train the MLP neural ranker")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    pipeline = CascadePipeline(config, args.config)
    if not pipeline.graphrag_path.exists():
        raise FileNotFoundError(f"Missing cached GraphRAG results: {pipeline.graphrag_path}")
    ranker_config = config.get("ranker", {})
    training = config.get("training", {})
    seed = int(training.get("seed", 13))
    set_seed(seed)
    feature_builder = RankerFeatureBuilder(create_embedder(config.get("embedding", {})))
    model = MLPRanker(feature_builder.dimension, int(ranker_config.get("hidden_dim", 256)), float(ranker_config.get("dropout", 0.1)))
    checkpoint = ranker_config.get("checkpoint") or str(pipeline.output_dir / "ranker.pt")
    summary = train_mlp(
        model,
        feature_builder,
        lambda: iter_results(pipeline.graphrag_path),
        checkpoint,
        epochs=int(training.get("epochs", 3)),
        learning_rate=float(training.get("learning_rate", 1e-3)),
        device=ranker_config.get("device", "cpu"),
        seed=seed,
    )
    print(json.dumps({"checkpoint": checkpoint, **summary}, indent=2))


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
