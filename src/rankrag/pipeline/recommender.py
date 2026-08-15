from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from rankrag.data.hotpotqa import HotpotQAAdapter
from rankrag.embedding import create_embedder
from rankrag.evaluation.evaluator import evaluate_results
from rankrag.graph.builder import HotpotQAGraphBuilder, KGExtractionIndex
from rankrag.graphrag.retriever import GraphRAGRetriever, RetrievalConfig
from rankrag.graphrag.scorer import WeightedCandidateScorer
from rankrag.io import iter_results, write_json, write_jsonl
from rankrag.llm.cache import LLMResponseCache
from rankrag.llm.client import create_provider
from rankrag.llm.reranker import LLMReranker
from rankrag.ranker.features import RankerFeatureBuilder
from rankrag.ranker.mlp import MLPRanker, NeuralReranker, load_checkpoint


class CascadePipeline:
    def __init__(self, config: dict[str, Any], config_path: str | Path | None = None) -> None:
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        dataset = config.get("dataset", {})
        if dataset.get("name", "hotpotqa") != "hotpotqa":
            raise ValueError("This release includes only the HotpotQA adapter")
        self.adapter = HotpotQAAdapter(dataset["path"])
        output = config.get("output", {})
        self.output_dir = Path(output.get("root", "outputs")) / dataset.get("name", "hotpotqa") / output.get("experiment", "baseline")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config_path:
            shutil.copyfile(self.config_path, self.output_dir / "config.yaml")
        self.embedder = create_embedder(config.get("embedding", {}))

    @property
    def graphrag_path(self) -> Path:
        return self.output_dir / "graphrag.jsonl"

    @property
    def neural_path(self) -> Path:
        return self.output_dir / "neural.jsonl"

    @property
    def llm_path(self) -> Path:
        return self.output_dir / "llm.jsonl"

    def run_graphrag(self, limit: int | None = None) -> Path:
        graph_config = self.config.get("graph", {})
        index = KGExtractionIndex(graph_config.get("extractions_path"))
        titles: set[str] = set()
        for instance in self.adapter.iter_instances(limit):
            titles.update(candidate.candidate_id for candidate in instance.candidates)
        index.load_for_titles(titles)
        retrieval = self.config.get("retrieval", {})
        retriever = GraphRAGRetriever(
            self.embedder,
            HotpotQAGraphBuilder(
                index,
                lexical_fallback=bool(graph_config.get("lexical_fallback", True)),
                max_fallback_terms=int(graph_config.get("max_fallback_terms", 10)),
            ),
            WeightedCandidateScorer(float(retrieval.get("semantic_weight", 0.5)), float(retrieval.get("graph_weight", 0.5))),
            RetrievalConfig(
                top_k=int(retrieval.get("top_k", 100)),
                hops=int(retrieval.get("hops", 2)),
                seed_k=int(retrieval.get("seed_k", 8)),
                max_paths_per_candidate=int(retrieval.get("max_paths_per_candidate", 3)),
            ),
        )
        write_jsonl(self.graphrag_path, (retriever.rank(instance) for instance in self.adapter.iter_instances(limit)))
        return self.graphrag_path

    def _build_neural_reranker(self) -> NeuralReranker:
        ranker_config = self.config.get("ranker", {})
        feature_builder = RankerFeatureBuilder(self.embedder)
        if ranker_config.get("model", "mlp") != "mlp":
            raise ValueError("Only the mlp ranker is included in this release")
        model = MLPRanker(feature_builder.dimension, int(ranker_config.get("hidden_dim", 256)), float(ranker_config.get("dropout", 0.1)))
        checkpoint = ranker_config.get("checkpoint")
        # After train.py, reuse the checkpoint in the current experiment even
        # when the YAML intentionally leaves ranker.checkpoint as null.
        if not checkpoint:
            default_checkpoint = self.output_dir / "ranker.pt"
            if default_checkpoint.exists():
                checkpoint = str(default_checkpoint)
        if checkpoint:
            if not Path(checkpoint).exists():
                raise FileNotFoundError(f"Ranker checkpoint not found: {checkpoint}")
            load_checkpoint(model, checkpoint, ranker_config.get("device", "cpu"))
        return NeuralReranker(
            model,
            feature_builder,
            top_k=int(ranker_config.get("top_k", 20)),
            device=ranker_config.get("device", "cpu"),
            representation_output_dim=int(ranker_config.get("representation_output_dim", 16)),
        )

    def run_neural(self) -> Path:
        if not self.graphrag_path.exists():
            raise FileNotFoundError(f"Run the GraphRAG stage first: {self.graphrag_path}")
        reranker = self._build_neural_reranker()
        write_jsonl(self.neural_path, (reranker.rank(result) for result in iter_results(self.graphrag_path)))
        return self.neural_path

    def run_llm(self) -> Path:
        if not self.neural_path.exists():
            raise FileNotFoundError(f"Run the neural stage first: {self.neural_path}")
        llm_config = self.config.get("llm", {})
        cache_dir = self.output_dir / llm_config.get("cache_dir", "llm_cache")
        reranker = LLMReranker(
            create_provider(llm_config),
            LLMResponseCache(cache_dir),
            top_k=int(llm_config.get("top_k", 10)),
            prompt_version=llm_config.get("prompt_version", "v1"),
            max_text_chars=int(llm_config.get("max_text_chars", 4000)),
        )
        write_jsonl(self.llm_path, (reranker.rank(result) for result in iter_results(self.neural_path)))
        return self.llm_path

    def evaluate(self) -> dict[str, dict[str, float | int]]:
        ks = [int(k) for k in self.config.get("evaluation", {}).get("ks", [5, 10])]
        metrics = {}
        for stage, path in (("graphrag", self.graphrag_path), ("neural", self.neural_path), ("llm", self.llm_path)):
            if path.exists():
                metrics[stage] = evaluate_results(iter_results(path), ks)
        write_json(self.output_dir / "metrics.json", metrics)
        return metrics
