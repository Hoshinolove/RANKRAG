from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from rankrag.data.hotpotqa import HotpotQAAdapter
from rankrag.data.paragraph_corpus import ParagraphCorpus
from rankrag.embedding import create_embedder
from rankrag.evaluation.evaluator import evaluate_results
from rankrag.graph.builder import HotpotQAGraphBuilder, KGExtractionIndex
from rankrag.graphrag.retriever import GraphRAGRetriever, RetrievalConfig
from rankrag.graphrag.global_assets import GlobalGraphIndex, SemanticParagraphIndex
from rankrag.graphrag.global_retrieval import (
    HybridCandidateConfig,
    HybridCandidateGenerator,
    HybridGraphRAGRetriever,
)
from rankrag.graphrag.scorer import WeightedCandidateScorer
from rankrag.io import iter_results, write_json, write_jsonl
from rankrag.llm.cache import LLMResponseCache
from rankrag.llm.client import create_provider
from rankrag.llm.reranker import LLMReranker
from rankrag.ranker.tensor_inference import iter_tensor_rankings


class CascadePipeline:
    def __init__(self, config: dict[str, Any], config_path: str | Path | None = None) -> None:
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        dataset = config.get("dataset", {})
        if dataset.get("name", "hotpotqa") != "hotpotqa":
            raise ValueError("This release includes only the HotpotQA adapter")
        global_retrieval = config.get("global_retrieval", {})
        self.adapter = HotpotQAAdapter(
            dataset["path"],
            use_paragraph_ids=bool(global_retrieval.get("enabled", False)),
        )
        output = config.get("output", {})
        self.output_dir = Path(output.get("root", "outputs")) / dataset.get("name", "hotpotqa") / output.get("experiment", "baseline")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config_path:
            shutil.copyfile(self.config_path, self.output_dir / "config.yaml")
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = create_embedder(self.config.get("embedding", {}))
        return self._embedder

    @property
    def graphrag_path(self) -> Path:
        return self.output_dir / "graphrag.jsonl"

    @property
    def neural_path(self) -> Path:
        return self.output_dir / "neural.jsonl"

    def neural_split_path(self, split: str = "validation") -> Path:
        return self.neural_path if split == "validation" else self.output_dir / f"neural.{split}.jsonl"

    @property
    def llm_path(self) -> Path:
        return self.output_dir / "llm.jsonl"

    @property
    def retrieval_stats_path(self) -> Path:
        return self.output_dir / "graphrag_retrieval_stats.json"

    def run_graphrag(self, limit: int | None = None) -> Path:
        global_config = self.config.get("global_retrieval", {})
        if global_config.get("enabled", False):
            return self._run_global_graphrag(global_config, limit)
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

    def _run_global_graphrag(self, global_config: dict[str, Any], limit: int | None) -> Path:
        asset_dir = Path(global_config.get("asset_dir", "outputs/global_retrieval"))
        corpus_path = Path(global_config.get("corpus_path", asset_dir / "corpus.jsonl"))
        embeddings_path = Path(global_config.get("embeddings_path", asset_dir / "paragraph_embeddings.npy"))
        faiss_path = Path(global_config.get("faiss_index_path", asset_dir / "paragraphs.faiss"))
        graph_path = Path(global_config.get("graph_index_path", asset_dir / "global_graph.sqlite"))
        manifest_path = Path(global_config.get("manifest_path", asset_dir / "manifest.json"))
        for name, path in {
            "corpus": corpus_path,
            "embeddings": embeddings_path,
            "FAISS index": faiss_path,
            "global graph": graph_path,
            "asset manifest": manifest_path,
        }.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing offline {name}: {path}. Run prepare_global_retrieval.py first")
        asset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if asset_manifest.get("embedding") != self.config.get("embedding", {}):
            raise ValueError("Offline embedding configuration differs from the query embedder; rebuild global assets")
        corpus = ParagraphCorpus.load(corpus_path)
        semantic_index = SemanticParagraphIndex(
            corpus,
            embeddings_path,
            faiss_path,
            backend=str(global_config.get("index_backend", "faiss")),
        )
        graph_index = GlobalGraphIndex(graph_path)
        candidate_generator = HybridCandidateGenerator(
            corpus,
            semantic_index,
            graph_index,
            self.embedder,
            HybridCandidateConfig(
                semantic_top_k=int(global_config.get("semantic_top_k", 500)),
                seed_paragraph_k=int(global_config.get("seed_paragraph_k", 20)),
                graph_hops=min(2, int(global_config.get("graph_hops", 2))),
                max_graph_candidates=int(global_config.get("max_graph_candidates", 1000)),
                max_entities_per_hop=int(global_config.get("max_entities_per_hop", 500)),
                max_relations_per_entity=int(global_config.get("max_relations_per_entity", 100)),
                max_paragraphs_per_entity=int(global_config.get("max_paragraphs_per_entity", 100)),
                max_paths_per_candidate=int(global_config.get("max_paths_per_candidate", 3)),
            ),
        )
        retrieval = self.config.get("retrieval", {})
        retriever = HybridGraphRAGRetriever(
            candidate_generator,
            WeightedCandidateScorer(float(retrieval.get("semantic_weight", 0.5)), float(retrieval.get("graph_weight", 0.5))),
            top_k=int(retrieval.get("top_k", 100)),
        )
        totals = {
            "queries": 0,
            "candidate_pool_size": 0.0,
            "gold_recall_before_graphrag": 0.0,
            "Recall@100": 0.0,
            "graph_expansion_candidate_count": 0.0,
            "retrieval_time_ms": 0.0,
        }

        def iter_ranked():
            for instance in self.adapter.iter_instances(limit):
                result = retriever.rank(instance)
                totals["queries"] += 1
                totals["candidate_pool_size"] += float(result.metadata["candidate_pool_size"])
                totals["gold_recall_before_graphrag"] += float(result.metadata["gold_recall_before_graphrag"])
                totals["Recall@100"] += float(result.metadata["recall_at_100"])
                totals["graph_expansion_candidate_count"] += float(result.metadata["graph_expansion_candidate_count"])
                totals["retrieval_time_ms"] += float(result.metadata["retrieval_time_ms"])
                yield result

        write_jsonl(self.graphrag_path, iter_ranked())
        semantic_index.close()
        graph_index.close()
        count = int(totals["queries"])
        summary = {
            "queries": count,
            "average_candidate_pool_size": totals["candidate_pool_size"] / count if count else 0.0,
            "gold_recall_before_graphrag": totals["gold_recall_before_graphrag"] / count if count else 0.0,
            "Recall@100": totals["Recall@100"] / count if count else 0.0,
            "average_graph_expansion_candidate_count": totals["graph_expansion_candidate_count"] / count if count else 0.0,
            "average_retrieval_time_ms": totals["retrieval_time_ms"] / count if count else 0.0,
            "total_retrieval_time_seconds": totals["retrieval_time_ms"] / 1000.0,
        }
        write_json(self.retrieval_stats_path, summary)
        return self.graphrag_path

    def run_neural(self, split: str | None = None) -> Path:
        split = split or str(self.config.get("ranker", {}).get("inference_split", "validation"))
        if not self.graphrag_path.exists():
            raise FileNotFoundError(f"Run the GraphRAG stage first: {self.graphrag_path}")
        manifest = self.config.get("ranker_dataset", {}).get("manifest")
        if not manifest:
            raise ValueError("Neural inference requires ranker_dataset.manifest; run prepare_ranker_dataset.py first")
        if not Path(manifest).exists():
            raise FileNotFoundError(f"Ranker tensor manifest not found: {manifest}. Run prepare_ranker_dataset.py first")
        output_path = self.neural_split_path(split)
        write_jsonl(output_path, iter_tensor_rankings(self.config, split=split))
        return output_path

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
