from __future__ import annotations

import hashlib
from itertools import islice
import json
import os
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
    create_graph_expansion_executor,
    default_graph_workers,
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

    def run_graphrag(self, limit: int | None = None, force: bool = False) -> Path:
        global_config = self.config.get("global_retrieval", {})
        if global_config.get("enabled", False):
            return self._run_global_graphrag(global_config, limit, force)
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

    def _run_global_graphrag(
        self,
        global_config: dict[str, Any],
        limit: int | None,
        force: bool = False,
    ) -> Path:
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
        candidate_config = HybridCandidateConfig(
            semantic_top_k=int(global_config.get("semantic_top_k", 500)),
            seed_paragraph_k=int(global_config.get("seed_paragraph_k", 20)),
            graph_hops=min(2, int(global_config.get("graph_hops", 2))),
            max_graph_candidates=int(global_config.get("max_graph_candidates", 1000)),
            max_entities_per_hop=int(global_config.get("max_entities_per_hop", 500)),
            max_relations_per_entity=int(global_config.get("max_relations_per_entity", 100)),
            max_paragraphs_per_entity=int(global_config.get("max_paragraphs_per_entity", 100)),
            max_paths_per_candidate=int(global_config.get("max_paths_per_candidate", 3)),
        )
        candidate_generator = HybridCandidateGenerator(
            corpus,
            semantic_index,
            graph_index,
            self.embedder,
            candidate_config,
        )
        retrieval = self.config.get("retrieval", {})
        retriever = HybridGraphRAGRetriever(
            candidate_generator,
            WeightedCandidateScorer(float(retrieval.get("semantic_weight", 0.5)), float(retrieval.get("graph_weight", 0.5))),
            top_k=int(retrieval.get("top_k", 100)),
        )
        query_batch_size = max(1, int(global_config.get("query_batch_size", 256)))
        output_shard_size = max(query_batch_size, int(global_config.get("output_shard_size", 1024)))
        raw_graph_workers = global_config.get("graph_workers")
        graph_workers = default_graph_workers() if raw_graph_workers is None else max(1, int(raw_graph_workers))
        shard_dir = self.output_dir / "graphrag_shards"
        shard_manifest_path = shard_dir / "manifest.json"
        dataset_path = Path(self.config["dataset"]["path"])
        dataset_stat = dataset_path.stat()
        signature_payload = {
            "pipeline_schema_version": 2,
            "dataset": {
                "path": str(dataset_path.resolve()),
                "size": dataset_stat.st_size,
                "mtime_ns": dataset_stat.st_mtime_ns,
            },
            "embedding": self.config.get("embedding", {}),
            "global_retrieval": global_config,
            "retrieval": retrieval,
            "asset_manifest": asset_manifest,
            "limit": limit,
            "output_shard_size": output_shard_size,
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if force:
            self.graphrag_path.unlink(missing_ok=True)
            if shard_dir.exists():
                for path in shard_dir.glob("part-*.jsonl"):
                    path.unlink()
                for path in shard_dir.glob("part-*.jsonl.tmp"):
                    path.unlink()
                shard_manifest_path.unlink(missing_ok=True)
        elif self.graphrag_path.exists():
            semantic_index.close()
            graph_index.close()
            return self.graphrag_path

        shard_dir.mkdir(parents=True, exist_ok=True)
        if shard_manifest_path.exists():
            shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
            if shard_manifest.get("signature") != signature:
                semantic_index.close()
                graph_index.close()
                raise ValueError(
                    "Existing GraphRAG shards were created from different data, assets, or configuration. "
                    "Run the GraphRAG stage with --force to start a clean run."
                )
        else:
            if any(shard_dir.glob("part-*.jsonl")):
                semantic_index.close()
                graph_index.close()
                raise ValueError("GraphRAG shard manifest is missing; use --force to discard unverified shards")
            shard_manifest = {
                "schema_version": 1,
                "signature": signature,
                "query_batch_size": query_batch_size,
                "output_shard_size": output_shard_size,
                "graph_workers": graph_workers,
                "complete": False,
            }
            write_json(shard_manifest_path, shard_manifest)

        timing_fields = (
            "query_embedding_time_ms",
            "semantic_search_time_ms",
            "graph_expansion_time_ms",
            "evidence_serialization_time_ms",
            "total_query_time_ms",
        )
        totals = {
            "queries": 0,
            "candidate_pool_size": 0.0,
            "gold_recall_before_graphrag": 0.0,
            "Recall@100": 0.0,
            "graph_expansion_candidate_count": 0.0,
            "retrieval_time_ms": 0.0,
            **{field: 0.0 for field in timing_fields},
        }
        shard_paths: list[Path] = []
        resumed_shards = 0
        computed_shards = 0
        graph_executor = None
        graph_executor_initialized = False

        def add_statistics(result) -> None:
            totals["queries"] += 1
            totals["candidate_pool_size"] += float(result.metadata["candidate_pool_size"])
            totals["gold_recall_before_graphrag"] += float(result.metadata["gold_recall_before_graphrag"])
            totals["Recall@100"] += float(result.metadata["recall_at_100"])
            totals["graph_expansion_candidate_count"] += float(result.metadata["graph_expansion_candidate_count"])
            totals["retrieval_time_ms"] += float(result.metadata.get("retrieval_time_ms", 0.0))
            for field in timing_fields:
                totals[field] += float(result.metadata.get(field, 0.0))

        try:
            instances = iter(self.adapter.iter_instances(limit))
            shard_index = 0
            while True:
                instance_shard = list(islice(instances, output_shard_size))
                if not instance_shard:
                    break
                shard_path = shard_dir / f"part-{shard_index:06d}.jsonl"
                expected_ids = [instance.query.query_id for instance in instance_shard]
                if shard_path.exists():
                    actual_ids = []
                    for result in iter_results(shard_path):
                        actual_ids.append(result.query_id)
                        add_statistics(result)
                    if actual_ids != expected_ids:
                        raise ValueError(
                            f"Cannot resume: query IDs in {shard_path} do not match the current dataset. "
                            "Use --force to rebuild GraphRAG shards."
                        )
                    resumed_shards += 1
                else:
                    if not graph_executor_initialized:
                        graph_executor = create_graph_expansion_executor(
                            graph_path,
                            candidate_config,
                            graph_workers,
                        )
                        graph_executor_initialized = True

                    def iter_computed_results():
                        for start in range(0, len(instance_shard), query_batch_size):
                            batch = instance_shard[start : start + query_batch_size]
                            for result in retriever.rank_batch(batch, graph_executor):
                                add_statistics(result)
                                yield result

                    write_jsonl(shard_path, iter_computed_results())
                    computed_shards += 1
                shard_paths.append(shard_path)
                shard_index += 1
                print(
                    f"graphrag_shard={shard_index} queries={int(totals['queries'])} "
                    f"resumed={resumed_shards} computed={computed_shards}",
                    flush=True,
                )

            temporary_output = self.graphrag_path.with_suffix(self.graphrag_path.suffix + ".tmp")
            with temporary_output.open("wb") as destination:
                for shard_path in shard_paths:
                    with shard_path.open("rb") as source:
                        shutil.copyfileobj(source, destination)
            os.replace(temporary_output, self.graphrag_path)
            shard_manifest.update(
                {
                    "complete": True,
                    "queries": int(totals["queries"]),
                    "shards": len(shard_paths),
                }
            )
            write_json(shard_manifest_path, shard_manifest)
        finally:
            if graph_executor is not None:
                graph_executor.shutdown(wait=True)
            semantic_index.close()
            graph_index.close()

        count = int(totals["queries"])
        summary = {
            "queries": count,
            "query_batch_size": query_batch_size,
            "graph_workers": graph_workers,
            "output_shards": len(shard_paths),
            "resumed_shards": resumed_shards,
            "computed_shards": computed_shards,
            "average_candidate_pool_size": totals["candidate_pool_size"] / count if count else 0.0,
            "gold_recall_before_graphrag": totals["gold_recall_before_graphrag"] / count if count else 0.0,
            "Recall@100": totals["Recall@100"] / count if count else 0.0,
            "average_graph_expansion_candidate_count": totals["graph_expansion_candidate_count"] / count if count else 0.0,
            "average_retrieval_time_ms": totals["retrieval_time_ms"] / count if count else 0.0,
            "total_retrieval_time_seconds": totals["retrieval_time_ms"] / 1000.0,
            **{field: totals[field] / count if count else 0.0 for field in timing_fields},
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
