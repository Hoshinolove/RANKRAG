from __future__ import annotations

import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import shutil
from typing import Any

from rankrag.data.factory import create_candidate_corpus, create_dataset_adapter
from rankrag.embedding import create_embedder
from rankrag.evaluation.evaluator import evaluate_results
from rankrag.graph.factory import create_candidate_graph_index, create_instance_graph_builder
from rankrag.graphrag.retriever import GraphRAGRetriever, RetrievalConfig
from rankrag.graphrag.global_assets import SemanticCandidateIndex
from rankrag.graphrag.global_retrieval import (
    HybridCandidateConfig,
    HybridCandidateGenerator,
    HybridGraphRAGRetriever,
    create_graph_expansion_executor,
    default_graph_workers,
)
from rankrag.graphrag.scorer import WeightedCandidateScorer
from rankrag.graphrag.seeds import create_seed_provider
from rankrag.io import iter_results, write_json, write_jsonl
from rankrag.llm.cache import LLMResponseCache
from rankrag.llm.client import create_provider
from rankrag.llm.reranker import LLMReranker
from rankrag.ranker.tensor_inference import iter_tensor_rankings
from rankrag.ranker.tensor_dataset import load_manifest, load_split_query_ids


class CascadePipeline:
    def __init__(self, config: dict[str, Any], config_path: str | Path | None = None) -> None:
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        dataset = config.get("dataset", {})
        self.adapter = create_dataset_adapter(config)
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

    def graphrag_split_path(self, split: str = "validation") -> Path:
        return self.output_dir / f"graphrag.{split}.jsonl"

    @property
    def graphrag_validation_path(self) -> Path:
        return self.graphrag_split_path("validation")

    @property
    def neural_path(self) -> Path:
        return self.output_dir / "neural.jsonl"

    def neural_split_path(self, split: str = "validation") -> Path:
        return self.neural_path if split == "validation" else self.output_dir / f"neural.{split}.jsonl"

    @property
    def llm_path(self) -> Path:
        return self.output_dir / "llm.jsonl"

    def llm_split_path(self, split: str = "validation") -> Path:
        return self.llm_path if split == "validation" else self.output_dir / f"llm.{split}.jsonl"

    @property
    def retrieval_stats_path(self) -> Path:
        return self.output_dir / "graphrag_retrieval_stats.json"

    def _evaluation_split(self) -> str:
        return str(
            self.config.get("evaluation", {}).get(
                "split",
                self.config.get("ranker", {}).get("inference_split", "validation"),
            )
        )

    def _split_query_ids(self, split: str) -> list[str]:
        manifest_path = self.config.get("ranker_dataset", {}).get("manifest")
        if not manifest_path:
            raise ValueError("A ranker_dataset.manifest is required for split-aligned evaluation")
        if not Path(manifest_path).exists():
            raise FileNotFoundError(f"Ranker tensor manifest not found: {manifest_path}")
        return load_split_query_ids(load_manifest(manifest_path), split)

    def prepare_graphrag_split(self, split: str | None = None, force: bool = False) -> Path:
        """Filter cached full GraphRAG results to a manifest-defined split without retrieval."""
        split = split or self._evaluation_split()
        if not self.graphrag_path.exists():
            raise FileNotFoundError(f"Full GraphRAG cache not found: {self.graphrag_path}")
        output_path = self.graphrag_split_path(split)
        if output_path.exists() and not force:
            return output_path
        expected_ids = self._split_query_ids(split)
        expected_set = set(expected_ids)
        selected_ids: list[str] = []
        seen_source_ids: set[str] = set()

        def iter_selected_results():
            for result in iter_results(self.graphrag_path):
                if result.query_id in seen_source_ids:
                    raise ValueError(f"Full GraphRAG cache contains duplicate query ID: {result.query_id}")
                seen_source_ids.add(result.query_id)
                if result.query_id in expected_set:
                    selected_ids.append(result.query_id)
                    yield result
            if selected_ids != expected_ids:
                missing = sorted(expected_set - set(selected_ids))
                raise ValueError(
                    f"GraphRAG {split!r} query order does not match the ranker manifest; "
                    f"selected={len(selected_ids)} expected={len(expected_ids)} missing={missing[:5]}"
                )

        write_jsonl(output_path, iter_selected_results())
        return output_path

    @staticmethod
    def _result_query_ids(path: Path) -> list[str]:
        query_ids = [result.query_id for result in iter_results(path)]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError(f"Ranking output contains duplicate query IDs: {path}")
        return query_ids

    def _validate_split_alignment(self, paths: dict[str, Path], split: str) -> None:
        expected_ids = self._split_query_ids(split)
        for stage, path in paths.items():
            actual_ids = self._result_query_ids(path)
            if actual_ids != expected_ids:
                raise ValueError(
                    f"{stage} output is not aligned to ranker split {split!r}: "
                    f"queries={len(actual_ids)} expected={len(expected_ids)} path={path}"
                )

    def run_graphrag(self, limit: int | None = None, force: bool = False) -> Path:
        global_config = self.config.get("global_retrieval", {})
        if global_config.get("enabled", False):
            return self._run_global_graphrag(global_config, limit, force)
        retrieval = self.config.get("retrieval", {})
        retriever = GraphRAGRetriever(
            self.embedder,
            create_instance_graph_builder(self.config, self.adapter, limit),
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
        corpus = create_candidate_corpus(self.config)
        semantic_index = SemanticCandidateIndex(
            corpus,
            embeddings_path,
            faiss_path,
            backend=str(global_config.get("index_backend", "faiss")),
        )
        graph_index = create_candidate_graph_index(self.config)
        candidate_config = HybridCandidateConfig(
            semantic_top_k=int(global_config.get("semantic_top_k", 500)),
            seed_candidate_k=int(global_config.get("seed_candidate_k", global_config.get("seed_paragraph_k", 20))),
            graph_hops=min(2, int(global_config.get("graph_hops", 2))),
            max_graph_candidates=int(global_config.get("max_graph_candidates", 1000)),
            max_nodes_per_hop=int(global_config.get("max_nodes_per_hop", global_config.get("max_entities_per_hop", 500))),
            max_neighbors_per_node=int(global_config.get("max_neighbors_per_node", global_config.get("max_relations_per_entity", 100))),
            max_candidates_per_node=int(global_config.get("max_candidates_per_node", global_config.get("max_paragraphs_per_entity", 100))),
            max_paths_per_candidate=int(global_config.get("max_paths_per_candidate", 3)),
            graph_aggregation=str(global_config.get("graph_aggregation", "max")),
        )
        seed_provider = create_seed_provider(self.config, candidate_config.seed_candidate_k)
        candidate_generator = HybridCandidateGenerator(
            corpus,
            semantic_index,
            graph_index,
            self.embedder,
            candidate_config,
            seed_provider=seed_provider,
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
        dataset_sources = []
        for source_path in self.adapter.source_paths():
            source_stat = source_path.stat()
            dataset_sources.append(
                {
                    "path": str(source_path.resolve()),
                    "size": source_stat.st_size,
                    "mtime_ns": source_stat.st_mtime_ns,
                }
            )
        signature_payload = {
            "pipeline_schema_version": 2,
            "dataset": {"config": self.config.get("dataset", {}), "sources": dataset_sources},
            "embedding": self.config.get("embedding", {}),
            "global_retrieval": global_config,
            "seed_provider": self.config.get("seed_provider", {}),
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
                            graph_backend=graph_index.backend,
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

    def run_llm(self, split: str | None = None) -> Path:
        split = split or self._evaluation_split()
        neural_path = self.neural_split_path(split)
        output_path = self.llm_split_path(split)
        if not neural_path.exists():
            raise FileNotFoundError(f"Run the neural stage first: {neural_path}")
        self._validate_split_alignment({"neural": neural_path}, split)
        llm_config = self.config.get("llm", {})
        cache_dir = self.output_dir / llm_config.get("cache_dir", "llm_cache")
        reranker = LLMReranker(
            create_provider(llm_config),
            LLMResponseCache(cache_dir),
            top_k=int(llm_config.get("top_k", 10)),
            prompt_version=llm_config.get("prompt_version", "v1"),
            max_text_chars=int(llm_config.get("max_text_chars", 4000)),
        )
        write_jsonl(output_path, (reranker.rank(result) for result in iter_results(neural_path)))
        return output_path

    def _evaluation_ks(self, stage: str) -> list[int]:
        evaluation = self.config.get("evaluation", {})
        stage_ks = evaluation.get("stage_ks", {})
        values = stage_ks.get(stage, evaluation.get("ks", [5, 10]))
        return [int(k) for k in values]

    def evaluate(self) -> dict[str, dict[str, float | int]]:
        split = self._evaluation_split()
        neural_path = self.neural_split_path(split)
        llm_path = self.llm_split_path(split)
        manifest_value = self.config.get("ranker_dataset", {}).get("manifest")
        manifest_exists = bool(manifest_value) and Path(manifest_value).exists()
        if manifest_exists:
            graphrag_split_path = self.prepare_graphrag_split(split)
            stage_paths = {
                "graphrag": graphrag_split_path,
                **({"neural": neural_path} if neural_path.exists() else {}),
                **({"llm": llm_path} if llm_path.exists() else {}),
            }
            self._validate_split_alignment(stage_paths, split)
        else:
            if neural_path.exists() or llm_path.exists():
                raise FileNotFoundError(
                    "Neural/LLM outputs exist but ranker_dataset.manifest is unavailable; "
                    "split-aligned evaluation cannot be verified"
                )
            stage_paths = {"graphrag": self.graphrag_path} if self.graphrag_path.exists() else {}
        metrics = {}
        for stage, path in stage_paths.items():
            metrics[stage] = evaluate_results(iter_results(path), self._evaluation_ks(stage))
        write_json(self.output_dir / "metrics.json", metrics)
        return metrics
