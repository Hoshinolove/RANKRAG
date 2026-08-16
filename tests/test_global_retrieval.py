import json
import sys

import numpy as np

import prepare_global_retrieval
from rankrag.data.paragraph_corpus import ParagraphCorpus, build_paragraph_corpus, paragraph_id
from rankrag.embedding import HashingEmbedder
from rankrag.graphrag.global_assets import (
    GlobalGraphIndex,
    SemanticParagraphIndex,
    build_global_graph_index,
    inspect_global_graph_index,
)
from rankrag.graphrag.global_retrieval import (
    CandidatePool,
    GeneratedCandidate,
    GlobalEvidencePath,
    HybridCandidateConfig,
    HybridCandidateGenerator,
    HybridGraphRAGRetriever,
    create_graph_expansion_executor,
)
from rankrag.graphrag.scorer import WeightedCandidateScorer
from rankrag.models import Query, RecommendationInstance
from rankrag.pipeline.recommender import CascadePipeline


class RecordingHashingEmbedder(HashingEmbedder):
    def __init__(self, dimension=32):
        super().__init__(dimension)
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return super().encode(texts)


def test_reused_corpus_recovers_missing_manifest_metadata(tmp_path, monkeypatch):
    source = tmp_path / "paragraphs.json"
    source.write_text(
        json.dumps(
            [
                {"title": "Alice", "text": "Alice lives in Paris."},
                {"title": "Alice", "text": "Alice lives in Paris."},
            ]
        ),
        encoding="utf-8",
    )
    asset_dir = tmp_path / "assets"
    corpus_path = asset_dir / "corpus.jsonl"
    build_paragraph_corpus([source], corpus_path)
    manifest_path = asset_dir / "manifest.json"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "global_retrieval": {
                    "enabled": True,
                    "asset_dir": str(asset_dir),
                    "corpus_inputs": [str(source)],
                    "corpus_path": str(corpus_path),
                    "manifest_path": str(manifest_path),
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_global_retrieval.py", "--config", str(config_path), "--stage", "corpus"],
    )
    prepare_global_retrieval.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["corpus"]["input_records"] == 2
    assert manifest["corpus"]["paragraph_count"] == 1
    assert manifest["corpus"]["duplicates_removed"] == 1


def test_global_corpus_graph_and_hybrid_retrieval_do_not_inject_positives(tmp_path):
    paragraphs = [
        {"title": "Noise", "text": "An unrelated paragraph."},
        {"title": "Alice", "text": "Alice lives in Paris."},
        {"title": "Paris", "text": "Paris is a city in France."},
        {"title": "Alice", "text": "Alice lives in Paris."},
    ]
    source = tmp_path / "paragraphs.json"
    source.write_text(json.dumps(paragraphs), encoding="utf-8")
    corpus_path = tmp_path / "corpus.jsonl"
    statistics = build_paragraph_corpus([source], corpus_path)
    assert statistics["paragraph_count"] == 3
    assert statistics["duplicates_removed"] == 1

    corpus = ParagraphCorpus.load(corpus_path)
    assert corpus.records == sorted(corpus.records, key=lambda record: record.paragraph_id)
    alice_id = paragraph_id("Alice", "Alice lives in Paris.")
    paris_id = paragraph_id("Paris", "Paris is a city in France.")

    kg_path = tmp_path / "kg.jsonl"
    kg_records = [
        {
            "id": alice_id,
            "title": "Alice",
            "entities": [
                {"entity_name": "Alice", "entity_type": "person", "description": ""},
                {"entity_name": "Paris", "entity_type": "location", "description": ""},
            ],
            "relationships": [{"src_id": "Alice", "tgt_id": "Paris", "keywords": "lives_in", "description": ""}],
        },
        {
            "id": paris_id,
            "title": "Paris",
            "entities": [
                {"entity_name": "Paris", "entity_type": "location", "description": ""},
                {"entity_name": "France", "entity_type": "location", "description": ""},
            ],
            "relationships": [{"src_id": "Paris", "tgt_id": "France", "keywords": "located_in", "description": ""}],
        },
    ]
    kg_path.write_text("".join(json.dumps(record) + "\n" for record in kg_records), encoding="utf-8")
    graph_path = tmp_path / "graph.sqlite"
    graph_statistics = build_global_graph_index(corpus_path, kg_path, graph_path)
    assert inspect_global_graph_index(graph_path) == graph_statistics
    graph = GlobalGraphIndex(graph_path)
    assert alice_id in graph.paragraphs_for_entities(["alice"], 10)["alice"]
    assert "paris" in {edge["target"] for edge in graph.adjacent_entities(["alice"], 10)["alice"]}

    embedder = RecordingHashingEmbedder(32)
    matrix = embedder.encode([f"{record.title}\n{record.text}" for record in corpus.records])
    embedder.calls.clear()
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, matrix)
    semantic = SemanticParagraphIndex(corpus, embeddings_path, None, backend="numpy")
    generator = HybridCandidateGenerator(
        corpus,
        semantic,
        graph,
        embedder,
        HybridCandidateConfig(semantic_top_k=1, seed_paragraph_k=1, graph_hops=2, max_graph_candidates=10),
    )
    retriever = HybridGraphRAGRetriever(generator, WeightedCandidateScorer(), top_k=2)
    query = Query("q", "Where does Alice live?")
    without_gold, with_gold = retriever.rank_batch(
        [
            RecommendationInstance(query, [], ["not-in-corpus"]),
            RecommendationInstance(query, [], [paris_id]),
        ]
    )

    assert [item.candidate_id for item in without_gold.candidates] == [item.candidate_id for item in with_gold.candidates]
    assert embedder.calls == [[query.text, query.text]]
    assert with_gold.metadata["candidate_pool_size"] >= 1
    assert "gold_recall_before_graphrag" in with_gold.metadata
    assert "recall_at_100" in with_gold.metadata
    assert "retrieval_time_ms" in with_gold.metadata
    assert "query_embedding_time_ms" in with_gold.metadata
    assert "semantic_search_time_ms" in with_gold.metadata
    assert "graph_expansion_time_ms" in with_gold.metadata
    assert "evidence_serialization_time_ms" in with_gold.metadata
    assert "total_query_time_ms" in with_gold.metadata

    executor = create_graph_expansion_executor(graph_path, generator.config, graph_workers=2)
    try:
        parallel_results = retriever.rank_batch(
            [
                RecommendationInstance(query, [], ["not-in-corpus"]),
                RecommendationInstance(query, [], [paris_id]),
            ],
            executor,
        )
    finally:
        executor.shutdown(wait=True)
    assert [candidate.to_dict() for candidate in parallel_results[0].candidates] == [
        candidate.to_dict() for candidate in without_gold.candidates
    ]
    assert [candidate.to_dict() for candidate in parallel_results[1].candidates] == [
        candidate.to_dict() for candidate in with_gold.candidates
    ]
    semantic.close()
    graph.close()


def test_semantic_search_batch_matches_reference_and_uses_one_index_call(tmp_path):
    paragraphs = [
        {"title": "A", "text": "alpha"},
        {"title": "B", "text": "beta"},
        {"title": "C", "text": "gamma"},
    ]
    source = tmp_path / "paragraphs.json"
    source.write_text(json.dumps(paragraphs), encoding="utf-8")
    corpus_path = tmp_path / "corpus.jsonl"
    build_paragraph_corpus([source], corpus_path)
    corpus = ParagraphCorpus.load(corpus_path)
    embedder = HashingEmbedder(16)
    matrix = embedder.encode([f"{record.title}\n{record.text}" for record in corpus.records])
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, matrix)
    semantic = SemanticParagraphIndex(corpus, embeddings_path, None, backend="numpy")
    vectors = embedder.encode(["alpha", "gamma"])

    expected = semantic.search_batch(vectors, 2)

    class RecordingIndex:
        def __init__(self):
            self.calls = []

        def search(self, query_vectors, top_k):
            self.calls.append(tuple(query_vectors.shape))
            scores = matrix @ query_vectors.T
            rows = np.argsort(-scores, axis=0, kind="stable")[:top_k].T
            ordered_scores = np.take_along_axis(scores.T, rows, axis=1)
            return ordered_scores.astype(np.float32), rows.astype(np.int64)

    recording_index = RecordingIndex()
    semantic.backend = "faiss"
    semantic.index = recording_index
    actual = semantic.search_batch(vectors, 2)

    assert recording_index.calls == [(2, 16)]
    assert [[pid for pid, _ in hits] for hits in actual] == [[pid for pid, _ in hits] for hits in expected]
    assert np.allclose(
        [[score for _, score in hits] for hits in actual],
        [[score for _, score in hits] for hits in expected],
    )
    semantic.close()


def test_evidence_metadata_is_loaded_only_for_selected_top_k(tmp_path):
    paragraphs = [
        {"title": "A", "text": "alpha"},
        {"title": "B", "text": "beta"},
        {"title": "C", "text": "gamma"},
    ]
    source = tmp_path / "paragraphs.json"
    source.write_text(json.dumps(paragraphs), encoding="utf-8")
    corpus_path = tmp_path / "corpus.jsonl"
    build_paragraph_corpus([source], corpus_path)
    corpus = ParagraphCorpus.load(corpus_path)
    embedder = HashingEmbedder(8)
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, embedder.encode([record.text for record in corpus.records]))
    semantic = SemanticParagraphIndex(corpus, embeddings_path, None, backend="numpy")

    class RecordingGraph:
        def __init__(self):
            self.requested_entities = set()

        def entity_metadata(self, entity_ids):
            self.requested_entities = set(entity_ids)
            return {entity_id: {"name": entity_id} for entity_id in entity_ids}

    graph = RecordingGraph()
    generator = HybridCandidateGenerator(
        corpus,
        semantic,
        graph,
        embedder,
        HybridCandidateConfig(),
    )
    retriever = HybridGraphRAGRetriever(generator, WeightedCandidateScorer(), top_k=2)
    ordered_ids = [record.paragraph_id for record in corpus.records]
    pool = CandidatePool(
        candidates=[
            GeneratedCandidate(
                paragraph_id=paragraph_id_value,
                semantic_score=score,
                paths=[
                    GlobalEvidencePath(
                        nodes=(f"paragraph::{paragraph_id_value}", f"entity::entity-{index}"),
                        relations=("mentions",),
                        score=score,
                    )
                ],
            )
            for index, (paragraph_id_value, score) in enumerate(zip(ordered_ids, [0.9, 0.8, 0.1], strict=True))
        ],
        semantic_candidate_count=3,
        graph_expansion_candidate_count=0,
    )

    result, _ = retriever._rank_pool(RecommendationInstance(Query("q", "query"), [], []), pool)

    assert len(result.candidates) == 2
    assert graph.requested_entities == {"entity-0", "entity-1"}
    assert "entity-2" not in graph.requested_entities
    semantic.close()


def test_global_pipeline_resumes_atomic_shards_in_query_order(tmp_path):
    dataset_path = "tests/fixtures/hotpot_tiny.json"
    paragraph_source = tmp_path / "paragraphs.json"
    paragraph_source.write_text(
        json.dumps(
            [
                {"title": "Noise", "text": "A completely unrelated paragraph."},
                {"title": "Alice", "text": "Alice lives in Paris."},
                {"title": "Paris", "text": "Paris is a city in France."},
                {"title": "Grass", "text": "Grass is green."},
                {"title": "Sky", "text": "The sky is blue."},
            ]
        ),
        encoding="utf-8",
    )
    asset_dir = tmp_path / "assets"
    corpus_path = asset_dir / "corpus.jsonl"
    build_paragraph_corpus([paragraph_source], corpus_path)
    corpus = ParagraphCorpus.load(corpus_path)
    embedder = HashingEmbedder(16)
    embeddings_path = asset_dir / "embeddings.npy"
    np.save(embeddings_path, embedder.encode([f"{record.title}\n{record.text}" for record in corpus.records]))
    faiss_path = asset_dir / "unused.faiss"
    faiss_path.write_bytes(b"numpy backend")
    kg_path = tmp_path / "empty_kg.jsonl"
    kg_path.write_text("", encoding="utf-8")
    graph_path = asset_dir / "graph.sqlite"
    build_global_graph_index(corpus_path, kg_path, graph_path)
    embedding_config = {"backend": "hashing", "dimension": 16}
    manifest_path = asset_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"embedding": embedding_config}), encoding="utf-8")
    config = {
        "dataset": {"name": "hotpotqa", "path": dataset_path},
        "embedding": embedding_config,
        "global_retrieval": {
            "enabled": True,
            "asset_dir": str(asset_dir),
            "corpus_path": str(corpus_path),
            "embeddings_path": str(embeddings_path),
            "faiss_index_path": str(faiss_path),
            "graph_index_path": str(graph_path),
            "manifest_path": str(manifest_path),
            "index_backend": "numpy",
            "query_batch_size": 1,
            "graph_workers": 1,
            "output_shard_size": 1,
            "semantic_top_k": 3,
            "seed_paragraph_k": 1,
            "graph_hops": 2,
        },
        "retrieval": {"top_k": 2, "semantic_weight": 0.5, "graph_weight": 0.5},
        "output": {"root": str(tmp_path), "experiment": "resume"},
    }
    pipeline = CascadePipeline(config)
    pipeline.run_graphrag(force=True)
    shard_paths = sorted((pipeline.output_dir / "graphrag_shards").glob("part-*.jsonl"))
    shard_mtimes = [path.stat().st_mtime_ns for path in shard_paths]
    pipeline.graphrag_path.unlink()

    def fail_if_recomputed(_texts):
        raise AssertionError("Completed GraphRAG shards must not recompute query embeddings")

    pipeline._embedder.encode = fail_if_recomputed

    pipeline.run_graphrag()

    assert [result["query_id"] for result in map(json.loads, pipeline.graphrag_path.read_text(encoding="utf-8").splitlines())] == [
        "q1",
        "q2",
    ]
    assert [path.stat().st_mtime_ns for path in shard_paths] == shard_mtimes
    statistics = json.loads(pipeline.retrieval_stats_path.read_text(encoding="utf-8"))
    assert statistics["resumed_shards"] == 2
    assert statistics["computed_shards"] == 0
    for field in (
        "query_embedding_time_ms",
        "semantic_search_time_ms",
        "graph_expansion_time_ms",
        "evidence_serialization_time_ms",
        "total_query_time_ms",
    ):
        assert field in statistics
