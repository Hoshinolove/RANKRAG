import json

import numpy as np

from prepare_scidocs import audit_protocol, build_corpus
from rankrag.data.candidate_corpus import load_candidate_corpus
from rankrag.data.scidocs import SCIDOCSAdapter
from rankrag.embedding import HashingEmbedder
from rankrag.graph.candidate_index import SQLiteCandidateGraphIndex
from rankrag.graph.scidocs import SCIDOCSGraphBuilder
from rankrag.graphrag.global_assets import SemanticCandidateIndex
from rankrag.graphrag.global_retrieval import HybridCandidateConfig, HybridCandidateGenerator
from rankrag.graphrag.seeds import create_seed_provider
from rankrag.models import Query
from rankrag.pipeline.recommender import CascadePipeline


def _write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_beir_adapter_uses_official_qrels_and_unrestricted_corpus(tmp_path):
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels" / "test.tsv"
    _write_jsonl(queries, [{"_id": "q1", "text": "Paper query title"}])
    qrels.parent.mkdir(parents=True)
    qrels.write_text(
        "query-id\tcorpus-id\tscore\nq1\tp1\t1\nq1\tp2\t1\nq1\tp3\t0\n",
        encoding="utf-8",
    )
    adapter = SCIDOCSAdapter("beir_test", "test", queries, qrels)
    instance = next(adapter.iter_instances())
    assert instance.query.text == "Paper query title"
    assert instance.positive_ids == ["p1", "p2"]
    assert instance.query.allowed_candidate_ids is None
    assert instance.query.excluded_candidate_ids == ("q1",)
    assert instance.candidates == []


def test_development_protocol_excludes_all_official_test_queries(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "q-train": {"title": "Train query"},
                "q-test": {"title": "Held out query"},
                "p1": {"title": "Positive"},
            }
        ),
        encoding="utf-8",
    )
    train = tmp_path / "train.csv"
    train.write_text("query_id,clicked_id,other_ids\nq-train,p1,p2\nq-test,p1,p2\n", encoding="utf-8")
    validation = tmp_path / "val.qrel"
    validation.write_text("q-train 0 p1 1\nq-test 0 p1 1\n", encoding="utf-8")
    forbidden = tmp_path / "beir-queries.jsonl"
    _write_jsonl(forbidden, [{"_id": "q-test", "text": "Held out query"}])
    adapter = SCIDOCSAdapter(
        "development",
        "all",
        metadata_paths=[metadata],
        recomm_train_path=train,
        citation_validation_qrels_path=validation,
        forbidden_query_ids_path=forbidden,
    )
    instances = list(adapter.iter_instances())
    assert [value.query.query_id for value in instances] == ["q-train", "q-train"]
    assert [value.metadata["split"] for value in instances] == ["train", "validation"]
    assert all(value.query.allowed_candidate_ids is None for value in instances)


def test_development_can_report_and_drop_positives_missing_from_metadata_corpus(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"q1": {"title": "Query"}}), encoding="utf-8")
    train = tmp_path / "train.csv"
    train.write_text("query_id,clicked_id\nq1,missing-paper\n", encoding="utf-8")
    validation = tmp_path / "val.qrel"
    validation.write_text("", encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"candidate_id": "q1", "text": "Query", "embedding_text": "Query"}])
    adapter = SCIDOCSAdapter(
        "development",
        "train",
        metadata_paths=[metadata],
        recomm_train_path=train,
        citation_validation_qrels_path=validation,
        candidate_corpus_path=corpus,
        missing_positive_policy="drop",
    )
    assert list(adapter.iter_instances()) == []
    assert adapter.dropped_positive_judgments == [
        {
            "query_id": "q1",
            "candidate_id": "missing-paper",
            "split": "train",
            "source": "original_scidocs_recomm",
        }
    ]
    assert adapter.dropped_queries == ["q1"]


def test_scidocs_graph_uses_generic_api_and_blocks_heldout_query_outgoing_citations(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {"candidate_id": "q1", "text": "Query", "embedding_text": "Query"},
            {"candidate_id": "p1", "text": "Positive", "embedding_text": "Positive"},
            {"candidate_id": "p2", "text": "Other", "embedding_text": "Other"},
        ],
    )
    enrichment = tmp_path / "enrichment.jsonl"
    _write_jsonl(
        enrichment,
        [
            {
                "paper_id": "q1",
                "authors": [{"authorId": "a1", "name": "Author"}],
                "fieldsOfStudy": ["IR"],
                "references": ["p1"],
            },
            {"paper_id": "p2", "authors": ["a1"], "references": ["p1"]},
        ],
    )
    graph_path = tmp_path / "graph.sqlite"
    stats = SCIDOCSGraphBuilder(
        load_candidate_corpus(corpus_path),
        enrichment,
        forbidden_outgoing_paper_ids={"q1"},
    ).build(graph_path)
    assert stats["skipped_forbidden_references"] == 1
    graph = SQLiteCandidateGraphIndex(graph_path)
    assert "author::a1" in graph.nodes_for_candidates(["q1"])["q1"]
    assert set(graph.candidates_for_nodes(["author::a1"], 10)["author::a1"]) == {"q1", "p2"}
    assert not graph.neighbors(["candidate::q1"], 10).get("candidate::q1")
    assert graph.neighbors(["candidate::p2"], 10)["candidate::p2"][0]["target"] == "candidate::p1"
    graph.close()


def test_generic_node_seeds_are_not_filtered_as_candidate_ids(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {"candidate_id": "p1", "text": "One", "embedding_text": "One"},
            {"candidate_id": "p2", "text": "Two", "embedding_text": "Two"},
        ],
    )
    enrichment = tmp_path / "enrichment.jsonl"
    _write_jsonl(enrichment, [{"paper_id": "p2", "authors": ["a1"]}])
    corpus = load_candidate_corpus(corpus_path)
    graph_path = tmp_path / "graph.sqlite"
    SCIDOCSGraphBuilder(corpus, enrichment).build(graph_path)
    graph = SQLiteCandidateGraphIndex(graph_path)
    embedder = HashingEmbedder(8)
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, embedder.encode(corpus.embedding_texts(0, len(corpus))))
    semantic = SemanticCandidateIndex(corpus, embeddings_path, None, backend="numpy")
    config = {
        "seed_provider": {
            "type": "graph_nodes",
            "semantic_seed_k": 0,
            "include_query_nodes": True,
        }
    }
    generator = HybridCandidateGenerator(
        corpus,
        semantic,
        graph,
        embedder,
        HybridCandidateConfig(semantic_top_k=2, seed_candidate_k=0, graph_hops=1),
        seed_provider=create_seed_provider(config, 0),
    )
    pool = generator.generate(Query("q", "query", seed_node_ids=("author::a1",)))
    assert "p2" in {candidate.candidate_id for candidate in pool.candidates}
    semantic.close()
    graph.close()


def test_protocol_audit_rejects_candidate_subsets_and_checks_exact_id_join(tmp_path):
    raw_corpus = tmp_path / "raw-corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels" / "test.tsv"
    _write_jsonl(raw_corpus, [{"_id": "p1", "title": "P1", "text": "Abstract"}])
    _write_jsonl(queries, [{"_id": "q1", "text": "Q1"}])
    qrels.parent.mkdir(parents=True)
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\tp1\t1\n", encoding="utf-8")
    config = {
        "dataset": {
            "name": "scidocs",
            "adapter": "scidocs",
            "protocol": "beir_test",
            "split": "test",
            "beir_corpus_path": str(raw_corpus),
            "beir_queries_path": str(queries),
            "beir_qrels_path": str(qrels),
            "expected_candidate_count": 1,
            "expected_query_count": 1,
        },
        "global_retrieval": {},
    }
    corpus_path = tmp_path / "corpus.jsonl"
    assert build_corpus(config, corpus_path)["candidate_count"] == 1
    report = audit_protocol(config, corpus_path)
    assert report["positive_in_corpus_rate"] == 1.0
    assert report["full_corpus_retrieval"] is True


def test_official_test_has_isolated_outputs_and_stage_specific_metrics(tmp_path):
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels" / "test.tsv"
    _write_jsonl(queries, [{"_id": "q1", "text": "Q1"}])
    qrels.parent.mkdir(parents=True)
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\tp1\t1\n", encoding="utf-8")
    config = {
        "dataset": {
            "name": "scidocs",
            "adapter": "scidocs",
            "protocol": "beir_test",
            "split": "test",
            "beir_queries_path": str(queries),
            "beir_qrels_path": str(qrels),
        },
        "ranker": {"inference_split": "test"},
        "evaluation": {
            "split": "test",
            "stage_ks": {"graphrag": [20, 50, 100], "neural": [5, 10, 20]},
        },
        "output": {"root": str(tmp_path / "outputs"), "experiment": "official"},
    }
    pipeline = CascadePipeline(config)
    assert pipeline.neural_split_path("test").name == "neural.test.jsonl"
    assert pipeline.llm_split_path("test").name == "llm.test.jsonl"
    assert pipeline._evaluation_ks("graphrag") == [20, 50, 100]
    assert pipeline._evaluation_ks("neural") == [5, 10, 20]
