import json

import numpy as np

from rankrag.data.paragraph_corpus import ParagraphCorpus, build_paragraph_corpus, paragraph_id
from rankrag.embedding import HashingEmbedder
from rankrag.graphrag.global_assets import GlobalGraphIndex, SemanticParagraphIndex, build_global_graph_index
from rankrag.graphrag.global_retrieval import HybridCandidateConfig, HybridCandidateGenerator, HybridGraphRAGRetriever
from rankrag.graphrag.scorer import WeightedCandidateScorer
from rankrag.models import Query, RecommendationInstance


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
    build_global_graph_index(corpus_path, kg_path, graph_path)
    graph = GlobalGraphIndex(graph_path)
    assert alice_id in graph.paragraphs_for_entities(["alice"], 10)["alice"]
    assert "paris" in {edge["target"] for edge in graph.adjacent_entities(["alice"], 10)["alice"]}

    embedder = HashingEmbedder(32)
    matrix = embedder.encode([f"{record.title}\n{record.text}" for record in corpus.records])
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
    without_gold = retriever.rank(RecommendationInstance(query, [], ["not-in-corpus"]))
    with_gold = retriever.rank(RecommendationInstance(query, [], [paris_id]))

    assert [item.candidate_id for item in without_gold.candidates] == [item.candidate_id for item in with_gold.candidates]
    assert with_gold.metadata["candidate_pool_size"] >= 1
    assert "gold_recall_before_graphrag" in with_gold.metadata
    assert "recall_at_100" in with_gold.metadata
    assert "retrieval_time_ms" in with_gold.metadata
    semantic.close()
    graph.close()
