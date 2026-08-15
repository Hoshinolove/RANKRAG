from rankrag.data.hotpotqa import HotpotQAAdapter
from rankrag.embedding import HashingEmbedder
from rankrag.graph.builder import HotpotQAGraphBuilder
from rankrag.graphrag.retriever import GraphRAGRetriever, RetrievalConfig
from rankrag.graphrag.scorer import WeightedCandidateScorer


def test_graphrag_returns_ordered_explainable_candidates():
    instance = next(HotpotQAAdapter("tests/fixtures/hotpot_tiny.json").iter_instances())
    retriever = GraphRAGRetriever(
        HashingEmbedder(64),
        HotpotQAGraphBuilder(),
        WeightedCandidateScorer(),
        RetrievalConfig(top_k=100, hops=2),
    )
    result = retriever.rank(instance)
    assert len(result.candidates) == 3
    assert [item.rank for item in result.candidates] == [1, 2, 3]
    assert all(item.evidence_nodes and item.evidence_edges and item.paths for item in result.candidates)
    assert result.candidates == sorted(result.candidates, key=lambda item: (-item.rag_score, item.candidate_id))
