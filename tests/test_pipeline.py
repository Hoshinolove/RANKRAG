from pathlib import Path

from rankrag.io import iter_results
from rankrag.pipeline.recommender import CascadePipeline


def test_all_stages_are_independently_persisted(tmp_path):
    config = {
        "dataset": {"name": "hotpotqa", "path": "tests/fixtures/hotpot_tiny.json"},
        "embedding": {"backend": "hashing", "dimension": 64},
        "retrieval": {"top_k": 100, "hops": 2},
        "ranker": {"top_k": 20, "hidden_dim": 32, "device": "cpu"},
        "llm": {"provider": "passthrough", "top_k": 10, "prompt_version": "v1"},
        "evaluation": {"ks": [1, 5, 10]},
        "output": {"root": str(tmp_path), "experiment": "test"},
    }
    pipeline = CascadePipeline(config)
    pipeline.run_graphrag()
    pipeline.run_neural()
    pipeline.run_llm()
    metrics = pipeline.evaluate()
    assert set(metrics) == {"graphrag", "neural", "llm"}
    assert len(list(iter_results(pipeline.graphrag_path))) == 2
    assert len(list(iter_results(pipeline.neural_path))) == 2
    llm_results = list(iter_results(pipeline.llm_path))
    assert len(llm_results) == 2
    assert all(result.metadata["llm_cache_key"] for result in llm_results)
    assert (pipeline.output_dir / "metrics.json").exists()
