import json

from rankrag.io import iter_results
from rankrag.pipeline.recommender import CascadePipeline
from rankrag.ranker.preprocessing import prepare_ranker_dataset
from rankrag.ranker.trainer import train_tensor_ranker


def test_validation_inference_excludes_train_queries_and_stages_persist(tmp_path):
    config = {
        "dataset": {"name": "hotpotqa", "path": "tests/fixtures/hotpot_tiny.json"},
        "embedding": {"backend": "hashing", "dimension": 64},
        "retrieval": {"top_k": 100, "hops": 2},
        "ranker": {
            "model": "set_transformer",
            "top_k": 20,
            "hidden_dim": 16,
            "num_heads": 4,
            "num_layers": 1,
            "feedforward_dim": 32,
            "device": "cpu",
        },
        "llm": {"provider": "passthrough", "top_k": 10, "prompt_version": "v1"},
        "evaluation": {"ks": [1, 5, 10]},
        "output": {"root": str(tmp_path), "experiment": "test"},
    }
    pipeline = CascadePipeline(config)
    pipeline.run_graphrag()
    dataset_dir = pipeline.output_dir / "ranker_dataset"
    checkpoint_dir = pipeline.output_dir / "ranker_checkpoints"
    config["ranker_dataset"] = {
        "graphrag_path": str(pipeline.graphrag_path),
        "output_dir": str(dataset_dir),
        "manifest": str(dataset_dir / "manifest.json"),
        "candidate_k": 5,
        "shard_size": 1,
        "storage_dtype": "float16",
        "validation_fraction": 0.5,
        "split_seed": 0,
    }
    config["training"] = {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "seed": 13,
        "loss": "listwise",
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "amp": False,
        "checkpoint_dir": str(checkpoint_dir),
    }
    prepare_ranker_dataset(config)
    train_tensor_ranker(config)
    pipeline.prepare_graphrag_split("validation")
    pipeline.run_neural(split="validation")
    pipeline.run_llm()
    metrics = pipeline.evaluate()
    assert set(metrics) == {"graphrag", "neural", "llm"}
    assert len(list(iter_results(pipeline.graphrag_path))) == 2
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    validation_ids = {
        json.loads(line)["query_id"]
        for shard in manifest["splits"]["validation"]["shards"]
        for line in (dataset_dir / shard["metadata"]).read_text(encoding="utf-8").splitlines()
    }
    train_ids = {
        json.loads(line)["query_id"]
        for shard in manifest["splits"]["train"]["shards"]
        for line in (dataset_dir / shard["metadata"]).read_text(encoding="utf-8").splitlines()
    }
    neural_results = list(iter_results(pipeline.neural_path))
    assert {result.query_id for result in neural_results} == validation_ids
    assert not ({result.query_id for result in neural_results} & train_ids)
    llm_results = list(iter_results(pipeline.llm_path))
    assert len(llm_results) == len(validation_ids)
    assert all(result.metadata["llm_cache_key"] for result in llm_results)
    validation_graphrag_results = list(iter_results(pipeline.graphrag_validation_path))
    assert [result.query_id for result in validation_graphrag_results] == [result.query_id for result in neural_results]
    assert metrics["graphrag"]["queries"] == len(validation_ids)
    assert metrics["neural"]["queries"] == len(validation_ids)
    assert metrics["llm"]["queries"] == len(validation_ids)
    assert (pipeline.output_dir / "metrics.json").exists()
