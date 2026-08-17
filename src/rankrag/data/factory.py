from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from rankrag.data.base import DatasetAdapter
from rankrag.data.candidate_corpus import CandidateCorpus, load_candidate_corpus
from rankrag.data.hotpotqa import HotpotQAAdapter
from rankrag.data.scidocs import SCIDOCSAdapter


AdapterFactory = Callable[[dict[str, Any]], DatasetAdapter]


def _hotpotqa_adapter(config: dict[str, Any]) -> DatasetAdapter:
    dataset = config["dataset"]
    return HotpotQAAdapter(
        dataset["path"],
        use_paragraph_ids=bool(config.get("global_retrieval", {}).get("enabled", False)),
    )


def _scidocs_adapter(config: dict[str, Any]) -> DatasetAdapter:
    dataset = config["dataset"]
    return SCIDOCSAdapter(
        protocol=str(dataset.get("protocol", "beir_test")),
        split=str(dataset.get("split", "all")),
        beir_queries_path=dataset.get("beir_queries_path"),
        beir_qrels_path=dataset.get("beir_qrels_path"),
        metadata_paths=list(dataset.get("metadata_paths", [])),
        recomm_train_path=dataset.get("recomm_train_path"),
        citation_validation_qrels_path=dataset.get("citation_validation_qrels_path"),
        forbidden_query_ids_path=dataset.get("forbidden_query_ids_path"),
        candidate_corpus_path=config.get("candidate_corpus", {}).get(
            "path",
            config.get("global_retrieval", {}).get("corpus_path"),
        ),
        missing_positive_policy=str(dataset.get("missing_positive_policy", "strict")),
    )


_ADAPTERS: dict[str, AdapterFactory] = {
    "hotpotqa": _hotpotqa_adapter,
    "scidocs": _scidocs_adapter,
}


def create_dataset_adapter(config: dict[str, Any]) -> DatasetAdapter:
    dataset = config.get("dataset", {})
    adapter_name = str(dataset.get("adapter", dataset.get("name", "hotpotqa")))
    try:
        factory = _ADAPTERS[adapter_name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset adapter: {adapter_name}") from exc
    return factory(config)


def create_candidate_corpus(config: dict[str, Any]) -> CandidateCorpus:
    corpus_config = config.get("candidate_corpus", {})
    global_config = config.get("global_retrieval", {})
    asset_dir = Path(global_config.get("asset_dir", "outputs/global_retrieval"))
    path = Path(corpus_config.get("path", global_config.get("corpus_path", asset_dir / "corpus.jsonl")))
    return load_candidate_corpus(path)
