from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rankrag.data.candidate_corpus import load_candidate_corpus
from rankrag.data.factory import create_dataset_adapter
from rankrag.data.scidocs import SCIDOCSAdapter, load_beir_qrels, load_beir_queries, load_paper_metadata
from rankrag.embedding import create_embedder
from rankrag.graph.scidocs import SCIDOCSGraphBuilder
from rankrag.graphrag.global_assets import build_semantic_assets
from rankrag.io import load_config, write_json


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    global_config = config["global_retrieval"]
    asset_dir = Path(global_config["asset_dir"])
    return {
        "asset_dir": asset_dir,
        "corpus": Path(global_config["corpus_path"]),
        "embeddings": Path(global_config["embeddings_path"]),
        "faiss": Path(global_config["faiss_index_path"]),
        "graph": Path(global_config["graph_index_path"]),
        "manifest": Path(global_config["manifest_path"]),
        "audit": asset_dir / "protocol_report.json",
    }


def _paper_text(title: str, abstract: str) -> str:
    return "\n".join(value for value in (title.strip(), abstract.strip()) if value).strip()


def _beir_corpus_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            candidate_id = str(value.get("_id", value.get("candidate_id", ""))).strip()
            if not candidate_id:
                raise ValueError(f"BEIR SCIDOCS corpus row is missing _id: {value}")
            title = str(value.get("title", "")).strip()
            abstract = str(value.get("text", value.get("abstract", ""))).strip()
            text = _paper_text(title, abstract)
            yield {
                "candidate_id": candidate_id,
                "text": text,
                "embedding_text": text,
                "metadata": {"title": title, "abstract": abstract},
                "sources": [str(path)],
            }


def _development_corpus_records(paths: list[Path]) -> Iterable[dict[str, Any]]:
    papers = load_paper_metadata([str(path) for path in paths])
    for candidate_id in sorted(papers):
        value = papers[candidate_id]
        title = str(value.get("title", "")).strip()
        abstract = str(value.get("abstract", value.get("text", ""))).strip()
        text = _paper_text(title, abstract)
        yield {
            "candidate_id": candidate_id,
            "text": text,
            "embedding_text": text,
            "metadata": {"title": title, "abstract": abstract},
            "sources": [str(path) for path in paths],
        }


def build_corpus(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    dataset = config["dataset"]
    protocol = str(dataset.get("protocol", "beir_test"))
    if protocol == "beir_test":
        raw_path = Path(dataset["beir_corpus_path"])
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing BEIR SCIDOCS corpus: {raw_path}")
        records = _beir_corpus_records(raw_path)
        sources = [str(raw_path)]
    elif protocol == "development":
        metadata_paths = [Path(path) for path in dataset.get("metadata_paths", [])]
        if not metadata_paths:
            raise ValueError("dataset.metadata_paths is required for SCIDOCS development corpus")
        records = _development_corpus_records(metadata_paths)
        sources = [str(path) for path in metadata_paths]
    else:
        raise ValueError("dataset.protocol must be development or beir_test")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    seen: set[str] = set()
    count = 0
    empty_text = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            candidate_id = record["candidate_id"]
            if candidate_id in seen:
                raise ValueError(f"Duplicate SCIDOCS candidate ID: {candidate_id}")
            seen.add(candidate_id)
            count += 1
            empty_text += not bool(record["text"])
            handle.write(_json(record) + "\n")
    os.replace(temporary, destination)
    return {"candidate_count": count, "empty_text_count": int(empty_text), "sources": sources}


def _configured_forbidden_ids(config: dict[str, Any]) -> set[str]:
    dataset = config["dataset"]
    result: set[str] = set()
    forbidden_path = dataset.get("forbidden_query_ids_path")
    if forbidden_path:
        result.update(row["query_id"] for row in load_beir_queries(forbidden_path))
    if str(dataset.get("protocol")) == "beir_test" and dataset.get("beir_queries_path"):
        result.update(row["query_id"] for row in load_beir_queries(dataset["beir_queries_path"]))
    return result


def audit_protocol(config: dict[str, Any], corpus_path: Path) -> dict[str, Any]:
    corpus = load_candidate_corpus(corpus_path)
    corpus_ids = {corpus.candidate_id_at(row) for row in range(len(corpus))}
    dataset = config["dataset"]
    expected_positives = dataset.get("expected_positive_judgment_count")
    if str(dataset.get("protocol")) == "beir_test":
        raw_positive_count = sum(
            len(values) for values in load_beir_qrels(dataset["beir_qrels_path"]).values()
        )
        if expected_positives is not None and raw_positive_count != int(expected_positives):
            raise ValueError(
                "SCIDOCS positive judgment count mismatch: "
                f"{raw_positive_count} != {expected_positives}. "
                "The qrels file is not the expected BEIR-SCIDOCS test.tsv."
            )
    adapter = create_dataset_adapter(config)
    instances = list(adapter.iter_instances())
    dropped_judgments = (
        list(adapter.dropped_positive_judgments) if isinstance(adapter, SCIDOCSAdapter) else []
    )
    dropped_queries = list(adapter.dropped_queries) if isinstance(adapter, SCIDOCSAdapter) else []
    query_ids = [instance.query.query_id for instance in instances]
    positives = [candidate_id for instance in instances for candidate_id in instance.positive_ids]
    positive_in_corpus = sum(candidate_id in corpus_ids for candidate_id in positives)
    raw_positive_count = len(positives) + len(dropped_judgments)
    query_split_counts = Counter(str(instance.metadata.get("split", "unknown")) for instance in instances)
    allowed_candidate_restrictions = sum(
        instance.query.allowed_candidate_ids is not None for instance in instances
    )
    report: dict[str, Any] = {
        "protocol": config["dataset"].get("protocol"),
        "candidate_count": len(corpus),
        "query_count": len(instances),
        "query_split_counts": dict(query_split_counts),
        "unique_query_count": len(set(query_ids)),
        "duplicate_query_count": len(query_ids) - len(set(query_ids)),
        "query_ids_in_candidate_corpus": sum(query_id in corpus_ids for query_id in query_ids),
        "positive_judgment_count": len(positives),
        "positive_ids_in_candidate_corpus": positive_in_corpus,
        "positive_in_corpus_rate": positive_in_corpus / len(positives) if positives else 0.0,
        "raw_positive_judgment_count": raw_positive_count,
        "raw_positive_in_corpus_rate": positive_in_corpus / raw_positive_count if raw_positive_count else 0.0,
        "dropped_missing_positive_judgment_count": len(dropped_judgments),
        "dropped_missing_positive_examples": dropped_judgments[:20],
        "dropped_query_count": len(dropped_queries),
        "dropped_query_examples": dropped_queries[:20],
        "queries_with_all_positives_in_corpus": sum(
            all(candidate_id in corpus_ids for candidate_id in instance.positive_ids)
            for instance in instances
        ),
        "allowed_candidate_restrictions": allowed_candidate_restrictions,
        "full_corpus_retrieval": allowed_candidate_restrictions == 0,
        "forbidden_outgoing_query_count": len(_configured_forbidden_ids(config)),
    }
    expected_candidates = dataset.get("expected_candidate_count")
    expected_queries = dataset.get("expected_query_count")
    if expected_candidates is not None and len(corpus) != int(expected_candidates):
        raise ValueError(f"SCIDOCS candidate count mismatch: {len(corpus)} != {expected_candidates}")
    if expected_queries is not None and len(instances) != int(expected_queries):
        raise ValueError(f"SCIDOCS query count mismatch: {len(instances)} != {expected_queries}")
    if allowed_candidate_restrictions:
        raise ValueError("SCIDOCS must use full-corpus retrieval; allowed_candidate_ids must stay unset")
    if positives and positive_in_corpus != len(positives):
        raise ValueError(
            f"SCIDOCS ID join failed: only {positive_in_corpus}/{len(positives)} positives are in the corpus"
        )
    return report


def build_graph(config: dict[str, Any], corpus_path: Path, destination: Path) -> dict[str, Any]:
    graph_config = config.get("scidocs_graph", {})
    enrichment_paths = graph_config.get("enrichment_paths")
    if enrichment_paths is None:
        enrichment_paths = [graph_config["enrichment_path"]] if graph_config.get("enrichment_path") else []
    missing = [str(path) for path in enrichment_paths if not Path(path).exists()]
    if graph_config.get("require_enrichment", False) and missing:
        raise FileNotFoundError(f"Missing required SCIDOCS graph enrichment: {missing}")
    return SCIDOCSGraphBuilder(
        load_candidate_corpus(corpus_path),
        enrichment_path=enrichment_paths,
        forbidden_outgoing_paper_ids=_configured_forbidden_ids(config),
        reverse_citations=bool(graph_config.get("reverse_citations", True)),
    ).build(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and build leakage-safe SCIDOCS retrieval assets")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=["audit", "corpus", "graph", "embeddings", "all"],
        default="audit",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("dataset", {}).get("adapter") != "scidocs":
        raise ValueError("prepare_scidocs.py requires dataset.adapter: scidocs")
    if "qrel" in _json(config.get("scidocs_graph", {})).casefold():
        raise ValueError("scidocs_graph must not reference qrels")

    paths = _paths(config)
    paths["asset_dir"].mkdir(parents=True, exist_ok=True)
    manifest = (
        json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if paths["manifest"].exists()
        else {"schema_version": 2}
    )
    if args.stage in {"all", "corpus"} and (args.force or not paths["corpus"].exists()):
        manifest["corpus"] = build_corpus(config, paths["corpus"])
    if not paths["corpus"].exists():
        raise FileNotFoundError("Build the SCIDOCS candidate corpus before audit/graph/embeddings")

    if args.stage in {"all", "audit", "corpus"}:
        report = audit_protocol(config, paths["corpus"])
        write_json(paths["audit"], report)
        manifest["protocol_audit"] = report
    if args.stage in {"all", "graph"} and (args.force or not paths["graph"].exists()):
        manifest["global_graph"] = build_graph(config, paths["corpus"], paths["graph"])
    if args.stage in {"all", "embeddings"} and (
        args.force or not (paths["embeddings"].exists() and paths["faiss"].exists())
    ):
        manifest["semantic_index"] = build_semantic_assets(
            paths["corpus"],
            paths["embeddings"],
            paths["faiss"],
            create_embedder(config.get("embedding", {})),
            batch_size=int(config["global_retrieval"].get("embedding_batch_size", 256)),
        )
    manifest["embedding"] = config.get("embedding", {})
    manifest["dataset"] = config.get("dataset", {})
    manifest["assets"] = {
        key: str(value) for key, value in paths.items() if key not in {"asset_dir", "audit"}
    }
    write_json(paths["manifest"], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
