from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
import csv
import json
from pathlib import Path
from typing import Any

from rankrag.data.base import DatasetAdapter
from rankrag.data.candidate_corpus import iter_candidate_records
from rankrag.models import Query, RecommendationInstance


def load_paper_metadata(paths: list[str | Path]) -> dict[str, dict[str, Any]]:
    """Load original SciDocs metadata files without assuming one JSON layout."""
    papers: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing SciDocs paper metadata: {path}")
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            rows = ((str(paper_id), metadata) for paper_id, metadata in value.items())
        elif isinstance(value, list):
            rows = (
                (str(metadata.get("paper_id", metadata.get("_id", metadata.get("id", "")))), metadata)
                for metadata in value
            )
        else:
            raise ValueError(f"Unsupported SciDocs metadata structure: {path}")
        for paper_id, metadata in rows:
            if not paper_id or not isinstance(metadata, dict):
                continue
            candidate = dict(metadata)
            previous = papers.get(paper_id)
            if previous is None:
                papers[paper_id] = candidate
                continue
            previous_text = str(previous.get("abstract", previous.get("text", "")))
            candidate_text = str(candidate.get("abstract", candidate.get("text", "")))
            if len(candidate_text) > len(previous_text):
                papers[paper_id] = {**previous, **candidate}
    return papers


def load_beir_queries(path: str | Path) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            query_id = str(value.get("_id", value.get("query_id", "")))
            text = str(value.get("text", value.get("title", ""))).strip()
            if not query_id:
                raise ValueError(f"SCIDOCS query is missing _id: {value}")
            queries.append({"query_id": query_id, "text": text})
    return queries


def load_beir_qrels(path: str | Path) -> dict[str, list[str]]:
    positives: dict[str, list[str]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"SCIDOCS qrels has no header: {path}")
        for row in reader:
            query_id = str(row.get("query-id", row.get("query_id", "")))
            candidate_id = str(row.get("corpus-id", row.get("corpus_id", "")))
            score = int(row.get("score", 0))
            if query_id and candidate_id and score > 0:
                positives[query_id].append(candidate_id)
    return {query_id: list(dict.fromkeys(values)) for query_id, values in positives.items()}


def load_original_qrels(path: str | Path) -> tuple[list[str], dict[str, list[str]]]:
    order: list[str] = []
    positives: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 4:
                raise ValueError(f"Invalid original SciDocs qrel row: {line.rstrip()}")
            query_id, candidate_id = parts[0], parts[2]
            try:
                score = int(parts[3])
            except ValueError:
                continue
            if query_id not in seen:
                order.append(query_id)
                seen.add(query_id)
            if score > 0:
                positives[query_id].append(candidate_id)
    return order, {query_id: list(dict.fromkeys(values)) for query_id, values in positives.items()}


class SCIDOCSAdapter(DatasetAdapter):
    """Map BEIR test and original SciDocs development data to one generic schema."""

    def __init__(
        self,
        protocol: str,
        split: str,
        beir_queries_path: str | Path | None = None,
        beir_qrels_path: str | Path | None = None,
        metadata_paths: list[str | Path] | None = None,
        recomm_train_path: str | Path | None = None,
        citation_validation_qrels_path: str | Path | None = None,
        forbidden_query_ids_path: str | Path | None = None,
        candidate_corpus_path: str | Path | None = None,
        missing_positive_policy: str = "strict",
    ) -> None:
        if protocol not in {"development", "beir_test"}:
            raise ValueError("SCIDOCS protocol must be development or beir_test")
        if split not in {"all", "train", "validation", "test"}:
            raise ValueError("SCIDOCS split must be all, train, validation, or test")
        if missing_positive_policy not in {"strict", "drop"}:
            raise ValueError("SCIDOCS missing_positive_policy must be strict or drop")
        self.protocol = protocol
        self.split = split
        self.beir_queries_path = Path(beir_queries_path) if beir_queries_path else None
        self.beir_qrels_path = Path(beir_qrels_path) if beir_qrels_path else None
        self.metadata_paths = [Path(path) for path in metadata_paths or []]
        self.recomm_train_path = Path(recomm_train_path) if recomm_train_path else None
        self.citation_validation_qrels_path = (
            Path(citation_validation_qrels_path) if citation_validation_qrels_path else None
        )
        self.forbidden_query_ids_path = Path(forbidden_query_ids_path) if forbidden_query_ids_path else None
        self.candidate_corpus_path = Path(candidate_corpus_path) if candidate_corpus_path else None
        self.missing_positive_policy = missing_positive_policy
        self._metadata: dict[str, dict[str, Any]] | None = None
        self._forbidden_query_ids: set[str] | None = None
        self._candidate_ids: set[str] | None = None
        self.dropped_positive_judgments: list[dict[str, str]] = []
        self.dropped_queries: list[str] = []

    def source_paths(self) -> tuple[Path, ...]:
        values = [
            *self.metadata_paths,
            self.beir_queries_path,
            self.beir_qrels_path,
            self.recomm_train_path,
            self.citation_validation_qrels_path,
            self.forbidden_query_ids_path,
        ]
        return tuple(path for path in values if path is not None)

    @property
    def metadata(self) -> dict[str, dict[str, Any]]:
        if self._metadata is None:
            self._metadata = load_paper_metadata([str(path) for path in self.metadata_paths])
        return self._metadata

    @property
    def forbidden_query_ids(self) -> set[str]:
        if self._forbidden_query_ids is None:
            if self.forbidden_query_ids_path is None:
                self._forbidden_query_ids = set()
            else:
                self._forbidden_query_ids = {
                    value["query_id"] for value in load_beir_queries(self.forbidden_query_ids_path)
                }
        return self._forbidden_query_ids

    @property
    def candidate_ids(self) -> set[str] | None:
        if self.candidate_corpus_path is None or not self.candidate_corpus_path.exists():
            return None
        if self._candidate_ids is None:
            self._candidate_ids = {
                str(value.get("candidate_id", value.get("_id", "")))
                for value in iter_candidate_records(self.candidate_corpus_path)
            }
            self._candidate_ids.discard("")
        return self._candidate_ids

    def _query_text(self, paper_id: str) -> str:
        metadata = self.metadata.get(paper_id, {})
        return str(metadata.get("title", metadata.get("text", paper_id))).strip() or paper_id

    def _instance(
        self,
        query_id: str,
        query_text: str,
        positive_ids: list[str],
        split: str,
        source: str,
    ) -> RecommendationInstance | None:
        positive_ids = list(dict.fromkeys(positive_ids))
        candidate_ids = self.candidate_ids
        missing_ids = (
            [candidate_id for candidate_id in positive_ids if candidate_id not in candidate_ids]
            if candidate_ids is not None
            else []
        )
        if missing_ids and self.missing_positive_policy == "strict":
            raise ValueError(
                f"SCIDOCS positive IDs are missing from the candidate corpus for query {query_id}: "
                f"{missing_ids[:10]}"
            )
        if missing_ids:
            self.dropped_positive_judgments.extend(
                {"query_id": query_id, "candidate_id": candidate_id, "split": split, "source": source}
                for candidate_id in missing_ids
            )
            positive_ids = [candidate_id for candidate_id in positive_ids if candidate_id in candidate_ids]
        if not positive_ids:
            self.dropped_queries.append(query_id)
            return None
        return RecommendationInstance(
            query=Query(
                query_id=query_id,
                text=query_text,
                # A query paper may seed its qrel-independent author/topic graph.
                # The generic generator silently drops this seed if it is not in the corpus.
                seed_candidate_ids=(query_id,),
                seed_candidate_weights=(1.0,),
                excluded_candidate_ids=(query_id,),
                allowed_candidate_ids=None,
                metadata={"split": split, "source": source},
            ),
            candidates=[],
            positive_ids=positive_ids,
            metadata={
                "split": split,
                "source": source,
                "dropped_positive_ids": missing_ids,
            },
        )

    def _iter_recommendation_train(self) -> Iterator[RecommendationInstance]:
        if self.recomm_train_path is None:
            raise ValueError("dataset.recomm_train_path is required for SCIDOCS development training")
        with self.recomm_train_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                return
            normalized = [value.strip().casefold() for value in header]
            has_header = "query_id" in normalized and "clicked_id" in normalized
            rows = reader if has_header else iter([header, *reader])
            query_index = normalized.index("query_id") if has_header else 0
            clicked_index = normalized.index("clicked_id") if has_header else 1
            for row in rows:
                if len(row) <= max(query_index, clicked_index):
                    continue
                query_id = str(row[query_index]).strip()
                clicked_id = str(row[clicked_index]).strip()
                if not query_id or not clicked_id or query_id in self.forbidden_query_ids:
                    continue
                instance = self._instance(
                    query_id,
                    self._query_text(query_id),
                    [clicked_id],
                    "train",
                    "original_scidocs_recomm",
                )
                if instance is not None:
                    yield instance

    def _iter_citation_validation(self) -> Iterator[RecommendationInstance]:
        if self.citation_validation_qrels_path is None:
            raise ValueError("dataset.citation_validation_qrels_path is required for validation")
        order, positives = load_original_qrels(self.citation_validation_qrels_path)
        for query_id in order:
            if query_id in self.forbidden_query_ids or not positives.get(query_id):
                continue
            instance = self._instance(
                query_id,
                self._query_text(query_id),
                positives[query_id],
                "validation",
                "original_scidocs_cite_val",
            )
            if instance is not None:
                yield instance

    def _iter_beir_test(self) -> Iterator[RecommendationInstance]:
        if self.beir_queries_path is None or self.beir_qrels_path is None:
            raise ValueError("dataset.beir_queries_path and dataset.beir_qrels_path are required")
        positives = load_beir_qrels(self.beir_qrels_path)
        for query in load_beir_queries(self.beir_queries_path):
            query_id = query["query_id"]
            if not positives.get(query_id):
                raise ValueError(f"Official SCIDOCS query has no positive qrel: {query_id}")
            instance = self._instance(
                query_id,
                query["text"],
                positives[query_id],
                "test",
                "beir_scidocs",
            )
            if instance is not None:
                yield instance

    def iter_instances(self, limit: int | None = None) -> Iterator[RecommendationInstance]:
        if self.protocol == "beir_test":
            sources = [self._iter_beir_test()]
        else:
            sources = []
            if self.split in {"all", "train"}:
                sources.append(self._iter_recommendation_train())
            if self.split in {"all", "validation"}:
                sources.append(self._iter_citation_validation())
        count = 0
        for source in sources:
            for instance in source:
                if limit is not None and count >= limit:
                    return
                yield instance
                count += 1
