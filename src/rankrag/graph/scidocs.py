from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from rankrag.data.candidate_corpus import CandidateCorpus
from rankrag.graph.candidate_index import create_candidate_graph_schema


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _stable_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _identifier(value: Any, keys: tuple[str, ...]) -> tuple[str, str]:
    if isinstance(value, dict):
        identifier = next((str(value[key]).strip() for key in keys if value.get(key)), "")
        label = str(value.get("name", value.get("label", identifier))).strip()
        return identifier, label
    text = str(value).strip()
    return text, text


def load_scidocs_enrichment(
    path: str | Path | Iterable[str | Path] | None,
) -> dict[str, dict[str, Any]]:
    """Load optional paper relations. This file must be independent of evaluation qrels."""
    records: dict[str, dict[str, Any]] = {}
    sources = [path] if isinstance(path, (str, Path)) else list(path or [])
    for raw_source in sources:
        source = Path(raw_source)
        if not source.exists():
            continue
        loaded: dict[str, dict[str, Any]] = {}
        with source.open("r", encoding="utf-8") as handle:
            if source.suffix.casefold() == ".jsonl":
                values = (json.loads(line) for line in handle if line.strip())
                for value in values:
                    paper_id = str(
                        value.get("paper_id", value.get("_id", value.get("id", "")))
                    ).strip()
                    if paper_id:
                        loaded[paper_id] = dict(value)
            else:
                value = json.load(handle)
                if isinstance(value, dict):
                    for paper_id, metadata in value.items():
                        if isinstance(metadata, dict):
                            loaded[str(paper_id)] = dict(metadata)
                elif isinstance(value, list):
                    for metadata in value:
                        if not isinstance(metadata, dict):
                            continue
                        paper_id = str(
                            metadata.get("paper_id", metadata.get("_id", metadata.get("id", "")))
                        ).strip()
                        if paper_id:
                            loaded[paper_id] = dict(metadata)
                else:
                    raise ValueError(f"Unsupported SCIDOCS enrichment structure: {source}")
        for paper_id, metadata in loaded.items():
            records[paper_id] = {**records.get(paper_id, {}), **metadata}
    return records


class SCIDOCSGraphBuilder:
    """Build a generic candidate graph from qrel-independent SciDocs metadata."""

    def __init__(
        self,
        corpus: CandidateCorpus,
        enrichment_path: str | Path | Iterable[str | Path] | None = None,
        forbidden_outgoing_paper_ids: Iterable[str] = (),
        reverse_citations: bool = True,
    ) -> None:
        self.corpus = corpus
        if isinstance(enrichment_path, (str, Path)):
            self.enrichment_paths = (Path(enrichment_path),)
        else:
            self.enrichment_paths = tuple(Path(path) for path in enrichment_path or [])
        self.forbidden_outgoing_paper_ids = set(forbidden_outgoing_paper_ids)
        self.reverse_citations = bool(reverse_citations)

    @staticmethod
    def _candidate_node_id(candidate_id: str) -> str:
        return f"candidate::{candidate_id}"

    def build(self, destination: str | Path) -> dict[str, Any]:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        enrichment = load_scidocs_enrichment(self.enrichment_paths)
        candidate_ids = {self.corpus.candidate_id_at(row) for row in range(len(self.corpus))}
        counts: Counter[str] = Counter()
        connection = sqlite3.connect(temporary)
        try:
            create_candidate_graph_schema(connection)
            for row in range(len(self.corpus)):
                candidate = self.corpus.candidate_at(row)
                candidate_id = candidate.candidate_id
                node_id = self._candidate_node_id(candidate_id)
                connection.execute(
                    "INSERT INTO nodes VALUES (?, ?, ?, ?)",
                    (node_id, "candidate", candidate.text, _json({"candidate_id": candidate_id})),
                )
                connection.execute(
                    "INSERT INTO candidate_nodes VALUES (?, ?, ?, ?, ?)",
                    (candidate_id, node_id, "represents", 1.0, "{}"),
                )
                counts["candidate_nodes"] += 1

            for paper_id, metadata in enrichment.items():
                if paper_id not in candidate_ids:
                    counts["enrichment_papers_outside_corpus"] += 1
                    continue
                for author in _stable_values(metadata.get("authors")):
                    author_id, label = _identifier(author, ("authorId", "author_id", "id", "name"))
                    if not author_id:
                        continue
                    node_id = f"author::{author_id}"
                    connection.execute(
                        "INSERT OR IGNORE INTO nodes VALUES (?, ?, ?, ?)",
                        (node_id, "author", label, _json({"author_id": author_id})),
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO candidate_nodes VALUES (?, ?, ?, ?, ?)",
                        (paper_id, node_id, "authored_by", 1.0, "{}"),
                    )
                    counts["author_links"] += 1

                fields = metadata.get("fields", metadata.get("fieldsOfStudy", metadata.get("topics")))
                for field in _stable_values(fields):
                    field_id, label = _identifier(field, ("field_id", "topic_id", "id", "name"))
                    if not field_id:
                        continue
                    node_id = f"field::{field_id.casefold()}"
                    connection.execute(
                        "INSERT OR IGNORE INTO nodes VALUES (?, ?, ?, ?)",
                        (node_id, "field", label, _json({"field_id": field_id})),
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO candidate_nodes VALUES (?, ?, ?, ?, ?)",
                        (paper_id, node_id, "has_field", 1.0, "{}"),
                    )
                    counts["field_links"] += 1

                references = metadata.get("references", metadata.get("outbound_citations", []))
                if paper_id in self.forbidden_outgoing_paper_ids:
                    counts["forbidden_outgoing_papers"] += 1
                    counts["skipped_forbidden_references"] += len(_stable_values(references))
                    continue
                for reference in _stable_values(references):
                    target_id, _ = _identifier(
                        reference,
                        ("paperId", "paper_id", "corpus_id", "id"),
                    )
                    if not target_id or target_id not in candidate_ids or target_id == paper_id:
                        continue
                    source_node = self._candidate_node_id(paper_id)
                    target_node = self._candidate_node_id(target_id)
                    connection.execute(
                        "INSERT OR IGNORE INTO graph_edges VALUES (?, ?, ?, ?, ?)",
                        (source_node, target_node, "cites", 1.0, "{}"),
                    )
                    counts["citation_edges"] += 1
                    if self.reverse_citations:
                        connection.execute(
                            "INSERT OR IGNORE INTO graph_edges VALUES (?, ?, ?, ?, ?)",
                            (target_node, source_node, "cited_by", 1.0, "{}"),
                        )
                        counts["reverse_citation_edges"] += 1

            connection.commit()
            result = {
                "candidate_count": len(candidate_ids),
                "enrichment_paths": [str(path) for path in self.enrichment_paths],
                "enriched_papers_in_corpus": sum(1 for paper_id in enrichment if paper_id in candidate_ids),
                "forbidden_outgoing_paper_count": len(self.forbidden_outgoing_paper_ids),
                "nodes": int(connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]),
                "candidate_node_edges": int(
                    connection.execute("SELECT COUNT(*) FROM candidate_nodes").fetchone()[0]
                ),
                "graph_edges": int(connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]),
                **dict(counts),
            }
        finally:
            connection.close()
        os.replace(temporary, destination)
        return result
