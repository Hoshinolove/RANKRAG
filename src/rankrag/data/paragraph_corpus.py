from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from rankrag.data.json_stream import iter_json_array
from rankrag.models import Candidate


def paragraph_id(title: str, text: str) -> str:
    """Return the stable ID shared by corpus, KG extraction, and evaluation."""
    return hashlib.sha1(f"{title.strip()}\n{text.strip()}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParagraphRecord:
    paragraph_id: str
    title: str
    text: str
    sources: tuple[str, ...] = ()

    def to_candidate(self) -> Candidate:
        return Candidate(
            candidate_id=self.paragraph_id,
            text=self.text,
            metadata={"title": self.title, "paragraph_id": self.paragraph_id, "sources": list(self.sources)},
        )


class ParagraphCorpus:
    """Immutable, row-addressable paragraph corpus used by the FAISS index."""

    def __init__(self, records: Sequence[ParagraphRecord]) -> None:
        self.records = list(records)
        self.id_to_row = {record.paragraph_id: row for row, record in enumerate(self.records)}
        if len(self.id_to_row) != len(self.records):
            raise ValueError("Paragraph corpus contains duplicate paragraph_id values")

    def __len__(self) -> int:
        return len(self.records)

    def candidate(self, paragraph_id_value: str) -> Candidate:
        return self.records[self.id_to_row[paragraph_id_value]].to_candidate()

    @classmethod
    def load(cls, path: str | Path) -> "ParagraphCorpus":
        records: list[ParagraphRecord] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                records.append(
                    ParagraphRecord(
                        paragraph_id=str(value["paragraph_id"]),
                        title=str(value["title"]),
                        text=str(value["text"]),
                        sources=tuple(str(item) for item in value.get("sources", [])),
                    )
                )
        return cls(records)


def build_paragraph_corpus(input_paths: Sequence[str | Path], output_path: str | Path) -> dict[str, Any]:
    """Deduplicate source paragraphs and write them in stable paragraph-ID order."""
    unique: dict[str, dict[str, Any]] = {}
    input_count = 0
    for raw_path in input_paths:
        path = Path(raw_path)
        for value in iter_json_array(path):
            input_count += 1
            title = str(value.get("title", "")).strip()
            text = str(value.get("text", "")).strip()
            if not title or not text:
                continue
            pid = paragraph_id(title, text)
            if pid not in unique:
                unique[pid] = {"paragraph_id": pid, "title": title, "text": text, "sources": [str(path)]}
            elif str(path) not in unique[pid]["sources"]:
                unique[pid]["sources"].append(str(path))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for pid in sorted(unique):
            line = json.dumps(unique[pid], ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    os.replace(temporary, destination)
    return {
        "input_records": input_count,
        "paragraph_count": len(unique),
        "duplicates_removed": input_count - len(unique),
        "corpus_sha256": digest.hexdigest(),
        "corpus_path": str(destination),
    }


def inspect_paragraph_corpus(input_paths: Sequence[str | Path], corpus_path: str | Path) -> dict[str, Any]:
    """Recover manifest statistics for an already-built paragraph corpus."""
    input_count = sum(1 for path in input_paths for _ in iter_json_array(path))
    paragraph_count = 0
    digest = hashlib.sha256()
    corpus = Path(corpus_path)
    with corpus.open("rb") as handle:
        for line in handle:
            if line.strip():
                paragraph_count += 1
            digest.update(line)
    return {
        "input_records": input_count,
        "paragraph_count": paragraph_count,
        "duplicates_removed": input_count - paragraph_count,
        "corpus_sha256": digest.hexdigest(),
        "corpus_path": str(corpus),
    }
