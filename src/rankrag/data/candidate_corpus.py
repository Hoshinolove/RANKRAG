from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator

from rankrag.models import Candidate


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    text: str
    embedding_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sources: tuple[str, ...] = ()

    def to_candidate(self) -> Candidate:
        return Candidate(self.candidate_id, self.text, dict(self.metadata))


class CandidateCorpus(ABC):
    """Row-addressable global candidate collection aligned with a semantic index."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def candidate(self, candidate_id: str) -> Candidate: ...

    @abstractmethod
    def candidate_at(self, row: int) -> Candidate: ...

    @abstractmethod
    def candidate_id_at(self, row: int) -> str: ...

    @abstractmethod
    def row_for_id(self, candidate_id: str) -> int: ...

    @abstractmethod
    def embedding_text_at(self, row: int) -> str: ...

    def embedding_texts(self, start: int, end: int) -> list[str]:
        return [self.embedding_text_at(row) for row in range(start, end)]


class JsonlCandidateCorpus(CandidateCorpus):
    """Generic corpus schema used by non-HotpotQA datasets."""

    def __init__(self, records: list[CandidateRecord]) -> None:
        self.records = records
        self.id_to_row = {record.candidate_id: row for row, record in enumerate(records)}
        if len(self.id_to_row) != len(records):
            raise ValueError("Candidate corpus contains duplicate candidate IDs")

    def __len__(self) -> int:
        return len(self.records)

    def candidate(self, candidate_id: str) -> Candidate:
        return self.records[self.row_for_id(candidate_id)].to_candidate()

    def candidate_at(self, row: int) -> Candidate:
        return self.records[row].to_candidate()

    def candidate_id_at(self, row: int) -> str:
        return self.records[row].candidate_id

    def row_for_id(self, candidate_id: str) -> int:
        return self.id_to_row[candidate_id]

    def embedding_text_at(self, row: int) -> str:
        return self.records[row].embedding_text

    @classmethod
    def load(cls, path: str | Path) -> "JsonlCandidateCorpus":
        records: list[CandidateRecord] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                records.append(
                    CandidateRecord(
                        candidate_id=str(value["candidate_id"]),
                        text=str(value.get("text", "")),
                        embedding_text=str(value.get("embedding_text", value.get("text", ""))),
                        metadata=dict(value.get("metadata", {})),
                        sources=tuple(str(item) for item in value.get("sources", [])),
                    )
                )
        return cls(records)


def _first_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"Candidate corpus is empty: {path}")


def load_candidate_corpus(path: str | Path) -> CandidateCorpus:
    """Load schema-v2 candidates or the unchanged legacy HotpotQA corpus."""
    corpus_path = Path(path)
    first = _first_record(corpus_path)
    if "candidate_id" in first:
        return JsonlCandidateCorpus.load(corpus_path)
    if "paragraph_id" in first:
        from rankrag.data.paragraph_corpus import ParagraphCorpus

        return ParagraphCorpus.load(corpus_path)
    raise ValueError(f"Unknown candidate corpus schema: {corpus_path}")


def iter_candidate_records(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
