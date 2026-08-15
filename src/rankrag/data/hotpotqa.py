from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rankrag.data.base import DatasetAdapter
from rankrag.data.json_stream import iter_json_array
from rankrag.data.paragraph_corpus import paragraph_id
from rankrag.models import Candidate, Query, RecommendationInstance


class HotpotQAAdapter(DatasetAdapter):
    """Maps HotpotQA questions and context paragraphs to the common ranking schema."""

    def __init__(self, path: str | Path, use_paragraph_ids: bool = False) -> None:
        self.path = Path(path)
        self.use_paragraph_ids = use_paragraph_ids

    def iter_instances(self, limit: int | None = None) -> Iterator[RecommendationInstance]:
        for index, record in enumerate(iter_json_array(self.path)):
            if limit is not None and index >= limit:
                return
            yield self._convert(record, use_paragraph_ids=self.use_paragraph_ids)

    @staticmethod
    def _convert(record: dict[str, Any], use_paragraph_ids: bool = False) -> RecommendationInstance:
        context = record.get("context", {})
        titles = context.get("title", [])
        sentence_groups = context.get("sentences", [])
        candidates = []
        for title, sentences in zip(titles, sentence_groups, strict=False):
            title_text = str(title).strip()
            paragraph_text = "\n".join(str(sentence) for sentence in sentences).strip()
            candidate_id = paragraph_id(title_text, paragraph_text) if use_paragraph_ids else title_text
            candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    text=paragraph_text,
                    metadata={
                        "title": title_text,
                        "paragraph_id": paragraph_id(title_text, paragraph_text),
                        "sentences": list(sentences),
                    },
                )
            )
        support = record.get("supporting_facts", {})
        positive_titles = set(str(title) for title in support.get("title", []))
        positives = [candidate.candidate_id for candidate in candidates if candidate.metadata["title"] in positive_titles]
        candidate_titles = {str(candidate.metadata["title"]) for candidate in candidates}
        missing = sorted(positive_titles - candidate_titles)
        return RecommendationInstance(
            query=Query(
                query_id=str(record.get("id", "")),
                text=str(record.get("question", "")),
                metadata={"answer": record.get("answer"), "type": record.get("type"), "level": record.get("level")},
            ),
            candidates=candidates,
            positive_ids=positives,
            metadata={"missing_positive_titles": missing},
        )
