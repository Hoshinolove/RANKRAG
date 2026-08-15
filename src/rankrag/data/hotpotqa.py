from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rankrag.data.base import DatasetAdapter
from rankrag.data.json_stream import iter_json_array
from rankrag.models import Candidate, Query, RecommendationInstance


class HotpotQAAdapter(DatasetAdapter):
    """Maps HotpotQA questions and context paragraphs to the common ranking schema."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def iter_instances(self, limit: int | None = None) -> Iterator[RecommendationInstance]:
        for index, record in enumerate(iter_json_array(self.path)):
            if limit is not None and index >= limit:
                return
            yield self._convert(record)

    @staticmethod
    def _convert(record: dict[str, Any]) -> RecommendationInstance:
        context = record.get("context", {})
        titles = context.get("title", [])
        sentence_groups = context.get("sentences", [])
        candidates = [
            Candidate(
                candidate_id=str(title),
                text="\n".join(str(sentence) for sentence in sentences).strip(),
                metadata={"title": str(title), "sentences": list(sentences)},
            )
            for title, sentences in zip(titles, sentence_groups, strict=False)
        ]
        support = record.get("supporting_facts", {})
        positive_set = set(str(title) for title in support.get("title", []))
        candidate_ids = {candidate.candidate_id for candidate in candidates}
        positives = [candidate.candidate_id for candidate in candidates if candidate.candidate_id in positive_set]
        missing = sorted(positive_set - candidate_ids)
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
