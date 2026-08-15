from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from rankrag.llm.base import LLMRequest, LLMResponse


class LLMResponseCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(request: LLMRequest, provider_identity: str) -> str:
        payload = {
            "query_id": request.query_id,
            "query_text": request.query_text,
            "candidates": request.candidates,
            "model": provider_identity,
            "prompt_version": request.prompt_version,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> LLMResponse | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return LLMResponse(value["ranking"], value.get("raw_response"), value.get("metadata", {}))

    def put(self, key: str, response: LLMResponse) -> None:
        path = self.directory / f"{key}.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                {"ranking": response.ranking, "raw_response": response.raw_response, "metadata": response.metadata},
                handle,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(temporary, path)
