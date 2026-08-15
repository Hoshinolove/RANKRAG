from __future__ import annotations

import json

from rankrag.llm.base import LLMRequest


SYSTEM_PROMPTS = {
    "v1": (
        "You are a ranking model. Rank every candidate by relevance to the query, using graph evidence "
        "when useful. Return JSON only with shape: "
        '{"ranking":[{"candidate_id":"...","score":0.0,"reason":"..."}]}. '
        "Use each supplied candidate_id exactly once."
    )
}


def build_messages(request: LLMRequest) -> list[dict[str, str]]:
    if request.prompt_version not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt version: {request.prompt_version}")
    payload = {"query": request.query_text, "candidates": request.candidates}
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[request.prompt_version]},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
