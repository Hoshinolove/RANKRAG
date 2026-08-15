from __future__ import annotations

import json
import os
import re
from urllib import request as urlrequest

from rankrag.llm.base import LLMProvider, LLMRequest, LLMResponse
from rankrag.llm.prompt import build_messages


class PassthroughProvider(LLMProvider):
    """Offline provider for reproducible pipeline tests; preserves neural order."""

    @property
    def identity(self) -> str:
        return "passthrough"

    def rerank(self, request: LLMRequest) -> LLMResponse:
        ranking = [
            {"candidate_id": item["candidate_id"], "score": float(len(request.candidates) - index), "reason": "neural_order"}
            for index, item in enumerate(request.candidates)
        ]
        return LLMResponse(ranking=ranking, metadata={"offline": True})


class OpenAICompatibleProvider(LLMProvider):
    """Works with OpenAI-compatible Qwen, DeepSeek, and self-hosted endpoints."""

    def __init__(self, model: str, base_url: str, api_key_env: str, timeout: int = 120) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout

    @property
    def identity(self) -> str:
        return f"openai-compatible:{self.base_url}:{self.model}"

    def rerank(self, request: LLMRequest) -> LLMResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        body = json.dumps(
            {
                "model": self.model,
                "messages": build_messages(request),
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        http_request = urlrequest.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(http_request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        parsed = _parse_json_object(content)
        if not isinstance(parsed.get("ranking"), list):
            raise ValueError("LLM response does not contain a ranking array")
        return LLMResponse(parsed["ranking"], raw_response=content, metadata={"usage": payload.get("usage", {})})


def _parse_json_object(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from LLM")
    return value


def create_provider(config: dict) -> LLMProvider:
    provider = config.get("provider", "passthrough")
    if provider in {None, "passthrough"}:
        return PassthroughProvider()
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(
            model=config["model"],
            base_url=config["base_url"],
            api_key_env=config.get("api_key_env", "OPENAI_API_KEY"),
            timeout=int(config.get("timeout", 120)),
        )
    raise ValueError(f"Unknown LLM provider: {provider}")
