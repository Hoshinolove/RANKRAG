from rankrag.llm.base import LLMRequest
from rankrag.llm.cache import LLMResponseCache


def test_cache_key_changes_with_model_prompt_or_candidates(tmp_path):
    cache = LLMResponseCache(tmp_path)
    base = LLMRequest("q", "question", [{"candidate_id": "a"}], "v1")
    assert cache.key(base, "model-a") == cache.key(base, "model-a")
    assert cache.key(base, "model-a") != cache.key(base, "model-b")
    changed = LLMRequest("q", "question", [{"candidate_id": "b"}], "v1")
    assert cache.key(base, "model-a") != cache.key(changed, "model-a")
