from rankrag.data.hotpotqa import HotpotQAAdapter
from rankrag.data.paragraph_corpus import paragraph_id


def test_hotpot_adapter_streams_common_instances():
    instances = list(HotpotQAAdapter("tests/fixtures/hotpot_tiny.json").iter_instances())
    assert len(instances) == 2
    assert instances[0].query.query_id == "q1"
    assert [candidate.candidate_id for candidate in instances[0].candidates] == ["Noise", "Alice", "Paris"]
    assert instances[0].positive_ids == ["Alice", "Paris"]


def test_hotpot_adapter_honors_limit():
    instances = list(HotpotQAAdapter("tests/fixtures/hotpot_tiny.json").iter_instances(limit=1))
    assert [instance.query.query_id for instance in instances] == ["q1"]


def test_hotpot_adapter_uses_stable_paragraph_ids_for_global_evaluation():
    instance = next(
        HotpotQAAdapter("tests/fixtures/hotpot_tiny.json", use_paragraph_ids=True).iter_instances()
    )
    alice_id = paragraph_id("Alice", "Alice lives in Paris.")
    paris_id = paragraph_id("Paris", "Paris is a city in France.")
    assert instance.positive_ids == [alice_id, paris_id]
    assert {candidate.candidate_id for candidate in instance.candidates} >= {alice_id, paris_id}
