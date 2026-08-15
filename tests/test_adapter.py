from rankrag.data.hotpotqa import HotpotQAAdapter


def test_hotpot_adapter_streams_common_instances():
    instances = list(HotpotQAAdapter("tests/fixtures/hotpot_tiny.json").iter_instances())
    assert len(instances) == 2
    assert instances[0].query.query_id == "q1"
    assert [candidate.candidate_id for candidate in instances[0].candidates] == ["Noise", "Alice", "Paris"]
    assert instances[0].positive_ids == ["Alice", "Paris"]


def test_hotpot_adapter_honors_limit():
    instances = list(HotpotQAAdapter("tests/fixtures/hotpot_tiny.json").iter_instances(limit=1))
    assert [instance.query.query_id for instance in instances] == ["q1"]
