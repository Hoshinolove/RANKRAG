from __future__ import annotations

from typing import Any, Callable

from rankrag.data.base import DatasetAdapter
from rankrag.graph.builder import GraphBuilder, HotpotQAGraphBuilder, KGExtractionIndex
from rankrag.graph.candidate_index import CandidateGraphIndex, open_candidate_graph_index


def create_candidate_graph_index(config: dict[str, Any]) -> CandidateGraphIndex:
    graph_config = config.get("graph_index", {})
    global_config = config.get("global_retrieval", {})
    path = graph_config.get("path", global_config.get("graph_index_path"))
    if not path:
        raise ValueError("graph_index.path or global_retrieval.graph_index_path is required")
    backend = str(graph_config.get("backend", "hotpot_legacy_sqlite"))
    return open_candidate_graph_index(path, backend)


InstanceGraphBuilderFactory = Callable[[dict[str, Any], DatasetAdapter, int | None], GraphBuilder]


def _create_hotpotqa_instance_graph_builder(
    config: dict[str, Any],
    adapter: DatasetAdapter,
    limit: int | None = None,
) -> GraphBuilder:
    graph_config = config.get("graph", {})
    index = KGExtractionIndex(graph_config.get("extractions_path"))
    candidate_ids: set[str] = set()
    for instance in adapter.iter_instances(limit):
        candidate_ids.update(candidate.candidate_id for candidate in instance.candidates)
    index.load_for_titles(candidate_ids)
    return HotpotQAGraphBuilder(
        index,
        lexical_fallback=bool(graph_config.get("lexical_fallback", True)),
        max_fallback_terms=int(graph_config.get("max_fallback_terms", 10)),
    )


_INSTANCE_GRAPH_BUILDERS: dict[str, InstanceGraphBuilderFactory] = {
    "hotpotqa": _create_hotpotqa_instance_graph_builder,
}


def create_instance_graph_builder(
    config: dict[str, Any],
    adapter: DatasetAdapter,
    limit: int | None = None,
) -> GraphBuilder:
    dataset = config.get("dataset", {})
    builder_name = str(
        config.get("graph", {}).get(
            "builder",
            dataset.get("adapter", dataset.get("name", "hotpotqa")),
        )
    )
    try:
        factory = _INSTANCE_GRAPH_BUILDERS[builder_name]
    except KeyError as exc:
        raise ValueError(
            f"No instance-local graph builder is registered for {builder_name!r}; use global retrieval"
        ) from exc
    return factory(config, adapter, limit)
