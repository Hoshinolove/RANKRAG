from __future__ import annotations

from typing import Any

from rankrag.graph.store import GraphStore


def serialize_path_evidence(store: GraphStore, paths: list[list[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node_values: dict[str, dict[str, Any]] = {}
    edge_values: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for node_id in path:
            node = store.node(node_id)
            if node:
                node_values[node_id] = {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "text": node.text,
                    "metadata": node.metadata,
                }
        for source, target in zip(path, path[1:], strict=False):
            edge = store.edge(source, target)
            if edge:
                key = tuple(sorted((source, target)))
                edge_values[key] = {
                    "source": edge.source,
                    "relation": edge.relation,
                    "target": edge.target,
                    "metadata": edge.metadata,
                }
    return list(node_values.values()), list(edge_values.values())
