from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import networkx as nx

from rankrag.models import Edge, Node


class GraphStore(ABC):
    @abstractmethod
    def add_nodes(self, nodes: Iterable[Node]) -> None: ...

    @abstractmethod
    def add_edges(self, edges: Iterable[Edge]) -> None: ...

    @abstractmethod
    def neighbors(self, node_id: str) -> list[str]: ...

    @abstractmethod
    def node(self, node_id: str) -> Node | None: ...

    @abstractmethod
    def edge(self, source: str, target: str) -> Edge | None: ...

    @abstractmethod
    def node_ids(self, node_type: str | None = None) -> list[str]: ...

    @abstractmethod
    def shortest_path(self, source: str, target: str, max_hops: int) -> list[str] | None: ...


class NetworkXGraphStore(GraphStore):
    def __init__(self) -> None:
        self.graph = nx.Graph()

    def add_nodes(self, nodes: Iterable[Node]) -> None:
        for node in nodes:
            self.graph.add_node(node.node_id, value=node)

    def add_edges(self, edges: Iterable[Edge]) -> None:
        for edge in edges:
            if edge.source not in self.graph:
                self.add_nodes([Node(edge.source, "entity", edge.source)])
            if edge.target not in self.graph:
                self.add_nodes([Node(edge.target, "entity", edge.target)])
            self.graph.add_edge(edge.source, edge.target, value=edge)

    def neighbors(self, node_id: str) -> list[str]:
        return list(self.graph.neighbors(node_id)) if node_id in self.graph else []

    def node(self, node_id: str) -> Node | None:
        return self.graph.nodes[node_id].get("value") if node_id in self.graph else None

    def edge(self, source: str, target: str) -> Edge | None:
        data = self.graph.get_edge_data(source, target)
        return data.get("value") if data else None

    def node_ids(self, node_type: str | None = None) -> list[str]:
        if node_type is None:
            return list(self.graph.nodes)
        return [node_id for node_id, data in self.graph.nodes(data=True) if data["value"].node_type == node_type]

    def shortest_path(self, source: str, target: str, max_hops: int) -> list[str] | None:
        try:
            path = nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return path if len(path) - 1 <= max_hops else None
