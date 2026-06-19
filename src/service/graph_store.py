"""Graph storage abstraction with a NetworkX + GraphML default backend."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import networkx as nx

from src.models.graph import GraphNodeView, NodeLabel

GRAPH_DB_DIR = ".graph_db"
GRAPH_FILE = "regulations.graphml"


def node_id(label: str | NodeLabel, name: str) -> str:
    """Build a stable node identifier from label and display name."""
    label_str = label.value if isinstance(label, NodeLabel) else label
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return f"{label_str.lower()}:{normalized or 'unknown'}"


@runtime_checkable
class GraphStore(Protocol):
    """Protocol for regulation knowledge graph backends."""

    def add_node(
        self,
        nid: str,
        label: str,
        name: str,
        *,
        source_doc: str = "",
        page: int = 0,
    ) -> None: ...

    def add_edge(self, source: str, target: str, relation: str) -> None: ...

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        subject_label: str | NodeLabel = NodeLabel.CONCEPT,
        object_label: str | NodeLabel = NodeLabel.CONCEPT,
        source_doc: str = "",
        page: int = 0,
    ) -> None: ...

    def neighbors(self, nid: str, hops: int = 2) -> list[GraphNodeView]: ...

    def find_nodes_by_name(
        self, name: str, label: str | NodeLabel | None = None
    ) -> list[GraphNodeView]: ...

    def find_path(self, source: str, target: str) -> list[str]: ...

    def get_node(self, nid: str) -> GraphNodeView | None: ...

    def save(self, path: Path) -> None: ...

    def node_count(self) -> int: ...

    def edge_count(self) -> int: ...


class NetworkXGraphStore:
    """In-memory directed graph persisted as GraphML."""

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    def add_node(
        self,
        nid: str,
        label: str,
        name: str,
        *,
        source_doc: str = "",
        page: int = 0,
    ) -> None:
        self._graph.add_node(
            nid,
            label=label,
            name=name,
            source_doc=source_doc,
            page=int(page),
        )

    def add_edge(self, source: str, target: str, relation: str) -> None:
        if source not in self._graph:
            return
        if target not in self._graph:
            return
        self._graph.add_edge(source, target, relation=relation)

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        subject_label: str | NodeLabel = NodeLabel.CONCEPT,
        object_label: str | NodeLabel = NodeLabel.CONCEPT,
        source_doc: str = "",
        page: int = 0,
    ) -> None:
        subj_id = node_id(subject_label, subject)
        obj_id = node_id(object_label, obj)
        subj_label = (
            subject_label.value
            if isinstance(subject_label, NodeLabel)
            else subject_label
        )
        obj_label = (
            object_label.value if isinstance(object_label, NodeLabel) else object_label
        )

        self.add_node(subj_id, subj_label, subject, source_doc=source_doc, page=page)
        self.add_node(obj_id, obj_label, obj, source_doc=source_doc, page=page)
        self.add_edge(subj_id, obj_id, predicate)

        if source_doc:
            doc_id = node_id(NodeLabel.DOCUMENT, source_doc)
            self.add_node(doc_id, NodeLabel.DOCUMENT.value, source_doc)
            section_id = node_id(NodeLabel.SECTION, f"{source_doc}:p{page}")
            self.add_node(
                section_id,
                NodeLabel.SECTION.value,
                f"{source_doc} p.{page}",
                source_doc=source_doc,
                page=page,
            )
            self.add_edge(doc_id, section_id, "HAS_SECTION")
            self.add_edge(section_id, subj_id, "MENTIONS")

    def neighbors(self, nid: str, hops: int = 2) -> list[GraphNodeView]:
        if nid not in self._graph:
            return []

        seen: set[str] = {nid}
        frontier: set[str] = {nid}
        collected: list[GraphNodeView] = []

        for _ in range(hops):
            next_frontier: set[str] = set()
            for current in frontier:
                adjacent = set(self._graph.successors(current)) | set(
                    self._graph.predecessors(current)
                )
                for neighbor in adjacent:
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    next_frontier.add(neighbor)
                    collected.append(self._to_view(neighbor))
            frontier = next_frontier

        return collected

    def find_nodes_by_name(
        self, name: str, label: str | NodeLabel | None = None
    ) -> list[GraphNodeView]:
        needle = name.strip().lower()
        if not needle:
            return []

        label_value = label.value if isinstance(label, NodeLabel) else label
        matches: list[GraphNodeView] = []

        for nid, attrs in self._graph.nodes(data=True):
            node_name = str(attrs.get("name", ""))
            node_label = str(attrs.get("label", ""))
            if label_value and node_label != label_value:
                continue
            if needle in node_name.lower() or needle in nid.lower():
                matches.append(self._to_view(nid))

        return matches

    def find_path(self, source: str, target: str) -> list[str]:
        if source not in self._graph or target not in self._graph:
            return []
        try:
            return nx.shortest_path(self._graph.to_undirected(), source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_node(self, nid: str) -> GraphNodeView | None:
        if nid not in self._graph:
            return None
        return self._to_view(nid)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self._graph, path)

    @classmethod
    def load(cls, path: Path) -> NetworkXGraphStore:
        store = cls()
        store._graph = nx.read_graphml(path)
        return store

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def _to_view(self, nid: str) -> GraphNodeView:
        attrs: dict[str, Any] = self._graph.nodes[nid]
        page_raw = attrs.get("page", 0)
        return GraphNodeView(
            node_id=nid,
            label=str(attrs.get("label", "")),
            name=str(attrs.get("name", nid)),
            source_doc=str(attrs.get("source_doc", "")),
            page=int(page_raw) if page_raw else 0,
        )


def graph_path() -> Path:
    return Path(GRAPH_DB_DIR) / GRAPH_FILE


def load_graph_store() -> NetworkXGraphStore | None:
    """Load persisted graph or return None if not indexed."""
    path = graph_path()
    if not path.exists():
        return None
    return NetworkXGraphStore.load(path)


def get_graph_store(create: bool = False) -> NetworkXGraphStore | None:
    """Load existing graph, optionally create an empty in-memory store."""
    store = load_graph_store()
    if store is not None:
        return store
    if create:
        return NetworkXGraphStore()
    return None
