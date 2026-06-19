"""Query the regulation knowledge graph for agent context."""

from __future__ import annotations

import re

from src.models.graph import GraphNodeView
from src.service.graph_store import GraphStore, load_graph_store


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]


def find_entities_in_query(query: str, store: GraphStore) -> list[GraphNodeView]:
    """Match graph nodes mentioned in a natural-language query."""
    matches: list[GraphNodeView] = []
    seen: set[str] = set()

    for token in _tokenize(query):
        for node in store.find_nodes_by_name(token):
            if node.node_id not in seen:
                seen.add(node.node_id)
                matches.append(node)

    lowered = query.lower()
    for phrase in ("fcr", "afrr", "mfrr", "so gl", "eb gl", "imbalance settlement"):
        if phrase in lowered:
            for node in store.find_nodes_by_name(phrase):
                if node.node_id not in seen:
                    seen.add(node.node_id)
                    matches.append(node)

    return matches


def format_neighbors(node: GraphNodeView, neighbors: list[GraphNodeView]) -> str:
    lines = [f"- {node.label} '{node.name}'"]
    if node.source_doc:
        lines[0] += f" ({node.source_doc} p.{node.page})" if node.page else ""

    for neighbor in neighbors[:8]:
        suffix = ""
        if neighbor.source_doc:
            suffix = f" [{neighbor.source_doc}"
            if neighbor.page:
                suffix += f" p.{neighbor.page}"
            suffix += "]"
        lines.append(f"  -> {neighbor.label} '{neighbor.name}'{suffix}")

    return "\n".join(lines)


def get_graph_context(query: str, hops: int = 2) -> str:
    """Return formatted graph neighborhood context for a query."""
    store = load_graph_store()
    if store is None or store.node_count() == 0:
        return ""

    entities = find_entities_in_query(query, store)
    if not entities:
        return ""

    sections: list[str] = []
    for entity in entities[:5]:
        neighbors = store.neighbors(entity.node_id, hops=hops)
        if neighbors:
            sections.append(format_neighbors(entity, neighbors))

    if not sections:
        return ""

    return "Knowledge graph relations:\n" + "\n\n".join(sections)


def query_graph(query: str, hops: int = 2) -> str:
    """CLI-friendly graph query output with matched entities and paths."""
    store = load_graph_store()
    if store is None or store.node_count() == 0:
        return "No knowledge graph indexed (run: vpp-rag index --with-graph)."

    entities = find_entities_in_query(query, store)
    if not entities:
        return f"No graph entities matched for: {query!r}"

    lines = [f"Matched {len(entities)} entit(y/ies) for: {query!r}", ""]
    for entity in entities:
        lines.append(
            format_neighbors(entity, store.neighbors(entity.node_id, hops=hops))
        )
        lines.append("")

    if len(entities) >= 2:
        path = store.find_path(entities[0].node_id, entities[1].node_id)
        if path:
            names: list[str] = []
            for node_id in path:
                node = store.get_node(node_id)
                if node:
                    names.append(node.name)
            if names:
                lines.append(f"Shortest path: {' -> '.join(names)}")

    return "\n".join(lines).strip()
