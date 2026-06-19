"""Query the regulation knowledge graph for agent context."""

from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx

from src.models.graph import GraphNodeView
from src.service.graph_store import GraphStore, NetworkXGraphStore, load_graph_store

COMMUNITY_CACHE = Path(".graph_db/community_summaries.json")


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


def _heuristic_community_summary(nodes: list[GraphNodeView]) -> str:
    labels: dict[str, list[str]] = {}
    for node in nodes:
        labels.setdefault(node.label, []).append(node.name)
    parts = [f"{label}: {', '.join(names[:6])}" for label, names in labels.items()]
    return "Related regulation concepts — " + "; ".join(parts)


def build_community_summaries(store: NetworkXGraphStore) -> dict[str, str]:
    """Detect Louvain communities and build summary strings per community."""
    if store.node_count() == 0:
        return {}

    graph = store._graph.to_undirected()
    if graph.number_of_nodes() == 0:
        return {}

    communities = list(nx.community.louvain_communities(graph))
    summaries: dict[str, str] = {}

    for index, member_ids in enumerate(communities):
        nodes = []
        for nid in member_ids:
            node = store.get_node(str(nid))
            if node:
                nodes.append(node)
        if nodes:
            summaries[f"community_{index}"] = _heuristic_community_summary(nodes)

    return summaries


def load_community_summaries() -> dict[str, str]:
    """Load cached community summaries from disk."""
    if not COMMUNITY_CACHE.exists():
        return {}
    try:
        return json.loads(COMMUNITY_CACHE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_community_summaries(summaries: dict[str, str]) -> Path:
    """Persist community summaries for reuse."""
    COMMUNITY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    COMMUNITY_CACHE.write_text(json.dumps(summaries, indent=2))
    return COMMUNITY_CACHE


def refresh_community_summaries() -> int:
    """Rebuild and cache community summaries from the indexed graph."""
    store = load_graph_store()
    if store is None:
        return 0
    summaries = build_community_summaries(store)
    save_community_summaries(summaries)
    return len(summaries)


def _relevant_community_summaries(
    query: str, store: GraphStore, summaries: dict[str, str]
) -> list[str]:
    entities = find_entities_in_query(query, store)
    if not entities or not summaries:
        return []

    entity_names = {e.name.lower() for e in entities}
    matched: list[str] = []
    for summary in summaries.values():
        lowered = summary.lower()
        if any(name in lowered for name in entity_names):
            matched.append(summary)
    return matched[:2]


def get_multi_hop_context(query: str, hops: int = 2) -> str:
    """Merge entity neighborhoods and community summaries for GraphRAG."""
    store = load_graph_store()
    if store is None or store.node_count() == 0:
        return ""

    sections: list[str] = []
    graph_ctx = get_graph_context(query, hops=hops)
    if graph_ctx:
        sections.append(graph_ctx)

    summaries = load_community_summaries()
    if not summaries:
        refresh_community_summaries()
        summaries = load_community_summaries()

    community_bits = _relevant_community_summaries(query, store, summaries)
    if community_bits:
        sections.append(
            "Community context:\n" + "\n".join(f"- {s}" for s in community_bits)
        )

    return "\n\n".join(sections)


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
