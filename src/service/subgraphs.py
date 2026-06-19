"""LangGraph subgraphs for price and regulation tool paths."""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, StateGraph

from src.models.internals import AgentState


def build_price_subgraph(fetch_prices: Callable[[AgentState], AgentState]):
    """Compile the price-fetch subgraph."""
    graph = StateGraph(AgentState)
    graph.add_node("fetch_prices", fetch_prices)
    graph.set_entry_point("fetch_prices")
    graph.add_edge("fetch_prices", END)
    return graph.compile()


def build_regulation_subgraph(
    search_regulations: Callable[[AgentState], AgentState],
):
    """Compile the regulation-search subgraph."""
    graph = StateGraph(AgentState)
    graph.add_node("search_regulations", search_regulations)
    graph.set_entry_point("search_regulations")
    graph.add_edge("search_regulations", END)
    return graph.compile()
