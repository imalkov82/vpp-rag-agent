"""Tests for graph search helpers."""

from src.models.graph import NodeLabel
from src.service.graph_search import (
    build_community_summaries,
    find_entities_in_query,
    get_graph_context,
    get_multi_hop_context,
)
from src.service.graph_store import NetworkXGraphStore


class TestGraphSearch:
    def _sample_store(self) -> NetworkXGraphStore:
        store = NetworkXGraphStore()
        store.add_triple(
            "FCR",
            "REGULATED_BY",
            "SO GL",
            subject_label=NodeLabel.BALANCING_PRODUCT,
            object_label=NodeLabel.NETWORK_CODE,
            source_doc="so_gl.pdf",
            page=4,
        )
        return store

    def test_find_entities_in_query(self):
        store = self._sample_store()
        entities = find_entities_in_query("What are the FCR requirements?", store)
        names = {entity.name for entity in entities}
        assert "FCR" in names

    def test_get_graph_context_empty_without_index(self, monkeypatch):
        monkeypatch.setattr(
            "src.service.graph_search.load_graph_store",
            lambda: None,
        )
        assert get_graph_context("FCR requirements") == ""

    def test_get_graph_context_formats_neighbors(self, monkeypatch):
        store = self._sample_store()
        monkeypatch.setattr(
            "src.service.graph_search.load_graph_store",
            lambda: store,
        )
        context = get_graph_context("FCR in SO GL")
        assert "Knowledge graph relations" in context
        assert "FCR" in context

    def test_build_community_summaries(self):
        store = self._sample_store()
        summaries = build_community_summaries(store)
        assert summaries
        assert any("FCR" in value for value in summaries.values())

    def test_get_multi_hop_context_includes_communities(self, monkeypatch, tmp_path):
        store = self._sample_store()
        monkeypatch.setattr(
            "src.service.graph_search.load_graph_store",
            lambda: store,
        )
        cache = tmp_path / "community_summaries.json"
        monkeypatch.setattr(
            "src.service.graph_search.COMMUNITY_CACHE",
            cache,
        )
        monkeypatch.setattr(
            "src.service.graph_search.load_community_summaries",
            lambda: {"c0": "BalancingProduct: FCR, SO GL"},
        )
        context = get_multi_hop_context("FCR requirements")
        assert "Knowledge graph relations" in context
        assert "Community context" in context
