"""Tests for graph search helpers."""

from src.models.graph import NodeLabel
from src.service.graph_search import find_entities_in_query, get_graph_context
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
