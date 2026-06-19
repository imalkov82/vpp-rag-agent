"""Tests for NetworkX graph store."""

from pathlib import Path

from src.models.graph import NodeLabel
from src.service.graph_store import NetworkXGraphStore, node_id


class TestNetworkXGraphStore:
    def test_add_triple_creates_nodes_and_edges(self):
        store = NetworkXGraphStore()
        store.add_triple(
            "FCR",
            "REGULATED_BY",
            "SO GL",
            subject_label=NodeLabel.BALANCING_PRODUCT,
            object_label=NodeLabel.NETWORK_CODE,
            source_doc="so_gl.pdf",
            page=3,
        )

        assert store.node_count() >= 2
        assert store.edge_count() >= 1

        fcr_id = node_id(NodeLabel.BALANCING_PRODUCT, "FCR")
        neighbors = store.neighbors(fcr_id, hops=1)
        neighbor_names = {n.name for n in neighbors}
        assert "SO GL" in neighbor_names

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        store = NetworkXGraphStore()
        store.add_triple(
            "FCR",
            "REFERENCES",
            "Balancing Reserve",
            source_doc="rules.pdf",
            page=1,
        )
        path = tmp_path / "test.graphml"
        store.save(path)

        loaded = NetworkXGraphStore.load(path)
        assert loaded.node_count() == store.node_count()
        assert loaded.get_node(node_id(NodeLabel.CONCEPT, "FCR")) is not None

    def test_find_nodes_by_name(self):
        store = NetworkXGraphStore()
        store.add_node(node_id(NodeLabel.CONCEPT, "FCR"), "Concept", "FCR")

        matches = store.find_nodes_by_name("fcr")
        assert len(matches) == 1
        assert matches[0].name == "FCR"
