"""Tests for graph ingest heuristics."""

from src.models.graph import NodeLabel
from src.service.entity_tags import entity_names_for_chunk
from src.service.graph_ingest import extract_triples_heuristic, is_graph_indexed


class TestGraphIngest:
    def test_extract_triples_heuristic_finds_balancing_terms(self):
        text = "The SO GL defines FCR and aFRR requirements for balancing."
        triples = extract_triples_heuristic(text, "so_gl.pdf", page=2)

        objects = {triple.object for triple in triples}
        predicates = {triple.predicate for triple in triples}

        assert "FCR" in objects
        assert "SO GL" in objects
        assert "MENTIONS" in predicates

    def test_entity_names_for_chunk(self):
        text = "FCR and SO GL requirements for balancing reserves."
        names = entity_names_for_chunk(text)
        assert "FCR" in names
        assert "SO GL" in names

    def test_extract_triples_heuristic_empty_text(self):
        assert extract_triples_heuristic("no matching terms here", "x.pdf", 1) == []

    def test_extract_triples_links_product_to_network_code(self):
        text = "Under SO GL, FCR capacity must be maintained."
        triples = extract_triples_heuristic(text, "so_gl.pdf", page=1)
        regulated = [
            t for t in triples if t.predicate == "REGULATED_BY" and t.subject == "FCR"
        ]
        assert regulated
        assert regulated[0].object == "SO GL"
        assert regulated[0].subject_label == NodeLabel.BALANCING_PRODUCT

    def test_is_graph_indexed_false_when_missing(self):
        assert is_graph_indexed() is False
