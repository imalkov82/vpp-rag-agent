"""Tests for hybrid retrieval and RRF fusion."""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.service.hybrid_retriever import (
    EntityBoostRetriever,
    reciprocal_rank_fusion,
    tokenize,
)
from src.service.retriever_config import (
    RETRIEVER_GRAPH,
    RETRIEVER_HYBRID,
    VALID_RETRIEVER_MODES,
    get_retriever_mode,
)


class TestReciprocalRankFusion:
    def test_promotes_docs_present_in_both_lists(self):
        shared = Document(
            page_content="fcr rules", metadata={"source": "a.pdf", "page": 1}
        )
        only_vector = Document(
            page_content="other", metadata={"source": "b.pdf", "page": 2}
        )
        only_bm25 = Document(
            page_content="frequency", metadata={"source": "c.pdf", "page": 3}
        )

        fused = reciprocal_rank_fusion(
            [[shared, only_vector], [shared, only_bm25]],
            k=2,
        )
        assert fused[0].page_content == "fcr rules"

    def test_tokenize_filters_short_tokens(self):
        assert "fcr" in tokenize("What is FCR?")


class TestEntityBoostRetriever:
    def test_boosts_chunks_with_entity_overlap(self):
        class StubRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
                return [
                    Document(
                        page_content="generic",
                        metadata={"source": "a.pdf", "page": 1, "entities": "Other"},
                    ),
                    Document(
                        page_content="fcr detail",
                        metadata={
                            "source": "b.pdf",
                            "page": 2,
                            "entities": "FCR,SO GL",
                        },
                    ),
                ]

        retriever = EntityBoostRetriever(
            base_retriever=StubRetriever(),
            query_entities=["FCR"],
            k=2,
        )
        docs = retriever.invoke("FCR requirements")
        assert docs[0].metadata["source"] == "b.pdf"


class TestRetrieverConfig:
    def test_defaults_to_vector(self, monkeypatch):
        monkeypatch.delenv("VPP_RETRIEVER", raising=False)
        assert get_retriever_mode() == "vector"

    def test_reads_env(self, monkeypatch):
        monkeypatch.setenv("VPP_RETRIEVER", "hybrid")
        assert get_retriever_mode() == RETRIEVER_HYBRID

    def test_invalid_mode_falls_back(self, monkeypatch):
        monkeypatch.setenv("VPP_RETRIEVER", "unknown")
        assert get_retriever_mode() == "vector"

    def test_valid_modes(self):
        assert RETRIEVER_GRAPH in VALID_RETRIEVER_MODES
