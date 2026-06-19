"""Tests for RAG retriever and LCEL chain."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.service.rag import EntsoeRagSystem, _format_docs


class TestFormatDocs:
    def test_format_docs_includes_source_and_page(self):
        docs = [
            Document(
                page_content="FCR capacity rules",
                metadata={"source": "so_gl.pdf", "page": 12},
            ),
        ]
        text = _format_docs(docs)
        assert "[Source: so_gl.pdf p.12]" in text
        assert "FCR capacity rules" in text


class TestEntsoeRagRetriever:
    @patch.object(EntsoeRagSystem, "__init__", lambda self, *a, **kw: None)
    def _make_rag(self):
        rag = EntsoeRagSystem()
        rag.vectorstore = MagicMock()
        return rag

    def test_get_retriever_delegates_to_vectorstore(self):
        rag = self._make_rag()
        mock_retriever = MagicMock()
        rag.vectorstore.as_retriever.return_value = mock_retriever

        retriever = rag.get_retriever(k=5)

        assert retriever is mock_retriever
        rag.vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 5})

    def test_get_context_via_retriever_uses_chain(self):
        rag = self._make_rag()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "[Source: test.pdf p.1]\ncontent"
        rag.get_retrieval_chain = MagicMock(return_value=mock_chain)

        result = rag.get_context_via_retriever("FCR", k=2)

        assert "content" in result
        rag.get_retrieval_chain.assert_called_once_with(k=2, mode="vector")
        mock_chain.invoke.assert_called_once_with("FCR")
