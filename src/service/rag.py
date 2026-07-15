"""RAG system for ENTSO-E grid regulation documents"""

import hashlib
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ConfigDict
from pypdf import PdfReader

from src.models.internals import DocumentChunk
from src.service.entity_tags import entity_names_for_chunk
from src.service.graph_search import (
    find_entities_in_query,
    get_multi_hop_context,
    load_community_summaries,
    refresh_community_summaries,
)
from src.service.graph_store import load_graph_store
from src.service.hybrid_retriever import (
    EntityBoostRetriever,
    GraphAugmentedRetriever,
    HybridRetriever,
    build_bm25_index,
    load_bm25_corpus,
    save_bm25_corpus,
)
from src.service.retriever_config import (
    RETRIEVER_ENTITY,
    RETRIEVER_GRAPH,
    RETRIEVER_HYBRID,
    RETRIEVER_VECTOR,
    get_retriever_mode,
)

load_dotenv()


def _chunk_id(source: str, page: int, content: str) -> str:
    """Stable, collision-resistant chunk id."""
    h = hashlib.sha256()
    h.update(source.encode())
    h.update(b"\0")
    h.update(str(page).encode())
    h.update(b"\0")
    h.update(content.encode())
    return h.hexdigest()[:16]


def _format_docs(docs: List[Document]) -> str:
    """Format retrieved documents for LLM context."""
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')} "
        f"p.{doc.metadata.get('page', 0)}]\n{doc.page_content}"
        for doc in docs
    )


class EntsoeRagSystem:
    CHROMA_DIR = ".chroma_db"
    COLLECTION = "entsoe_regulations"

    def __init__(
        self,
        pdf_dir: str = "data/pdfs",
        embedding_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
    ):
        self.pdf_dir = Path(pdf_dir)
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_url,
        )
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.CHROMA_DIR,
            collection_name=self.COLLECTION,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". "],
        )

    def get_vector_retriever(self, k: int = 4) -> BaseRetriever:
        """Return a LangChain retriever over the regulation vector store."""
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_retriever(self, k: int = 4, mode: str | None = None) -> BaseRetriever:
        """Return retriever for the requested GraphRAG mode."""
        resolved = get_retriever_mode(mode)

        if resolved == RETRIEVER_VECTOR:
            return self.vectorstore.as_retriever(search_kwargs={"k": k})

        corpus = load_bm25_corpus(self.CHROMA_DIR)
        if not corpus:
            return self.vectorstore.as_retriever(search_kwargs={"k": k})

        bm25, docs = build_bm25_index(corpus)
        vector = self.get_vector_retriever(k=max(k, 4))
        hybrid = HybridRetriever(
            vector_retriever=vector,
            bm25=bm25,
            corpus=docs,
            k=k,
            fetch_k=max(k * 2, 8),
        )

        if resolved == RETRIEVER_HYBRID:
            return hybrid

        entity_boost = EntityBoostRetriever(
            base_retriever=hybrid,
            query_entities=[],
            k=k,
        )

        if resolved == RETRIEVER_ENTITY:
            return _QueryEntityRetriever(inner=entity_boost, k=k)

        summaries = load_community_summaries()
        if not summaries and load_graph_store() is not None:
            refresh_community_summaries()
            summaries = load_community_summaries()

        graph_retriever = GraphAugmentedRetriever(
            base_retriever=entity_boost,
            community_summaries=list(summaries.values()),
            k=k,
        )
        return _QueryEntityRetriever(
            inner=graph_retriever,
            k=k,
            include_communities=True,
        )

    def get_retrieval_chain(self, k: int = 4, mode: str | None = None):
        """LCEL chain: query -> retriever -> formatted context string."""
        return self.get_retriever(k=k, mode=mode) | RunnableLambda(_format_docs)

    def load_pdf(self, file_path: Path) -> List[Document]:
        """Load and chunk a PDF file"""
        reader = PdfReader(file_path)
        documents: List[Document] = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path.name, "page": page_num},
                    )
                )

        return documents

    def load_all_pdfs(self) -> List[Document]:
        """Load all PDFs from the pdf directory"""
        documents: List[Document] = []
        if not self.pdf_dir.exists():
            return documents

        for pdf_file in sorted(self.pdf_dir.glob("*.pdf")):
            documents.extend(self.load_pdf(pdf_file))

        return documents

    def is_indexed(self) -> bool:
        """Return True if the vector store already has chunks."""
        try:
            return self.vectorstore._collection.count() > 0
        except Exception:
            return False

    def _annotate_chunks(self, chunks: List[Document]) -> List[Document]:
        """Attach entity tags used by entity-boosted retrieval."""
        for chunk in chunks:
            names = entity_names_for_chunk(chunk.page_content)
            if names:
                chunk.metadata["entities"] = ",".join(names)
        return chunks

    def index_documents(self, force_rebuild: bool = False) -> int:
        """Index all PDFs into the vector store and BM25 corpus."""
        if force_rebuild:
            chroma_path = Path(self.CHROMA_DIR)
            if chroma_path.exists():
                shutil.rmtree(chroma_path)
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.CHROMA_DIR,
                collection_name=self.COLLECTION,
            )
        elif self.is_indexed():
            return self.vectorstore._collection.count()

        documents = self.load_all_pdfs()
        if not documents:
            return 0

        chunks = self.text_splitter.split_documents(documents)
        chunks = self._annotate_chunks(chunks)
        ids = [
            _chunk_id(
                c.metadata.get("source", "unknown"),
                c.metadata.get("page", 0),
                c.page_content,
            )
            for c in chunks
        ]
        self.vectorstore.add_documents(chunks, ids=ids)
        save_bm25_corpus(chunks, self.CHROMA_DIR)
        return len(chunks)

    def search(self, query: str, k: int = 4) -> List[DocumentChunk]:
        """Search for relevant document chunks"""
        results = self.vectorstore.similarity_search(query, k=k)

        chunks: List[DocumentChunk] = []
        for doc in results:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0)
            chunks.append(
                DocumentChunk(
                    content=doc.page_content,
                    source=source,
                    page=page,
                    chunk_id=_chunk_id(source, page, doc.page_content),
                )
            )

        return chunks

    def get_context(self, query: str, k: int = 4) -> str:
        """Get formatted context string for LLM (direct similarity search)."""
        chunks = self.search(query, k=k)
        return "\n\n".join(
            f"[Source: {c.source} p.{c.page}]\n{c.content}" for c in chunks
        )

    def get_context_via_retriever(
        self, query: str, k: int = 4, mode: str | None = None
    ) -> str:
        """Get formatted context via configured retriever chain."""
        resolved = get_retriever_mode(mode)
        chain = self.get_retrieval_chain(k=k, mode=resolved)
        context = chain.invoke(query)

        if resolved == RETRIEVER_GRAPH:
            graph_ctx = get_multi_hop_context(query)
            if graph_ctx:
                context = graph_ctx + "\n\n" + context

        return context


class _QueryEntityRetriever(BaseRetriever):
    """Inject per-query entity names into entity-boosted retrievers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: BaseRetriever
    k: int = 4
    include_communities: bool = False

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        store = load_graph_store()
        entities: list[str] = []
        if store is not None:
            entities = [node.name for node in find_entities_in_query(query, store)]

        if isinstance(self.inner, GraphAugmentedRetriever):
            summaries = load_community_summaries()
            if not summaries and store is not None:
                refresh_community_summaries()
                summaries = load_community_summaries()
            relevant = []
            lowered = query.lower()
            for summary in summaries.values():
                if any(name.lower() in summary.lower() for name in entities):
                    relevant.append(summary)
                elif any(token in lowered for token in ("fcr", "afrr", "mfrr")):
                    if any(
                        token in summary.lower() for token in ("fcr", "afrr", "mfrr")
                    ):
                        relevant.append(summary)
            self.inner.community_summaries = relevant[:2]

        if isinstance(self.inner, EntityBoostRetriever):
            self.inner.query_entities = entities
        elif isinstance(self.inner, GraphAugmentedRetriever) and isinstance(
            self.inner.base_retriever, EntityBoostRetriever
        ):
            self.inner.base_retriever.query_entities = entities

        docs = self.inner.invoke(query)
        return docs[: self.k]


def get_default_rag() -> EntsoeRagSystem:
    """Get configured RAG system"""
    return EntsoeRagSystem()
