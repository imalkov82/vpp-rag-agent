"""RAG system for ENTSO-E grid regulation documents"""

import hashlib
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.models.internals import DocumentChunk

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

    def get_retriever(self, k: int = 4) -> BaseRetriever:
        """Return a LangChain retriever over the regulation vector store."""
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_retrieval_chain(self, k: int = 4):
        """LCEL chain: query -> retriever -> formatted context string."""
        return self.get_retriever(k=k) | RunnableLambda(_format_docs)

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

    def index_documents(self, force_rebuild: bool = False) -> int:
        """Index all PDFs into the vector store.

        Idempotent: if the collection already has content and ``force_rebuild``
        is False, this is a no-op and returns the existing chunk count.
        """
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
        ids = [
            _chunk_id(
                c.metadata.get("source", "unknown"),
                c.metadata.get("page", 0),
                c.page_content,
            )
            for c in chunks
        ]
        self.vectorstore.add_documents(chunks, ids=ids)
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

    def get_context_via_retriever(self, query: str, k: int = 4) -> str:
        """Get formatted context via LangChain retriever + LCEL chain."""
        chain = self.get_retrieval_chain(k=k)
        return chain.invoke(query)


def get_default_rag() -> EntsoeRagSystem:
    """Get configured RAG system"""
    return EntsoeRagSystem()
