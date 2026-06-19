"""Hybrid BM25 + vector retrieval with reciprocal rank fusion."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from rank_bm25 import BM25Okapi

BM25_CORPUS_FILE = "bm25_corpus.pkl"


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


def doc_key(doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", 0))
    return f"{source}:{page}:{hash(doc.page_content)}"


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    *,
    k: int = 4,
    rrf_k: int = 60,
) -> list[Document]:
    """Fuse multiple ranked document lists with RRF."""
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    for results in ranked_lists:
        for rank, doc in enumerate(results):
            key = doc_key(doc)
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [docs_by_key[key] for key, _ in ordered[:k]]


def save_bm25_corpus(chunks: list[Document], chroma_dir: str) -> Path:
    """Persist chunk corpus for BM25 rebuilds."""
    path = Path(chroma_dir) / BM25_CORPUS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"page_content": doc.page_content, "metadata": dict(doc.metadata)}
        for doc in chunks
    ]
    path.write_bytes(pickle.dumps(payload))
    return path


def load_bm25_corpus(chroma_dir: str) -> list[Document]:
    """Load persisted BM25 corpus."""
    path = Path(chroma_dir) / BM25_CORPUS_FILE
    if not path.exists():
        return []
    payload: list[dict[str, Any]] = pickle.loads(path.read_bytes())
    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in payload
    ]


def build_bm25_index(corpus: list[Document]) -> tuple[BM25Okapi, list[Document]]:
    """Build a BM25 index over tokenized corpus documents."""
    if not corpus:
        raise ValueError("BM25 corpus is empty")
    tokenized = [tokenize(doc.page_content) for doc in corpus]
    return BM25Okapi(tokenized), corpus


def bm25_search(
    query: str, bm25: BM25Okapi, corpus: list[Document], k: int
) -> list[Document]:
    """Return top-k BM25 matches for a query."""
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [corpus[i] for i in ranked[:k] if scores[i] > 0] or [
        corpus[i] for i in ranked[:k]
    ]


class HybridRetriever(BaseRetriever):
    """Fuse vector and BM25 rankings with reciprocal rank fusion."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_retriever: BaseRetriever
    bm25: BM25Okapi
    corpus: list[Document]
    k: int = 4
    rrf_k: int = 60
    fetch_k: int = Field(default=8)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = bm25_search(query, self.bm25, self.corpus, self.fetch_k)
        return reciprocal_rank_fusion(
            [vector_docs, bm25_docs],
            k=self.k,
            rrf_k=self.rrf_k,
        )


def entity_overlap_score(query_entities: set[str], doc: Document) -> float:
    """Score chunk relevance by overlapping entity tags."""
    raw = str(doc.metadata.get("entities", ""))
    if not raw or not query_entities:
        return 0.0
    doc_entities = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not doc_entities:
        return 0.0
    return len(query_entities & doc_entities) / len(query_entities)


class EntityBoostRetriever(BaseRetriever):
    """Re-rank a base retriever using entity metadata overlap."""

    base_retriever: BaseRetriever
    query_entities: list[str] = Field(default_factory=list)
    k: int = 4
    fetch_k: int = Field(default=12)
    boost: float = Field(default=0.35)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        docs = self.base_retriever.invoke(query)
        if len(docs) > self.fetch_k:
            docs = docs[: self.fetch_k]

        query_entities = {e.lower() for e in self.query_entities}
        if not query_entities:
            return docs[: self.k]

        scored: list[tuple[float, int, Document]] = []
        for index, doc in enumerate(docs):
            overlap = entity_overlap_score(query_entities, doc)
            scored.append((1.0 + overlap * self.boost, index, doc))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc for _, _, doc in scored[: self.k]]


class GraphAugmentedRetriever(BaseRetriever):
    """Entity-boosted retrieval plus community summary pseudo-chunks."""

    base_retriever: BaseRetriever
    community_summaries: list[str] = Field(default_factory=list)
    k: int = 4

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        docs = self.base_retriever.invoke(query)
        extras: list[Document] = []
        for index, summary in enumerate(self.community_summaries[:2]):
            extras.append(
                Document(
                    page_content=summary,
                    metadata={"source": "graph_community", "page": index},
                )
            )
        merged = extras + docs
        return merged[: self.k]
