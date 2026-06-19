"""Build the regulation knowledge graph from indexed PDF chunks."""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.models.graph import ChunkExtraction, NodeLabel, RegulationTriple
from src.service.graph_store import (
    GRAPH_DB_DIR,
    NetworkXGraphStore,
    graph_path,
    load_graph_store,
)
from src.service.rag import EntsoeRagSystem, get_default_rag

KNOWN_CONCEPTS: dict[str, NodeLabel] = {
    "fcr": NodeLabel.BALANCING_PRODUCT,
    "afrr": NodeLabel.BALANCING_PRODUCT,
    "mfrr": NodeLabel.BALANCING_PRODUCT,
    "imbalance settlement": NodeLabel.CONCEPT,
    "balancing reserve": NodeLabel.CONCEPT,
    "capacity allocation": NodeLabel.CONCEPT,
    "day-ahead": NodeLabel.CONCEPT,
    "network code": NodeLabel.NETWORK_CODE,
}

KNOWN_NETWORK_CODES: dict[str, str] = {
    "so gl": "SO GL",
    "eb gl": "EB GL",
    "nc rfg": "NC RfG",
    "nc cacm": "NC CACM",
}

EXTRACTION_PROMPT = (
    "Extract regulation knowledge as subject-predicate-object triples from the "
    "chunk. Focus on balancing products, network codes, grid requirements, and "
    "cross-references. Return an empty list when nothing useful is present."
)


def _find_known_terms(text: str) -> list[tuple[str, NodeLabel]]:
    lowered = text.lower()
    found: list[tuple[str, NodeLabel]] = []
    seen: set[str] = set()

    for term, label in sorted(KNOWN_CONCEPTS.items(), key=lambda item: -len(item[0])):
        if term in lowered and term not in seen:
            found.append((term.upper() if len(term) <= 4 else term.title(), label))
            seen.add(term)

    for term, display in sorted(
        KNOWN_NETWORK_CODES.items(), key=lambda item: -len(item[0])
    ):
        if term in lowered and display not in seen:
            found.append((display, NodeLabel.NETWORK_CODE))
            seen.add(display)

    return found


def extract_triples_heuristic(
    text: str, source_doc: str, page: int
) -> list[RegulationTriple]:
    """Extract triples using domain keyword matching (no LLM required)."""
    terms = _find_known_terms(text)
    if not terms:
        return []

    doc_name = Path(source_doc).stem
    network_codes = [
        (name, label) for name, label in terms if label == NodeLabel.NETWORK_CODE
    ]
    balancing_products = [
        (name, label) for name, label in terms if label == NodeLabel.BALANCING_PRODUCT
    ]

    triples: list[RegulationTriple] = []
    for name, label in terms:
        triples.append(
            RegulationTriple(
                subject=doc_name,
                predicate="MENTIONS",
                object=name,
                subject_label=NodeLabel.DOCUMENT,
                object_label=label,
                source_doc=source_doc,
                page=page,
            )
        )

    for product_name, _ in balancing_products:
        for code_name, _ in network_codes:
            triples.append(
                RegulationTriple(
                    subject=product_name,
                    predicate="REGULATED_BY",
                    object=code_name,
                    subject_label=NodeLabel.BALANCING_PRODUCT,
                    object_label=NodeLabel.NETWORK_CODE,
                    source_doc=source_doc,
                    page=page,
                )
            )

    return triples


def extract_triples_with_llm(
    text: str,
    source_doc: str,
    page: int,
    model: str = "deepseek-r1:8b",
    base_url: str = "http://localhost:11434",
) -> list[RegulationTriple]:
    """Extract triples with structured LLM output; falls back to heuristics."""
    llm = ChatOllama(model=model, temperature=0.0, base_url=base_url)
    extractor = llm.with_structured_output(ChunkExtraction)

    try:
        raw = extractor.invoke(
            [
                SystemMessage(content=EXTRACTION_PROMPT),
                HumanMessage(
                    content=(
                        f"Source: {source_doc} page {page}\n\nChunk:\n{text[:3000]}"
                    )
                ),
            ]
        )
        result = (
            raw
            if isinstance(raw, ChunkExtraction)
            else ChunkExtraction.model_validate(raw)
        )
        triples = result.triples
        for triple in triples:
            triple.source_doc = source_doc
            triple.page = page
        if triples:
            return triples
    except Exception:
        pass

    return extract_triples_heuristic(text, source_doc, page)


def _apply_triples(store: NetworkXGraphStore, triples: list[RegulationTriple]) -> None:
    for triple in triples:
        store.add_triple(
            triple.subject,
            triple.predicate,
            triple.object,
            subject_label=triple.subject_label,
            object_label=triple.object_label,
            source_doc=triple.source_doc,
            page=triple.page,
        )


def index_graph(
    force_rebuild: bool = False,
    use_llm: bool = False,
    rag: EntsoeRagSystem | None = None,
) -> int:
    """Build or rebuild the regulation knowledge graph from PDF chunks.

    Returns the number of graph nodes after indexing.
    """
    rag = rag or get_default_rag()

    if force_rebuild:
        graph_dir = Path(GRAPH_DB_DIR)
        if graph_dir.exists():
            shutil.rmtree(graph_dir)

    existing = load_graph_store()
    if existing is not None and existing.node_count() > 0 and not force_rebuild:
        return existing.node_count()

    chunk_count = rag.index_documents(force_rebuild=force_rebuild)
    if chunk_count == 0:
        store = NetworkXGraphStore()
        store.save(graph_path())
        return 0

    store = NetworkXGraphStore()
    documents = rag.load_all_pdfs()
    chunks = rag.text_splitter.split_documents(documents)

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        page = int(chunk.metadata.get("page", 0))
        if use_llm:
            triples = extract_triples_with_llm(chunk.page_content, source, page)
        else:
            triples = extract_triples_heuristic(chunk.page_content, source, page)
        _apply_triples(store, triples)

    store.save(graph_path())
    return store.node_count()


def is_graph_indexed() -> bool:
    store = load_graph_store()
    return store is not None and store.node_count() > 0
