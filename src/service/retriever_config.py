"""Retriever mode selection for GraphRAG."""

from __future__ import annotations

import os

RETRIEVER_VECTOR = "vector"
RETRIEVER_HYBRID = "hybrid"
RETRIEVER_ENTITY = "entity"
RETRIEVER_GRAPH = "graph"

VALID_RETRIEVER_MODES = {
    RETRIEVER_VECTOR,
    RETRIEVER_HYBRID,
    RETRIEVER_ENTITY,
    RETRIEVER_GRAPH,
}


def get_retriever_mode(mode: str | None = None) -> str:
    """Resolve retriever mode from argument or VPP_RETRIEVER env."""
    resolved = (mode or os.getenv("VPP_RETRIEVER", RETRIEVER_VECTOR)).lower()
    if resolved not in VALID_RETRIEVER_MODES:
        return RETRIEVER_VECTOR
    return resolved
