"""Health checks for external dependencies."""

import os
from pathlib import Path

import requests

from src.models.internals import HealthResult


def check_ollama(
    base_url: str = "http://localhost:11434", timeout: float = 2.0
) -> HealthResult:
    """Verify the local Ollama daemon is reachable."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout)
        r.raise_for_status()
        return HealthResult("ollama", True, f"reachable at {base_url}")
    except Exception as e:
        return HealthResult("ollama", False, f"unreachable: {e}")


def check_entsoe_api_key() -> HealthResult:
    """Verify ENTSOE_API_KEY is configured (does not call the API)."""
    if os.getenv("ENTSOE_API_KEY"):
        return HealthResult("entsoe", True, "ENTSOE_API_KEY set")
    return HealthResult("entsoe", False, "ENTSOE_API_KEY missing")


def check_chroma(chroma_dir: str = ".chroma_db") -> HealthResult:
    """Verify the vector store has been persisted."""
    path = Path(chroma_dir)
    if path.exists() and any(path.iterdir()):
        return HealthResult("chroma", True, f"persisted at {chroma_dir}")
    return HealthResult(
        "chroma", False, f"no index at {chroma_dir} (run with --rebuild-index)"
    )


def check_pdf_corpus(pdf_dir: str = "data/pdfs") -> HealthResult:
    """Verify there are PDFs available to index."""
    path = Path(pdf_dir)
    if not path.exists():
        return HealthResult("pdfs", False, f"{pdf_dir} does not exist")
    pdfs = list(path.glob("*.pdf"))
    if not pdfs:
        return HealthResult("pdfs", False, f"{pdf_dir} has no PDFs")
    return HealthResult("pdfs", True, f"{len(pdfs)} PDF(s) in {pdf_dir}")


def check_graph(graph_path: str = ".graph_db/regulations.graphml") -> HealthResult:
    """Verify the regulation knowledge graph has been persisted."""
    path = Path(graph_path)
    if not path.exists():
        return HealthResult(
            "graph",
            False,
            f"no graph at {graph_path} (run: vpp-rag index --with-graph)",
        )
    try:
        from src.service.graph_store import NetworkXGraphStore

        store = NetworkXGraphStore.load(path)
        count = store.node_count()
        if count == 0:
            return HealthResult("graph", False, f"empty graph at {graph_path}")
        return HealthResult("graph", True, f"{count} node(s) at {graph_path}")
    except Exception as e:
        return HealthResult("graph", False, f"failed to load graph: {e}")


def check_health() -> list[HealthResult]:
    """Run every check and return the aggregated results."""
    return [
        check_ollama(),
        check_entsoe_api_key(),
        check_chroma(),
        check_graph(),
        check_pdf_corpus(),
    ]
