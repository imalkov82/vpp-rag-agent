"""Health checks for external dependencies — mirrors vppems-cmd's app/health.py."""

import os
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class HealthResult:
    name: str
    ok: bool
    detail: str


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


def check_health() -> list[HealthResult]:
    """Run every check and return the aggregated results."""
    return [
        check_ollama(),
        check_entsoe_api_key(),
        check_chroma(),
        check_pdf_corpus(),
    ]
