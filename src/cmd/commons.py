"""Shared helpers for CLI commands."""

from src.utils.console import console

__all__ = ["console", "format_source"]


def format_source(source: dict) -> str:
    """Render a source dict as a single human-readable line."""
    if source.get("type") == "price":
        return f"price feed (area={source.get('area', 'unknown')})"
    if source.get("type") == "regulation":
        snippet = source.get("context", "").splitlines()
        head = next((line for line in snippet if line.strip()), "")
        return f"regulation excerpt: {head[:160]}"
    return str(source)
