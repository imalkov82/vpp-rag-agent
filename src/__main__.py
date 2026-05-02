"""
VPP RAG Agent - CLI Entry Point

Electricity price and grid regulation Q&A system using LangGraph and RAG.
"""

import os

import click
from dotenv import load_dotenv

from src.agent.graph import VppAgent, AgentInput
from src.health import check_health
from src.utils.console import console

load_dotenv()

DEFAULT_ZONE = os.getenv("DEFAULT_AREA", "10YDE-EL------O")


def _format_source(source: dict) -> str:
    if source.get("type") == "price":
        return f"price feed (area={source.get('area', 'unknown')})"
    if source.get("type") == "regulation":
        snippet = source.get("context", "").splitlines()
        head = next((line for line in snippet if line.strip()), "")
        return f"regulation excerpt: {head[:160]}"
    return str(source)


@click.group()
def main() -> None:
    """VPP RAG Agent - Electricity Price & Grid Regulation Q&A."""


@main.command(name="ask")
@click.argument("query")
@click.option(
    "--zone",
    default=DEFAULT_ZONE,
    show_default=True,
    help="Bidding zone (override with $DEFAULT_AREA).",
)
@click.option(
    "--no-index",
    is_flag=True,
    help="Skip RAG indexing entirely (assume the store is already populated).",
)
@click.option(
    "--rebuild-index",
    is_flag=True,
    help="Wipe and rebuild the vector store from data/pdfs/.",
)
@click.pass_context
def ask_command(
    ctx: click.Context,
    query: str,
    zone: str,
    no_index: bool,
    rebuild_index: bool,
) -> None:
    """Ask the agent a question (price / regulation / both)."""
    try:
        if not no_index:
            from src.rag.vectorstore import get_default_rag

            rag = get_default_rag()
            count = rag.index_documents(force_rebuild=rebuild_index)
            if count == 0:
                console.print(
                    "Warning: no PDF chunks indexed (place PDFs under data/pdfs/).",
                    style="yellow",
                )
            else:
                console.print(f"Vector store ready ({count} chunks).", style="dim")

        agent = VppAgent()
        result = agent.run(AgentInput(query=query, bidding_zone=zone))

        console.rule("[bold]ANSWER")
        console.print(result.answer)

        if result.prices and result.prices.get("prices"):
            console.rule("[bold]PRICES")
            for p in result.prices["prices"][:12]:
                ts = p["timestamp"][:16]
                console.print(f"  {ts}: [cyan]{p['price']:.2f}[/] EUR/MWh")

        if result.sources:
            console.rule("[bold]SOURCES")
            for s in result.sources:
                console.print(f"  - {_format_source(s)}")

        if result.error:
            console.print(f"\nNote: {result.error}", style="yellow")
            ctx.exit(2)

    except Exception as e:
        console.print(f"Error: {e}", style="red")
        ctx.exit(1)


@main.command(name="health")
@click.pass_context
def health_command(ctx: click.Context) -> None:
    """Check connectivity to Ollama, ENTSO-E config, vector store, and PDF corpus."""
    results = check_health()
    failed = 0
    for r in results:
        style = "green" if r.ok else "red"
        marker = "OK " if r.ok else "FAIL"
        console.print(f"[{style}]{marker}[/] {r.name}: {r.detail}")
        if not r.ok:
            failed += 1
    if failed:
        ctx.exit(1)


@main.command(name="index")
@click.option(
    "--rebuild",
    is_flag=True,
    help="Wipe and rebuild from scratch instead of incremental indexing.",
)
def index_command(rebuild: bool) -> None:
    """Build (or rebuild) the RAG vector store from data/pdfs/."""
    from src.rag.vectorstore import get_default_rag

    rag = get_default_rag()
    count = rag.index_documents(force_rebuild=rebuild)
    if count == 0:
        console.print(
            "No PDF chunks indexed (place PDFs under data/pdfs/).",
            style="yellow",
        )
    else:
        console.print(f"Vector store ready ({count} chunks).", style="green")


if __name__ == "__main__":
    main()
