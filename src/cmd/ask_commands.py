"""'ask' command — query the agent."""

import os

import click
from dotenv import load_dotenv

from src.cmd.commons import console, format_source
from src.models.internals import AgentInput
from src.service.agent import VppAgent

load_dotenv()

DEFAULT_ZONE = os.getenv("DEFAULT_AREA", "10YDE-EL------O")


@click.command(name="ask")
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
            from src.service.rag import get_default_rag

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
                console.print(f"  - {format_source(s)}")

        if result.error:
            console.print(f"\nNote: {result.error}", style="yellow")
            ctx.exit(2)

    except Exception as e:
        console.print(f"Error: {e}", style="red")
        ctx.exit(1)
