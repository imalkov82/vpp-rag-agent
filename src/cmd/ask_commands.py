"""'ask' command — query the agent."""

import os

import click
from dotenv import load_dotenv

from src.cmd.commons import console, format_source
from src.models.internals import AgentInput
from src.service.agent import VppAgent

load_dotenv()

DEFAULT_ZONE = os.getenv("DEFAULT_AREA", "10YDE-EL------O")


def _ensure_index(no_index: bool, rebuild_index: bool) -> None:
    if no_index:
        return
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


def _print_result(result) -> None:
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
@click.option(
    "--thread",
    "thread_id",
    default=None,
    help="Conversation thread id for Sqlite checkpointing.",
)
@click.option(
    "--stream",
    is_flag=True,
    help="Stream answer tokens live (falls back to node updates if needed).",
)
@click.option(
    "--react",
    is_flag=True,
    help="Use LangGraph ReAct tool-loop agent instead of classify-and-route.",
)
@click.pass_context
def ask_command(
    ctx: click.Context,
    query: str,
    zone: str,
    no_index: bool,
    rebuild_index: bool,
    thread_id: str | None,
    stream: bool,
    react: bool,
) -> None:
    """Ask the agent a question (price / regulation / both)."""
    try:
        _ensure_index(no_index, rebuild_index)

        agent = VppAgent(use_react=react)
        agent_input = AgentInput(query=query, bidding_zone=zone)

        if stream:
            console.rule("[bold]ANSWER")
            streamed = False
            try:
                for token in agent.stream_run(agent_input, thread_id=thread_id):
                    console.print(token, end="")
                    streamed = True
                if streamed:
                    console.print()
            except Exception:
                streamed = False

            if not streamed:
                for update in agent.stream_updates(agent_input, thread_id=thread_id):
                    for node_update in update.values():
                        answer = node_update.get("final_answer")
                        if answer:
                            console.print(answer)
        else:
            result = agent.run(agent_input, thread_id=thread_id)
            _print_result(result)

        config = agent._config(thread_id)
        snapshot = agent.graph.get_state(config)
        result = (
            agent._to_output(snapshot.values)
            if snapshot.values
            else agent.run(agent_input, thread_id=thread_id)
        )

        if stream:
            if result.prices and result.prices.get("prices"):
                console.rule("[bold]PRICES")
                for p in result.prices["prices"][:12]:
                    ts = p["timestamp"][:16]
                    console.print(f"  {ts}: [cyan]{p['price']:.2f}[/] EUR/MWh")
            if result.sources:
                console.rule("[bold]SOURCES")
                for s in result.sources:
                    console.print(f"  - {format_source(s)}")

        if thread_id:
            console.print(f"Thread: {thread_id}", style="dim")

        if result.error:
            console.print(f"\nNote: {result.error}", style="yellow")
            ctx.exit(2)

    except Exception as e:
        console.print(f"Error: {e}", style="red")
        ctx.exit(1)
