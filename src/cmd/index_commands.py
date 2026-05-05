"""'index' command — build the RAG vector store."""

import click

from src.cmd.commons import console


@click.command(name="index")
@click.option(
    "--rebuild",
    is_flag=True,
    help="Wipe and rebuild from scratch instead of incremental indexing.",
)
def index_command(rebuild: bool) -> None:
    """Build (or rebuild) the RAG vector store from data/pdfs/."""
    from src.service.rag import get_default_rag

    rag = get_default_rag()
    count = rag.index_documents(force_rebuild=rebuild)
    if count == 0:
        console.print(
            "No PDF chunks indexed (place PDFs under data/pdfs/).",
            style="yellow",
        )
    else:
        console.print(f"Vector store ready ({count} chunks).", style="green")
