"""'index' command — build the RAG vector store."""

import click

from src.cmd.commons import console


@click.command(name="index")
@click.option(
    "--rebuild",
    is_flag=True,
    help="Wipe and rebuild from scratch instead of incremental indexing.",
)
@click.option(
    "--with-graph",
    is_flag=True,
    help="Also build the regulation knowledge graph (.graph_db/).",
)
@click.option(
    "--use-llm",
    is_flag=True,
    help="Use LLM extraction for graph triples (requires Ollama).",
)
def index_command(rebuild: bool, with_graph: bool, use_llm: bool) -> None:
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

    if with_graph:
        from src.service.graph_ingest import index_graph

        node_count = index_graph(force_rebuild=rebuild, use_llm=use_llm, rag=rag)
        if node_count == 0:
            console.print(
                "Knowledge graph empty (no extractable entities in PDFs).",
                style="yellow",
            )
        else:
            console.print(
                f"Knowledge graph ready ({node_count} nodes).",
                style="green",
            )
