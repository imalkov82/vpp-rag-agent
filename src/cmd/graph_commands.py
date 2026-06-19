"""'graph' command group — inspect the regulation knowledge graph."""

import click

from src.cmd.commons import console
from src.service.graph_search import query_graph


@click.group(name="graph")
def graph_group() -> None:
    """Inspect the regulation knowledge graph."""


@graph_group.command(name="query")
@click.argument("query")
@click.option("--hops", default=2, show_default=True, help="Traversal depth.")
def graph_query_command(query: str, hops: int) -> None:
    """Match entities in QUERY and show graph neighborhoods."""
    console.print(query_graph(query, hops=hops))
