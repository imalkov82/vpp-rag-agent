"""VPP RAG Agent — CLI entry point.

Electricity price and grid regulation Q&A system using LangGraph and RAG.
"""

import click

from src.cmd.ask_commands import ask_command
from src.cmd.eval_commands import eval_group
from src.cmd.health_commands import health_command
from src.cmd.index_commands import index_command


@click.group()
def main() -> None:
    """VPP RAG Agent - Electricity Price & Grid Regulation Q&A."""


main.add_command(ask_command)
main.add_command(health_command)
main.add_command(index_command)
main.add_command(eval_group)


if __name__ == "__main__":
    main()
