"""'health' command — verify external dependencies."""

import click

from src.cmd.commons import console
from src.health import check_health


@click.command(name="health")
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
