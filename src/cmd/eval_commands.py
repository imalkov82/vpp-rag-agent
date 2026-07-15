"""'eval' command group — measure retrieval and agent quality."""

from pathlib import Path

import click
from rich.table import Table

from src.cmd.commons import console
from src.eval.dataset import load_eval_set
from src.eval.metrics import LLMJudge
from src.eval.report import write_report
from src.eval.runner import run_eval


@click.group(name="eval")
def eval_group() -> None:
    """Run evaluation against the hand-curated gold dataset."""


@eval_group.command(name="run")
@click.option(
    "--dataset",
    "dataset_path",
    default="tests/eval/regulation_eval_set.jsonl",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Path to the JSONL gold eval set.",
)
@click.option("--k", default=4, show_default=True, help="Retriever top-k.")
@click.option(
    "--label",
    default="baseline-vector",
    show_default=True,
    help="Label recorded in the eval report.",
)
@click.option(
    "--no-judge",
    is_flag=True,
    help="Skip LLM faithfulness judging (retrieval metrics only).",
)
@click.option(
    "--retrieval-only",
    is_flag=True,
    help="Skip full agent runs; measure retriever metrics only.",
)
@click.option(
    "--report",
    default="docs/eval_report.md",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Where to write the markdown report.",
)
@click.option(
    "--retriever",
    "retriever_mode",
    default=None,
    type=click.Choice(["vector", "hybrid", "entity", "graph"]),
    help="Retriever mode (defaults to VPP_RETRIEVER or vector).",
)
@click.option(
    "--multiturn",
    is_flag=True,
    help="Run follow-up turns with checkpointed thread ids (requires follow_up in dataset).",
)
def eval_run_command(
    dataset_path: Path,
    k: int,
    label: str,
    no_judge: bool,
    retrieval_only: bool,
    report: Path,
    retriever_mode: str | None,
    multiturn: bool,
) -> None:
    """Run the gold eval set and write docs/eval_report.md."""
    from src.service.rag import get_default_rag

    rag = get_default_rag()
    if not rag.is_indexed():
        console.print(
            "Vector store is empty. Index PDFs first: uv run vpp-rag index",
            style="red",
        )
        raise SystemExit(1)

    from src.service.retriever_config import get_retriever_mode

    mode = get_retriever_mode(retriever_mode)
    if label == "baseline-vector" and mode != "vector":
        label = f"baseline-{mode}"

    cases = load_eval_set(dataset_path)
    judge = None if no_judge or retrieval_only else LLMJudge()

    console.print(
        f"Running {len(cases)} eval cases (k={k}, label={label})...",
        style="bold",
    )
    eval_report = run_eval(
        cases,
        k=k,
        label=label,
        judge=judge,
        run_agent=not retrieval_only,
        retriever_mode=mode,
        multiturn=multiturn,
    )

    out = write_report(eval_report, report)
    console.print(f"Wrote report to {out}", style="green")

    table = Table(title="Eval summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Recall@k", f"{eval_report.mean_recall_at_k:.2f}")
    table.add_row("MRR", f"{eval_report.mean_mrr:.2f}")
    table.add_row("Context precision", f"{eval_report.mean_context_precision:.2f}")
    if eval_report.mean_faithfulness is not None:
        table.add_row("Faithfulness", f"{eval_report.mean_faithfulness:.2f}")
    if eval_report.mean_answer_contains is not None:
        table.add_row("Answer contains", f"{eval_report.mean_answer_contains:.2f}")
    table.add_row("p95 latency (ms)", f"{eval_report.p95_latency_ms:.0f}")
    console.print(table)
