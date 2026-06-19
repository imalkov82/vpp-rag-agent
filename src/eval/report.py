"""Evaluation report models and markdown rendering."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.eval.dataset import EvalCase
from src.eval.metrics import context_precision, mrr, recall_at_k


class CaseResult(BaseModel):
    """Per-question evaluation outcome."""

    case_id: str
    question: str
    category: str
    recall_at_k: float
    mrr: float
    context_precision: float
    faithfulness: float | None = None
    answer_contains: float | None = None
    latency_ms: float


class EvalReport(BaseModel):
    """Aggregated evaluation report."""

    label: str
    k: int
    case_count: int
    mean_recall_at_k: float
    mean_mrr: float
    mean_context_precision: float
    mean_faithfulness: float | None = None
    mean_answer_contains: float | None = None
    p95_latency_ms: float
    results: list[CaseResult]
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, int(round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def aggregate_results(
    label: str,
    k: int,
    results: list[CaseResult],
) -> EvalReport:
    """Build an EvalReport from per-case results."""
    return EvalReport(
        label=label,
        k=k,
        case_count=len(results),
        mean_recall_at_k=sum(r.recall_at_k for r in results) / len(results),
        mean_mrr=sum(r.mrr for r in results) / len(results),
        mean_context_precision=sum(r.context_precision for r in results)
        / len(results),
        mean_faithfulness=_mean([r.faithfulness for r in results]),
        mean_answer_contains=_mean([r.answer_contains for r in results]),
        p95_latency_ms=_p95([r.latency_ms for r in results]),
        results=results,
    )


def render_markdown(report: EvalReport) -> str:
    """Render the report as markdown with a baseline summary row."""
    ts = report.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    faithfulness = (
        f"{report.mean_faithfulness:.2f}"
        if report.mean_faithfulness is not None
        else "n/a"
    )
    answer_contains = (
        f"{report.mean_answer_contains:.2f}"
        if report.mean_answer_contains is not None
        else "n/a"
    )

    lines = [
        "# Evaluation Report",
        "",
        f"Generated: {ts}",
        "",
        "## Summary",
        "",
        "| Label | k | Cases | Recall@k | MRR | Context precision | "
        "Faithfulness | Answer contains | p95 latency (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {report.label} | {report.k} | {report.case_count} | "
            f"{report.mean_recall_at_k:.2f} | {report.mean_mrr:.2f} | "
            f"{report.mean_context_precision:.2f} | {faithfulness} | "
            f"{answer_contains} | {report.p95_latency_ms:.0f} |"
        ),
        "",
        "## Per-case results",
        "",
        "| ID | Category | Recall@k | MRR | Ctx prec | Faith | Answer | ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in report.results:
        faith = (
            f"{result.faithfulness:.2f}"
            if result.faithfulness is not None
            else "n/a"
        )
        answer = (
            f"{result.answer_contains:.2f}"
            if result.answer_contains is not None
            else "n/a"
        )
        lines.append(
            f"| {result.case_id} | {result.category} | "
            f"{result.recall_at_k:.2f} | {result.mrr:.2f} | "
            f"{result.context_precision:.2f} | {faith} | {answer} | "
            f"{result.latency_ms:.0f} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_report(report: EvalReport, path: Path | str = "docs/eval_report.md") -> Path:
    """Write markdown report to disk."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(report))
    return out


def case_result_from_docs(
    case: EvalCase,
    docs: list,
    *,
    k: int,
    latency_ms: float,
    faithfulness: float | None = None,
    answer_contains: float | None = None,
) -> CaseResult:
    """Build a CaseResult from retrieved documents."""
    documents: list[Document] = []
    for item in docs:
        if isinstance(item, Document):
            documents.append(item)
    return CaseResult(
        case_id=case.id,
        question=case.question,
        category=case.category,
        recall_at_k=recall_at_k(documents, case, k),
        mrr=mrr(documents, case, k),
        context_precision=context_precision(documents, case, k),
        faithfulness=faithfulness,
        answer_contains=answer_contains,
        latency_ms=latency_ms,
    )
