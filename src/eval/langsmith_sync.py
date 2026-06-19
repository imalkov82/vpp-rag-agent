"""Optional LangSmith dataset and experiment sync."""

from __future__ import annotations

import os

from src.eval.dataset import EvalCase
from src.eval.report import EvalReport


def langsmith_enabled() -> bool:
    return bool(os.getenv("LANGCHAIN_TRACING_V2")) and bool(
        os.getenv("LANGSMITH_API_KEY")
    )


def upload_dataset(
    cases: list[EvalCase],
    dataset_name: str = "vpp-rag-regulation-eval",
) -> str:
    """Create or update a LangSmith dataset from gold eval cases."""
    from langsmith import Client

    client = Client()
    existing = list(client.list_datasets(dataset_name=dataset_name))
    dataset = (
        existing[0]
        if existing
        else client.create_dataset(dataset_name=dataset_name)
    )

    for case in cases:
        client.create_example(
            inputs={"question": case.question, "category": case.category},
            outputs={
                "expected_doc_substrings": case.expected_doc_substrings,
                "expected_pages": case.expected_pages,
                "answer_must_contain": case.answer_must_contain,
            },
            dataset_id=dataset.id,
        )

    return str(dataset.id)


def log_eval_summary(
    report: EvalReport,
    *,
    experiment_name: str | None = None,
) -> None:
    """No-op placeholder: agent runs trace via LANGCHAIN_TRACING_V2 when enabled.

    Aggregate metrics are persisted locally in docs/eval_report.md.
    """
    _ = (report, experiment_name)
