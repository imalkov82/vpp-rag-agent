"""Gold evaluation dataset loading."""

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """Hand-curated evaluation case for regulation RAG."""

    id: str
    question: str
    expected_doc_substrings: list[str] = Field(default_factory=list)
    expected_pages: list[int] = Field(default_factory=list)
    answer_must_contain: list[str] = Field(default_factory=list)
    category: str = "regulation"
    follow_up: str | None = None
    follow_up_must_contain: list[str] = Field(default_factory=list)


DEFAULT_EVAL_SET = Path("tests/eval/regulation_eval_set.jsonl")


def load_eval_set(path: Path | str | None = None) -> list[EvalCase]:
    """Load evaluation cases from a JSONL file."""
    eval_path = Path(path) if path is not None else DEFAULT_EVAL_SET
    if not eval_path.exists():
        raise FileNotFoundError(f"eval set not found: {eval_path}")

    cases: list[EvalCase] = []
    for line_no, line in enumerate(eval_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
            cases.append(EvalCase.model_validate(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid eval case at {eval_path}:{line_no}") from exc

    if not cases:
        raise ValueError(f"eval set is empty: {eval_path}")

    return cases
