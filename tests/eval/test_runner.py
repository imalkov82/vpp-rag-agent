"""Tests for eval dataset loading and runner."""

from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.eval.dataset import load_eval_set
from src.eval.runner import run_eval


class TestLoadEvalSet:
    def test_loads_jsonl(self):
        cases = load_eval_set(Path("tests/eval/regulation_eval_set.jsonl"))
        assert len(cases) >= 30
        assert cases[0].id


class TestRunEval:
    def test_run_eval_with_mock_retriever(self):
        cases = load_eval_set(Path("tests/eval/regulation_eval_set.jsonl"))[:3]
        retriever = MagicMock()
        retriever.invoke.return_value = [
            Document(
                page_content="FCR frequency containment reserve",
                metadata={"source": "so_gl.pdf", "page": 1},
            )
        ]

        report = run_eval(
            cases,
            k=2,
            label="test",
            retriever=retriever,
            run_agent=False,
        )

        assert report.case_count == 3
        assert 0.0 <= report.mean_recall_at_k <= 1.0
        assert report.p95_latency_ms >= 0.0
