"""Tests for multiturn eval runner."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.eval.dataset import EvalCase, load_eval_set
from src.eval.runner import run_eval


class TestMultiturnEval:
    def test_multiturn_runs_follow_up_on_same_thread(self):
        cases = load_eval_set("tests/eval/multiturn_eval_set.jsonl")
        agent = MagicMock()
        agent.run.side_effect = [
            MagicMock(answer="FCR rules", sources=[]),
            MagicMock(answer="SO GL covers FCR", sources=[]),
        ] * 3

        retriever = MagicMock()
        retriever.invoke.return_value = [
            Document(
                page_content="FCR frequency containment",
                metadata={"source": "so_gl.pdf", "page": 1},
            )
        ]

        report = run_eval(
            cases[:1],
            k=2,
            label="multiturn-test",
            retriever=retriever,
            agent=agent,
            multiturn=True,
        )

        assert report.case_count == 1
        assert agent.run.call_count == 2
        thread_ids = [
            call.kwargs.get("thread_id")
            for call in agent.run.call_args_list
        ]
        assert thread_ids[0] == thread_ids[1] == "eval-mt-001"

    def test_load_multiturn_cases(self):
        cases = load_eval_set("tests/eval/multiturn_eval_set.jsonl")
        assert all(case.follow_up for case in cases)
