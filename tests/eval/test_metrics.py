"""Tests for retrieval metric math."""

from langchain_core.documents import Document

from src.eval.dataset import EvalCase
from src.eval.metrics import (
    answer_contains_score,
    context_precision,
    mrr,
    recall_at_k,
)


def _case(**kwargs) -> EvalCase:
    return EvalCase(id="t1", question="q", **kwargs)


class TestRecallAtK:
    def test_hit_on_first_rank(self):
        case = _case(expected_doc_substrings=["so_gl"])
        docs = [
            Document(
                page_content="FCR rules",
                metadata={"source": "so_gl.pdf", "page": 1},
            ),
            Document(page_content="other", metadata={"source": "other.pdf", "page": 2}),
        ]
        assert recall_at_k(docs, case, k=2) == 1.0

    def test_miss_returns_zero(self):
        case = _case(expected_doc_substrings=["missing"])
        docs = [
            Document(page_content="x", metadata={"source": "a.pdf", "page": 1}),
        ]
        assert recall_at_k(docs, case, k=3) == 0.0


class TestMRR:
    def test_second_rank(self):
        case = _case(expected_doc_substrings=["target"])
        docs = [
            Document(page_content="noise", metadata={"source": "a.pdf", "page": 1}),
            Document(
                page_content="target doc",
                metadata={"source": "target.pdf", "page": 2},
            ),
        ]
        assert mrr(docs, case, k=2) == 0.5


class TestContextPrecision:
    def test_half_relevant(self):
        case = _case(expected_doc_substrings=["good"])
        docs = [
            Document(page_content="good", metadata={"source": "good.pdf", "page": 1}),
            Document(page_content="bad", metadata={"source": "bad.pdf", "page": 2}),
        ]
        assert context_precision(docs, case, k=2) == 0.5


class TestAnswerContains:
    def test_partial_match(self):
        assert answer_contains_score("FCR and aFRR rules", ["FCR", "missing"]) == 0.5

    def test_empty_requirements(self):
        assert answer_contains_score("answer", []) is None
