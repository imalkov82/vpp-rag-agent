"""Retrieval and faithfulness metrics for evaluation."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.eval.dataset import EvalCase


def _doc_matches(case: EvalCase, doc: Document) -> bool:
    source = str(doc.metadata.get("source", "")).lower()
    page = int(doc.metadata.get("page", 0) or 0)
    content = doc.page_content.lower()

    if case.expected_doc_substrings:
        if any(
            sub.lower() in source or sub.lower() in content
            for sub in case.expected_doc_substrings
        ):
            return True

    if case.expected_pages and page in case.expected_pages:
        return True

    return not case.expected_doc_substrings and not case.expected_pages


def recall_at_k(docs: list[Document], case: EvalCase, k: int) -> float:
    """1.0 if any relevant document appears in the top-k results."""
    if not case.expected_doc_substrings and not case.expected_pages:
        return 1.0
    for doc in docs[:k]:
        if _doc_matches(case, doc):
            return 1.0
    return 0.0


def mrr(docs: list[Document], case: EvalCase, k: int) -> float:
    """Reciprocal rank of the first relevant document in top-k."""
    if not case.expected_doc_substrings and not case.expected_pages:
        return 1.0
    for rank, doc in enumerate(docs[:k], start=1):
        if _doc_matches(case, doc):
            return 1.0 / rank
    return 0.0


def context_precision(docs: list[Document], case: EvalCase, k: int) -> float:
    """Fraction of top-k retrieved chunks that are relevant."""
    top_k = docs[:k]
    if not top_k:
        return 0.0
    if not case.expected_doc_substrings and not case.expected_pages:
        return 1.0
    relevant = sum(1 for doc in top_k if _doc_matches(case, doc))
    return relevant / len(top_k)


def answer_contains_score(answer: str, required: list[str]) -> float | None:
    """Fraction of required substrings present in the generated answer."""
    if not required:
        return None
    lowered = answer.lower()
    hits = sum(1 for token in required if token.lower() in lowered)
    return hits / len(required)


class FaithfulnessJudgment(BaseModel):
    """LLM structured judgment for answer faithfulness to context."""

    faithful: bool
    reasoning: str = ""


class LLMJudge:
    """Score whether an answer is grounded in retrieved context."""

    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        base_url: str = "http://localhost:11434",
    ):
        self._llm = ChatOllama(model=model, temperature=0.0, base_url=base_url)
        self._judge = self._llm.with_structured_output(FaithfulnessJudgment)

    def score_faithfulness(
        self, question: str, context: str, answer: str
    ) -> float | None:
        """Return 1.0 for faithful, 0.0 for unfaithful, None on judge failure."""
        if not context.strip() or not answer.strip():
            return None

        prompt = (
            "Decide if the answer is fully supported by the context. "
            "Return faithful=false if the answer adds unsupported facts."
        )
        try:
            raw = self._judge.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(
                        content=(
                            f"Question: {question}\n\n"
                            f"Context:\n{context}\n\n"
                            f"Answer:\n{answer}"
                        )
                    ),
                ]
            )
            result = (
                raw
                if isinstance(raw, FaithfulnessJudgment)
                else FaithfulnessJudgment.model_validate(raw)
            )
            return 1.0 if result.faithful else 0.0
        except Exception:
            return None
