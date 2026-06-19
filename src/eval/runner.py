"""Run evaluation over the gold dataset."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from langchain_core.retrievers import BaseRetriever

from src.eval.dataset import EvalCase
from src.eval.metrics import LLMJudge, answer_contains_score
from src.eval.report import (
    CaseResult,
    EvalReport,
    aggregate_results,
    case_result_from_docs,
)
from src.models.internals import AgentInput

if TYPE_CHECKING:
    from src.service.agent import VppAgent


def run_eval(
    cases: list[EvalCase],
    *,
    k: int = 4,
    label: str = "baseline",
    retriever: BaseRetriever | None = None,
    agent: VppAgent | None = None,
    judge: LLMJudge | None = None,
    run_agent: bool = True,
    retriever_mode: str | None = None,
) -> EvalReport:
    """Evaluate retrieval and optionally full agent answers on gold cases."""
    if retriever is None:
        from src.service.rag import get_default_rag

        retriever = get_default_rag().get_retriever(k=k, mode=retriever_mode)

    if run_agent and agent is None:
        from src.service.agent import get_default_agent

        agent = get_default_agent()

    results: list[CaseResult] = []

    for case in cases:
        start = time.perf_counter()
        docs = retriever.invoke(case.question)

        faithfulness: float | None = None
        answer_score: float | None = None
        latency_ms = (time.perf_counter() - start) * 1000

        if run_agent and agent is not None:
            output = agent.run(AgentInput(query=case.question))
            latency_ms = (time.perf_counter() - start) * 1000

            answer_score = answer_contains_score(
                output.answer, case.answer_must_contain
            )

            context = ""
            for source in output.sources:
                if source.get("type") == "regulation":
                    context = str(source.get("context", ""))
                    break

            if judge is not None:
                faithfulness = judge.score_faithfulness(
                    case.question, context, output.answer
                )

        results.append(
            case_result_from_docs(
                case,
                docs,
                k=k,
                latency_ms=latency_ms,
                faithfulness=faithfulness,
                answer_contains=answer_score,
            )
        )

    return aggregate_results(label=label, k=k, results=results)
