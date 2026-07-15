"""Tests for markdown report rendering."""

from src.eval.report import CaseResult, EvalReport, render_markdown


class TestRenderMarkdown:
    def test_contains_summary_row(self):
        report = EvalReport(
            label="baseline-vector",
            k=4,
            case_count=1,
            mean_recall_at_k=0.5,
            mean_mrr=0.5,
            mean_context_precision=0.5,
            mean_faithfulness=1.0,
            mean_answer_contains=0.5,
            p95_latency_ms=120.0,
            results=[
                CaseResult(
                    case_id="reg-001",
                    question="What is FCR?",
                    category="balancing",
                    recall_at_k=0.5,
                    mrr=0.5,
                    context_precision=0.5,
                    faithfulness=1.0,
                    answer_contains=0.5,
                    latency_ms=120.0,
                )
            ],
        )
        md = render_markdown(report)
        assert "baseline-vector" in md
        assert "Recall@k" in md
        assert "reg-001" in md
