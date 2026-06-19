"""Tests for Phase 4 agent maturity features."""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.models.internals import AgentInput, QueryClassification, QueryType
from src.service.agent import VppAgent
from src.service.react_agent import extract_react_answer
from src.service.subgraphs import build_price_subgraph


class TestSubgraphs:
    def test_price_subgraph_runs_fetch_node(self):
        def fetch(state):
            state["prices"] = {"area": "test", "prices": []}
            return state

        graph = build_price_subgraph(fetch)
        result = graph.invoke(
            {
                "messages": [HumanMessage(content="price?")],
                "query_type": QueryType.PRICE,
                "prices": None,
                "regulation_context": None,
                "final_answer": None,
                "error": None,
                "degraded": False,
                "bidding_zone": "10YDE-EL------O",
            }
        )
        assert result["prices"]["area"] == "test"


class TestAgentMaturity:
    @patch.object(VppAgent, "__init__", lambda self, *a, **kw: None)
    def _make_agent(self):
        agent = VppAgent()
        agent.llm_with_tools = MagicMock()
        return agent

    @patch("src.service.agent.get_electricity_prices")
    def test_get_prices_retries_before_degraded(self, mock_tool):
        agent = self._make_agent()
        mock_tool.invoke.side_effect = [
            RuntimeError("timeout"),
            RuntimeError("timeout"),
            json.dumps({"area": "10YDE-EL------O", "prices": []}),
        ]

        state = {
            "bidding_zone": "10YDE-EL------O",
            "error": None,
            "degraded": False,
        }
        result = agent._get_prices(state)

        assert result["prices"]["area"] == "10YDE-EL------O"
        assert mock_tool.invoke.call_count == 3

    @patch("src.service.agent.get_electricity_prices")
    def test_get_prices_marks_degraded_after_retries(self, mock_tool):
        agent = self._make_agent()
        mock_tool.invoke.side_effect = RuntimeError("down")

        state = {
            "bidding_zone": "10YDE-EL------O",
            "error": None,
            "degraded": False,
        }
        result = agent._get_prices(state)

        assert result["degraded"] is True
        assert "price fetch failed" in result["error"]

    def test_generate_answer_degraded_without_context(self):
        agent = self._make_agent()
        state = {
            "messages": [HumanMessage(content="prices?")],
            "error": "price fetch failed",
            "degraded": True,
            "prices": None,
            "regulation_context": None,
        }
        result = agent._generate_answer(state)
        assert "could not retrieve" in result["final_answer"].lower()

    def test_checkpoint_thread_appends_messages(self):
        with patch.object(VppAgent, "__init__", lambda self, *a, **kw: None):
            agent = VppAgent()
        agent.classifier = MagicMock()
        agent.classifier.invoke.return_value = QueryClassification(
            query_type=QueryType.UNKNOWN,
            reasoning="",
        )
        agent.llm_with_tools = MagicMock()
        agent.llm_with_tools.invoke.return_value = AIMessage(content="hello")
        agent.use_react = False
        agent.checkpointer = MemorySaver()
        agent.graph = agent._build_graph()

        first = agent.run(AgentInput(query="What is FCR?"), thread_id="t1")
        assert first.answer

        snapshot = agent.graph.get_state({"configurable": {"thread_id": "t1"}})
        assert len(snapshot.values["messages"]) >= 2

    def test_extract_react_answer(self):
        answer = extract_react_answer(
            [
                HumanMessage(content="q"),
                AIMessage(content="final response"),
            ]
        )
        assert answer == "final response"
