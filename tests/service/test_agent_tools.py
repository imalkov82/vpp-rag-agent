"""Tests for LangChain tools and agent routing."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.models.internals import QueryClassification, QueryType
from src.service.agent import VppAgent
from src.service.tools import VPP_TOOLS, get_electricity_prices, search_regulations


class TestVppTools:
    def test_tools_registered(self):
        names = {t.name for t in VPP_TOOLS}
        assert names == {"get_electricity_prices", "search_regulations"}

    @patch("src.service.tools.get_default_client")
    def test_get_electricity_prices_tool(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        price = MagicMock()
        price.timestamp.isoformat.return_value = "2024-01-15T12:00:00"
        price.price = 85.5

        data = MagicMock()
        data.area = "10YDE-EL------O"
        data.prices = [price]
        mock_client.get_day_ahead_prices.return_value = data

        raw = get_electricity_prices.invoke({"bidding_zone": "10YDE-EL------O"})
        payload = json.loads(raw)

        assert payload["area"] == "10YDE-EL------O"
        assert payload["prices"][0]["price"] == 85.5
        mock_client.get_day_ahead_prices.assert_called_once_with("10YDE-EL------O")

    @patch("src.service.tools.get_default_rag")
    def test_search_regulations_tool(self, mock_get_rag):
        mock_rag = MagicMock()
        mock_rag.get_context_via_retriever.return_value = (
            "[Source: test.pdf p.1]\nFCR rules"
        )
        mock_get_rag.return_value = mock_rag

        result = search_regulations.invoke({"query": "FCR requirements", "k": 3})

        assert "FCR rules" in result
        mock_rag.get_context_via_retriever.assert_called_once_with(
            "FCR requirements", k=3
        )


class TestVppAgentRouting:
    @patch.object(VppAgent, "__init__", lambda self, *a, **kw: None)
    def _make_agent(self):
        agent = VppAgent()
        agent.classifier = MagicMock()
        agent.llm = MagicMock()
        agent.graph = MagicMock()
        return agent

    def test_classify_query_structured(self):
        agent = self._make_agent()
        agent.classifier.invoke.return_value = QueryClassification(
            query_type=QueryType.REGULATION,
            reasoning="asks about grid code",
        )

        state = {
            "messages": [MagicMock(content="Explain the balancing reserve rules")],
            "query_type": None,
        }
        result = agent._classify_query(state)

        assert result["query_type"] == QueryType.REGULATION.value

    def test_classify_query_fallback_on_error(self):
        agent = self._make_agent()
        agent.classifier.invoke.side_effect = RuntimeError("ollama down")

        state = {
            "messages": [MagicMock(content="hello")],
            "query_type": None,
        }
        result = agent._classify_query(state)

        assert result["query_type"] == QueryType.UNKNOWN.value

    @pytest.mark.parametrize(
        ("query_type", "expected_route"),
        [
            (QueryType.PRICE, "price_subgraph"),
            (QueryType.REGULATION, "regulation_subgraph"),
            (QueryType.BOTH, "get_both"),
            (QueryType.UNKNOWN, "general"),
        ],
    )
    def test_route_query(self, query_type, expected_route):
        agent = self._make_agent()
        state = {"query_type": query_type}
        assert agent._route_query(state) == expected_route

    @patch("src.service.agent.get_electricity_prices")
    def test_get_prices_uses_tool(self, mock_tool):
        agent = self._make_agent()
        mock_tool.invoke.return_value = json.dumps(
            {"area": "10YDE-EL------O", "prices": [{"timestamp": "t", "price": 1.0}]}
        )

        state = {"bidding_zone": "10YDE-EL------O", "error": None}
        result = agent._get_prices(state)

        assert result["prices"]["area"] == "10YDE-EL------O"
        mock_tool.invoke.assert_called_once_with({"bidding_zone": "10YDE-EL------O"})

    @patch("src.service.agent.search_regulations")
    def test_search_regs_uses_tool(self, mock_tool):
        agent = self._make_agent()
        mock_tool.invoke.return_value = "regulation context"

        state = {"messages": [MagicMock(content="balancing rules")], "error": None}
        result = agent._search_regs(state)

        assert result["regulation_context"] == "regulation context"
        mock_tool.invoke.assert_called_once_with({"query": "balancing rules", "k": 3})
