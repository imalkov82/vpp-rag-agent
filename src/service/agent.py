"""LangGraph agent for electricity price and grid regulation Q&A"""

import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.models.internals import (
    AgentInput,
    AgentOutput,
    AgentState,
    QueryClassification,
    QueryType,
)
from src.service.tools import VPP_TOOLS, get_electricity_prices, search_regulations

load_dotenv()

CLASSIFY_SYSTEM_PROMPT = (
    "Classify the user query for a European electricity market assistant. "
    "Use 'price' for electricity price or cost questions, 'regulation' for "
    "grid codes, balancing, reserves, or policy questions, 'both' when the "
    "query clearly needs live prices and regulation documents, and 'unknown' "
    "for general questions that do not need either data source."
)


class VppAgent:
    """LangGraph agent for VPP electricity queries"""

    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        temperature: float = 0.3,
        base_url: str = "http://localhost:11434",
    ):
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
        )
        self.classifier = self.llm.with_structured_output(QueryClassification)
        self.llm_with_tools = self.llm.bind_tools(VPP_TOOLS)
        self.graph = self._build_graph()

    def _classify_query(self, state: AgentState) -> AgentState:
        """Classify query intent with structured LLM output."""
        query = state["messages"][-1].content
        try:
            raw = self.classifier.invoke(
                [
                    SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
                    HumanMessage(content=query),
                ]
            )
            result = (
                raw
                if isinstance(raw, QueryClassification)
                else QueryClassification.model_validate(raw)
            )
            state["query_type"] = result.query_type
        except Exception:
            state["query_type"] = QueryType.UNKNOWN

        return state

    def _route_query(self, state: AgentState) -> str:
        """Route to appropriate handler"""
        qt = state.get("query_type") or QueryType.UNKNOWN
        return {
            QueryType.PRICE: "get_prices",
            QueryType.REGULATION: "search_regs",
            QueryType.BOTH: "get_both",
        }.get(qt, "general")

    def _get_prices(self, state: AgentState) -> AgentState:
        """Fetch electricity prices via LangChain tool."""
        try:
            zone = state.get("bidding_zone", "10YDE-EL------O")
            raw = get_electricity_prices.invoke({"bidding_zone": zone})
            state["prices"] = json.loads(raw)
        except Exception as e:
            state["error"] = f"price fetch failed: {e}"

        return state

    def _search_regs(self, state: AgentState) -> AgentState:
        """Search regulations via LangChain tool + retriever chain."""
        try:
            query = state["messages"][-1].content
            state["regulation_context"] = search_regulations.invoke(
                {"query": query, "k": 3}
            )
        except Exception as e:
            state["error"] = f"regulation search failed: {e}"

        return state

    def _get_both(self, state: AgentState) -> AgentState:
        """Handle both price and regulation queries"""
        state = self._get_prices(state)
        state = self._search_regs(state)
        return state

    def _general(self, state: AgentState) -> AgentState:
        """General fallback"""
        return state

    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate final answer using LLM with tools bound."""
        query = state["messages"][-1].content

        system_msg = SystemMessage(
            content=(
                "You are an expert on European electricity markets and grid "
                "regulations. Answer user questions based on real-time price "
                "data and ENTSO-E regulation documents. Always cite your "
                "sources. Be concise and informative."
            )
        )

        context_parts: list[str] = []

        if state.get("error"):
            context_parts.append(f"Note: a data lookup failed — {state['error']}")

        prices = state.get("prices")
        if prices and prices.get("prices"):
            points = prices["prices"]
            latest = points[-1]
            context_parts.append(
                f"Latest price ({prices['area']}): "
                f"{latest['price']:.2f} EUR/MWh at {latest['timestamp']}"
            )

        if state.get("regulation_context"):
            context_parts.append(
                f"Relevant regulations:\n{state['regulation_context']}"
            )

        if context_parts:
            prompt = f"{query}\n\nContext:\n" + "\n\n".join(context_parts)
        else:
            prompt = query

        response = self.llm_with_tools.invoke(
            [system_msg, HumanMessage(content=prompt)]
        )
        content = response.content
        state["final_answer"] = content if isinstance(content, str) else str(content)
        return state

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph state machine"""
        graph = StateGraph(AgentState)

        graph.add_node("classify", self._classify_query)
        graph.add_node("get_prices", self._get_prices)
        graph.add_node("search_regs", self._search_regs)
        graph.add_node("get_both", self._get_both)
        graph.add_node("general", self._general)
        graph.add_node("answer", self._generate_answer)

        graph.set_entry_point("classify")
        graph.add_conditional_edges(
            "classify",
            self._route_query,
            {
                "get_prices": "get_prices",
                "search_regs": "search_regs",
                "get_both": "get_both",
                "general": "general",
            },
        )

        graph.add_edge("get_prices", "answer")
        graph.add_edge("search_regs", "answer")
        graph.add_edge("get_both", "answer")
        graph.add_edge("general", "answer")
        graph.add_edge("answer", END)

        return graph.compile()

    def run(self, input: AgentInput) -> AgentOutput:
        """Run the agent"""
        initial_state: AgentState = {
            "messages": [HumanMessage(content=input.query)],
            "query_type": None,
            "prices": None,
            "regulation_context": None,
            "final_answer": None,
            "error": None,
            "bidding_zone": input.bidding_zone,
        }

        result = self.graph.invoke(initial_state)

        sources: list[dict] = []
        if result.get("regulation_context"):
            sources.append(
                {
                    "type": "regulation",
                    "context": result["regulation_context"][:500],
                }
            )
        if result.get("prices"):
            sources.append({"type": "price", "area": result["prices"]["area"]})

        return AgentOutput(
            answer=result.get("final_answer", "No answer generated"),
            sources=sources,
            prices=result.get("prices"),
            error=result.get("error"),
        )


def get_default_agent() -> VppAgent:
    """Get configured agent"""
    return VppAgent()
