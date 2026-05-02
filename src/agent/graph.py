"""LangGraph agent for electricity price and grid regulation Q&A"""

from enum import Enum
from typing import TypedDict, Optional, List
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from src.entsoe.client import get_default_client
from src.rag.vectorstore import get_default_rag

load_dotenv()


PRICE_KEYWORDS = {"price", "prices", "cost", "spot", "eur", "euro", "mwh", "kwh"}
REGULATION_KEYWORDS = {
    "regulation",
    "regulations",
    "rule",
    "rules",
    "policy",
    "policies",
    "grid",
    "code",
    "codes",
    "balancing",
    "reserve",
    "capacity",
    "entsoe",
}


def _tokenize(text: str) -> set[str]:
    return set(
        word
        for word in "".join(c.lower() if c.isalnum() else " " for c in text).split()
    )


class QueryType(str, Enum):
    """Types of queries the agent can handle"""

    PRICE = "price"
    REGULATION = "regulation"
    BOTH = "both"
    UNKNOWN = "unknown"


class AgentState(TypedDict):
    """State of the agent"""

    messages: List
    query_type: Optional[QueryType]
    prices: Optional[dict]
    regulation_context: Optional[str]
    final_answer: Optional[str]
    error: Optional[str]
    bidding_zone: str


class AgentInput(BaseModel):
    """Input to the agent"""

    query: str
    bidding_zone: str = "10YDE-EL------O"


class AgentOutput(BaseModel):
    """Output from the agent"""

    answer: str
    sources: List[dict]
    prices: Optional[dict]
    error: Optional[str] = None


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
        self.graph = self._build_graph()

    def _classify_query(self, state: AgentState) -> AgentState:
        """Classify the query type using whole-word token matching."""
        tokens = _tokenize(state["messages"][-1].content)
        has_price = bool(tokens & PRICE_KEYWORDS)
        has_reg = bool(tokens & REGULATION_KEYWORDS)

        if has_price and has_reg:
            state["query_type"] = QueryType.BOTH
        elif has_price:
            state["query_type"] = QueryType.PRICE
        elif has_reg:
            state["query_type"] = QueryType.REGULATION
        else:
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
        """Fetch electricity prices"""
        try:
            zone = state.get("bidding_zone", "10YDE-EL------O")
            client = get_default_client()
            data = client.get_day_ahead_prices(zone)

            state["prices"] = {
                "area": data.area,
                "prices": [
                    {"timestamp": p.timestamp.isoformat(), "price": p.price}
                    for p in data.prices
                ],
            }
        except Exception as e:
            state["error"] = f"price fetch failed: {e}"

        return state

    def _search_regs(self, state: AgentState) -> AgentState:
        """Search regulations"""
        try:
            query = state["messages"][-1].content
            rag = get_default_rag()
            state["regulation_context"] = rag.get_context(query, k=3)
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
        """Generate final answer using LLM"""
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

        response = self.llm.invoke([system_msg, HumanMessage(content=prompt)])
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
