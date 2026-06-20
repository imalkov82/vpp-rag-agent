"""LangGraph agent for electricity price and grid regulation Q&A"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
from src.service.checkpoints import get_checkpointer
from src.service.react_agent import build_react_graph, extract_react_answer
from src.service.subgraphs import build_price_subgraph, build_regulation_subgraph
from src.service.tools import get_electricity_prices, search_regulations

load_dotenv()

CLASSIFY_SYSTEM_PROMPT = (
    "Classify the user query for a European electricity market assistant. "
    "Use 'price' for electricity price or cost questions, 'regulation' for "
    "grid codes, balancing, reserves, or policy questions, 'both' when the "
    "query clearly needs live prices and regulation documents, and 'unknown' "
    "for general questions that do not need either data source."
)

ANSWER_SYSTEM_PROMPT = (
    "You are an expert on European electricity markets and grid "
    "regulations. Answer user questions based on real-time price "
    "data and ENTSO-E regulation documents. Always cite your "
    "sources. Be concise and informative."
)

DEGRADED_ANSWER_TEMPLATE = (
    "I could not retrieve all requested data ({error}). "
    "Based on what is available:\n\n{partial}"
)


class VppAgent:
    """LangGraph agent for VPP electricity queries."""

    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        temperature: float = 0.3,
        base_url: str = "http://localhost:11434",
        *,
        use_react: bool = False,
        checkpointer: Any | None = None,
    ):
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
        )
        self.classifier = self.llm.with_structured_output(QueryClassification)
        self.use_react = use_react
        self.checkpointer = (
            checkpointer if checkpointer is not None else get_checkpointer()
        )
        if use_react:
            self.graph = build_react_graph(self.llm, checkpointer=self.checkpointer)
        else:
            self.graph = self._build_graph()

    def _config(self, thread_id: str | None) -> RunnableConfig:
        if not thread_id:
            return cast(RunnableConfig, {})
        return cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})

    def _parse_query_type(self, raw: str | QueryType | None) -> QueryType:
        """Normalize checkpoint-safe query_type strings back to enum."""
        if raw is None:
            return QueryType.UNKNOWN
        if isinstance(raw, QueryType):
            return raw
        try:
            return QueryType(raw)
        except ValueError:
            return QueryType.UNKNOWN

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
            state["query_type"] = result.query_type.value
        except Exception:
            state["query_type"] = QueryType.UNKNOWN.value

        return state

    def _route_query(self, state: AgentState) -> str:
        """Route to appropriate handler."""
        qt = self._parse_query_type(state.get("query_type"))
        return {
            QueryType.PRICE: "price_subgraph",
            QueryType.REGULATION: "regulation_subgraph",
            QueryType.BOTH: "get_both",
        }.get(qt, "general")

    def _get_prices(self, state: AgentState) -> AgentState:
        """Fetch electricity prices with retry and degraded fallback."""
        zone = state.get("bidding_zone", "10YDE-EL------O")
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                raw = get_electricity_prices.invoke({"bidding_zone": zone})
                state["prices"] = json.loads(raw)
                return state
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))

        state["error"] = f"price fetch failed after retries: {last_error}"
        state["degraded"] = True
        return state

    def _search_regs(self, state: AgentState) -> AgentState:
        """Search regulations via LangChain tool + retriever chain."""
        try:
            query = state["messages"][-1].content
            state["regulation_context"] = search_regulations.invoke(
                {"query": query, "k": 3}
            )
        except Exception as exc:
            state["error"] = f"regulation search failed: {exc}"
            state["degraded"] = True

        return state

    def _get_both(self, state: AgentState) -> AgentState:
        """Handle both price and regulation queries."""
        state = self._get_prices(state)
        state = self._search_regs(state)
        return state

    def _general(self, state: AgentState) -> AgentState:
        """General fallback."""
        return state

    def _build_answer_prompt(self, state: AgentState) -> str:
        query = state["messages"][-1].content
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
            return f"{query}\n\nContext:\n" + "\n\n".join(context_parts)
        return query

    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate final answer from retrieved context (tools already ran upstream)."""
        prompt = self._build_answer_prompt(state)
        system_msg = SystemMessage(content=ANSWER_SYSTEM_PROMPT)

        if (
            state.get("degraded")
            and not state.get("prices")
            and not state.get("regulation_context")
        ):
            state["final_answer"] = DEGRADED_ANSWER_TEMPLATE.format(
                error=state.get("error", "unknown error"),
                partial="No live data or regulation context was available.",
            )
            return state

        response = self.llm.invoke([system_msg, HumanMessage(content=prompt)])
        content = response.content
        answer = content if isinstance(content, str) else str(content)

        if state.get("degraded") and state.get("error"):
            answer = DEGRADED_ANSWER_TEMPLATE.format(
                error=state["error"],
                partial=answer,
            )

        state["final_answer"] = answer
        state["messages"] = state["messages"] + [AIMessage(content=answer)]
        return state

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph state machine with price/regulation subgraphs."""
        graph = StateGraph(AgentState)

        graph.add_node("classify", self._classify_query)
        graph.add_node("price_subgraph", build_price_subgraph(self._get_prices))
        graph.add_node(
            "regulation_subgraph",
            build_regulation_subgraph(self._search_regs),
        )
        graph.add_node("get_both", self._get_both)
        graph.add_node("general", self._general)
        graph.add_node("answer", self._generate_answer)

        graph.set_entry_point("classify")
        graph.add_conditional_edges(
            "classify",
            self._route_query,
            {
                "price_subgraph": "price_subgraph",
                "regulation_subgraph": "regulation_subgraph",
                "get_both": "get_both",
                "general": "general",
            },
        )

        graph.add_edge("price_subgraph", "answer")
        graph.add_edge("regulation_subgraph", "answer")
        graph.add_edge("get_both", "answer")
        graph.add_edge("general", "answer")
        graph.add_edge("answer", END)

        return graph.compile(checkpointer=self.checkpointer)

    def _initial_state(self, input: AgentInput, thread_id: str | None) -> AgentState:
        """Build initial state, appending to checkpoint history when continuing."""
        new_message = HumanMessage(content=input.query)
        messages: list[Any] = [new_message]

        if thread_id and not self.use_react:
            snapshot = self.graph.get_state(self._config(thread_id))
            if snapshot.values and snapshot.values.get("messages"):
                messages = list(snapshot.values["messages"]) + [new_message]

        return {
            "messages": messages,
            "query_type": None,
            "prices": None,
            "regulation_context": None,
            "final_answer": None,
            "error": None,
            "degraded": False,
            "bidding_zone": input.bidding_zone,
        }

    def _react_input(self, input: AgentInput, thread_id: str | None) -> dict[str, Any]:
        if thread_id:
            snapshot = self.graph.get_state(self._config(thread_id))
            prior = list(snapshot.values.get("messages", [])) if snapshot.values else []
            return {"messages": prior + [HumanMessage(content=input.query)]}
        return {"messages": [HumanMessage(content=input.query)]}

    def run(self, input: AgentInput, *, thread_id: str | None = None) -> AgentOutput:
        """Run the agent, optionally continuing a checkpointed thread."""
        config = self._config(thread_id)

        if self.use_react:
            result = self.graph.invoke(self._react_input(input, thread_id), config)
            messages = result.get("messages", [])
            answer = extract_react_answer(messages)
            return AgentOutput(
                answer=answer,
                sources=[],
                prices=None,
                error=None,
            )

        result = self.graph.invoke(self._initial_state(input, thread_id), config)
        return self._to_output(result)

    def stream_run(
        self, input: AgentInput, *, thread_id: str | None = None
    ) -> Iterator[str]:
        """Stream LLM tokens from graph execution via astream_events."""

        async def _token_stream() -> AsyncIterator[str]:
            config = self._config(thread_id)
            payload = (
                self._react_input(input, thread_id)
                if self.use_react
                else self._initial_state(input, thread_id)
            )
            async for event in self.graph.astream_events(payload, config, version="v2"):
                if event.get("event") != "on_chat_model_stream":
                    continue
                chunk = event.get("data", {}).get("chunk")
                if chunk is None:
                    continue
                content = getattr(chunk, "content", "")
                if content:
                    yield content if isinstance(content, str) else str(content)

        loop = asyncio.new_event_loop()
        generator = _token_stream()
        try:
            while True:
                yield loop.run_until_complete(generator.__anext__())
        except StopAsyncIteration:
            pass
        finally:
            loop.close()

    def stream_updates(
        self, input: AgentInput, *, thread_id: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """Stream per-node state updates (sync fallback)."""
        config = self._config(thread_id)
        payload = (
            self._react_input(input, thread_id)
            if self.use_react
            else self._initial_state(input, thread_id)
        )
        for update in self.graph.stream(payload, config, stream_mode="updates"):
            yield update

    def _to_output(self, result: dict[str, Any]) -> AgentOutput:
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


def get_default_agent(*, use_react: bool = False) -> VppAgent:
    """Get configured agent."""
    return VppAgent(use_react=use_react)
