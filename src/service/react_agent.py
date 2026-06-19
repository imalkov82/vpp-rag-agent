"""ReAct agent built on LangGraph prebuilt tool loop."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from src.service.tools import VPP_TOOLS

REACT_SYSTEM_PROMPT = (
    "You are an expert on European electricity markets and ENTSO-E grid "
    "regulations. Use tools to fetch live prices or search regulation PDFs "
    "when needed. Cite sources and be concise."
)


def build_react_graph(
    llm: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Create a ReAct agent graph with VPP tools."""
    return create_react_agent(
        llm,
        VPP_TOOLS,
        prompt=SystemMessage(content=REACT_SYSTEM_PROMPT),
        checkpointer=checkpointer,
    )


def extract_react_answer(messages: list[Any]) -> str:
    """Pull the final assistant message from a ReAct run."""
    for message in reversed(messages):
        role = getattr(message, "type", None) or getattr(message, "role", None)
        if role in ("ai", "assistant"):
            content = getattr(message, "content", "")
            if content:
                return content if isinstance(content, str) else str(content)
    return "No answer generated"
