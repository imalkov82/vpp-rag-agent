"""Domain exceptions — mirrors the vppems-cmd app/utils/exceptions.py pattern."""


class VppRagException(Exception):
    """Base exception for the VPP RAG agent."""


class EntsoeError(VppRagException):
    """Errors talking to the ENTSO-E Transparency Platform API."""


class RagIndexError(VppRagException):
    """Errors building or querying the RAG vector store."""


class AgentError(VppRagException):
    """Errors raised by the LangGraph agent."""
