"""Pydantic / dataclass models used across the package."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, TypedDict

from pydantic import BaseModel


class QueryType(str, Enum):
    """Types of queries the agent can handle"""

    PRICE = "price"
    REGULATION = "regulation"
    BOTH = "both"
    UNKNOWN = "unknown"


class QueryClassification(BaseModel):
    """LLM-structured query intent for routing."""

    query_type: QueryType
    reasoning: str = ""


class AgentState(TypedDict):
    """State of the agent"""

    messages: List
    query_type: Optional[str]
    prices: Optional[dict]
    regulation_context: Optional[str]
    final_answer: Optional[str]
    error: Optional[str]
    degraded: bool
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


class DocumentChunk(BaseModel):
    """A chunk of a document with metadata"""

    content: str
    source: str
    page: int
    chunk_id: str


class PricePoint(BaseModel):
    """Single electricity price point"""

    timestamp: datetime
    price: float
    currency: str = "EUR"
    unit: str = "MWh"


class DayAheadPrices(BaseModel):
    """Day-ahead electricity prices for a bidding zone"""

    area: str
    prices: list[PricePoint]
    fetched_at: datetime


@dataclass
class HealthResult:
    name: str
    ok: bool
    detail: str
