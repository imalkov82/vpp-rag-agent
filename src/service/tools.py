"""LangChain tools for electricity prices and regulation search."""

import json

from langchain_core.tools import tool

from src.clients.entsoe_client import get_default_client
from src.service.rag import get_default_rag

VPP_TOOLS = []  # populated after tool definitions


@tool
def get_electricity_prices(bidding_zone: str = "10YDE-EL------O") -> str:
    """Fetch day-ahead electricity prices for an ENTSO-E bidding zone.

    Args:
        bidding_zone: ENTSO-E bidding zone EIC code (e.g. 10YDE-EL------O for Germany).
    """
    client = get_default_client()
    data = client.get_day_ahead_prices(bidding_zone)
    return json.dumps(
        {
            "area": data.area,
            "prices": [
                {"timestamp": p.timestamp.isoformat(), "price": p.price}
                for p in data.prices
            ],
        }
    )


@tool
def search_regulations(query: str, k: int = 4) -> str:
    """Search ENTSO-E grid regulation PDFs for excerpts relevant to the query.

    Args:
        query: Natural-language question about grid codes, balancing, reserves, etc.
        k: Number of document chunks to retrieve.
    """
    rag = get_default_rag()
    return rag.get_context_via_retriever(query, k=k)


VPP_TOOLS = [get_electricity_prices, search_regulations]
