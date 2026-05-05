"""ENTSO-E Transparency Platform API Client"""

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

from src.models.internals import DayAheadPrices, PricePoint

load_dotenv()


_RESOLUTION_RE = re.compile(r"PT(\d+)([MH])")


def _resolution_to_timedelta(resolution: str) -> timedelta:
    """Convert ISO 8601 duration like PT60M / PT15M / PT1H to a timedelta."""
    m = _RESOLUTION_RE.fullmatch(resolution.strip())
    if not m:
        raise ValueError(f"Unsupported resolution: {resolution}")
    value, unit = int(m.group(1)), m.group(2)
    return timedelta(minutes=value) if unit == "M" else timedelta(hours=value)


class EntsoeClient:
    BASE_URL = "https://web-api.tp.entsoe.eu/api"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ENTSOE_API_KEY")
        if not self.api_key:
            raise ValueError("ENTSOE_API_KEY required")

    def _make_request(self, params: dict) -> str:
        """Make request to ENTSO-E API"""
        params["securityToken"] = self.api_key
        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _format_interval(start: datetime, end: datetime) -> str:
        """Format a UTC interval as required by ENTSO-E (yyyy-MM-ddTHH:mmZ)."""
        return (
            f"{start.strftime('%Y-%m-%dT%H:%M')}Z/" f"{end.strftime('%Y-%m-%dT%H:%M')}Z"
        )

    def get_day_ahead_prices(
        self,
        bidding_zone: str,
        date: Optional[datetime] = None,
    ) -> DayAheadPrices:
        """
        Fetch day-ahead prices for a bidding zone.

        Common bidding zones:
        - 10YDE-EL------O: Germany
        - 10YAT-APG------L: Austria
        - 10YCH----------C: Switzerland
        - 10YFR-1-----R: France
        - 10YGB-2--------: Great Britain
        """
        if date is None:
            date = datetime.now(timezone.utc)
        elif date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        else:
            date = date.astimezone(timezone.utc)

        date_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)

        params = {
            "documentType": "A44",
            "in_Domain": bidding_zone,
            "out_Domain": bidding_zone,
            "periodStart": date_start.strftime("%Y%m%d%H%M"),
            "periodEnd": date_end.strftime("%Y%m%d%H%M"),
        }

        xml_text = self._make_request(params)
        return self._parse_prices_xml(xml_text, bidding_zone)

    @staticmethod
    def _findtext_local(elem, local_name: str) -> Optional[str]:
        """Namespace-agnostic findtext by local-name."""
        result = elem.xpath(".//*[local-name()=$n]", n=local_name)
        if not result:
            return None
        text = result[0].text
        return text.strip() if text else None

    def _parse_prices_xml(self, xml: str, area: str) -> DayAheadPrices:
        """Parse ENTSO-E XML response into structured data.

        Uses namespace-agnostic XPath because ENTSO-E publication documents
        ship with versioned IEC 62325 namespaces that change between API
        versions.
        """
        from lxml import etree

        root = etree.fromstring(xml.encode())

        prices: list[PricePoint] = []
        timeseries_nodes = root.xpath("//*[local-name()='TimeSeries']")
        for timeseries in timeseries_nodes:
            for period in timeseries.xpath(".//*[local-name()='Period']"):
                start_text = self._findtext_local(
                    period.xpath(".//*[local-name()='timeInterval']")[0],
                    "start",
                )
                resolution_text = self._findtext_local(period, "resolution")
                if not start_text or not resolution_text:
                    continue

                interval_start = datetime.fromisoformat(
                    start_text.replace("Z", "+00:00")
                )
                step = _resolution_to_timedelta(resolution_text)

                for point in period.xpath(".//*[local-name()='Point']"):
                    pos_text = self._findtext_local(point, "position")
                    amount_text = self._findtext_local(point, "price.amount")
                    if pos_text is None or amount_text is None:
                        continue
                    position = int(pos_text)
                    timestamp = interval_start + step * (position - 1)
                    prices.append(
                        PricePoint(
                            timestamp=timestamp,
                            price=float(amount_text),
                        )
                    )

        return DayAheadPrices(
            area=area,
            prices=prices,
            fetched_at=datetime.now(timezone.utc),
        )

    def get_current_price(self, bidding_zone: str) -> Optional[PricePoint]:
        """Get the most recent price point at or before now (UTC)."""
        data = self.get_day_ahead_prices(bidding_zone)
        if not data.prices:
            return None

        now = datetime.now(timezone.utc)
        for price in reversed(data.prices):
            if price.timestamp <= now:
                return price
        return data.prices[0]


def get_default_client() -> EntsoeClient:
    """Get configured ENTSO-E client"""
    return EntsoeClient()
