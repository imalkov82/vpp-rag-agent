"""End-to-end flow tests with HTTP mocks."""

import json
from unittest.mock import MagicMock, patch

from src.clients.entsoe_client import EntsoeClient

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <TimeSeries>
    <Period>
      <timeInterval><start>2024-01-15T00:00Z</start><end>2024-01-16T00:00Z</end></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>42.5</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>
"""


class TestEntsoeE2E:
    @patch("src.clients.entsoe_client.requests.get")
    def test_day_ahead_prices_parsed_from_api(self, mock_get):
        response = MagicMock()
        response.text = SAMPLE_XML
        response.raise_for_status = MagicMock()
        mock_get.return_value = response

        client = EntsoeClient(api_key="test-key")
        data = client.get_day_ahead_prices("10YDE-EL------O")

        assert mock_get.called
        assert data.area == "10YDE-EL------O"
        assert len(data.prices) == 1
        assert data.prices[0].price == 42.5

    @patch("src.service.tools.get_default_client")
    def test_price_tool_json_roundtrip(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        point = MagicMock()
        point.timestamp.isoformat.return_value = "2024-01-15T00:00:00"
        point.price = 42.5
        data = MagicMock()
        data.area = "10YDE-EL------O"
        data.prices = [point]
        client.get_day_ahead_prices.return_value = data

        from src.service.tools import get_electricity_prices

        raw = get_electricity_prices.invoke({"bidding_zone": "10YDE-EL------O"})
        payload = json.loads(raw)
        assert payload["area"] == "10YDE-EL------O"
        assert payload["prices"][0]["price"] == 42.5
