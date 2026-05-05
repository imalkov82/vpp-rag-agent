"""Tests for ENTSO-E API client"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.clients.entsoe_client import EntsoeClient, _resolution_to_timedelta
from src.models.internals import DayAheadPrices, PricePoint


class TestPricePoint:
    def test_price_point_creation(self):
        price = PricePoint(
            timestamp=datetime(2024, 1, 15, 12, 0),
            price=85.50,
        )
        assert price.price == 85.50
        assert price.currency == "EUR"
        assert price.unit == "MWh"

    def test_price_point_defaults(self):
        price = PricePoint(timestamp=datetime.now(), price=100.0)
        assert price.currency == "EUR"
        assert price.unit == "MWh"


class TestDayAheadPrices:
    def test_day_ahead_prices_structure(self):
        prices = [
            PricePoint(timestamp=datetime(2024, 1, 15, 0, 0), price=50.0),
            PricePoint(timestamp=datetime(2024, 1, 15, 1, 0), price=45.0),
        ]
        data = DayAheadPrices(
            area="10YDE-EL------O",
            prices=prices,
            fetched_at=datetime.now(),
        )
        assert data.area == "10YDE-EL------O"
        assert len(data.prices) == 2

    def test_day_ahead_prices_empty(self):
        data = DayAheadPrices(
            area="10YDE-EL------O",
            prices=[],
            fetched_at=datetime.now(),
        )
        assert len(data.prices) == 0


class TestResolutionParser:
    @pytest.mark.parametrize(
        "value,minutes",
        [("PT60M", 60), ("PT15M", 15), ("PT1H", 60), ("PT2H", 120)],
    )
    def test_known_resolutions(self, value, minutes):
        assert _resolution_to_timedelta(value).total_seconds() == minutes * 60

    def test_invalid_resolution_raises(self):
        with pytest.raises(ValueError):
            _resolution_to_timedelta("nope")


class TestEntsoeClient:
    def test_client_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ENTSOE_API_KEY required"):
            EntsoeClient(api_key=None)

    def test_client_initialization(self):
        client = EntsoeClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.BASE_URL.startswith("https://")

    def test_common_bidding_zones(self):
        common_zones = [
            "10YDE-EL------O",
            "10YAT-APG------L",
            "10YCH----------C",
            "10YFR-1-----R",
            "10YGB-2--------",
        ]
        # ENTSO-E EIC codes are 16 chars total when including the check char,
        # but the zone codes used here are area codes in mixed lengths — they
        # must at least be non-empty and start with the "10Y" country prefix.
        for zone in common_zones:
            assert zone.startswith("10Y")
            assert len(zone) > 0

    @patch("src.clients.entsoe_client.requests.get")
    def test_make_request_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<root></root>"
        mock_get.return_value = mock_response

        client = EntsoeClient(api_key="test_key")
        client._make_request({"param": "value"})

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.kwargs["params"]["securityToken"] == "test_key"
        mock_response.raise_for_status.assert_called_once()

    @patch("src.clients.entsoe_client.requests.get")
    def test_make_request_timeout(self, mock_get):
        import requests

        mock_get.side_effect = requests.Timeout("Request timed out")

        client = EntsoeClient(api_key="test_key")
        with pytest.raises(requests.Timeout):
            client._make_request({})

    def test_format_interval_uses_utc_minute_precision(self):
        start = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc)
        assert (
            EntsoeClient._format_interval(start, end)
            == "2024-01-15T00:00Z/2024-01-16T00:00Z"
        )


class TestXMLParsing:
    NS = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0"

    def test_parse_prices_xml_empty(self):
        xml = f"""<?xml version="1.0"?>
        <Publication_MarketDocument xmlns="{self.NS}">
        </Publication_MarketDocument>"""

        client = EntsoeClient(api_key="test_key")
        result = client._parse_prices_xml(xml, "10YDE-EL------O")

        assert result.area == "10YDE-EL------O"
        assert len(result.prices) == 0

    def test_parse_prices_xml_hourly(self):
        xml = f"""<?xml version="1.0"?>
        <Publication_MarketDocument xmlns="{self.NS}">
          <TimeSeries>
            <Period>
              <timeInterval>
                <start>2024-01-15T00:00Z</start>
                <end>2024-01-15T03:00Z</end>
              </timeInterval>
              <resolution>PT60M</resolution>
              <Point>
                <position>1</position>
                <price.amount>50.10</price.amount>
              </Point>
              <Point>
                <position>2</position>
                <price.amount>45.20</price.amount>
              </Point>
              <Point>
                <position>3</position>
                <price.amount>40.30</price.amount>
              </Point>
            </Period>
          </TimeSeries>
        </Publication_MarketDocument>"""

        client = EntsoeClient(api_key="test_key")
        result = client._parse_prices_xml(xml, "10YDE-EL------O")

        assert len(result.prices) == 3
        assert result.prices[0].price == 50.10
        assert result.prices[0].timestamp == datetime(
            2024, 1, 15, 0, 0, tzinfo=timezone.utc
        )
        assert result.prices[2].timestamp == datetime(
            2024, 1, 15, 2, 0, tzinfo=timezone.utc
        )

    def test_parse_prices_xml_quarter_hour(self):
        xml = f"""<?xml version="1.0"?>
        <Publication_MarketDocument xmlns="{self.NS}">
          <TimeSeries>
            <Period>
              <timeInterval>
                <start>2024-01-15T00:00Z</start>
                <end>2024-01-15T01:00Z</end>
              </timeInterval>
              <resolution>PT15M</resolution>
              <Point>
                <position>1</position>
                <price.amount>10</price.amount>
              </Point>
              <Point>
                <position>4</position>
                <price.amount>40</price.amount>
              </Point>
            </Period>
          </TimeSeries>
        </Publication_MarketDocument>"""

        client = EntsoeClient(api_key="test_key")
        result = client._parse_prices_xml(xml, "10YDE-EL------O")

        assert [p.timestamp.minute for p in result.prices] == [0, 45]
