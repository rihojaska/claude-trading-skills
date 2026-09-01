"""FMP /stable-only helpers: live quote, dividend yield, HV price history.

These used to try `/stable/...` and then a second `/api/v3/...` rung. fmp_compat
rewrites a v3 URL straight back to the equivalent /stable endpoint, so that rung
was never a distinct upstream — only a second rate-limited call on the SAME one
(WPP-20260831-004). The v3 entries are deleted.

This file pins the DIRECT-requests fallback used by a standalone .skill install
(fmp_compat not importable). The production fmp_compat path is pinned at the
transport seam in test_fmp_stable_only.py.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import black_scholes  # noqa: E402
from black_scholes import (  # noqa: E402
    fetch_historical_prices_for_hv,
    get_current_stock_price,
    get_dividend_yield,
)


@pytest.fixture(autouse=True)
def _direct_stable_transport(monkeypatch):
    """Force the no-fmp_compat fallback so `black_scholes.requests` is the seam."""
    monkeypatch.setattr(black_scholes, "fmp_get", None, raising=False)


def _resp(status_code, json_payload):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_payload
    return resp


class TestCurrentStockPrice:
    @patch("black_scholes.requests")
    def test_uses_stable_quote(self, mock_requests):
        mock_requests.get.return_value = _resp(200, [{"symbol": "AAPL", "price": 150.0}])
        assert get_current_stock_price("AAPL", "key") == 150.0
        call = mock_requests.get.call_args
        assert call[0][0].endswith("/stable/quote")
        assert call[1]["params"] == {"symbol": "AAPL"}

    @patch("black_scholes.requests")
    def test_no_v3_rung_on_failure(self, mock_requests):
        mock_requests.get.return_value = _resp(403, {})
        assert get_current_stock_price("AAPL", "key") is None
        assert mock_requests.get.call_count == 1
        urls = [c[0][0] for c in mock_requests.get.call_args_list]
        assert not any("/api/v3/" in u for u in urls)


class TestDividendYield:
    @patch("black_scholes.requests")
    def test_reads_stable_lastDividend_field(self, mock_requests):
        # /stable/profile uses lastDividend (no lastDiv). Yield = 2.0 / 100.
        mock_requests.get.return_value = _resp(
            200, [{"symbol": "AAPL", "lastDividend": 2.0, "price": 100.0}]
        )
        assert get_dividend_yield("AAPL", "key") == 0.02
        call = mock_requests.get.call_args
        assert call[0][0].endswith("/stable/profile")
        assert call[1]["params"] == {"symbol": "AAPL"}

    @patch("black_scholes.requests")
    def test_legacy_lastDiv_field_still_read(self, mock_requests):
        # Older payloads (and cached fixtures) still carry lastDiv.
        mock_requests.get.return_value = _resp(
            200, [{"symbol": "AAPL", "lastDiv": 4.0, "price": 100.0}]
        )
        assert get_dividend_yield("AAPL", "key") == 0.04

    @patch("black_scholes.requests")
    def test_zero_when_no_data(self, mock_requests):
        mock_requests.get.return_value = _resp(200, [])
        assert get_dividend_yield("AAPL", "key") == 0

    @patch("black_scholes.requests")
    def test_no_v3_rung_on_failure(self, mock_requests):
        mock_requests.get.return_value = _resp(403, {})
        assert get_dividend_yield("AAPL", "key") == 0
        assert mock_requests.get.call_count == 1


class TestHistoricalPricesForHV:
    @patch("black_scholes.requests")
    def test_stable_flat_list_is_parsed(self, mock_requests):
        """The /stable EOD endpoint returns a FLAT list.

        The pre-migration dict-only parsing dropped it silently and fell
        through to the v3 rung; the direct path now handles both shapes.
        """
        mock_requests.get.return_value = _resp(
            200,
            [
                {"date": "2026-05-19", "close": 101.0},
                {"date": "2026-05-18", "close": 100.0},
            ],
        )
        # Returned oldest-first (reversed from FMP's most-recent-first order).
        assert fetch_historical_prices_for_hv("AAPL", "key", days=90) == [100.0, 101.0]
        assert mock_requests.get.call_args[0][0].endswith("/stable/historical-price-eod/full")

    @patch("black_scholes.requests")
    def test_legacy_dict_shape_is_parsed(self, mock_requests):
        mock_requests.get.return_value = _resp(
            200, {"symbol": "AAPL", "historical": [{"date": "2026-05-19", "adjClose": 42.0}]}
        )
        assert fetch_historical_prices_for_hv("AAPL", "key", days=90) == [42.0]

    @patch("black_scholes.requests")
    def test_no_v3_rung_on_failure(self, mock_requests):
        mock_requests.get.return_value = _resp(403, {})
        assert fetch_historical_prices_for_hv("AAPL", "key", days=90) is None
        assert mock_requests.get.call_count == 1
