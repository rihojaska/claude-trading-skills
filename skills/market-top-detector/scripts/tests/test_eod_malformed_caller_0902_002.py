"""WPP-20260902-002 caller-level pin — through the ROOT `fmp_compat.fmp_get`
(which market-top's client imports), a 200-OK EOD payload with one malformed
element is refused whole (None) and the client falls through to its yfinance
rung; with no yfinance either, the caller gets None — never a silently
shortened series.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import fmp_compat  # noqa: E402
import fmp_client  # noqa: E402

MALFORMED = [{"symbol": "SPY", "date": "2026-09-04", "close": 1.0}, "junk", {"symbol": "SPY", "close": 2.0}]
YF_BARS = {"symbol": "SPY", "historical": [{"date": "2026-09-04", "open": 1, "high": 1, "low": 1, "close": 1.0, "volume": 1}]}


class _Resp:
    status_code = 200
    ok = True
    text = ""
    headers = {}
    url = "https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=SPY"

    def json(self):
        return MALFORMED


def _client():
    with patch.dict(os.environ, {"FMP_API_KEY": "k", "FMP_FALLBACK_API_KEY": ""}):  # pragma: allowlist secret
        return fmp_client.FMPClient()


def test_client_routes_through_the_root_shim():
    assert fmp_client.fmp_get is fmp_compat.fmp_get


def test_malformed_fmp_payload_falls_through_to_yfinance(capsys):
    c = _client()
    with (
        patch.object(fmp_compat, "_original_get", return_value=_Resp()),
        patch.dict(os.environ, {"FMP_API_KEY": "k"}),  # pragma: allowlist secret
        patch.dict(type(c)._rate_limited_get.__globals__, {"_yf_history": lambda *a, **k: dict(YF_BARS)}),
    ):
        data = c.get_historical_prices("SPY", days=5)
    assert data and len(data["historical"]) == 1
    assert c.data_sources["historical:SPY"] == "yfinance"
    assert "malformed at index 1" in capsys.readouterr().err


def test_malformed_fmp_payload_without_yfinance_is_none(capsys):
    c = _client()
    with (
        patch.object(fmp_compat, "_original_get", return_value=_Resp()),
        patch.dict(os.environ, {"FMP_API_KEY": "k"}),  # pragma: allowlist secret
        patch.dict(type(c)._rate_limited_get.__globals__, {"_yf_history": lambda *a, **k: None}),
    ):
        assert c.get_historical_prices("SPY", days=5) is None
    assert "refusing the whole series" in capsys.readouterr().err
