"""Tests for run_20pct_study.FMPClient.

FMPClient used to try v3 endpoints as a fallback when the /stable call
failed. FMP retired v3 for keys issued after 2025-08-31, and a v3 URL
requested through fmp_compat is rewritten straight back to the identical
/stable endpoint, so that fallback was never a distinct upstream
(WPP-20260831-004 / WPP-20260901-016) — it has been deleted. These tests pin
`fmp_get = None` to exercise the standalone (no-fmp_compat) transport
directly; the fmp_get path is covered at the real transport seam in
scripts/tests/test_v3_rungs_gone_0901_016.py.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_20pct_study.py"
spec = importlib.util.spec_from_file_location("run_20pct_study", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


@pytest.fixture(autouse=True)
def _direct_stable_transport(monkeypatch):
    monkeypatch.setattr(mod, "fmp_get", None, raising=False)


def response(status, payload):
    res = MagicMock()
    res.status_code = status
    res.text = "body"
    res.json.return_value = payload
    return res


def test_fmp_universe_uses_stable_company_screener_first(monkeypatch):
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    session = MagicMock()
    session.get.return_value = response(
        200,
        [
            {"symbol": "AAPL", "exchangeShortName": "NASDAQ", "price": 200},
            {"symbol": "SPY", "exchangeShortName": "NYSEARCA", "price": 600, "isEtf": True},
        ],
    )
    client = mod.FMPClient(api_key="test", max_api_calls=10)
    client.session = session

    symbols = client.get_stock_list(limit=20)

    assert symbols == ["AAPL"]
    assert session.get.call_args_list[0][0][0].endswith("/stable/company-screener")


def test_fmp_universe_stable_failure_returns_empty_no_second_request(monkeypatch):
    """A failed stable call makes no second (v3) request."""
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    session = MagicMock()
    session.get.return_value = response(403, {})
    client = mod.FMPClient(api_key="test", max_api_calls=10)
    client.session = session

    symbols = client.get_stock_list(limit=10)

    assert symbols == []
    assert session.get.call_count == 1
    url = session.get.call_args_list[0][0][0]
    assert url.endswith("/stable/company-screener")
    assert "api/v3" not in url


def test_fmp_historical_accepts_stable_flat_list(monkeypatch):
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    session = MagicMock()
    session.get.return_value = response(
        200,
        [
            {
                "symbol": "AAPL",
                "date": "2026-01-02",
                "open": 11,
                "high": 12,
                "low": 10,
                "close": 11.5,
                "volume": 2000,
            },
            {
                "symbol": "AAPL",
                "date": "2026-01-01",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 1000,
            },
        ],
    )
    client = mod.FMPClient(api_key="test", max_api_calls=10)
    client.session = session

    bars = client.get_historical_prices("AAPL", days=2)

    assert [bar["date"] for bar in bars] == ["2026-01-01", "2026-01-02"]
    assert session.get.call_args_list[0][0][0].endswith("/stable/historical-price-eod/full")


def test_fmp_historical_empty_stable_returns_empty_no_second_request(monkeypatch):
    """An empty stable response makes no second (v3) request."""
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    session = MagicMock()
    session.get.return_value = response(200, [])
    client = mod.FMPClient(api_key="test", max_api_calls=10)
    client.session = session

    bars = client.get_historical_prices("AAPL", days=2)

    assert bars == []
    assert session.get.call_count == 1
    url = session.get.call_args_list[0][0][0]
    assert url.endswith("/stable/historical-price-eod/full")
    assert "api/v3" not in url


def test_fmp_requires_key_for_live_path(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)

    with pytest.raises(ValueError, match="FMP API key required"):
        mod.FMPClient()


def test_fmp_api_budget_exhaustion():
    client = mod.FMPClient(api_key="test", max_api_calls=0)

    with pytest.raises(mod.ApiCallBudgetExceeded):
        client.get_stock_list(limit=1)
