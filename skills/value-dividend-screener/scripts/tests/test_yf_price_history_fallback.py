"""WPP-20260906-007 — yfinance price-history leg for get_historical_prices.

Pins: FMP hit → yfinance never consulted; every FMP rung missed → the
yfinance rows (most-recent-first, ≤days, raw close, NaN bars dropped) reach
the RSI consumer in the FMP row shape; yfinance empty / raising → the old
empty contract (never a synthetic series); the FMP miss still feeds the
endpoint breaker.
"""
import os
import sys
import types
from datetime import datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import screen_dividend_stocks as mod  # noqa: E402

EMPTY = []


def _fake_yfinance(monkeypatch, frame=None, raise_exc=None):
    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            if raise_exc is not None:
                raise raise_exc
            return frame

    fake = types.ModuleType("yfinance")
    fake.Ticker = _Ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    return fake


def _frame(n=45, nan_at=None):
    end = datetime(2026, 9, 4)
    idx = pd.DatetimeIndex([end - timedelta(days=n - 1 - i) for i in range(n)])
    closes = [100.0 + i for i in range(n)]
    if nan_at is not None:
        closes[nan_at] = float("nan")
    return pd.DataFrame({"Close": closes, "Adj Close": [c * 0.9 for c in closes]}, index=idx)


def test_helper_rows_are_most_recent_first_raw_close_capped_and_nan_free(monkeypatch):
    _fake_yfinance(monkeypatch, frame=_frame(45, nan_at=44))
    rows = mod._yf_price_history("ALV.DE", days=30)
    assert len(rows) == 30
    assert rows[0]["date"] == "2026-09-03"  # the 09-04 NaN bar is dropped, not padded
    assert rows[0]["close"] == 143.0  # raw Close, not Adj Close
    assert rows[-1]["date"] < rows[0]["date"]
    assert {r["data_source"] for r in rows} == {"yfinance"}
    assert [r["date"] for r in rows] == sorted((r["date"] for r in rows), reverse=True)


@pytest.mark.parametrize("frame,exc", [(pd.DataFrame(), None), (None, None), (None, RuntimeError("boom"))])
def test_helper_fails_closed(monkeypatch, frame, exc):
    _fake_yfinance(monkeypatch, frame=frame, raise_exc=exc)
    assert mod._yf_price_history("ALV.DE", days=30) == []


def test_helper_without_yfinance_is_empty(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    assert mod._yf_price_history("ALV.DE", days=30) == []


def _client(monkeypatch, fmp_payload):
    client = mod.FMPClient(api_key="dummy")
    monkeypatch.setattr(client, "_request", lambda *a, **k: fmp_payload)
    return client


def test_fmp_hit_never_consults_yfinance(monkeypatch):
    bars = [{"date": "2026-09-04", "close": 10.0}, {"date": "2026-09-03", "close": 9.0}]
    client = _client(monkeypatch, bars)
    monkeypatch.setattr(mod, "_yf_price_history", lambda *a, **k: pytest.fail("yfinance consulted on an FMP hit"))
    assert client.get_historical_prices("PEP", days=30) == bars


def test_fmp_miss_falls_through_to_yfinance_and_records_the_miss(monkeypatch):
    client = _client(monkeypatch, [])
    yf_rows = [{"date": "2026-09-04", "close": 10.0, "data_source": "yfinance"}]
    seen = {}

    def fake_hist(symbol, days):
        seen["args"] = (symbol, days)
        return yf_rows

    monkeypatch.setattr(mod, "_yf_price_history", fake_hist)
    assert client.get_historical_prices("ALV.DE", days=30) == yf_rows
    assert seen["args"] == ("ALV.DE", 30)
    assert client._endpoint_failures[mod._FMP_HIST_ENDPOINTS[0]] == 1


def test_fmp_miss_and_yfinance_empty_keeps_the_empty_contract(monkeypatch):
    client = _client(monkeypatch, [])
    monkeypatch.setattr(mod, "_yf_price_history", lambda *a, **k: [])
    assert client.get_historical_prices("ALV.DE", days=30) == EMPTY
