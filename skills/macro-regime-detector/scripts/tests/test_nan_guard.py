"""WPP-20260827-009 — non-finite market data must never read as a valid observation.

Macro half of the guard (market-top's sibling, per-file implementation): the
regime score is built from cross-asset ratio series, and a NaN close silently
propagates through rolling ratio math into a wrong-but-plausible regime call.
All tests offline — the yfinance surface is a stub module in `sys.modules`,
the FMP transport a patched `fmp_get`.
"""

import datetime as _dt
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fmp_client as fc  # noqa: E402

NAN = float("nan")


class _Cols:
    pass  # no `levels` attribute -> droplevel branch skipped


class _FakeDF:
    def __init__(self, rows):
        self._rows = list(rows)
        self.columns = _Cols()

    @property
    def empty(self):
        return not self._rows

    def iterrows(self):
        return iter(self._rows)


class _FakeYF:
    def __init__(self, df):
        self._df = df

    def download(self, symbol, **kwargs):
        return self._df


def _bar_row(day, o=100.0, h=101.0, low=99.0, c=100.5, v=1_000_000):
    return (
        _dt.datetime(2026, 8, day),
        {"Open": o, "High": h, "Low": low, "Close": c, "Volume": v},
    )


def _with_yf(rows):
    return patch.dict(sys.modules, {"yfinance": _FakeYF(_FakeDF(rows))})


def test_finite_predicates_reject_never_raise():
    assert fc._finite(1.5)
    assert not fc._finite(NAN)
    assert not fc._finite(True)
    assert not fc._finite("3.5")
    assert not fc._finite(10**400)  # OverflowError class — must not raise
    assert fc._finite_positive(0.01)
    assert not fc._finite_positive(0)


def test_yf_history_all_finite_passes():
    with _with_yf([_bar_row(20), _bar_row(21, c=102.0)]):
        out = fc._yf_history("SPY", 30)
    assert out is not None
    assert out["data_source"] == "yfinance"
    assert [b["date"] for b in out["historical"]] == ["2026-08-21", "2026-08-20"]


def test_yf_history_nan_close_rejects_whole_fetch(capsys):
    with _with_yf([_bar_row(20, c=NAN), _bar_row(21)]):
        assert fc._yf_history("SPY", 30) is None
    assert "rejecting the whole fetch" in capsys.readouterr().err


def test_yf_history_nan_in_later_bar_rejects_whole_fetch():
    with _with_yf([_bar_row(20), _bar_row(21, low=NAN)]):
        assert fc._yf_history("SPY", 30) is None


def test_yf_history_nan_volume_is_typed_rejection_not_crash():
    with _with_yf([_bar_row(20, v=NAN)]):
        assert fc._yf_history("SPY", 30) is None


def test_yf_history_zero_volume_bar_is_accepted():
    with _with_yf([_bar_row(20, v=0)]):
        out = fc._yf_history("SPY", 30)
    assert out is not None and out["historical"][0]["volume"] == 0


def _payload(bars):
    return {"symbol": "SPY", "historical": bars}


def _fmp_bar(c=100.0, o=99.0, h=101.0, low=98.0, v=1000):
    return {"date": "2026-08-20", "open": o, "high": h, "low": low, "close": c, "volume": v}


def test_history_values_ok_all_or_none():
    assert fc._history_values_ok(_payload([_fmp_bar()]))
    assert not fc._history_values_ok(_payload([_fmp_bar(c=NAN)]))
    assert not fc._history_values_ok(_payload([_fmp_bar(), _fmp_bar(v=-1)]))
    assert not fc._history_values_ok(_payload([_fmp_bar(low=0)]))
    assert not fc._history_values_ok(None)
    assert not fc._history_values_ok(_payload([]))


def test_fmp_nan_close_falls_back_to_yf_through_real_loop():
    client = fc.FMPClient(api_key="test_key")  # pragma: allowlist secret
    fmp_rows = [
        {"symbol": "SPY", "date": "2026-08-21", "open": 1.0, "high": 2.0, "low": 0.5, "close": NAN, "volume": 10}
    ]

    def fake_fmp_get(url, params=None, **_kw):
        return fmp_rows if "historical-price-eod" in url else None

    transport = patch.dict(type(client)._rate_limited_get.__globals__, {"fmp_get": fake_fmp_get})
    with transport, _with_yf([_bar_row(20)]):
        data = client.get_historical_prices("SPY", days=30)
    assert data is not None
    assert client.data_sources["historical:SPY"] == "yfinance"  # NaN FMP payload refused


def test_fmp_nan_with_no_yf_yields_none_not_repair():
    client = fc.FMPClient(api_key="test_key")  # pragma: allowlist secret
    fmp_rows = [
        {"symbol": "SPY", "date": "2026-08-21", "open": 1.0, "high": 2.0, "low": 0.5, "close": NAN, "volume": 10}
    ]

    def fake_fmp_get(url, params=None, **_kw):
        return fmp_rows if "historical-price-eod" in url else None

    transport = patch.dict(type(client)._rate_limited_get.__globals__, {"fmp_get": fake_fmp_get})
    with transport, patch.object(fc, "_yf_history", lambda *a, **k: None):
        assert client.get_historical_prices("SPY", days=30) is None


def test_fmp_batch_stocklist_nan_falls_back_to_yf(capsys):
    # The historicalStockList branch has its own return path — the boundary
    # must hold there too, and the rejection must be LOUD.
    client = fc.FMPClient(api_key="test_key")  # pragma: allowlist secret
    payload = {
        "historicalStockList": [
            {
                "symbol": "SPY",
                "historical": [
                    {"date": "2026-08-21", "open": 1.0, "high": 2.0, "low": 0.5, "close": NAN, "volume": 10}
                ],
            }
        ]
    }

    def fake_fmp_get(url, params=None, **_kw):
        return payload if "historical-price-eod" in url else None

    transport = patch.dict(type(client)._rate_limited_get.__globals__, {"fmp_get": fake_fmp_get})
    with transport, _with_yf([_bar_row(20)]):
        data = client.get_historical_prices("SPY", days=30)
    assert data is not None
    assert client.data_sources["historical:SPY"] == "yfinance"
    assert "failed the value boundary" in capsys.readouterr().err


def test_rejection_is_loud_and_caches_nothing(capsys):
    client = fc.FMPClient(api_key="test_key")  # pragma: allowlist secret
    fmp_rows = [
        {"symbol": "SPY", "date": "2026-08-21", "open": 1.0, "high": 2.0, "low": 0.5, "close": NAN, "volume": 10}
    ]

    def fake_fmp_get(url, params=None, **_kw):
        return fmp_rows if "historical-price-eod" in url else None

    transport = patch.dict(type(client)._rate_limited_get.__globals__, {"fmp_get": fake_fmp_get})
    with transport, patch.object(fc, "_yf_history", lambda *a, **k: None):
        assert client.get_historical_prices("SPY", days=30) is None
    assert "failed the value boundary" in capsys.readouterr().err
    assert client.cache == {}
    assert "historical:SPY" not in client.data_sources


def test_history_values_ok_present_adjclose_must_be_real():
    # codex gate r1: macro's monthly downsampling prefers adjClose — a present
    # NaN/None/0 adjClose must reject even when close is finite.
    good = _fmp_bar()
    good["adjClose"] = 100.0
    assert fc._history_values_ok(_payload([good]))
    assert fc._history_values_ok(_payload([_fmp_bar()]))  # absent adjClose acceptable
    for bad_val in (NAN, None, 0):
        bad = _fmp_bar()
        bad["adjClose"] = bad_val
        assert not fc._history_values_ok(_payload([bad])), bad_val
