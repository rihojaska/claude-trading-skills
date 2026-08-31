"""WPP-20260827-009 — non-finite market data must never read as a valid observation.

Sibling of ftd-detector's WPP-20260818-009 guard, re-implemented per-file (the
clients share a lineage, not code): `_yf_history` passed NaN closes through
`float(row["Close"])`, and `_yf_quote` derived yearHigh via `max()` over the
window — where a NaN in the FIRST bar seeds `max` and yields NaN while a later
NaN is silently skipped, so yearHigh was quietly wrong rather than visibly
broken (consumed at market_top_detector.py's 52-week positioning).

Every test here is offline: the yfinance surface is a stub module injected into
`sys.modules`, and the FMP transport is `fmp_get` patched in the module
namespace. A test that could pass by fetching live data proves nothing (the
recurring network-false-green trap).
"""

import datetime as _dt
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fmp_client as fc  # noqa: E402

NAN = float("nan")


# ---------------------------------------------------------------------------
# yfinance fake: `_yf_history` calls `yf.download(...)`, drops a MultiIndex
# level if `df.columns` has `levels`, then iterates `df.iterrows()` with
# `row["Open"]`-style access and an index carrying `strftime`. Reproduce that
# exact surface so the real parse loop runs.
# ---------------------------------------------------------------------------


class _Cols:
    pass  # no `levels` attribute -> droplevel branch skipped


class _FakeDF:
    def __init__(self, rows):
        # rows: list of (datetime, dict) pairs
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
        self.calls = []

    def download(self, symbol, **kwargs):
        self.calls.append(symbol)
        return self._df


def _bar_row(day, o=100.0, h=101.0, low=99.0, c=100.5, v=1_000_000):
    return (
        _dt.datetime(2026, 8, day),
        {"Open": o, "High": h, "Low": low, "Close": c, "Volume": v},
    )


def _with_yf(rows):
    return patch.dict(sys.modules, {"yfinance": _FakeYF(_FakeDF(rows))})


# ---------------------------------------------------------------------------
# _finite / _finite_positive: predicates must reject, never raise
# ---------------------------------------------------------------------------


def test_finite_rejects_nan_bool_str_and_overflow_int():
    assert fc._finite(1.5)
    assert fc._finite(0)
    assert not fc._finite(NAN)
    assert not fc._finite(float("inf"))
    assert not fc._finite(True)  # bool is not a price
    assert not fc._finite("3.5")
    assert not fc._finite(None)
    assert not fc._finite(10**400)  # OverflowError class — must not raise


def test_finite_positive_rejects_zero_and_negative():
    assert fc._finite_positive(0.01)
    assert not fc._finite_positive(0)
    assert not fc._finite_positive(-1.0)
    assert not fc._finite_positive(NAN)


# ---------------------------------------------------------------------------
# _yf_history: all-or-none over the real parse loop
# ---------------------------------------------------------------------------


def test_yf_history_all_finite_passes_and_is_most_recent_first():
    with _with_yf([_bar_row(20), _bar_row(21, c=102.0)]):
        out = fc._yf_history("^GSPC", 30)
    assert out is not None
    assert out["data_source"] == "yfinance"
    assert [b["date"] for b in out["historical"]] == ["2026-08-21", "2026-08-20"]
    assert out["historical"][0]["close"] == 102.0


def test_yf_history_nan_close_in_first_bar_rejects_whole_fetch(capsys):
    with _with_yf([_bar_row(20, c=NAN), _bar_row(21)]):
        assert fc._yf_history("^GSPC", 30) is None
    assert "rejecting the whole fetch" in capsys.readouterr().err


def test_yf_history_nan_in_later_bar_rejects_whole_fetch():
    # The 0827-009 shape: a LATER NaN was silently skipped by max() — the
    # boundary must be order-independent.
    with _with_yf([_bar_row(20), _bar_row(21, h=NAN)]):
        assert fc._yf_history("^GSPC", 30) is None


def test_yf_history_nan_volume_is_typed_rejection_not_crash():
    with _with_yf([_bar_row(20, v=NAN)]):
        assert fc._yf_history("^GSPC", 30) is None


def test_yf_history_zero_volume_index_bar_is_accepted():
    # ^VIX-class indexes legitimately report volume 0 — must NOT be rejected.
    with _with_yf([_bar_row(20, v=0)]):
        out = fc._yf_history("^VIX", 30)
    assert out is not None and out["historical"][0]["volume"] == 0


def test_yf_history_nonpositive_price_rejects():
    with _with_yf([_bar_row(20, low=0.0)]):
        assert fc._yf_history("^GSPC", 30) is None
    with _with_yf([_bar_row(20, o=-5.0)]):
        assert fc._yf_history("^GSPC", 30) is None


# ---------------------------------------------------------------------------
# _yf_quote: NaN-safe by invariant (validated bars or None)
# ---------------------------------------------------------------------------


def test_yf_quote_none_on_nan_poisoned_history():
    # Previously: NaN in bar 0 seeded max() -> NaN yearHigh; NaN later was
    # skipped -> silently wrong yearHigh. Both shapes must now yield None.
    with _with_yf([_bar_row(20, h=NAN), _bar_row(21)]):
        assert fc._yf_quote("^GSPC") is None
    with _with_yf([_bar_row(20), _bar_row(21, h=NAN)]):
        assert fc._yf_quote("^GSPC") is None


def test_yf_quote_finite_window_produces_finite_year_bounds():
    with _with_yf([_bar_row(20, h=105.0, low=95.0), _bar_row(21, h=110.0, low=98.0)]):
        q = fc._yf_quote("^GSPC")
    assert q is not None
    assert q["yearHigh"] == 110.0 and q["yearLow"] == 95.0
    assert fc._finite(q["price"])


# ---------------------------------------------------------------------------
# _history_values_ok: the FMP-side half of the same boundary
# ---------------------------------------------------------------------------


def _payload(bars):
    return {"symbol": "^GSPC", "historical": bars}


def _fmp_bar(c=100.0, o=99.0, h=101.0, low=98.0, v=1000):
    return {"date": "2026-08-20", "open": o, "high": h, "low": low, "close": c, "volume": v}


def test_history_values_ok_accepts_clean_and_rejects_each_bad_field():
    assert fc._history_values_ok(_payload([_fmp_bar()]))
    assert not fc._history_values_ok(_payload([_fmp_bar(c=NAN)]))
    assert not fc._history_values_ok(_payload([_fmp_bar(o=None)]))
    assert not fc._history_values_ok(_payload([_fmp_bar(h="101")]))  # str is not a price
    assert not fc._history_values_ok(_payload([_fmp_bar(low=0)]))
    assert not fc._history_values_ok(_payload([_fmp_bar(v=NAN)]))
    assert not fc._history_values_ok(_payload([_fmp_bar(v=-1)]))
    assert not fc._history_values_ok(_payload([]))
    assert not fc._history_values_ok(_payload([_fmp_bar(), _fmp_bar(c=NAN)]))  # one bad bar = all


def test_history_values_ok_rejects_non_dict_shapes():
    assert not fc._history_values_ok(None)
    assert not fc._history_values_ok({"historical": "nope"})
    assert not fc._history_values_ok({"historical": [["not", "a", "dict"]]})


# ---------------------------------------------------------------------------
# Integration: FMP payload carrying NaN falls through to the yf fallback
# (and to None when yf is also unavailable) — through the REAL
# _request_with_fallback loop, offline.
# ---------------------------------------------------------------------------


def _client_with_fmp(payload_by_key):
    client = fc.FMPClient(api_key="test_key")  # pragma: allowlist secret

    def fake_fmp_get(url, params=None, **_kw):
        for marker, payload in payload_by_key.items():
            if marker in url:
                return payload
        return None

    return client, patch.dict(
        type(client)._rate_limited_get.__globals__, {"fmp_get": fake_fmp_get}
    )


def test_fmp_nan_close_falls_back_to_yf():
    fmp_rows = [{"symbol": "^GSPC", "date": "2026-08-21", "open": 1.0, "high": 2.0, "low": 0.5, "close": NAN, "volume": 10}]
    client, transport = _client_with_fmp({"historical-price-eod": fmp_rows})
    with transport, _with_yf([_bar_row(20)]):
        data = client.get_historical_prices("^GSPC", days=30)
    assert data is not None
    assert data["data_source"] == "yfinance"  # NaN FMP payload was refused


def test_fmp_nan_close_with_no_yf_yields_none_not_repair():
    fmp_rows = [{"symbol": "^GSPC", "date": "2026-08-21", "open": 1.0, "high": 2.0, "low": 0.5, "close": NAN, "volume": 10}]
    client, transport = _client_with_fmp({"historical-price-eod": fmp_rows})
    with transport, patch.object(fc, "_yf_history", lambda *a, **k: None):
        assert client.get_historical_prices("^GSPC", days=30) is None


def test_fmp_quote_without_real_price_falls_back_to_yf():
    quote = [{"symbol": "^VIX", "price": NAN}]
    client, transport = _client_with_fmp({"quote": quote})
    with transport, _with_yf([_bar_row(20), _bar_row(21)]):
        quotes = client.get_quote("^VIX")
    assert quotes is not None
    assert quotes[0]["data_source"] == "yfinance"


def test_fmp_finite_payload_still_passes_untouched():
    fmp_rows = [
        {"symbol": "^GSPC", "date": "2026-08-21", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10}
    ]
    client, transport = _client_with_fmp({"historical-price-eod": fmp_rows})
    with transport, patch.object(fc, "_yf_history", lambda *a, **k: None):
        data = client.get_historical_prices("^GSPC", days=30)
    assert data is not None
    assert data["historical"][0]["close"] == 1.5
    assert data["data_source"] == "fmp"


def test_fmp_batch_stocklist_nan_falls_back_to_yf(capsys):
    # The historicalStockList branch has its own return path — the boundary
    # must hold there too, and the rejection must be LOUD (an operator has to
    # be able to tell "guard caught bad data" from "endpoint unreachable").
    payload = {
        "historicalStockList": [
            {
                "symbol": "^GSPC",
                "historical": [
                    {"date": "2026-08-21", "open": 1.0, "high": 2.0, "low": 0.5, "close": NAN, "volume": 10}
                ],
            }
        ]
    }
    client, transport = _client_with_fmp({"historical-price-eod": payload})
    with transport, _with_yf([_bar_row(20)]):
        data = client.get_historical_prices("^GSPC", days=30)
    assert data is not None and data["data_source"] == "yfinance"
    assert "failed the value boundary" in capsys.readouterr().err


def test_fmp_side_rejection_is_loud(capsys):
    fmp_rows = [
        {"symbol": "^GSPC", "date": "2026-08-21", "open": 1.0, "high": 2.0, "low": 0.5, "close": NAN, "volume": 10}
    ]
    client, transport = _client_with_fmp({"historical-price-eod": fmp_rows})
    with transport, patch.object(fc, "_yf_history", lambda *a, **k: None):
        assert client.get_historical_prices("^GSPC", days=30) is None
    assert "failed the value boundary" in capsys.readouterr().err


def test_rejection_caches_nothing_and_stamps_no_provenance():
    fmp_rows = [
        {"symbol": "^GSPC", "date": "2026-08-21", "open": 1.0, "high": 2.0, "low": 0.5, "close": NAN, "volume": 10}
    ]
    client, transport = _client_with_fmp({"historical-price-eod": fmp_rows})
    with transport, patch.object(fc, "_yf_history", lambda *a, **k: None):
        assert client.get_historical_prices("^GSPC", days=30) is None
    assert client.cache == {}
    assert "historical:^GSPC" not in client.data_sources


def test_mixed_quote_batch_keeps_only_finite_priced_rows():
    # Quotes are independent per-symbol records: in a MULTI-symbol request a
    # NaN-priced row is dropped, the finite one survives (deliberate deviation
    # from the historical bars' all-or-none rule).
    quotes = [{"symbol": "AAA", "price": NAN}, {"symbol": "BBB", "price": 42.0}]
    client, transport = _client_with_fmp({"quote": quotes})
    with transport:
        result = client.get_quote("AAA,BBB")
    assert result == [{"symbol": "BBB", "price": 42.0}]


def test_single_symbol_quote_drops_unrelated_rows():
    # Identity before price (ftd gate-r3 lesson): a foreign row with a valid
    # price must not leak into a single-symbol response — element [0] and the
    # batch dict key by symbol downstream.
    quotes = [{"symbol": "UNRELATED", "price": 42.0}, {"symbol": "^GSPC", "price": 5000.0}]
    client, transport = _client_with_fmp({"quote": quotes})
    with transport:
        result = client.get_quote("^GSPC")
    assert result == [{"symbol": "^GSPC", "price": 5000.0}]
    batch = client.get_batch_quotes(["^GSPC"])
    assert set(batch) == {"^GSPC"}


def test_quote_values_ok_present_year_fields_must_be_real():
    # codex gate r1: a present yearHigh/yearLow of NaN/None/0 sailed past the
    # price-only filter and fed 52-week distance + basket selection.
    assert fc._quote_values_ok({"price": 100.0, "yearHigh": 120.0, "yearLow": 80.0})
    assert fc._quote_values_ok({"price": 100.0})  # absent fields acceptable
    assert not fc._quote_values_ok({"price": 100.0, "yearHigh": NAN})
    assert not fc._quote_values_ok({"price": 100.0, "yearHigh": None})
    assert not fc._quote_values_ok({"price": 100.0, "yearLow": 0})
    assert not fc._quote_values_ok({"price": NAN, "yearHigh": 120.0})


def test_fmp_quote_with_nan_year_high_falls_back_to_yf():
    quotes = [{"symbol": "^GSPC", "price": 5000.0, "yearHigh": NAN}]
    client, transport = _client_with_fmp({"quote": quotes})
    with transport, _with_yf([_bar_row(20), _bar_row(21)]):
        result = client.get_quote("^GSPC")
    assert result is not None
    assert result[0]["data_source"] == "yfinance"


def test_history_values_ok_present_adjclose_must_be_real():
    # codex gate r1: consumers prefer adjClose (`d.get("close", d.get("adjClose", 0))`
    # and macro downsampling) — a present NaN/None/0 adjClose must reject.
    good = _fmp_bar()
    good["adjClose"] = 100.0
    assert fc._history_values_ok(_payload([good]))
    assert fc._history_values_ok(_payload([_fmp_bar()]))  # absent adjClose acceptable
    for bad_val in (NAN, None, 0):
        bad = _fmp_bar()
        bad["adjClose"] = bad_val
        assert not fc._history_values_ok(_payload([bad])), bad_val
