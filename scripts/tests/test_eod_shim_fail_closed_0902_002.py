"""WPP-20260902-002 — `_normalize_eod_flat_list` fails CLOSED on a malformed row.

A 200-OK EOD payload with valid bars plus one non-dict / date-less element used
to be silently shortened and accepted on every `fmp_get` history path. Now the
shim refuses the whole series (returns None, one stderr line naming the symbol
and the offending index) — the same shape callers already treat as "no data".
Also pins the etf-holder v3→stable rewrite (WPP-20260901-016).
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fmp_compat  # noqa: E402

CLEAN = [
    {"symbol": "AAPL", "date": "2026-01-03", "close": 3.0},
    {"symbol": "AAPL", "date": "2026-01-02", "close": 1.0},
]


def test_clean_list_keeps_its_shape_and_limit():
    out = fmp_compat._normalize_eod_flat_list(CLEAN, "AAPL")
    assert out == {"symbol": "AAPL", "historical": [{"date": "2026-01-03", "close": 3.0}, {"date": "2026-01-02", "close": 1.0}]}
    assert fmp_compat._normalize_eod_flat_list(CLEAN, "AAPL", limit=1)["historical"] == [{"date": "2026-01-03", "close": 3.0}]


@pytest.mark.parametrize(
    "payload, index",
    [
        ([{"date": "2026-01-02", "close": 1.0}, "junk", {"close": 2.0}], 1),
        ([{"date": "2026-01-02", "close": 1.0}, {"close": 2.0}], 1),
        ([None], 0),
    ],
)
def test_malformed_row_refuses_the_whole_series(payload, index, capsys):
    assert fmp_compat._normalize_eod_flat_list(payload, "AAPL") is None
    err = capsys.readouterr().err
    assert "EOD payload for AAPL malformed at index " + str(index) in err
    assert "refusing the whole series" in err


def test_empty_and_non_list_unchanged():
    assert fmp_compat._normalize_eod_flat_list([], "AAPL") == {"symbol": "AAPL", "historical": []}
    assert fmp_compat._normalize_eod_flat_list({"symbol": "AAPL", "historical": []}, "AAPL") == {"symbol": "AAPL", "historical": []}


def test_etf_holder_rewrites_to_stable_query_form():
    rewritten = fmp_compat._rewrite_v3_to_stable("https://financialmodelingprep.com/api/v3/etf-holder/SPY") \
        if hasattr(fmp_compat, "_rewrite_v3_to_stable") else None
    if rewritten is None:
        pytest.skip("rewrite helper name differs — pinned via the map instead")
    assert rewritten.endswith("/stable/etf/holdings?symbol=SPY")


def test_etf_holder_in_map():
    assert any("etf-holder" in k and v.startswith("/stable/etf/holdings?symbol=") for k, v in fmp_compat._V3_TO_STABLE.items())
