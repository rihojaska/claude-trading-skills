"""WPP-20260901-019 — the VIX term-structure input carries a freshness leg
only when its date is KNOWN: auto-detection uses the older of the two quote
dates (both required); `--vix-term-date` is an optional CLI partner; an
undated CLI `--vix-term` keeps the by-design exclusion and is never charged
0.70; `decision_grade` never reads the freshness table.
"""
import os
import sys
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fmp_client  # noqa: E402
from fmp_client import _quote_date  # noqa: E402
from market_top_detector import _FLAG_PAIRS, _reconcile_flag_pairs, compute_data_freshness  # noqa: E402


def _client():
    with patch.dict(os.environ, {"FMP_API_KEY": "k"}):  # pragma: allowlist secret
        return fmp_client.FMPClient()


@pytest.mark.parametrize(
    "quote, expected",
    [
        ({"date": "2026-09-04"}, "2026-09-04"),
        ({"date": "2026-09-04 16:00:00"}, "2026-09-04"),
        ({"timestamp": 1_757_000_000}, "2025-09-04"),
        ({"timestamp": True}, None),
        ({"timestamp": -5}, None),
        ({"price": 1.0}, None),
        (None, None),
    ],
)
def test_quote_date_vectors(quote, expected):
    assert _quote_date(quote) == expected


def test_auto_term_structure_carries_the_older_quote_date():
    c = _client()
    c.get_quote = MagicMock(
        side_effect=lambda s: [{"price": 16.0, "date": "2026-09-03"}] if "VIX3M" in s else [{"price": 14.0, "date": "2026-09-04"}]
    )
    r = c.get_vix_term_structure()
    assert r["classification"] == "contango" and r["date"] == "2026-09-03"


def test_auto_term_structure_half_dated_reports_none():
    c = _client()
    c.get_quote = MagicMock(side_effect=lambda s: [{"price": 16.0}] if "VIX3M" in s else [{"price": 14.0, "date": "2026-09-04"}])
    assert c.get_vix_term_structure()["date"] is None


def test_yf_quote_carries_the_bar_date():
    bars = [{"date": f"2026-09-{d:02d}", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1} for d in (4, 3, 2)]
    with patch.object(fmp_client, "_yf_history", return_value={"historical": bars}):
        assert fmp_client._yf_quote("^VIX")["date"] == "2026-09-04"


def test_dated_leg_enters_freshness_and_undated_does_not():
    today = date.today().isoformat()
    with_leg = compute_data_freshness({"vix_term_date": today})
    assert with_leg["vix_term"]["factor"] == 1.0 and with_leg["vix_term"]["date"] == today
    assert "vix_term" not in compute_data_freshness({"vix_term_date": None})
    assert "vix_term" not in compute_data_freshness({})


def test_undated_cli_vix_term_is_never_charged():
    assert "vix_term" not in _FLAG_PAIRS  # not a reconciled pair by design
    args = MagicMock(vix_term="contango", vix_term_date=None, breadth_200dma=None, breadth_200dma_date=None,
                     breadth_50dma=None, breadth_50dma_date=None, put_call=None, put_call_date=None,
                     margin_debt_yoy=None, margin_debt_date=None)
    assert "vix_term" not in _reconcile_flag_pairs(args)
    assert "vix_term" not in compute_data_freshness({"vix_term_date": None}, undated_present=_reconcile_flag_pairs(args))


def test_decision_grade_never_reads_the_freshness_table():
    import inspect

    import market_top_detector as mtd

    src = inspect.getsource(mtd.build_data_coverage)
    assert "freshness" not in src and "vix_term" not in src
