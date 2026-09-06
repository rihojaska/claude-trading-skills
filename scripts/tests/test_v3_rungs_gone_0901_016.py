"""Pins for WPP-20260901-016: no hand-rolled FMP "stable -> v3" fallback rungs.

Eight scripts used to carry two-rung endpoint lists, a `_stable_then_v3`
helper, or raw `https://financialmodelingprep.com/api/v3/...` URLs, and none
of them imported the repo-root `fmp_compat` module. Through `fmp_compat.fmp_get`
a v3 URL is rewritten back to the identical /stable/ URL (see `_V3_TO_STABLE`
in fmp_compat.py), so a "v3 fallback" was only ever a second, rate-limited
query of the SAME endpoint; on raw requests `api/v3` is a permanently-403
upstream (retired 2025-08-31).

(a) A source pin: none of the 8 files contains the literal "api/v3" anymore.
(b) A source pin: each of the 8 files imports `fmp_get` from the repo-root
    `fmp_compat` module with a standalone-install `except ImportError` guard.
(c) For three cheap-to-import files, a transport-seam test: monkeypatching
    the module's `fmp_get` routes the FMP call through it with a /stable/
    URL, and the module's own low-level HTTP call (`requests.get` /
    `urllib.request.urlopen`) is never reached.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Paths (relative to the nested repo root) of the 8 files this fix touched.
TARGET_FILES = [
    "skills/pair-trade-screener/scripts/find_pairs.py",
    "skills/pair-trade-screener/scripts/analyze_spread.py",
    "skills/downtrend-duration-analyzer/scripts/analyze_downtrends.py",
    "skills/trader-memory-core/scripts/fmp_price_adapter.py",
    "skills/stockbee-episodic-pivot-analyzer/scripts/analyze_ep.py",
    "skills/signal-postmortem/scripts/postmortem_recorder.py",
    "skills/theme-detector/scripts/representative_stock_selector.py",
    "skills/stockbee-20pct-study/scripts/run_20pct_study.py",
]


def _source(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


class TestNoV3Literal:
    """(a) No file references the retired v3 API path anymore."""

    def test_no_api_v3_literal(self):
        offenders = [p for p in TARGET_FILES if "api/v3" in _source(p)]
        assert offenders == [], f"still reference api/v3: {offenders}"


class TestImportsFmpCompat:
    """(b) Each file imports fmp_get with the standalone-install fallback."""

    def test_imports_fmp_get_with_import_error_guard(self):
        missing_import = [
            p for p in TARGET_FILES if "from fmp_compat import fmp_get" not in _source(p)
        ]
        missing_guard = [p for p in TARGET_FILES if "except ImportError" not in _source(p)]
        assert missing_import == [], f"missing 'from fmp_compat import fmp_get': {missing_import}"
        assert missing_guard == [], f"missing 'except ImportError' guard: {missing_guard}"


# ---------------------------------------------------------------------------
# (c) Transport-seam tests: fmp_get, not requests/urllib, carries the call.
# ---------------------------------------------------------------------------


def _load(skill_scripts_rel_dir: str, module_name: str):
    """Import a skill script module with its own scripts/ dir on sys.path."""
    scripts_dir = REPO_ROOT / skill_scripts_rel_dir
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module(module_name)


class TestFmpPriceAdapterTransportSeam:
    def test_get_daily_closes_routes_through_fmp_get(self, monkeypatch):
        fmp_price_adapter = _load("skills/trader-memory-core/scripts", "fmp_price_adapter")

        calls = []

        def _recorder(url, params=None, **kwargs):
            calls.append((url, params))
            return {
                "symbol": "AAPL",
                "historical": [
                    {"date": "2026-01-02", "close": 151.0},
                    {"date": "2026-01-01", "close": 150.0},
                ],
            }

        monkeypatch.setattr(fmp_price_adapter, "fmp_get", _recorder)

        with patch("urllib.request.urlopen") as mock_urlopen:
            adapter = fmp_price_adapter.FMPPriceAdapter(api_key="test-key")
            result = adapter.get_daily_closes("AAPL", "2026-01-01", "2026-01-02")

        assert not mock_urlopen.called
        assert len(calls) == 1
        url, params = calls[0]
        assert url.startswith("https://financialmodelingprep.com/stable/")
        assert params["symbol"] == "AAPL"
        assert [row["date"] for row in result] == ["2026-01-01", "2026-01-02"]


class TestPostmortemRecorderTransportSeam:
    def test_fetch_price_data_routes_through_fmp_get(self, monkeypatch):
        postmortem_recorder = _load("skills/signal-postmortem/scripts", "postmortem_recorder")

        calls = []

        def _recorder(url, params=None, **kwargs):
            calls.append((url, params))
            return {"historical": [{"date": "2026-01-01", "close": 149.0}]}

        monkeypatch.setattr(postmortem_recorder, "fmp_get", _recorder)

        with patch.object(postmortem_recorder.requests, "get") as mock_get:
            result = postmortem_recorder.fetch_price_data(
                "AAPL", "2026-01-01", "2026-01-02", "test-key"
            )

        assert not mock_get.called
        assert len(calls) == 1
        url, params = calls[0]
        assert url == "https://financialmodelingprep.com/stable/historical-price-eod/full"
        assert params["symbol"] == "AAPL"
        assert result == {"2026-01-01": 149.0}


class TestRepresentativeStockSelectorTransportSeam:
    def test_fetch_etf_holdings_routes_through_fmp_get(self, monkeypatch):
        representative_stock_selector = _load(
            "skills/theme-detector/scripts", "representative_stock_selector"
        )

        calls = []

        def _recorder(url, params=None, **kwargs):
            calls.append((url, params))
            return [{"asset": "AAPL", "marketValue": 1_000_000}]

        monkeypatch.setattr(representative_stock_selector, "fmp_get", _recorder)

        sel = representative_stock_selector.RepresentativeStockSelector(fmp_api_key="test-key")

        with (
            patch.object(representative_stock_selector.requests, "get") as mock_get,
            patch.object(sel, "_rate_limit"),
        ):
            result = sel._fetch_etf_holdings("XLK", limit=5)

        assert not mock_get.called
        assert len(calls) == 1
        url, params = calls[0]
        assert url == "https://financialmodelingprep.com/stable/etf/holdings"
        assert params == {"symbol": "XLK"}
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["source"] == "etf_holdings"
