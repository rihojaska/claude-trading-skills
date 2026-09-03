"""Tests for the frozen/thin price series guard in the RSI gate.

Founding case: SDR.L printed RSI 33.5 on 2026-09-03 while sitting 0.92% off
its 52-week high; its last twelve closes were 587.9/588.4/588.9/588.4/588.4/
588.4/588.4/588.4/588.9/589.4/584.5/584.0 (4-month range 2.04%). RSI is
scale-free, so a frozen/thinly-updated series manufactures a false oversold
reading. Same-day true positive: IGG.L gapped 1706->1460->1371->1360->1350->
1312 (a real -21% move, RSI 18) and must NOT be flagged frozen.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "screen_dividend_growth_rsi.py"


def _load_script() -> ModuleType:
    """Import screen_dividend_growth_rsi as a module."""
    spec = importlib.util.spec_from_file_location("screen_dg_frozen", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules.setdefault("requests", MagicMock())
    sys.modules.setdefault("pandas", MagicMock())
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def mod():
    if not SCRIPT_PATH.exists():
        pytest.skip(f"Script not found: {SCRIPT_PATH}")
    return _load_script()


# SDR.L-shaped: 22 closes confined to a ~2.04% band (founding false-positive case).
SDR_L_SHAPED_PRICES = [
    587.9, 588.4, 588.9, 588.4, 588.4, 588.4, 588.4, 588.4, 588.9, 589.4,
    584.5, 584.0, 585.0, 586.0, 587.0, 588.0, 589.0, 588.5, 587.5, 586.5,
    585.5, 584.9,
]

# IGG.L-shaped: real -30% gap-down move (true-positive, must not be flagged).
IGG_L_SHAPED_PRICES = [
    1706, 1706, 1700, 1695, 1690, 1685, 1680, 1675, 1670, 1665,
    1660, 1655, 1650, 1645, 1640, 1635, 1630, 1500, 1460, 1400,
    1371, 1360,
]


class TestFrozenSeriesPredicate:
    def test_sdr_l_shaped_series_is_frozen(self, mod):
        assert mod._series_is_frozen_30d(SDR_L_SHAPED_PRICES) is True

    def test_igg_l_shaped_series_is_not_frozen(self, mod):
        assert mod._series_is_frozen_30d(IGG_L_SHAPED_PRICES) is False

    def test_too_few_bars_is_not_classifiable(self, mod):
        """Fewer than min_bars means "not classifiable", never True."""
        prices = [100.0] * 10
        assert mod._series_is_frozen_30d(prices) is False

    def test_exact_threshold_edge_below_is_frozen(self, mod):
        """A range just under 3.0% counts as frozen."""
        # max=100, min=97.02 -> range = 2.98%
        prices = [100.0] * 15 + [97.02] * 7
        assert mod._series_is_frozen_30d(prices, max_range_pct=3.0) is True

    def test_exact_threshold_edge_above_is_not_frozen(self, mod):
        """A range just over 3.0% does not count as frozen."""
        # max=100, min=96.98 -> range = 3.02%
        prices = [100.0] * 15 + [96.98] * 7
        assert mod._series_is_frozen_30d(prices, max_range_pct=3.0) is False


def test_frozen_series_candidate_is_skipped_not_emitted(monkeypatch):
    """A SDR.L-shaped frozen series must never surface as a screener candidate."""
    mod = _load_script()

    client = MagicMock()
    client.rate_limit_reached = False
    client.get_quote_with_profile.return_value = {
        "symbol": "SDR.L",
        "companyName": "Schroders plc",
        "price": 584.0,
        "sector": "Financial Services",
    }
    client.get_dividend_history.return_value = [{}]
    client.get_historical_prices.return_value = [
        {"close": p} for p in reversed(SDR_L_SHAPED_PRICES)
    ]

    analyzer = MagicMock()
    analyzer.analyze_dividend_growth.return_value = (15.0, True, 20.0, 5)

    monkeypatch.setattr(mod, "FMPClient", MagicMock(return_value=client))
    monkeypatch.setattr(mod, "StockAnalyzer", MagicMock(return_value=analyzer))
    monkeypatch.setattr(
        mod,
        "RSICalculator",
        MagicMock(return_value=MagicMock(calculate_rsi=MagicMock(return_value=33.5))),
    )

    results = mod.screen_dividend_growth_pullbacks("test-key", universe_symbols=["SDR.L"])

    assert results == []
    symbols_emitted = [r.get("symbol") for r in results]
    assert "SDR.L" not in symbols_emitted
