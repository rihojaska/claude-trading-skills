"""WPP-20260603-031 residual guards — dividend-growth-pullback screener.

Pins: a missing / non-positive / non-finite P/E–P/B is `None` (never a
cheapest-possible 0) and earns 0 valuation points in the composite score,
so an un-ratioed candidate ranks below an otherwise-identical priced peer;
the local-universe builder reports selected / attempted / priced /
un-priceable / unattempted on stderr, and a rate-limit break shows up as
`unattempted`, never as un-priceable.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import screen_dividend_growth_rsi as mod  # noqa: E402


@pytest.mark.parametrize("raw", [None, 0, -3.2, "n/a", float("nan"), float("inf")])
def test_invalid_ratio_is_none(raw):
    assert mod._valid_ratio(raw) is None


def test_valid_ratio_passes_through_low_multiples_unchanged():
    assert mod._valid_ratio(0.008) == 0.008  # cheap is not corrupt (codex P2 #5)
    assert mod._valid_ratio("12.5") == 12.5


def _stock(**over):
    base = {
        "dividend_yield": 4.0, "dividend_cagr_3y": 8.0, "dividend_consistent": True,
        "rsi": 30.0, "payout_ratio": 50.0, "fcf_payout_ratio": 60.0,
        "revenue_cagr_3y": 5.0, "eps_cagr_3y": 5.0, "pe_ratio": 12.0, "pb_ratio": 2.0,
    }
    base.update(over)
    return base


def test_missing_ratios_earn_zero_valuation_points_and_rank_below_a_priced_peer():
    score = mod.StockAnalyzer.calculate_composite_score
    priced = score(_stock())
    unratioed = score(_stock(pe_ratio=None, pb_ratio=None))
    zeroed = score(_stock(pe_ratio=0, pb_ratio=0))  # the old default shape
    assert priced - unratioed == 10
    assert zeroed == unratioed, "a 0 ratio must not score as the cheapest multiple"


def test_coverage_line_separates_unpriceable_from_unattempted(capsys):
    client = mod.FMPClient(api_key="dummy")
    calls = []

    def fake_quote(symbol):
        calls.append(symbol)
        if symbol == "AHT.L":
            return None
        if symbol == "PEP":
            client.rate_limit_reached = True
        return {"symbol": symbol, "price": 10.0}

    client.get_quote_with_profile = fake_quote
    out = mod.build_candidates_from_universe(["GSK.L", "AHT.L", "PEP", "KMB", "ADP"], client)
    assert [c["symbol"] for c in out] == ["GSK.L", "PEP"]
    err = capsys.readouterr().err
    assert "Coverage: selected 5 · attempted 3 · priced 2 · un-priceable 1 [AHT.L] · unattempted 2 (rate-limit break)" in err
