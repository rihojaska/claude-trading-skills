"""WPP-20260603-031 residual guards — value-dividend screener.

Pins: the FINVIZ-candidate ratios site applies the same `0 < x` validity as
the local-universe site (a non-positive P/E–P/B never survives as a cheap
one), the report's pe_ratio / pb_ratio are `None` rather than 0 when absent,
and the local-universe builder reports selected / attempted / priced /
un-priceable / unattempted on stderr.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import screen_dividend_stocks as mod  # noqa: E402


@pytest.mark.parametrize("raw", [None, 0, -1.5, "x", float("nan")])
def test_invalid_ratio_is_none(raw):
    assert mod._valid_ratio(raw) is None


def test_low_but_positive_ratio_is_kept():
    assert mod._valid_ratio(0.008) == 0.008


def test_coverage_line_counts_yfinance_miss_as_unpriceable_and_break_as_unattempted(capsys, monkeypatch):
    client = mod.FMPClient(api_key="dummy")
    seen = []

    def fake_get(endpoint, params=None, quiet=False):
        seen.append(endpoint)
        sym = endpoint.split("/")[-1]
        if sym == "KMB":
            client.rate_limit_reached = True
            return [{"symbol": "KMB", "price": 1.0}]
        if sym == "PEP":
            return [{"symbol": "PEP", "price": 1.0}]
        return None

    monkeypatch.setattr(client, "_get", fake_get)
    monkeypatch.setattr(client, "get_company_profile", lambda s: None)
    monkeypatch.setattr(mod, "_yf_quote_profile", lambda s: None)
    out = mod.build_candidates_from_universe(["PEP", "PHNX.L", "KMB", "ADP"], client)
    assert [c["symbol"] for c in out] == ["PEP", "KMB"]
    err = capsys.readouterr().err
    assert "Coverage: selected 4 · attempted 3 · priced 2 · un-priceable 1 [PHNX.L] · unattempted 1 (rate-limit break)" in err
