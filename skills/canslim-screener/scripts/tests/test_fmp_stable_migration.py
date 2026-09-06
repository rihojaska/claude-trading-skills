"""FMP /stable migration for canslim-screener.

Keys issued after 2025-08-31 lose v3 access (403), so every FMP call is now
stable-only, delegated through `fmp_compat.fmp_get_typed` (S-FMPCLIENT-3,
2026-09-06 — no more client-owned session, no v3 rung):
- get_income_statement: /stable/income-statement?symbol=&period=
- get_profile: /stable/profile?symbol= (+ mktCap alias)
- get_institutional_holders: /stable institutional-ownership summary (count +
  ownership%) + top-holders page (superinvestor names), returned as an
  aggregate dict — no v3 full-holder-list rung any more.
"""

import os
from datetime import date
from unittest.mock import MagicMock, patch

import fmp_client


def _make_client():
    with patch.dict(os.environ, {"FMP_API_KEY": "test_key"}):  # pragma: allowlist secret
        client = fmp_client.FMPClient(api_key="test_key")
    client.RATE_LIMIT_DELAY = 0  # no sleep in tests
    return client


def _resp(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = ""
    return resp


class TestIncomeStatementProfile:
    def test_income_statement_uses_stable_query(self, monkeypatch):
        client = _make_client()
        captured = {}

        def fake_get_typed(url, params=None, timeout=None, max_retries_per_key=None):
            captured["url"], captured["params"] = url, params
            return [{"symbol": "AAPL", "period": "Q2", "eps": 2.0}], None

        monkeypatch.setattr(fmp_client, "fmp_get_typed", fake_get_typed)
        client.get_income_statement("AAPL", period="quarter", limit=8)
        assert captured["url"].endswith("/stable/income-statement")
        assert captured["params"]["symbol"] == "AAPL"
        assert captured["params"]["period"] == "quarter"

    def test_profile_stable_with_mktcap_alias(self, monkeypatch):
        client = _make_client()
        # /stable/profile returns marketCap, not mktCap.
        monkeypatch.setattr(
            fmp_client,
            "fmp_get_typed",
            lambda *a, **k: (
                [{"symbol": "AAPL", "companyName": "Apple", "marketCap": 4_000_000_000_000}],
                None,
            ),
        )
        profile = client.get_profile("AAPL")
        assert profile[0]["mktCap"] == 4_000_000_000_000  # aliased for downstream


class TestInstitutionalHolders:
    def test_summary_plus_top_holders(self, monkeypatch):
        def fake_get_typed(url, params=None, timeout=None, max_retries_per_key=None):
            if "symbol-positions-summary" in url:
                return [{"investorsHolding": 6170, "ownershipPercent": 61.6}], None
            if "extract-analytics/holder" in url:
                return (
                    [
                        {
                            "investorName": "VANGUARD GROUP INC",
                            "sharesNumber": 100,
                            "changeInSharesNumber": 5,
                        },
                        {
                            "investorName": "BLACKROCK, INC.",
                            "sharesNumber": 90,
                            "changeInSharesNumber": -2,
                        },
                    ],
                    None,
                )
            raise AssertionError(f"unexpected {url}")

        client = _make_client()
        monkeypatch.setattr(fmp_client, "fmp_get_typed", fake_get_typed)
        result = client.get_institutional_holders("AAPL")
        assert result["num_holders"] == 6170
        assert result["ownership_pct"] == 61.6
        assert result["top_holders"][0] == {
            "holder": "VANGUARD GROUP INC",
            "shares": 100,
            "change": 5,
        }

    def test_no_holders_when_stable_empty_every_quarter(self, monkeypatch):
        """No stable 13F data for any of the walked-back quarters -> None.

        The v3 institutional-holder rung is gone (S-FMPCLIENT-3): a
        stable-empty result no longer falls back to a full v3 holder list.
        """

        def fake_get_typed(url, params=None, timeout=None, max_retries_per_key=None):
            return [], None  # no stable 13F data for any quarter

        client = _make_client()
        monkeypatch.setattr(fmp_client, "fmp_get_typed", fake_get_typed)
        result = client.get_institutional_holders("AAPL")
        assert result is None

    def test_recent_13f_quarters_walk_back(self):
        from fmp_client import FMPClient

        # May 2026 -> most recent completed quarter is Q1 2026, then walk back.
        quarters = list(FMPClient._recent_13f_quarters(as_of=date(2026, 5, 20), count=3))
        assert quarters == [(2026, 1), (2025, 4), (2025, 3)]


class TestInstitutionalCalculatorDictShape:
    def test_uses_aggregate_count_and_ownership(self):
        from calculators.institutional_calculator import calculate_institutional_sponsorship

        agg = {
            "num_holders": 6170,
            "ownership_pct": 61.6,
            # SUPERINVESTORS are famous active managers (Berkshire, Baupost, ...),
            # not passive index funds, so this name triggers the bonus.
            "top_holders": [
                {"holder": "BLACKROCK, INC.", "shares": 90, "change": 1},
                {"holder": "BERKSHIRE HATHAWAY INC", "shares": 50, "change": 0},
            ],
        }
        result = calculate_institutional_sponsorship(agg, profile={}, use_finviz_fallback=False)
        assert result["num_holders"] == 6170
        assert result["ownership_pct"] == 61.6
        assert result["superinvestor_present"] is True  # matched Berkshire
        assert result["score"] > 0
