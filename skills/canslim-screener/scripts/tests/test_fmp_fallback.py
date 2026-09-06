#!/usr/bin/env python3
"""
Tests for the FMP transport in canslim-screener.

FMP has a single (stable) endpoint per key as of WPP-20260827-012 — the
former "v3 fallback" entry was removed because, through fmp_compat, a v3 URL
is rewritten back to the equivalent stable endpoint anyway, so it was never a
distinct endpoint. As of S-FMPCLIENT-3 (2026-09-06) the client also no longer
owns its own transport (session/retry/EOD-fold) — it delegates every call to
`fmp_compat.fmp_get_typed`, which is driven here through a stubbed
lowest-level `_original_get` so the genuine key-failover/fold/truncate logic
executes end-to-end, not a re-implementation of it at the client layer.

Tier A: shape validation (quote / historical)
Tier B: failure -> None with a typed `_last_error`
Caller regression: screen_canslim.py behavior on failure/fallback
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import fmp_compat  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client():
    """Create FMPClient with a fake API key."""
    with patch.dict(os.environ, {"FMP_API_KEY": "test_key"}):  # pragma: allowlist secret
        from fmp_client import FMPClient

        client = FMPClient(api_key="test_key")
    client.RATE_LIMIT_DELAY = 0  # no sleep in tests
    return client


def _resp(status_code=200, json_data=None):
    """A minimal stand-in for a `requests.Response`."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = status_code < 400
    resp.json.return_value = json_data
    resp.text = ""
    return resp


def _drive_real_transport(monkeypatch, get_response):
    """Wire the client's `fmp_get_typed` to the REAL `fmp_compat` transport,
    stubbed at the lowest level (`_original_get`) so key-failover, retry, and
    the EOD fold/truncate logic all run for real. `get_response(url, params)`
    returns a `_resp(...)`."""
    monkeypatch.setattr(fmp_compat, "_original_get", get_response)
    monkeypatch.setattr(fmp_compat, "get_fmp_keys", lambda: ["test_key"])
    monkeypatch.setattr(fmp_compat.time, "sleep", lambda *_: None)


# ---------------------------------------------------------------------------
# Tier A — shape validation
# ---------------------------------------------------------------------------


class TestQuoteShape:
    def test_quote_success_single_stable_call(self, monkeypatch):
        """Stable 200 returns data via exactly one upstream HTTP attempt."""
        client = _make_client()
        calls = []

        def get_response(url, params=None, timeout=None):
            calls.append((url, dict(params or {})))
            return _resp(200, [{"symbol": "^GSPC", "price": 5000}])

        _drive_real_transport(monkeypatch, get_response)
        result = client.get_quote("^GSPC")
        assert result == [{"symbol": "^GSPC", "price": 5000}]
        assert len(calls) == 1
        url, params = calls[0]
        assert url == "https://financialmodelingprep.com/stable/quote"
        assert params["symbol"] == "^GSPC"
        assert client._last_error is None

    def test_quote_all_keys_fail_returns_none_with_typed_error(self, monkeypatch):
        client = _make_client()

        def get_response(url, params=None, timeout=None):
            return _resp(403, None)

        _drive_real_transport(monkeypatch, get_response)
        result = client.get_quote("^GSPC")
        assert result is None
        assert client._last_error is not None
        assert fmp_compat.reason_kind(client._last_error) == "rate_limited"

    def test_quote_symbol_mismatch_rejected(self, monkeypatch):
        """Single-symbol quote returning the wrong symbol is rejected client-side."""
        client = _make_client()

        def get_response(url, params=None, timeout=None):
            return _resp(200, [{"symbol": "SPY", "price": 500.0}])

        _drive_real_transport(monkeypatch, get_response)
        result = client.get_quote("^GSPC")
        assert result is None

    def test_batch_quote_fans_out_per_symbol(self, monkeypatch):
        """A comma list is served as one /stable/quote request PER symbol and
        merged (stable does not batch; codex nested gate r2 P2, 2026-09-06)."""
        client = _make_client()
        by_symbol = {"^GSPC": [{"symbol": "^GSPC", "price": 5000}], "^VIX": [{"symbol": "^VIX", "price": 20}]}
        seen = []

        def get_response(url, params=None, timeout=None):
            seen.append(params["symbol"])
            return _resp(200, by_symbol[params["symbol"]])

        _drive_real_transport(monkeypatch, get_response)
        result = client.get_quote("^GSPC,^VIX")
        assert result == by_symbol["^GSPC"] + by_symbol["^VIX"]
        assert seen == ["^GSPC", "^VIX"]


class TestHistoricalShape:
    def test_historical_folds_and_truncates(self, monkeypatch):
        """A flat stable EOD list is folded to {"symbol", "historical"} and
        truncated to the requested `days`, via the real fmp_compat pipeline."""
        client = _make_client()
        rows = [
            {
                "symbol": "^GSPC",
                "date": f"2026-03-{20 - i:02d}",
                "open": 5000.0,
                "high": 5010.0,
                "low": 4990.0,
                "close": 5000.0 + i,
                "volume": 1000,
            }
            for i in range(5)
        ]

        def get_response(url, params=None, timeout=None):
            return _resp(200, rows)

        _drive_real_transport(monkeypatch, get_response)
        result = client.get_historical_prices("^GSPC", days=2)
        assert result is not None
        assert result["symbol"] == "^GSPC"
        assert len(result["historical"]) == 2
        assert result["historical"][0]["date"] == "2026-03-20"
        assert result["historical"][0]["close"] == 5000.0
        assert "symbol" not in result["historical"][0]

    def test_historical_symbol_mismatch_yields_empty_history(self, monkeypatch):
        """No row matches the requested symbol -> fmp_compat folds to an
        empty `historical` list under the requested symbol (not a refusal —
        `_normalize_eod_flat_list` only drops non-matching rows)."""
        client = _make_client()

        def get_response(url, params=None, timeout=None):
            return _resp(200, [{"symbol": "SPY", "date": "2026-03-20", "close": 500.0}])

        _drive_real_transport(monkeypatch, get_response)
        result = client.get_historical_prices("^GSPC", days=10)
        assert result == {"symbol": "^GSPC", "historical": []}

    def test_historical_all_keys_fail_returns_none(self, monkeypatch):
        client = _make_client()

        def get_response(url, params=None, timeout=None):
            return _resp(403, None)

        _drive_real_transport(monkeypatch, get_response)
        result = client.get_historical_prices("^GSPC", days=80)
        assert result is None


# ---------------------------------------------------------------------------
# Caller regression
# ---------------------------------------------------------------------------


class TestCallerRegression:
    """Verify screen_canslim.py behavior when FMP endpoints fail."""

    @pytest.mark.parametrize("institutional_holders", [None, []])
    def test_analyze_stock_calls_institutional_calculator_when_holders_missing(
        self, institutional_holders
    ):
        """Missing FMP holders must still reach the Finviz-capable calculator."""
        import screen_canslim

        profile = {
            "companyName": "Test Corp",
            "sector": "Technology",
            "mktCap": 200_000_000_000,
            "price": 200.0,
            "sharesOutstanding": 1_000_000_000,
        }
        client = MagicMock()
        client.get_profile.return_value = [profile]
        client.get_quote.return_value = [
            {
                "symbol": "TEST",
                "price": 200.0,
                "yearHigh": 210.0,
                "yearLow": 120.0,
                "volume": 2_000_000,
                "avgVolume": 1_000_000,
            }
        ]
        client.get_income_statement.return_value = [
            {"eps": 1.0, "revenue": 100_000_000, "date": "2026-03-31"}
        ] * 8
        client.get_historical_prices.return_value = {
            "historical": [
                {
                    "date": f"2026-01-{(i % 28) + 1:02d}",
                    "close": 100.0 + ((-1) ** i),
                    "volume": 1_000_000 + (i * 1_000),
                }
                for i in range(90)
            ]
        }
        client.get_institutional_holders.return_value = institutional_holders
        market_data = {"score": 80, "trend": "uptrend"}

        with patch.object(
            screen_canslim,
            "calculate_institutional_sponsorship",
            return_value={"score": 50, "data_source": "Finviz"},
        ) as mock_calculator:
            result = screen_canslim.analyze_stock(
                "TEST",
                client,
                market_data,
                rs_benchmark_historical=None,
                rs_benchmark="^GSPC",
                disable_rs=True,
            )

        assert result["i_component"]["data_source"] == "Finviz"
        mock_calculator.assert_called_once_with(
            institutional_holders, profile, symbol="TEST", use_finviz_fallback=True
        )

    def test_canslim_exits_on_quote_failure(self):
        """get_quote("^GSPC") → None causes sys.exit(1)."""
        with patch.dict(os.environ, {"FMP_API_KEY": "test_key"}):  # pragma: allowlist secret
            from fmp_client import FMPClient

            with patch.object(FMPClient, "get_quote", return_value=None):
                with patch("sys.argv", ["screen_canslim.py", "--max-candidates", "1"]):
                    import screen_canslim

                    with pytest.raises(SystemExit) as exc_info:
                        screen_canslim.main()
                    assert exc_info.value.code == 1

    def test_canslim_continues_on_historical_failure(self, capsys, tmp_path):
        """get_historical_prices("^GSPC") → None prints EMA fallback warning and continues."""
        with patch.dict(os.environ, {"FMP_API_KEY": "test_key"}):  # pragma: allowlist secret
            from fmp_client import FMPClient

            mock_quote = [
                {
                    "symbol": "^GSPC",
                    "price": 5000.0,
                    "yearHigh": 5200.0,
                    "yearLow": 4200.0,
                    "changesPercentage": 0.5,
                }
            ]
            mock_vix = [{"symbol": "^VIX", "price": 15.0}]

            def mock_get_quote(symbols):
                if "^GSPC" in symbols and "^VIX" not in symbols:
                    return mock_quote
                if "^VIX" in symbols:
                    return mock_vix
                return mock_quote

            with (
                patch.object(FMPClient, "get_quote", side_effect=mock_get_quote),
                patch.object(FMPClient, "get_historical_prices", return_value=None),
                patch.object(FMPClient, "get_income_statement", return_value=None),
                patch.object(FMPClient, "get_profile", return_value=None),
                patch.object(FMPClient, "get_institutional_holders", return_value=None),
                patch(
                    "sys.argv",
                    [
                        "screen_canslim.py",
                        "--max-candidates",
                        "1",
                        "--universe",
                        "AAPL",
                        "--output-dir",
                        str(tmp_path),
                    ],
                ),
            ):
                import screen_canslim

                # Should NOT raise SystemExit — historical failure is non-fatal
                try:
                    screen_canslim.main()
                except SystemExit:
                    pytest.fail("screen_canslim.main() should not exit when historical prices fail")

            captured = capsys.readouterr()
            assert "EMA fallback" in captured.out or "historical data unavailable" in captured.out
