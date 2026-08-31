"""Tests for FMP client endpoint fallback.

FMP has a single (stable) endpoint per key as of WPP-20260827-012 — the
former "v3 fallback" entry was removed because, through fmp_compat, a v3 URL
is rewritten back to the equivalent stable endpoint anyway, so it was never a
distinct endpoint. The cross-provider fallback is yfinance.

Tier A: Fallback logic (4 tests)
Tier B: Response normalization (4 tests)
Tier B+: Shape validation (2 tests)
Tier C: Caller regression (2 tests)
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fmp_client import FMPClient

# ---------------------------------------------------------------------------
# Transport harness: the client transports via module-level fmp_compat.fmp_get
# (which swallows HTTP status — 402/403/429 and a genuine empty all surface as
# None), not requests.Session. Route fmp_get through each test's
# `client.session.get` mock (200 -> parsed JSON, else None) so the fallback
# tests keep exercising _request_with_fallback hermetically.
# ---------------------------------------------------------------------------

_ACTIVE: dict = {}


def _fake_fmp_get(url, params=None, timeout=30, **_kw):
    client = _ACTIVE.get("client")
    assert client is not None, "transport used before _make_client()"
    resp = client.session.get(url, params=params, timeout=timeout)
    if resp is None or getattr(resp, "status_code", 200) != 200:
        return None
    return resp.json()


def _wire_fake_transport(client):
    """Register the client and patch fmp_get in its module's namespace.

    Patching `type(client)._rate_limited_get.__globals__` targets the exact
    namespace the method resolves `fmp_get` against, which stays correct even
    when conftest evicts and re-imports `fmp_client` between skill suites.
    """
    _ACTIVE["client"] = client
    globs = type(client)._rate_limited_get.__globals__
    if globs.get("fmp_get") is not _fake_fmp_get:
        _ACTIVE.setdefault("patched", []).append((globs, globs["fmp_get"]))
        globs["fmp_get"] = _fake_fmp_get


def _no_yf(client):
    """Context manager: yfinance fallback unavailable (helpers return None)."""
    return patch.dict(
        type(client)._rate_limited_get.__globals__,
        {"_yf_quote": lambda *a, **k: None, "_yf_history": lambda *a, **k: None},
    )


def _with_yf(client, *, quote=None, history=None):
    """Context manager: stub the yfinance fallback to return fixed values.

    Since WPP-20260827-012 removed the v3 endpoint rung, yfinance is the
    ONLY cross-provider fallback FMP's stable endpoint has left, so tests
    that pin "stable fails -> next provider is tried" stub this instead of a
    second FMP response.
    """
    return patch.dict(
        type(client)._rate_limited_get.__globals__,
        {"_yf_quote": lambda *a, **k: quote, "_yf_history": lambda *a, **k: history},
    )


@pytest.fixture(autouse=True)
def _restore_transport():
    yield
    for globs, orig in _ACTIVE.get("patched", []):
        globs["fmp_get"] = orig
    _ACTIVE.clear()


def _make_client():
    """Create an FMPClient with a fake API key and zero rate-limit delay."""
    client = FMPClient(api_key="test_key")
    client.RATE_LIMIT_DELAY = 0
    _wire_fake_transport(client)
    return client


def _mock_response(status_code, json_data=None):
    """Create a mock response object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = f"HTTP {status_code}"
    return resp


# =========================================================================
# Tier A — Fallback logic
# =========================================================================


class TestFallbackLogic:
    """Tier A: stable -> v3 fallback mechanics."""

    def test_quote_stable_success(self):
        """Stable 200 returns data; v3 not called."""
        client = _make_client()
        quote_data = [{"symbol": "^GSPC", "price": 5000.0}]
        stable_resp = _mock_response(200, quote_data)

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return stable_resp

        client.session.get = MagicMock(side_effect=side_effect)
        result = client.get_quote("^GSPC")
        assert result == quote_data
        # Only stable endpoint called (1 call)
        assert call_count == 1

    def test_quote_stable_403_falls_back_to_yfinance(self):
        """Stable 403 -> yfinance provides the quote (no v3 rung left to try)."""
        client = _make_client()
        stable_resp = _mock_response(403)
        client.session.get = MagicMock(return_value=stable_resp)

        yf_quote = {"symbol": "^GSPC", "price": 5000.0, "data_source": "yfinance"}
        with _with_yf(client, quote=yf_quote):
            result = client.get_quote("^GSPC")
        assert result == [yf_quote]
        assert client.session.get.call_count == 1

    def test_quote_both_fail(self):
        """Stable 403 and yfinance unavailable -> returns None."""
        client = _make_client()
        stable_resp = _mock_response(403)
        client.session.get = MagicMock(return_value=stable_resp)

        with _no_yf(client):
            result = client.get_quote("^GSPC")
        assert result is None
        assert client.session.get.call_count == 1

    def test_historical_fallback_to_yfinance(self):
        """Stable 403 -> yfinance provides historical data (no v3 rung left to try)."""
        client = _make_client()
        stable_resp = _mock_response(403)
        client.session.get = MagicMock(return_value=stable_resp)

        yf_hist = {
            "symbol": "^GSPC",
            "historical": [{"date": "2026-03-20", "close": 5000.0}],
            "data_source": "yfinance",
        }
        with _with_yf(client, history=yf_hist):
            result = client.get_historical_prices("^GSPC", days=80)
        assert result == yf_hist
        assert client.session.get.call_count == 1


# =========================================================================
# Tier B — Response normalization
# =========================================================================


class TestResponseNormalization:
    """Tier B: historicalStockList normalization and passthrough."""

    def test_historical_stable_v3_format_passthrough(self):
        """Stable 200 with {"historical": [...]} -> returned as-is."""
        client = _make_client()
        hist_data = {
            "symbol": "^GSPC",
            "historical": [{"date": "2026-03-20", "close": 5000.0}],
        }
        resp = _mock_response(200, hist_data)
        client.session.get = MagicMock(return_value=resp)

        result = client.get_historical_prices("^GSPC", days=80)
        assert result == hist_data

    def test_historical_stable_batch_format_exact_match(self):
        """Stable 200 with historicalStockList matching symbol -> normalized."""
        client = _make_client()
        batch_data = {
            "historicalStockList": [
                {
                    "symbol": "^GSPC",
                    "historical": [{"date": "2026-03-20", "close": 5000.0}],
                }
            ]
        }
        resp = _mock_response(200, batch_data)
        client.session.get = MagicMock(return_value=resp)

        result = client.get_historical_prices("^GSPC", days=80)
        assert result is not None
        assert "historical" in result
        assert result["historical"] == [{"date": "2026-03-20", "close": 5000.0}]
        assert result["symbol"] == "^GSPC"

    def test_historical_stable_batch_no_match_falls_back_to_yfinance(self):
        """Stable batch no match -> falls through to yfinance (no v3 rung left)."""
        client = _make_client()
        # Stable returns batch with a different symbol
        batch_data = {
            "historicalStockList": [
                {
                    "symbol": "SPY",
                    "historical": [{"date": "2026-03-20", "close": 500.0}],
                }
            ]
        }
        stable_resp = _mock_response(200, batch_data)
        client.session.get = MagicMock(return_value=stable_resp)

        yf_hist = {
            "symbol": "^GSPC",
            "historical": [{"date": "2026-03-20", "close": 5000.0}],
            "data_source": "yfinance",
        }
        with _with_yf(client, history=yf_hist):
            result = client.get_historical_prices("^GSPC", days=80)
        assert result == yf_hist
        assert client.session.get.call_count == 1

    def test_historical_batch_no_match_returns_none_when_yfinance_unavailable(self):
        """Stable batch no match + yfinance unavailable -> None."""
        client = _make_client()
        batch_data = {
            "historicalStockList": [
                {
                    "symbol": "SPY",
                    "historical": [{"date": "2026-03-20", "close": 500.0}],
                }
            ]
        }
        stable_resp = _mock_response(200, batch_data)

        client.session.get = MagicMock(return_value=stable_resp)

        with _no_yf(client):
            result = client.get_historical_prices("^GSPC", days=80)
        assert result is None
        assert client.session.get.call_count == 1


# =========================================================================
# Tier B+ — Shape validation
# =========================================================================


class TestShapeValidation:
    """Tier B+: Reject truthy-but-wrong-shape responses."""

    def test_quote_rejects_non_list_response(self):
        """Stable returns truthy dict -> rejected, falls back to yfinance."""
        client = _make_client()
        # Stable returns a dict (wrong shape for quote)
        error_data = {"Error Message": "Invalid API call"}
        stable_resp = _mock_response(200, error_data)
        client.session.get = MagicMock(return_value=stable_resp)

        yf_quote = {"symbol": "^GSPC", "price": 5000.0, "data_source": "yfinance"}
        with _with_yf(client, quote=yf_quote):
            result = client.get_quote("^GSPC")
        assert result == [yf_quote]
        assert client.session.get.call_count == 1

    def test_historical_rejects_non_dict_response(self):
        """Stable returns truthy list -> rejected, falls back to yfinance."""
        client = _make_client()
        # Stable returns a list (wrong shape for historical)
        wrong_data = [1, 2, 3]
        stable_resp = _mock_response(200, wrong_data)
        client.session.get = MagicMock(return_value=stable_resp)

        yf_hist = {
            "symbol": "^GSPC",
            "historical": [{"date": "2026-03-20", "close": 5000.0}],
            "data_source": "yfinance",
        }
        with _with_yf(client, history=yf_hist):
            result = client.get_historical_prices("^GSPC", days=80)
        assert result == yf_hist
        assert client.session.get.call_count == 1


# =========================================================================
# Tier B++ — Symbol mismatch protection
# =========================================================================


class TestSymbolMismatch:
    """Reject responses where returned symbol doesn't match the request."""

    def test_quote_symbol_mismatch_falls_back(self):
        """Single-symbol quote returning wrong symbol is rejected, falls back to yfinance."""
        client = _make_client()
        wrong = _mock_response(200, [{"symbol": "SPY", "price": 500.0}])
        client.session.get = MagicMock(return_value=wrong)

        yf_quote = {"symbol": "^GSPC", "price": 5000.0, "data_source": "yfinance"}
        with _with_yf(client, quote=yf_quote):
            result = client.get_quote("^GSPC")
        assert result == [yf_quote]
        assert client.session.get.call_count == 1

    def test_historical_symbol_mismatch_falls_back(self):
        """Single-symbol historical returning wrong symbol is rejected, falls back to yfinance."""
        client = _make_client()
        wrong = _mock_response(200, {"symbol": "SPY", "historical": [{"close": 500}]})
        client.session.get = MagicMock(return_value=wrong)

        yf_hist = {"symbol": "^GSPC", "historical": [{"close": 5000}], "data_source": "yfinance"}
        with _with_yf(client, history=yf_hist):
            result = client.get_historical_prices("^GSPC", days=80)
        assert result["symbol"] == "^GSPC"
        assert client.session.get.call_count == 1

    def test_batch_quote_skips_symbol_check(self):
        """Multi-symbol (batch) quote does not apply symbol mismatch check."""
        client = _make_client()
        batch_data = [{"symbol": "^GSPC", "price": 5000}, {"symbol": "^VIX", "price": 20}]
        resp = _mock_response(200, batch_data)
        client.session.get = MagicMock(return_value=resp)

        result = client.get_quote("^GSPC,^VIX")
        assert result == batch_data
        assert client.session.get.call_count == 1


# =========================================================================
# Tier C — Caller regression
# =========================================================================


class TestCallerRegression:
    """Tier C: Verify ftd_detector.main() handles FMPClient failures correctly."""

    # IMPORTANT: patch `ftd_detector.FMPClient` (the symbol AS USED by main()),
    # not the module-level `FMPClient` imported at the top of this file.
    # When pytest runs both ftd-detector and market-top-detector test files in
    # the same session, conftest evicts and re-imports `fmp_client` during
    # skill switches. This produces multiple class objects from the same source
    # file. Patching the test-file-level `FMPClient` reference would miss the
    # class that `ftd_detector.main()` actually uses. Patching via
    # `ftd_detector.FMPClient` always hits the class bound inside ftd_detector
    # at the time main() executes.

    def test_ftd_detector_exits_on_historical_failure(self):
        """get_historical_prices -> None => main() calls sys.exit(1) (fatal)."""
        with (
            patch.dict(os.environ, {"FMP_API_KEY": "test_key"}),  # pragma: allowlist secret
            patch("sys.argv", ["ftd_detector.py"]),
        ):
            # Import inside patch to pick up env var
            import ftd_detector

            with (
                patch.object(ftd_detector.FMPClient, "get_historical_prices", return_value=None),
                patch.object(
                    ftd_detector.FMPClient,
                    "get_quote",
                    return_value=[{"symbol": "^GSPC", "price": 5000.0}],
                ),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    ftd_detector.main()
                assert exc_info.value.code == 1

    def test_ftd_detector_continues_on_quote_failure(self):
        """get_quote -> None => main() continues with warning (non-fatal)."""
        with (
            patch.dict(os.environ, {"FMP_API_KEY": "test_key"}),  # pragma: allowlist secret
            patch("sys.argv", ["ftd_detector.py"]),
        ):
            import ftd_detector

            sp500_hist = {
                "historical": [
                    {
                        "date": f"2026-03-{20 - i:02d}",
                        "open": 5000.0,
                        "high": 5010.0,
                        "low": 4990.0,
                        "close": 5000.0 - i * 10,
                        "volume": 3_000_000_000,
                    }
                    for i in range(80)
                ]
            }
            qqq_hist = {
                "historical": [
                    {
                        "date": f"2026-03-{20 - i:02d}",
                        "open": 450.0,
                        "high": 455.0,
                        "low": 445.0,
                        "close": 450.0 - i,
                        "volume": 50_000_000,
                    }
                    for i in range(80)
                ]
            }

            def mock_hist(symbol, days=365):
                if symbol == "^GSPC":
                    return sp500_hist
                elif symbol == "QQQ":
                    return qqq_hist
                return None

            with (
                patch.object(
                    ftd_detector.FMPClient, "get_historical_prices", side_effect=mock_hist
                ),
                patch.object(ftd_detector.FMPClient, "get_quote", return_value=None),
                # WPP-20260818-009: the two writers were replaced by one
                # pair-atomic entry point — both artifacts render before either
                # is written, so a markdown failure can no longer leave a dated
                # JSON from a failed run for the promoter to publish.
                patch.object(ftd_detector, "write_reports"),
            ):
                # Should NOT raise SystemExit — quote failure is non-fatal
                ftd_detector.main()


class TestEODFlatListSuccess:
    """Issue #64: stable EOD flat list -> public method success (regression)."""

    @patch("fmp_client.requests.Session")
    def test_get_historical_prices_normalizes_flat_list(self, mock_session_class):
        """Flat list response from new EOD endpoint -> dict contract preserved."""
        mock_session = MagicMock()
        mock_session.get.return_value = _mock_response(
            200,
            [
                {
                    "symbol": "SPY",
                    "date": "2026-04-29",
                    "open": 500.0,
                    "high": 502.0,
                    "low": 499.0,
                    "close": 501.0,
                    "volume": 1_000_000,
                },
                {
                    "symbol": "SPY",
                    "date": "2026-04-28",
                    "open": 498.0,
                    "high": 501.0,
                    "low": 497.0,
                    "close": 500.0,
                    "volume": 1_100_000,
                },
            ],
        )
        mock_session_class.return_value = mock_session

        client = _make_client()
        client.session = mock_session
        client.max_retries = 0

        result = client.get_historical_prices("SPY", days=2)
        assert isinstance(result, dict), f"expected dict, got {type(result).__name__}"
        assert result["symbol"] == "SPY"
        assert len(result["historical"]) == 2
        assert result["historical"][0]["close"] == 501.0

        # URL regression: must hit /historical-price-eod/full with from/to (not timeseries)
        first_call = mock_session.get.call_args_list[0]
        url = first_call[0][0]
        params = first_call[1]["params"]
        assert "historical-price-eod/full" in url
        assert "from" in params and "to" in params
        assert "timeseries" not in params
