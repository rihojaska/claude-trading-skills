"""Issue #64: stable/historical-price-eod/full normalization for vcp-screener.

The client no longer folds/truncates the EOD flat-list itself — it delegates
to `fmp_compat.fmp_get_typed`, which folds and truncates before returning
(S-FMPCLIENT-3, 2026-09-06). These tests drive the REAL `fmp_get_typed`
through a stubbed lowest-level `_original_get` so that pipeline runs
end-to-end.
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import fmp_compat  # noqa: E402
from fmp_client import FMPClient  # noqa: E402


def _make_client():
    client = FMPClient(api_key="test_key")
    client.max_retries = 0
    return client


def _mock_response(status_code, json_payload, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = status_code < 400
    resp.json.return_value = json_payload
    resp.text = text
    return resp


def _drive_real_transport(monkeypatch, get_response):
    monkeypatch.setattr(fmp_compat, "_original_get", get_response)
    monkeypatch.setattr(fmp_compat, "get_fmp_keys", lambda: ["test_key"])
    monkeypatch.setattr(fmp_compat.time, "sleep", lambda *_: None)


class TestEODFlatListSuccess:
    def test_get_historical_prices_normalizes_flat_list(self, monkeypatch):
        """Flat list response -> dict contract preserved, request carries
        `timeseries` through to fmp_compat's own from/to conversion."""
        calls = []

        def get_response(url, params=None, timeout=None):
            calls.append((url, dict(params or {})))
            return _mock_response(
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

        _drive_real_transport(monkeypatch, get_response)
        client = _make_client()

        result = client.get_historical_prices("SPY", days=2)
        assert isinstance(result, dict), f"expected dict, got {type(result).__name__}"
        assert result["symbol"] == "SPY"
        assert len(result["historical"]) == 2
        assert result["historical"][0]["close"] == 501.0

        assert len(calls) == 1
        url, params = calls[0]
        assert "historical-price-eod/full" in url
        assert params.get("symbol") == "SPY"
        # The client passes `timeseries` straight through; fmp_compat's own
        # `_prepare_params_for_url` converts it to from/to before the wire
        # call (and pops it into an internal `_tm_limit` truncation bound).
        assert "from" in params and "to" in params
        assert "timeseries" not in params


class TestFailureIsSilentAndTyped:
    """FMP has no v3 fallback any more (WPP-20260827-012) — a failed call
    surfaces `None` with a typed `_last_error`, not a second endpoint."""

    def test_stable_failure_yields_none_with_typed_error(self, monkeypatch):
        def get_response(url, params=None, timeout=None):
            return _mock_response(
                403, None, text='{"Error Message": "Special Endpoint: not available..."}'
            )

        _drive_real_transport(monkeypatch, get_response)
        client = _make_client()

        result = client.get_historical_prices("GOOG", days=10)
        assert result is None
        assert client._last_error is not None
        assert fmp_compat.reason_kind(client._last_error) == "rate_limited"

    def test_stable_success_suppresses_warning(self, monkeypatch, capsys):
        """Happy path: no spurious warnings on stderr."""

        def get_response(url, params=None, timeout=None):
            return _mock_response(
                200,
                [
                    {
                        "symbol": "AAPL",
                        "date": "2026-01-01",
                        "open": 1.0,
                        "high": 1.0,
                        "low": 1.0,
                        "close": 1.0,
                        "volume": 1000,
                    }
                ],
            )

        _drive_real_transport(monkeypatch, get_response)
        client = _make_client()

        result = client.get_historical_prices("AAPL", days=1)
        assert result is not None

        captured = capsys.readouterr()
        assert "WARN" not in captured.err, f"unexpected stderr:\n{captured.err}"
        assert "fallback" not in captured.err.lower()
