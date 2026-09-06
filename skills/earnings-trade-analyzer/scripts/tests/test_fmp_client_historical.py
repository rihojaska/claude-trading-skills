"""Issue #64: stable/historical-price-eod/full normalization for earnings-trade-analyzer.

This skill's `get_historical_prices()` returns Optional[list[dict]] (NOT dict),
unlike the other 6 fmp_client implementations. The public method extracts
`data["historical"]` from the normalizer output.

The client no longer folds/truncates the EOD flat-list itself — it delegates
to `fmp_compat.fmp_get_typed`, which folds and truncates before returning
(S-FMPCLIENT-3, 2026-09-06). These tests drive the REAL `fmp_get_typed`
through a stubbed lowest-level `_original_get` so the pipeline runs
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
    client = FMPClient(api_key="test_key", max_api_calls=200)
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
    def test_get_historical_prices_returns_list(self, monkeypatch):
        """Flat list response -> public method returns list (NOT dict)."""
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
        # CRITICAL: this skill returns list, not dict
        assert isinstance(result, list), (
            f"expected list (skill contract), got {type(result).__name__}"
        )
        assert len(result) == 2
        assert result[0]["date"] == "2026-04-29"
        assert result[0]["close"] == 501.0
        assert "symbol" not in result[0], "row-level symbol should be stripped"

        # URL regression
        assert len(calls) == 1
        url, params = calls[0]
        assert "historical-price-eod/full" in url
        assert "from" in params and "to" in params
        assert "timeseries" not in params
