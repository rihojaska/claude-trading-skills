"""FMP /api/v3 -> /stable migration tests for vcp-screener.

Every FMP call is now stable-only and delegated through
`fmp_compat.fmp_get_typed` (S-FMPCLIENT-3, 2026-09-06) — no client-owned
session, no v3 rung. These tests drive the REAL `fmp_compat.fmp_get_typed`
through a stubbed lowest-level `_original_get` so the genuine transport
(key failover, retries) executes end-to-end.
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
    return FMPClient(api_key="test_key")  # pragma: allowlist secret


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


class TestHardcodedCallSiteMigratedToStable:
    """Methods that used to bypass the fallback list build /stable URLs."""

    def test_sp500_constituents_hits_stable(self, monkeypatch):
        client = _make_client()
        seen = []

        def get_response(url, params=None, timeout=None):
            seen.append((url, params or {}))
            return _mock_response(200, [{"symbol": "AAPL"}, {"symbol": "MSFT"}])

        _drive_real_transport(monkeypatch, get_response)
        result = client.get_sp500_constituents()

        assert len(seen) == 1
        url, _ = seen[0]
        assert "/stable/sp500-constituent" in url
        assert "/api/v3/" not in url
        assert result == [{"symbol": "AAPL"}, {"symbol": "MSFT"}]


class TestNoV3Rung:
    """FMP retired v3 for non-legacy keys — there is no second endpoint to
    fall back to any more (WPP-20260827-012); a failed stable call goes
    straight to None (or the public-CSV fallback for constituents)."""

    def test_quote_stable_403_yields_none(self, monkeypatch):
        client = _make_client()
        calls = []

        def get_response(url, params=None, timeout=None):
            calls.append(url)
            return _mock_response(403, None, text="Forbidden")

        _drive_real_transport(monkeypatch, get_response)
        result = client.get_quote("^GSPC")

        assert result is None
        assert all("/api/v3/" not in u for u in calls)
