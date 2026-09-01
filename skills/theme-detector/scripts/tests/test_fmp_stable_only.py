"""Transport-seam pins for the theme-detector FMP path (WPP-20260831-004).

etf_scanner used to carry a second `/api/v3/...` rung behind every /stable
endpoint. fmp_compat rewrites a v3 URL straight back to the equivalent /stable
endpoint, so that rung was never a distinct upstream — only a second
rate-limited call on the SAME one, with no key failover.

These tests exercise the REAL `fmp_compat.fmp_get` path (the production one)
and assert at the transport seam: exactly one upstream call per miss, every
URL on /stable, and no `/api/v3/` string ever entering the compat layer.
test_etf_scanner.py covers the direct-requests fallback used by a standalone
.skill install.
"""

import json
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import etf_scanner  # noqa: E402
import fmp_compat  # noqa: E402
from etf_scanner import ETFScanner  # noqa: E402


def _response(status, payload):
    """A real requests.Response so .ok/.json() behave like production."""
    resp = requests.Response()
    resp.status_code = status
    resp._content = json.dumps(payload).encode()
    resp.headers["Content-Type"] = "application/json"
    return resp


@pytest.fixture
def seam(monkeypatch):
    """Record every URL entering fmp_compat and every upstream GET."""
    monkeypatch.setenv("FMP_API_KEY", "dummy-primary")
    monkeypatch.setenv("FMP_FALLBACK_API_KEY", "dummy-fallback")
    monkeypatch.setattr(fmp_compat.time, "sleep", lambda *_a, **_k: None)
    assert etf_scanner.fmp_get is not None, "fmp_compat must be importable here"

    state = {"upstream": [], "translated": [], "responses": []}

    real_translate = fmp_compat._translate_url

    def recording_translate(url):
        state["translated"].append(url)
        return real_translate(url)

    def fake_get(url, params=None, timeout=None, **_kw):
        state["upstream"].append(url)
        return state["responses"].pop(0) if state["responses"] else _response(404, {})

    monkeypatch.setattr(fmp_compat, "_translate_url", recording_translate)
    monkeypatch.setattr(fmp_compat, "_original_get", fake_get)
    return state


def _assert_stable_only(state):
    assert state["upstream"], "no upstream call recorded"
    for url in state["upstream"]:
        assert "/stable/" in url
        assert "/api/v3/" not in url
    for url in state["translated"]:
        assert "/api/v3/" not in url


class TestTransportSeam:
    def test_quote_hit_is_exactly_one_stable_call(self, seam):
        seam["responses"] = [_response(200, [{"symbol": "AAPL", "pe": 30, "price": 150}])]
        scanner = ETFScanner(fmp_api_key="caller-key", rate_limit_sec=0)

        data = scanner._fmp_request("quote", "AAPL")

        assert data == [{"symbol": "AAPL", "pe": 30, "price": 150}]
        assert len(seam["upstream"]) == 1
        assert "/stable/quote" in seam["upstream"][0]
        _assert_stable_only(seam)

    def test_quote_miss_is_exactly_one_stable_call(self, seam):
        seam["responses"] = [_response(404, {})]
        scanner = ETFScanner(fmp_api_key="caller-key", rate_limit_sec=0)

        assert scanner._fmp_request("quote", "AAPL") is None
        assert len(seam["upstream"]) == 1  # no v3 second rung
        assert scanner.backend_stats()["fmp_failures"] == 1
        _assert_stable_only(seam)

    def test_historical_hit_normalizes_to_v3_shape(self, seam):
        bars = [
            {"symbol": "AAPL", "date": "2026-05-19", "close": 298.97, "volume": 10},
            {"symbol": "AAPL", "date": "2026-05-18", "close": 296.0, "volume": 11},
        ]
        seam["responses"] = [_response(200, bars)]
        scanner = ETFScanner(fmp_api_key="caller-key", rate_limit_sec=0)

        data = scanner._fmp_request("historical", "AAPL", {"timeseries": 20})

        assert isinstance(data, dict)
        assert [row["close"] for row in data["historical"]] == [298.97, 296.0]
        assert len(seam["upstream"]) == 1
        assert "/stable/historical-price-eod/full" in seam["upstream"][0]
        _assert_stable_only(seam)

    def test_empty_historical_payload_counts_as_a_miss(self, seam):
        """fmp_get returns a truthy {"historical": []} for an empty list.

        Pre-migration a raw `[]` recorded an endpoint failure; the adapter must
        keep that, or the circuit breaker silently stops tripping.
        """
        seam["responses"] = [_response(200, [])]
        scanner = ETFScanner(fmp_api_key="caller-key", rate_limit_sec=0)

        assert scanner._fmp_request("historical", "AAPL", {"timeseries": 20}) is None
        assert scanner.backend_stats()["fmp_failures"] == 1
        _assert_stable_only(seam)


class TestAdapter:
    def test_none_from_fmp_get_drives_the_failure_path(self, monkeypatch):
        monkeypatch.setattr(etf_scanner, "fmp_get", lambda *_a, **_k: None)
        scanner = ETFScanner(fmp_api_key="k", rate_limit_sec=0)

        assert scanner._fmp_request("quote", "AAPL") is None
        assert scanner.backend_stats()["fmp_failures"] == 1

    def test_parsed_list_from_fmp_get_is_returned(self, monkeypatch):
        monkeypatch.setattr(etf_scanner, "fmp_get", lambda *_a, **_k: [{"symbol": "AAPL"}])
        scanner = ETFScanner(fmp_api_key="k", rate_limit_sec=0)

        assert scanner._fmp_request("quote", "AAPL") == [{"symbol": "AAPL"}]
        assert scanner.backend_stats()["fmp_failures"] == 0

    def test_circuit_breaker_disables_after_threshold(self, monkeypatch):
        monkeypatch.setattr(etf_scanner, "fmp_get", lambda *_a, **_k: None)
        scanner = ETFScanner(fmp_api_key="k", rate_limit_sec=0)

        for _ in range(ETFScanner._ENDPOINT_FAILURE_THRESHOLD):
            scanner._fmp_request("quote", "AAPL")
        assert scanner._disabled_endpoints  # still trips with a single rung


@pytest.fixture(autouse=True)
def _isolate_fmp_env(monkeypatch):
    """The constructor now ASSIGNS FMP_API_KEY (see TestCallerKeyWins), so every
    ETFScanner construction would otherwise leak a key into the rest of the
    session."""
    monkeypatch.setenv("FMP_API_KEY", "ambient-house-key")
    monkeypatch.setenv("FMP_FALLBACK_API_KEY", "ambient-fallback")


class TestCallerKeyWins:
    """`setdefault` could never fire: importing fmp_compat self-loads the house
    credential at import, so FMP_API_KEY is always already set. A
    caller-supplied credential must OVERRIDE the ambient one."""

    def test_caller_key_overrides_a_preset_ambient_key(self):
        ETFScanner(fmp_api_key="caller-key", rate_limit_sec=0)
        assert fmp_compat.get_fmp_keys()[0] == "caller-key"

    def test_no_caller_key_leaves_the_ambient_key_alone(self):
        ETFScanner(rate_limit_sec=0)
        assert fmp_compat.get_fmp_keys()[0] == "ambient-house-key"
