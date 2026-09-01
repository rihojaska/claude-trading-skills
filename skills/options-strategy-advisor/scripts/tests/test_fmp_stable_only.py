"""Transport-seam pins for the options-strategy-advisor FMP path.

black_scholes carried a second `/api/v3/...` rung behind each /stable endpoint.
fmp_compat rewrites a v3 URL straight back to the equivalent /stable endpoint,
so the rung was never a distinct upstream — only a second rate-limited call on
the SAME one, with no key failover (WPP-20260831-004).

These tests drive the REAL `fmp_compat.fmp_get` and assert at the transport
seam: exactly one upstream call per miss, every URL on /stable, and no
`/api/v3/` string ever entering the compat layer.
"""

import json
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import black_scholes  # noqa: E402
import fmp_compat  # noqa: E402


def _response(status, payload):
    """A real requests.Response so .ok/.json() behave like production."""
    resp = requests.Response()
    resp.status_code = status
    resp._content = json.dumps(payload).encode()
    resp.headers["Content-Type"] = "application/json"
    return resp


@pytest.fixture
def seam(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "dummy-primary")
    monkeypatch.setenv("FMP_FALLBACK_API_KEY", "dummy-fallback")
    monkeypatch.setattr(fmp_compat.time, "sleep", lambda *_a, **_k: None)
    assert black_scholes.fmp_get is not None, "fmp_compat must be importable here"

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
        seam["responses"] = [_response(200, [{"symbol": "AAPL", "price": 150.0}])]
        assert black_scholes.get_current_stock_price("AAPL", "caller-key") == 150.0
        assert len(seam["upstream"]) == 1
        assert "/stable/quote" in seam["upstream"][0]
        _assert_stable_only(seam)

    def test_quote_miss_is_exactly_one_stable_call(self, seam):
        seam["responses"] = [_response(404, {})]
        assert black_scholes.get_current_stock_price("AAPL", "caller-key") is None
        assert len(seam["upstream"]) == 1  # no v3 second rung
        _assert_stable_only(seam)

    def test_profile_hit_is_exactly_one_stable_call(self, seam):
        seam["responses"] = [_response(200, [{"lastDividend": 2.0, "price": 100.0}])]
        assert black_scholes.get_dividend_yield("AAPL", "caller-key") == 0.02
        assert len(seam["upstream"]) == 1
        assert "/stable/profile" in seam["upstream"][0]
        _assert_stable_only(seam)

    def test_historical_flat_list_is_normalized_by_fmp_compat(self, seam):
        seam["responses"] = [
            _response(
                200,
                [
                    {"symbol": "AAPL", "date": "2026-05-19", "close": 101.0},
                    {"symbol": "AAPL", "date": "2026-05-18", "close": 100.0},
                ],
            )
        ]
        prices = black_scholes.fetch_historical_prices_for_hv("AAPL", "caller-key", days=90)
        assert prices == [100.0, 101.0]  # oldest-first
        assert len(seam["upstream"]) == 1
        assert "/stable/historical-price-eod/full" in seam["upstream"][0]
        _assert_stable_only(seam)


class TestAdapter:
    def test_none_from_fmp_get_is_the_failure_path(self, monkeypatch):
        monkeypatch.setattr(black_scholes, "fmp_get", lambda *_a, **_k: None)
        assert black_scholes.get_current_stock_price("AAPL", "k") is None
        assert black_scholes.get_dividend_yield("AAPL", "k") == 0
        assert black_scholes.fetch_historical_prices_for_hv("AAPL", "k") is None

    def test_parsed_list_from_fmp_get_is_used(self, monkeypatch):
        monkeypatch.setattr(black_scholes, "fmp_get", lambda *_a, **_k: [{"price": 12.5}])
        assert black_scholes.get_current_stock_price("AAPL", "k") == 12.5
