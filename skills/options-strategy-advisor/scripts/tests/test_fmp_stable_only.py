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


@pytest.fixture(autouse=True)
def _isolate_fmp_env(monkeypatch):
    """A house credential is always present in production (fmp_compat self-loads
    it at import), so the caller-key tests below need a known ambient baseline
    to override AND to be restored to."""
    monkeypatch.setenv("FMP_API_KEY", "ambient-house-key")
    monkeypatch.setenv("FMP_FALLBACK_API_KEY", "ambient-fallback")


class TestErrorObjectDegradesInsteadOfRaising:
    """FMP answers an invalid key with a 200-OK error OBJECT. It is truthy and
    subscripting it raises — the adapter must degrade, not explode."""

    ERROR_OBJECT = {"Error Message": "Invalid API KEY."}

    def test_quote_error_object_is_a_miss(self, monkeypatch, capsys):
        monkeypatch.setattr(black_scholes, "fmp_get", lambda *_a, **_k: self.ERROR_OBJECT)
        assert black_scholes.get_current_stock_price("AAPL", "k") is None
        assert "returned no data" in capsys.readouterr().out

    def test_profile_error_object_is_a_zero_yield(self, monkeypatch):
        monkeypatch.setattr(black_scholes, "fmp_get", lambda *_a, **_k: self.ERROR_OBJECT)
        assert black_scholes.get_dividend_yield("AAPL", "k") == 0

    def test_historical_error_object_is_a_miss(self, monkeypatch):
        monkeypatch.setattr(black_scholes, "fmp_get", lambda *_a, **_k: self.ERROR_OBJECT)
        assert black_scholes.fetch_historical_prices_for_hv("AAPL", "k") is None

    def test_malformed_historical_rows_are_a_miss(self, monkeypatch):
        """Codex gate r3: a 200-OK payload with rows lacking a numeric close, a
        non-dict row, or a non-list `historical` must degrade to None, never
        raise KeyError/TypeError outside the failure boundary."""
        cases = [
            {"symbol": "AAPL", "historical": [{"date": "2026-08-01"}]},
            {"symbol": "AAPL", "historical": [{"close": "n/a"}]},
            {"symbol": "AAPL", "historical": ["oops"]},
            {"symbol": "AAPL", "historical": {"close": 1.0}},
            [{"close": 1.0}, "oops"],
            {"symbol": "AAPL", "historicalStockList": [None]},
            {"symbol": "AAPL", "historicalStockList": "nope"},
            {"symbol": "AAPL", "historical": [{"close": float("nan")}]},
            {"symbol": "AAPL", "historical": [{"close": float("inf")}]},
            {"symbol": "AAPL", "historical": [{"close": 0}]},
            {"symbol": "AAPL", "historical": [{"close": -3.5}]},
            {"symbol": "AAPL", "historical": [{"adjClose": True, "close": True}]},
        ]
        for payload in cases:
            monkeypatch.setattr(black_scholes, "fmp_get", lambda *_a, **_k: payload)
            assert black_scholes.fetch_historical_prices_for_hv("AAPL", "k") is None, payload
        monkeypatch.setattr(
            black_scholes, "fmp_get",
            lambda *_a, **_k: {"symbol": "AAPL", "historical": [{"close": 2.0}, {"adjClose": 1.5, "close": 1.0}]},
        )
        assert black_scholes.fetch_historical_prices_for_hv("AAPL", "k") == [1.5, 2.0]

    def test_non_dict_list_element_is_a_miss(self, monkeypatch):
        """MUTANT: guard only the dict case -> `["oops"][0].get` raises."""
        monkeypatch.setattr(black_scholes, "fmp_get", lambda *_a, **_k: ["oops"])
        assert black_scholes.get_current_stock_price("AAPL", "k") is None
        assert black_scholes.get_dividend_yield("AAPL", "k") == 0

    def test_quote_row_without_a_price_is_a_miss(self, monkeypatch):
        monkeypatch.setattr(black_scholes, "fmp_get", lambda *_a, **_k: [{"symbol": "AAPL"}])
        assert black_scholes.get_current_stock_price("AAPL", "k") is None


class TestEmptyHistoricalIsAMiss:
    """An empty payload carries no bars; HV over it would be fabricated."""

    def test_empty_flat_list_is_a_miss(self, monkeypatch):
        monkeypatch.setattr(black_scholes, "fmp_get", lambda *_a, **_k: [])
        assert black_scholes.fetch_historical_prices_for_hv("AAPL", "k") is None

    def test_empty_historical_dict_is_a_miss(self, monkeypatch):
        monkeypatch.setattr(black_scholes, "fmp_get", lambda *_a, **_k: {"historical": []})
        assert black_scholes.fetch_historical_prices_for_hv("AAPL", "k") is None

    def test_empty_historical_stock_list_is_a_miss(self, monkeypatch):
        monkeypatch.setattr(
            black_scholes, "fmp_get", lambda *_a, **_k: {"historicalStockList": []}
        )
        assert black_scholes.fetch_historical_prices_for_hv("AAPL", "k") is None


class TestCallerKeyWins:
    """A caller-supplied credential must OVERRIDE the ambient house key for the
    duration of the call, and MUST NOT outlive it: a process-global assignment
    made two callers passing different keys share the last one written (codex
    gate P2). The scoping lives in `fmp_compat.key_override`."""

    def _keys_seen(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            black_scholes,
            "fmp_get",
            lambda *_a, **_k: seen.append(fmp_compat.get_fmp_keys()) or None,
        )
        return seen

    def test_caller_key_is_in_effect_during_the_call(self, monkeypatch):
        seen = self._keys_seen(monkeypatch)
        black_scholes.get_current_stock_price("AAPL", "caller-key")
        assert seen and seen[0][0] == "caller-key"

    def test_ambient_key_is_restored_after_the_call(self, monkeypatch):
        """MUTANT: assign the caller key process-globally -> the NEXT call, made
        by a different caller with a different key, silently uses this one."""
        self._keys_seen(monkeypatch)
        black_scholes.get_current_stock_price("AAPL", "caller-key")
        assert fmp_compat.get_fmp_keys()[0] == "ambient-house-key"

    def test_no_caller_key_leaves_the_ambient_key_alone(self, monkeypatch):
        seen = self._keys_seen(monkeypatch)
        black_scholes.get_current_stock_price("AAPL", "")
        assert seen and seen[0][0] == "ambient-house-key"
        assert fmp_compat.get_fmp_keys()[0] == "ambient-house-key"
