"""Transport-seam pins for the value-dividend-screener FMP path (WPP-20260831-004).

`FMPClient._get` used to append a second `/api/v3/{endpoint}` rung behind the
/stable attempt, and `get_historical_prices` carried a v3 entry in
`_FMP_HIST_ENDPOINTS`. fmp_compat rewrites a v3 URL straight back to the
equivalent /stable endpoint, so neither rung was a distinct upstream — each was
a second rate-limited call on the SAME one.

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

import fmp_compat  # noqa: E402
import screen_dividend_stocks as mod  # noqa: E402
from screen_dividend_stocks import FMPClient  # noqa: E402


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
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(fmp_compat.time, "sleep", lambda *_a, **_k: None)
    assert mod.fmp_get is not None, "fmp_compat must be importable here"

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
    def test_profile_hit_is_exactly_one_stable_call(self, seam):
        seam["responses"] = [_response(200, [{"symbol": "AAPL", "sector": "Tech"}])]
        client = FMPClient("caller-key")

        assert client.get_company_profile("AAPL")["sector"] == "Tech"
        assert len(seam["upstream"]) == 1
        assert "/stable/profile" in seam["upstream"][0]
        _assert_stable_only(seam)

    def test_profile_miss_is_exactly_one_stable_call(self, seam):
        seam["responses"] = [_response(404, {})]
        client = FMPClient("caller-key")

        assert client.get_company_profile("AAPL") is None
        assert len(seam["upstream"]) == 1  # no v3 second rung
        _assert_stable_only(seam)

    def test_historical_hit_is_exactly_one_stable_call(self, seam):
        bars = [
            {"symbol": "AAPL", "date": "2026-05-19", "close": 100.0},
            {"symbol": "AAPL", "date": "2026-05-18", "close": 99.0},
        ]
        seam["responses"] = [_response(200, bars)]
        client = FMPClient("caller-key")

        rows = client.get_historical_prices("AAPL", days=30)

        assert [r["close"] for r in rows] == [100.0, 99.0]
        assert len(seam["upstream"]) == 1
        assert "/stable/historical-price-eod/full" in seam["upstream"][0]
        _assert_stable_only(seam)

    def test_historical_endpoint_list_is_stable_only(self):
        assert mod._FMP_HIST_ENDPOINTS
        assert all("/stable/" in u for u in mod._FMP_HIST_ENDPOINTS)
        assert not any("/api/v3/" in u for u in mod._FMP_HIST_ENDPOINTS)


class TestAdapter:
    def test_none_from_fmp_get_records_an_endpoint_failure(self, monkeypatch):
        monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
        monkeypatch.setattr(mod, "fmp_get", lambda *_a, **_k: None)
        client = FMPClient("k")

        assert not client.get_historical_prices("AAPL", days=30)
        assert client._endpoint_failures[mod._FMP_HIST_ENDPOINTS[0]] == 1

    def test_empty_historical_payload_counts_as_a_miss(self, monkeypatch):
        """fmp_get returns a truthy {"historical": []} for an empty list."""
        monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
        monkeypatch.setattr(
            mod, "fmp_get", lambda *_a, **_k: {"symbol": "AAPL", "historical": []}
        )
        client = FMPClient("k")

        assert not client.get_historical_prices("AAPL", days=30)
        assert client._endpoint_failures[mod._FMP_HIST_ENDPOINTS[0]] == 1

    def test_parsed_dict_from_fmp_get_is_sliced_to_days(self, monkeypatch):
        monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
        rows = [{"date": f"2026-05-{20 - i:02d}", "close": 100.0 + i} for i in range(10)]
        monkeypatch.setattr(
            mod, "fmp_get", lambda *_a, **_k: {"symbol": "AAPL", "historical": rows}
        )
        client = FMPClient("k")

        assert client.get_historical_prices("AAPL", days=3) == rows[:3]

    def test_endpoint_without_a_stable_equivalent_returns_none(self, monkeypatch, capsys):
        """No v3 escape hatch: an unmapped endpoint fails loud, not silently."""
        called = []
        monkeypatch.setattr(mod, "fmp_get", lambda *a, **k: called.append(a) or [])
        client = FMPClient("k")

        assert client._get("some-unmapped-endpoint") is None
        assert called == []  # never reached a transport
        assert "no /stable equivalent" in capsys.readouterr().err
