"""kanchi `FMPClient._get` routes through `fmp_compat.fmp_get_typed`
(WPP-20260906-004): key failover on a rate-limit signal, every attempt charged
to `api_calls`, a typed miss → None with one WARNING line, and the raw
standalone path only when fmp_compat is absent.
"""
from __future__ import annotations

import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import build_entry_signals as mod  # noqa: E402
import fmp_compat  # noqa: E402


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code, self.ok, self.text, self.headers = status_code, status_code < 400, "", {}
        self.url = "https://financialmodelingprep.com/stable/x"
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def seam(monkeypatch):
    script, calls = [], []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        nxt = script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    monkeypatch.setattr(fmp_compat, "_original_get", fake_get)
    monkeypatch.setattr(fmp_compat, "get_fmp_keys", lambda: ["k1", "k2"])
    monkeypatch.setattr(fmp_compat.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setenv("FMP_API_KEY", "k1")
    return script, calls


def test_failover_success_is_returned_and_every_attempt_is_charged(seam):
    script, calls = seam
    script.extend([_Resp(429), _Resp(200, [{"symbol": "PEP", "price": 1.0}])])
    client = mod.FMPClient(api_key="k1", sleep_seconds=0)
    assert client._get("quote", {"symbol": "PEP"}) == [{"symbol": "PEP", "price": 1.0}]
    assert client.api_calls == 2
    assert [c[1]["apikey"] for c in calls] == ["k1", "k2"]
    assert all(c[0].startswith("https://financialmodelingprep.com/stable/quote") for c in calls)


def test_typed_miss_is_none_with_one_warning(seam, capsys):
    script, _ = seam
    script.append(_Resp(404))
    client = mod.FMPClient(api_key="k1", sleep_seconds=0)
    assert client._get("dividends", {"symbol": "ZZZZ"}) is None
    err = capsys.readouterr().err
    assert err.count("WARNING: FMP request failed (http_4xx:404) for dividends") == 1


def test_ctor_key_that_differs_from_env_is_scoped_to_the_call(seam):
    script, calls = seam
    script.append(_Resp(200, []))
    client = mod.FMPClient(api_key="other-key", sleep_seconds=0)
    seen = {}

    real_keys = fmp_compat.get_fmp_keys

    def keys_from_env():
        seen["primary"] = os.environ.get("FMP_API_KEY")
        return ["k1", "k2"]

    fmp_compat.get_fmp_keys = keys_from_env
    try:
        client._get("quote", {"symbol": "PEP"})
    finally:
        fmp_compat.get_fmp_keys = real_keys
    assert seen["primary"] == "other-key"
    assert os.environ.get("FMP_API_KEY") == "k1", "override must not leak past the call"


def test_standalone_raw_path_when_fmp_compat_is_absent(monkeypatch):
    monkeypatch.setattr(mod, "fmp_get_typed", None)
    client = mod.FMPClient(api_key="k1", sleep_seconds=0)

    class _Session:
        def get(self, url, params=None, headers=None, timeout=None):
            assert headers == {"apikey": "k1"} and "apikey" not in (params or {})
            return _Resp(200, [{"symbol": "PEP"}])

    client.session = _Session()
    assert client._get("quote", {"symbol": "PEP"}) == [{"symbol": "PEP"}]
    assert client.api_calls == 1
