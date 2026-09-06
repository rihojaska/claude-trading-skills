"""`fmp_compat.fmp_get_typed` — typed miss contract (WPP-20260601-020).

Pins: every collapse-to-None path of the transport returns `(None, reason)`
with a base token in `FMP_REASONS`; success returns `(data, None)`; the
invariant `data is None <=> reason is not None` holds; `fmp_get` stays the
bare-`None` back-compat wrapper (same signature, same `_warned_no_keys`
attribute, one attempt per fake response); a rate-limit fall-through carries
the LAST signal seen and network/5xx-only fall-through is `unreachable`, never
`rate_limited` (codex plan review r1 P1, 2026-09-06).
"""
from __future__ import annotations

import inspect
import pathlib
import sys

import pytest
import requests

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fmp_compat  # noqa: E402


class _Resp:
    def __init__(self, status_code=200, payload=None, text="", raw_json=False):
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = text
        self.headers = {}
        self.url = "https://financialmodelingprep.com/stable/x"
        self._payload = payload
        self._raw_json = raw_json

    def json(self):
        if self._raw_json:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def seam(monkeypatch):
    """Script the transport: each entry is a `_Resp` or an exception to raise."""
    script: list = []
    calls: list = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        nxt = script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    monkeypatch.setattr(fmp_compat, "_original_get", fake_get)
    monkeypatch.setattr(fmp_compat, "get_fmp_keys", lambda: ["k1", "k2"])
    monkeypatch.setattr(fmp_compat.time, "sleep", lambda *_a, **_k: None)
    return script, calls


def _kind(reason):
    return fmp_compat.reason_kind(reason)


# ── success / invariant ──────────────────────────────────────────────────────

def test_success_returns_data_and_no_reason(seam):
    script, calls = seam
    script.append(_Resp(200, [{"symbol": "AAPL", "price": 1.0}]))
    data, reason = fmp_compat.fmp_get_typed("/stable/quote", {"symbol": "AAPL"})
    assert data == [{"symbol": "AAPL", "price": 1.0}] and reason is None
    assert len(calls) == 1 and calls[0][1]["apikey"] == "k1"


def test_empty_list_is_a_success_not_a_miss(seam):
    script, _ = seam
    script.append(_Resp(200, []))
    assert fmp_compat.fmp_get_typed("/stable/quote", {"symbol": "ZZZZ"}) == ([], None)


def test_json_null_is_null_payload(seam):
    script, _ = seam
    script.append(_Resp(200, None))
    assert fmp_compat.fmp_get_typed("/stable/quote", {"symbol": "AAPL"}) == (None, "null_payload")


def test_bad_json(seam):
    script, _ = seam
    script.append(_Resp(200, raw_json=True))
    assert fmp_compat.fmp_get_typed("/stable/quote", {"symbol": "AAPL"}) == (None, "bad_json")


def test_plain_4xx_bails_after_one_attempt_with_status(seam):
    script, calls = seam
    script.append(_Resp(404))
    data, reason = fmp_compat.fmp_get_typed("/stable/quote", {"symbol": "AAPL"})
    assert (data, reason) == (None, "http_4xx:404")
    assert _kind(reason) == "http_4xx" and len(calls) == 1


def test_no_keys_warns_once_on_fmp_get_attribute(monkeypatch, capsys):
    monkeypatch.setattr(fmp_compat, "get_fmp_keys", lambda: [])
    monkeypatch.setattr(fmp_compat.fmp_get, "_warned_no_keys", False, raising=False)
    assert fmp_compat.fmp_get_typed("/stable/quote", {"symbol": "AAPL"}) == (None, "no_keys")
    assert fmp_compat.fmp_get("/stable/quote", {"symbol": "AAPL"}) is None
    err = capsys.readouterr().err
    assert err.count("no FMP keys in environment") == 1
    assert fmp_compat.fmp_get._warned_no_keys is True


def test_attempt_budget(seam, capsys):
    script, calls = seam
    script.extend([requests.ConnectionError("down")] * 4)
    data, reason = fmp_compat.fmp_get_typed("/stable/profile", {"symbol": "AAPL"}, max_attempts=1)
    assert (data, reason) == (None, "attempt_budget") and len(calls) == 1
    assert "attempt budget (1) exhausted" in capsys.readouterr().err


# ── rate-limit vs unreachable ────────────────────────────────────────────────

def test_network_only_fall_through_is_unreachable_not_rate_limited(seam):
    script, calls = seam
    script.extend([requests.ConnectionError("down")] * 4)  # 2 keys x 2 retries
    data, reason = fmp_compat.fmp_get_typed("/stable/profile", {"symbol": "AAPL"})
    assert (data, reason) == (None, "unreachable") and len(calls) == 4


def test_5xx_only_fall_through_is_unreachable(seam):
    script, _ = seam
    script.extend([_Resp(503)] * 4)
    assert fmp_compat.fmp_get_typed("/stable/profile", {"symbol": "AAPL"}) == (None, "unreachable")


def test_429_on_both_keys_is_rate_limited_429(seam):
    script, calls = seam
    script.extend([_Resp(429), _Resp(429)])
    data, reason = fmp_compat.fmp_get_typed("/stable/profile", {"symbol": "AAPL"})
    assert (data, reason) == (None, "rate_limited:429")
    assert [c[1]["apikey"] for c in calls] == ["k1", "k2"]


def test_402_entitlement_on_both_keys_is_rate_limited_402_not_429(seam):
    script, _ = seam
    script.extend([_Resp(402), _Resp(402)])
    _, reason = fmp_compat.fmp_get_typed("/stable/key-metrics", {"symbol": "ACN"})
    assert reason == "rate_limited:402" and _kind(reason) == "rate_limited"


def test_soft_quota_then_network_carries_the_last_signal(seam):
    script, _ = seam
    script.extend([
        _Resp(200, {"Error Message": "Limit Reach . Please upgrade your plan"}),
        requests.ConnectionError("down"), requests.ConnectionError("down"),
    ])
    _, reason = fmp_compat.fmp_get_typed("/stable/profile", {"symbol": "AAPL"})
    assert reason == "rate_limited:quota"


def test_failover_success_after_429_is_a_plain_success(seam):
    script, calls = seam
    script.extend([_Resp(429), _Resp(200, [{"symbol": "AAPL"}])])
    assert fmp_compat.fmp_get_typed("/stable/profile", {"symbol": "AAPL"}) == ([{"symbol": "AAPL"}], None)
    assert calls[1][1]["apikey"] == "k2"


# ── normalizer refusal ───────────────────────────────────────────────────────

def test_eod_fold_refusing_a_malformed_series_is_normalize_refused(seam, capsys):
    script, _ = seam
    script.append(_Resp(200, [{"symbol": "AAPL", "date": "2026-09-05", "close": 1.0}, {"symbol": "AAPL"}]))
    data, reason = fmp_compat.fmp_get_typed(
        "/stable/historical-price-eod/full", {"symbol": "AAPL", "from": "2026-01-01", "to": "2026-09-05"}
    )
    assert data is None and reason == "normalize_refused"
    assert "refusing the whole series" in capsys.readouterr().err


def test_eod_fold_success_is_typed_success(seam):
    script, _ = seam
    script.append(_Resp(200, [{"symbol": "AAPL", "date": "2026-09-05", "close": 1.0}]))
    data, reason = fmp_compat.fmp_get_typed(
        "/stable/historical-price-eod/full", {"symbol": "AAPL", "from": "2026-01-01", "to": "2026-09-05"}
    )
    assert reason is None and data["symbol"] == "AAPL" and len(data["historical"]) == 1


# ── contract: base tokens, invariant, wrapper ────────────────────────────────

@pytest.mark.parametrize("reason", [
    "no_keys", "attempt_budget", "http_4xx:404", "bad_json", "null_payload",
    "rate_limited:429", "rate_limited:quota", "unreachable", "normalize_refused",
])
def test_every_reason_base_token_is_declared(reason):
    assert fmp_compat.reason_kind(reason) in fmp_compat.FMP_REASONS


def test_reason_kind_of_none_is_none():
    assert fmp_compat.reason_kind(None) is None


def test_fmp_get_is_the_bare_none_wrapper_with_the_same_signature(seam):
    script, _ = seam
    assert inspect.signature(fmp_compat.fmp_get) == inspect.signature(fmp_compat.fmp_get_typed).replace(
        return_annotation=inspect.signature(fmp_compat.fmp_get).return_annotation
    )
    script.extend([_Resp(429), _Resp(429)])
    assert fmp_compat.fmp_get("/stable/profile", {"symbol": "AAPL"}) is None
    script.append(_Resp(200, {"ok": 1}))
    assert fmp_compat.fmp_get("/stable/profile", {"symbol": "AAPL"}) == {"ok": 1}
