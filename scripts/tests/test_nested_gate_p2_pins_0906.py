"""S-PULSE-3 nested codex gate P2 pins (2026-09-06).

1. `fmp_compat.request_count()` counts every HTTP attempt (retries and key
   failover), and the two budgeted clients (stockbee-20pct-study,
   stockbee-episodic-pivot-analyzer) charge the DELTA around one `fmp_get`
   call — never a flat 1 — so `--max-api-calls` cannot be exceeded silently.
2. The standalone (no `fmp_compat`) EOD paths of find_pairs / analyze_spread /
   analyze_downtrends / postmortem_recorder refuse a malformed flat list
   WHOLE, mirroring the shared shim (WPP-20260902-002), instead of filtering.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from unittest.mock import patch

import pytest
import requests

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fmp_compat  # noqa: E402

SKILLS = REPO_ROOT / "skills"


def _load(relpath: str, name: str):
    path = SKILLS / relpath
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve cls.__module__ through sys.modules
    spec.loader.exec_module(mod)
    return mod


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = ""
        self.headers = {}
        self.url = "https://financialmodelingprep.com/stable/x"
        self._payload = payload

    def json(self):
        return self._payload


# ── 1. request accounting ────────────────────────────────────────────────────

def test_request_count_counts_every_attempt(monkeypatch):
    calls = {"n": 0}

    def flaky(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("blip")
        return _Resp(200, {"ok": True})

    monkeypatch.setattr(fmp_compat, "_original_get", flaky)
    monkeypatch.setattr(fmp_compat, "get_fmp_keys", lambda: ["k1"])
    monkeypatch.setattr(fmp_compat.time, "sleep", lambda *_: None)
    before = fmp_compat.request_count()
    assert fmp_compat.fmp_get("/stable/profile", {"symbol": "AAPL"}) == {"ok": True}
    assert fmp_compat.request_count() - before == 2 == calls["n"]


@pytest.mark.parametrize(
    "relpath, name, counter",
    [
        ("stockbee-20pct-study/scripts/run_20pct_study.py", "run_20pct_study_p2", "api_calls_made"),
        ("stockbee-episodic-pivot-analyzer/scripts/analyze_ep.py", "analyze_ep_p2", "api_calls"),
    ],
)
def test_budget_charges_the_request_delta(relpath, name, counter, monkeypatch):
    mod = _load(relpath, name)
    state = {"n": 0}

    def fake_fmp_get(url, params=None, timeout=None, **kw):
        state["n"] += 2  # one retry + one key failover = 2 HTTP attempts
        return {"symbol": "AAPL"}

    monkeypatch.setattr(mod, "fmp_get", fake_fmp_get)
    monkeypatch.setattr(mod, "request_count", lambda: state["n"])
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    client = mod.FMPClient(api_key="k", max_api_calls=10)
    client._get("https://financialmodelingprep.com/stable/profile", {"symbol": "AAPL"})
    assert getattr(client, counter) == 2


# ── 2. standalone paths refuse malformed lists whole ─────────────────────────

MALFORMED = [{"date": "2026-09-04", "close": 1.0}, "junk", {"close": 2.0}]
CLEAN = [{"date": "2026-09-04", "close": 1.0}, {"date": "2026-09-03", "close": 2.0}]


@pytest.mark.parametrize(
    "relpath, name",
    [
        ("pair-trade-screener/scripts/find_pairs.py", "find_pairs_p2"),
        ("pair-trade-screener/scripts/analyze_spread.py", "analyze_spread_p2"),
    ],
)
def test_pair_screener_standalone_refuses_whole(relpath, name, monkeypatch, capsys):
    try:
        mod = _load(relpath, name)
    except ModuleNotFoundError as e:  # scipy is absent in this venv (pre-existing)
        pytest.skip(f"pre-existing import gap: {e}")
    monkeypatch.setattr(mod, "fmp_get", None, raising=False)
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp(200, MALFORMED))
    assert mod._fetch_raw_historical("AAPL", "k") is None
    assert "refusing the whole series" in capsys.readouterr().err
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp(200, CLEAN))
    assert mod._fetch_raw_historical("AAPL", "k") == {"symbol": "AAPL", "historical": CLEAN}


def test_downtrends_standalone_refuses_whole(monkeypatch, capsys):
    mod = _load("downtrend-duration-analyzer/scripts/analyze_downtrends.py", "analyze_downtrends_p2")
    monkeypatch.setattr(mod, "fmp_get", None, raising=False)
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp(200, MALFORMED))
    out = mod.fetch_historical_prices("AAPL", "k", "2026-08-01", "2026-09-04")
    assert out is None or len(out) == 0
    assert "refusing the whole series" in capsys.readouterr().err


def test_postmortem_standalone_refuses_whole(monkeypatch, capsys):
    mod = _load("signal-postmortem/scripts/postmortem_recorder.py", "postmortem_recorder_p2")
    monkeypatch.setattr(mod, "fmp_get", None, raising=False)
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp(200, MALFORMED))
    assert mod.fetch_price_data("AAPL", "2026-08-01", "2026-09-04", "k") == {}
    assert "refusing the whole series" in capsys.readouterr().err
