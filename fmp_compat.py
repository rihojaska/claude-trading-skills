"""
fmp_compat.py — FMP API compatibility + resilience shim.

Responsibilities
----------------
1. **v3 → stable URL rewriting** (legacy reason this file exists): FMP deprecated
   their v3 API in August 2025. Some skills/scripts still hit `/api/v3/...` URLs;
   this shim patches `requests.get` and `requests.Session.get` to translate
   them transparently to `/stable/...` endpoints.

2. **Dual-key failover** (added 2026-04-18): Riho runs on FMP's free tier, which
   has a tight daily request cap. Two keys are registered:
     - `FMP_API_KEY`           — primary
     - `FMP_FALLBACK_API_KEY`  — secondary (separate free-tier account)

   When the primary key hits a rate-limit response, the shim transparently
   swaps in the fallback key and retries once before giving up. Rate-limit
   signals recognised:
     - HTTP 429 (Too Many Requests)
     - HTTP 403 (Forbidden / quota exhausted)
     - HTTP 402 (Payment Required — seen on some premium endpoints)
     - JSON body `{"Error Message": "...Limit Reach..."}` (200 OK but capped)

Public API
----------
- `build_url(path)`         → full stable URL (used by `dividend_projection.py`)
- `fmp_get(path, params)`   → JSON result or None, handles retry+failover
- `get_fmp_keys()`          → list of available keys in priority order
- `FMP_BASE`, `_V3_TO_STABLE`, `_translate_url`  → legacy/internal

Side effects on import
----------------------
`requests.get` and `Session.get` are monkey-patched so legacy code that builds
v3 URLs still works. This is imported for its side effects by scripts that do
`from fmp_compat import build_url` or `_load_compat_shim()`.
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import date, timedelta
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from requests import Session

FMP_BASE = "https://financialmodelingprep.com"

# ── Secrets self-load (added 2026-06-04) ──────────────────────────────────────
# Populate os.environ from a discovered `secrets.env` at import so consumers
# don't need a shell `source secrets.env` first (the Investment credential hook
# refuses that for interactive Bash, and the idiom is brittle across runner
# contexts). Self-contained by design: this shim ships in a public, standalone
# repo, so it does NOT depend on the private app's loader. setdefault semantics —
# a pre-existing env var (shell / cloud runner) always wins.


def _load_secrets_env() -> int:
    """Inject KEY=VALUE pairs from a discovered creds file into os.environ.

    Discovery order: ``$CLAUDE_PROJECT_DIR`` → each parent of this file → ``$PWD``.
    Only fills gaps (never overrides a pre-set var). Silent-safe; returns the
    count of keys injected.
    """
    from pathlib import Path

    fname = "secrets.env"
    candidates: list[Any] = []
    proj = os.environ.get("CLAUDE_PROJECT_DIR", "").strip()
    if proj:
        candidates.append(Path(proj) / fname)
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / fname)
    candidates.append(Path.cwd() / fname)
    injected = 0
    try:
        env_path = next((c for c in candidates if c.is_file()), None)
        if env_path is None:
            return 0
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val
                injected += 1
    except Exception:
        return injected
    return injected


_load_secrets_env()

# ── v3 → stable endpoint map ──────────────────────────────────────────────────

_V3_TO_STABLE = {
    "/api/v3/earning_calendar": "/stable/earnings-calendar",
    "/api/v3/profile/([^?]+)": "/stable/profile?symbol=\\1",
    "/api/v3/income-statement/([^?]+)": "/stable/income-statement?symbol=\\1",
    "/api/v3/balance-sheet-statement/([^?]+)": "/stable/balance-sheet-statement?symbol=\\1",
    "/api/v3/cash-flow-statement/([^?]+)": "/stable/cash-flow-statement?symbol=\\1",
    "/api/v3/key-metrics/([^?]+)": "/stable/key-metrics?symbol=\\1",
    "/api/v3/ratios/([^?]+)": "/stable/ratios?symbol=\\1",
    "/api/v3/stock_screener": "/stable/stock-screener",
    "/api/v3/stock-screener": "/stable/stock-screener",
    "/api/v3/historical-price-full/stock_dividend/([^?]+)": "/stable/dividends?symbol=\\1",
    "/api/v3/historical-price-full/([^?]+)": "/stable/historical-price-eod/full?symbol=\\1",
    "/api/v3/quote/([^?]+)": "/stable/quote?symbol=\\1",
    "/api/v3/dividends/([^?]+)": "/stable/dividends?symbol=\\1",
    "/api/v3/analyst-estimates/([^?]+)": "/stable/analyst-estimates?symbol=\\1",
    "/api/v3/institutional_holder/([^?]+)": "/stable/institutional-ownership/list?symbol=\\1",
    "/api/v3/institutional-holder/([^?]+)": "/stable/institutional-ownership/list?symbol=\\1",
}


def _normalise_stable_path(path: str) -> str:
    """Return the current stable path for known renamed/deprecated endpoints."""
    return path.replace("/stable/historical-price-full", "/stable/historical-price-eod/full")


def _fix_duplicate_query_markers(url: str) -> str:
    """Convert accidental second/later '?' query separators to '&'."""
    first_q = url.find("?")
    if first_q == -1:
        return url
    return url[: first_q + 1] + url[first_q + 1 :].replace("?", "&")


def _prepare_query_for_url(url: str) -> str:
    """Adjust legacy query-string params embedded directly in the URL."""
    parts = urlsplit(url)
    if not parts.query:
        return url

    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    changed = False

    if "historical-price-eod/full" in parts.path:
        if "from_date" in query and "from" not in query:
            query["from"] = query.pop("from_date")
            changed = True
        if "to_date" in query and "to" not in query:
            query["to"] = query.pop("to_date")
            changed = True
        if "timeseries" in query and ("from" not in query or "to" not in query):
            days = int(query.pop("timeseries"))
            end = date.today()
            start = end - timedelta(days=days * 2)
            query["from"] = start.isoformat()
            query["to"] = end.isoformat()
            changed = True

    if parts.path.endswith("/stable/dividends") and query.get("limit"):
        clamped = str(min(int(query["limit"]), 5))
        if clamped != query["limit"]:
            query["limit"] = clamped
            changed = True

    if not changed:
        return url
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _prepare_params_for_url(url: str, params: dict | None) -> dict:
    """Adjust legacy params to match the current stable endpoint contract."""
    prepared = dict(params or {})
    if "historical-price-eod/full" in url:
        if "from_date" in prepared and "from" not in prepared:
            prepared["from"] = prepared.pop("from_date")
        if "to_date" in prepared and "to" not in prepared:
            prepared["to"] = prepared.pop("to_date")
        if "timeseries" in prepared and ("from" not in prepared or "to" not in prepared):
            days = int(prepared.pop("timeseries"))
            end = date.today()
            start = end - timedelta(days=days * 2)
            prepared["from"] = start.isoformat()
            prepared["to"] = end.isoformat()
            prepared["_tm_limit"] = days
    if "/stable/dividends" in url and prepared.get("limit"):
        # Free-tier FMP currently allows at most 5 rows on this endpoint.
        prepared["limit"] = min(int(prepared["limit"]), 5)
    return prepared


def _normalize_eod_flat_list(data: Any, symbol: str | None, limit: int | None = None) -> Any:
    """Convert stable historical EOD flat-list data to the legacy v3 shape."""
    if not isinstance(data, list):
        return data
    if not data:
        return {"symbol": symbol, "historical": []}

    norm_target = symbol.replace("-", ".") if symbol else None
    matched_symbol = None
    historical = []
    for row in data:
        if not isinstance(row, dict) or "date" not in row:
            continue
        row_symbol = row.get("symbol") or symbol
        if norm_target and row_symbol and row_symbol.replace("-", ".") != norm_target:
            continue
        matched_symbol = matched_symbol or row_symbol
        historical.append({k: v for k, v in row.items() if k != "symbol"})

    if limit is not None and limit > 0:
        historical = historical[:limit]
    return {"symbol": matched_symbol or symbol, "historical": historical}


def _normalize_dividends_flat_list(data: Any, symbol: str | None) -> Any:
    """Convert stable dividend rows to the legacy stock_dividend shape."""
    if not isinstance(data, list):
        return data
    historical = []
    for row in data:
        if not isinstance(row, dict) or not row.get("date"):
            continue
        dividend = row.get("dividend", row.get("adjDividend", row.get("dividendAmount", 0)))
        try:
            dividend = float(dividend or 0)
        except (TypeError, ValueError):
            dividend = 0
        historical.append(
            {
                "date": row.get("date"),
                "label": row.get("label", row.get("date")),
                "adjDividend": dividend,
                "dividend": dividend,
                "recordDate": row.get("recordDate"),
                "paymentDate": row.get("paymentDate"),
                "declarationDate": row.get("declarationDate"),
            }
        )
    return {"symbol": symbol, "historical": historical}


def _translate_url(url: str) -> str:
    """Rewrite v3 paths to stable paths. No-op for already-stable URLs."""
    if FMP_BASE not in url:
        return url
    path_part = url.replace(FMP_BASE, "")
    for pattern, replacement in _V3_TO_STABLE.items():
        if re.search(pattern, path_part):
            new_path = re.sub(pattern, replacement, path_part)
            return _prepare_query_for_url(
                _fix_duplicate_query_markers(_normalise_stable_path(FMP_BASE + new_path))
            )
    if "/api/v3/" in url:
        return _prepare_query_for_url(
            _fix_duplicate_query_markers(_normalise_stable_path(url.replace("/api/v3/", "/stable/")))
        )
    return _prepare_query_for_url(_fix_duplicate_query_markers(_normalise_stable_path(url)))


def build_url(path: str) -> str:
    """
    Return full stable URL for a relative path.
    Examples:
      build_url("dividends?symbol=AAPL&limit=8")
        → https://financialmodelingprep.com/stable/dividends?symbol=AAPL&limit=8
      build_url("/stable/profile?symbol=AAPL")
        → https://financialmodelingprep.com/stable/profile?symbol=AAPL
    """
    if path.startswith("http"):
        return _normalise_stable_path(_translate_url(path))
    if path.startswith("/stable/") or path.startswith("/api/"):
        return _normalise_stable_path(_translate_url(FMP_BASE + path))
    # Bare path — prepend /stable/
    return _prepare_query_for_url(_normalise_stable_path(FMP_BASE + "/stable/" + path.lstrip("/")))


# ── Dual-key failover ─────────────────────────────────────────────────────────

_RATE_LIMIT_STATUS = {429, 403, 402}
_RATE_LIMIT_MARKERS = (
    "limit reach",
    "rate limit",
    "quota",
    "exceeded",
    "subscribe",
    "upgrade",
)


def get_fmp_keys() -> list[str]:
    """
    Return available FMP API keys in priority order.
    Primary first, fallback second. Empty list if neither set.
    """
    keys: list[str] = []
    primary = os.environ.get("FMP_API_KEY", "").strip()
    fallback = os.environ.get("FMP_FALLBACK_API_KEY", "").strip()
    if primary:
        keys.append(primary)
    if fallback and fallback != primary:
        keys.append(fallback)
    return keys


def _looks_like_quota_error(response: requests.Response) -> bool:
    """True if the response body signals a rate-limit hit despite 200 OK."""
    try:
        data = response.json()
    except Exception:
        return False
    if isinstance(data, dict):
        msg = str(data.get("Error Message", "") or data.get("error", "")).lower()
        if any(marker in msg for marker in _RATE_LIMIT_MARKERS):
            return True
    return False


def fmp_get(
    path: str,
    params: dict | None = None,
    *,
    timeout: int = 15,
    max_retries_per_key: int = 2,
) -> Any | None:
    """
    GET an FMP endpoint with automatic key failover.

    Tries primary key first; on rate-limit signals, swaps to fallback key.
    Returns parsed JSON (list or dict), or None on unrecoverable failure.

    Rate-limit detection:
      - HTTP 429 / 403 / 402 → swap key, retry
      - HTTP 200 with JSON "Error Message" containing limit/quota markers → swap
      - Within a single key, retries up to `max_retries_per_key` with 5s backoff

    Usage:
        data = fmp_get("/stable/key-metrics-ttm", {"symbol": "AAPL"})
        data = fmp_get("dividends", {"symbol": "AAPL", "limit": 8})
    """
    keys = get_fmp_keys()
    if not keys:
        if not getattr(fmp_get, "_warned_no_keys", False):
            print(
                "WARN: fmp_get() — no FMP keys in environment. "
                "Set FMP_API_KEY (and optionally FMP_FALLBACK_API_KEY), "
                "or source an env file that uses `export`.",
                file=sys.stderr,
            )
            fmp_get._warned_no_keys = True
        return None

    url = build_url(path) if not path.startswith("http") else _translate_url(path)
    base_params = _prepare_params_for_url(url, params)
    historical_limit = base_params.pop("_tm_limit", None)

    for key_idx, key in enumerate(keys):
        call_params = dict(base_params)
        call_params["apikey"] = key

        for attempt in range(max_retries_per_key):
            try:
                r = _original_get(url, params=call_params, timeout=timeout)
            except requests.RequestException:
                # Network-level failure — retry within same key, then move on
                time.sleep(2 * (attempt + 1))
                continue

            # Hard rate-limit signals → swap key
            if r.status_code in _RATE_LIMIT_STATUS:
                break  # exit inner loop, advance to next key

            if r.status_code >= 500:
                # Server-side transient — backoff within same key
                time.sleep(5 * (attempt + 1))
                continue

            if not r.ok:
                # 4xx other than rate-limit — probably a real error, bail
                return None

            # 200 OK — check for soft quota error
            if _looks_like_quota_error(r):
                break  # swap key

            try:
                data = r.json()
            except ValueError:
                return None
            if "/stable/dividends" in url and isinstance(data, list):
                return _normalize_dividends_flat_list(data, base_params.get("symbol"))
            if "historical-price-eod/full" in url and isinstance(data, list):
                return _normalize_eod_flat_list(data, base_params.get("symbol"), historical_limit)
            return data

        # If we got here, inner loop ended without return → try next key
        if key_idx < len(keys) - 1:
            # brief pause before retrying with fallback
            time.sleep(1.0)

    return None


# ── Monkey-patch requests.get / Session.get for v3 URL rewriting ─────────────
#
# Scripts that hit v3 URLs directly (via requests.get) will be transparently
# rewritten. Note: this does NOT add key failover to direct requests.get calls —
# only URL rewriting. Use fmp_get() for failover.

_original_get = requests.get
_original_session_get = Session.get


def _patched_get(url, **kwargs):
    translated = _translate_url(url)
    if "params" in kwargs:
        kwargs["params"] = _prepare_params_for_url(translated, kwargs.get("params"))
        kwargs["params"].pop("_tm_limit", None)
    return _original_get(translated, **kwargs)


def _patched_session_get(self, url, **kwargs):
    translated = _translate_url(url)
    if "params" in kwargs:
        kwargs["params"] = _prepare_params_for_url(translated, kwargs.get("params"))
        kwargs["params"].pop("_tm_limit", None)
    return _original_session_get(self, translated, **kwargs)


requests.get = _patched_get
Session.get = _patched_session_get
