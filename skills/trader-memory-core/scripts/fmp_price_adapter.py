"""Thin FMP price adapter for MAE/MFE calculation.

Single-purpose: fetch daily close prices.  Does not reuse existing
fmp_client modules (which vary in return shape across skills).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

SKILL_ROOT = Path(__file__).resolve().parents[3]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

try:
    from fmp_compat import fmp_get, key_override
except ImportError:  # standalone .skill install without the repo-root module
    fmp_get = None
    key_override = None
    print(
        "NOTE: fmp_compat not importable — FMP calls go direct to /stable; "
        "dual-key failover unavailable.",
        file=sys.stderr,
    )

# /stable/historical-price-eod/full only. FMP retired v3 for keys issued
# after 2025-08-31, and a v3 URL requested through fmp_compat is rewritten
# straight back to the equivalent /stable endpoint — so a second "v3 rung"
# here was never a distinct upstream (WPP-20260831-004 / WPP-20260901-016).
_FMP_HIST_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"


class FMPPriceAdapter:
    """Fetch daily adjusted close prices from FMP API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP API key required. Set FMP_API_KEY env var or pass api_key.")

    def get_daily_closes(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        """Return daily close prices, oldest first.

        Args:
            ticker: Stock symbol (e.g., "AAPL").
            from_date: Start date "YYYY-MM-DD".
            to_date: End date "YYYY-MM-DD".

        Returns:
            List of {"date": "YYYY-MM-DD", "close": float}, oldest first.

        Raises:
            urllib.error.URLError: On network/API errors (only if all endpoints fail).
            ValueError: On invalid response.
        """
        params = {"symbol": ticker, "from": from_date, "to": to_date}

        if fmp_get is not None:
            with key_override(self.api_key):
                data = fmp_get(_FMP_HIST_URL, params=params, timeout=30)
            historical = self._extract_historical(data, ticker) if data is not None else []
            return self._to_closes(historical, ticker, from_date, to_date)

        url = f"{_FMP_HIST_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"apikey": self.api_key})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            logger.error("FMP API error for %s: %s", ticker, e)
            raise

        historical = self._extract_historical(data, ticker)
        return self._to_closes(historical, ticker, from_date, to_date)

    def _to_closes(
        self, historical: list[dict], ticker: str, from_date: str, to_date: str
    ) -> list[dict]:
        if not historical:
            logger.warning("No price data returned for %s (%s to %s)", ticker, from_date, to_date)
            return []
        # FMP returns newest first; reverse to oldest first.
        # Stable EOD endpoint no longer exposes `adjClose`; fall back to `close`.
        # Patched 2026-05-22 (stable shape).
        return [
            {"date": item["date"], "close": item.get("adjClose") or item["close"]}
            for item in reversed(historical)
            if "date" in item and ("adjClose" in item or "close" in item)
        ]

    @staticmethod
    def _extract_historical(data, ticker: str) -> list[dict]:
        """Extract historical array from FMP response (stable list / dict)."""
        # New stable EOD endpoint returns a flat list of dicts directly.
        # Patched 2026-05-22 (stable shape).
        if isinstance(data, list):
            norm = ticker.replace("-", ".")
            return [
                row
                for row in data
                if isinstance(row, dict) and row.get("symbol", ticker).replace("-", ".") == norm
            ]
        if not isinstance(data, dict):
            return []
        if "historicalStockList" in data:
            # Legacy batch shape (standalone path; the shared shim folds it in
            # fmp_compat): EXACT normalized symbol match only, never a
            # symbol-less entry (codex nested gate r4).
            norm = ticker.replace("-", ".")
            for entry in data.get("historicalStockList") or []:
                if isinstance(entry, dict) and str(entry.get("symbol") or "").replace("-", ".") == norm:
                    return entry.get("historical") or []
            return []
        return data.get("historical") or []
