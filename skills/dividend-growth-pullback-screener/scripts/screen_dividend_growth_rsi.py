#!/usr/bin/env python3
"""
Dividend Growth Pullback Screener using FINVIZ + Financial Modeling Prep API

Two-stage screening approach:
1. FINVIZ Elite API: Pre-screen stocks with dividend growth + RSI criteria (fast, cost-effective)
2. FMP API: Detailed analysis of pre-screened candidates (comprehensive)

Screens for high-quality dividend growth stocks (12%+ dividend CAGR, 1.5%+ yield)
that are experiencing temporary pullbacks identified by RSI oversold conditions (RSI ≤40).

Usage:
    # Two-stage screening with FINVIZ (RECOMMENDED)
    python3 screen_dividend_growth_rsi.py --use-finviz

    # FMP-only screening (original method)
    python3 screen_dividend_growth_rsi.py

Environment variables:
    export FMP_API_KEY=your_fmp_key_here
    export FINVIZ_API_KEY=your_finviz_key_here  # Required for --use-finviz
"""

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import requests

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


def _yf_dividend_history(symbol: str) -> Optional[dict]:
    """Fetch dividend history from yfinance in legacy FMP stock_dividend shape."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        dividends = yf.Ticker(symbol).dividends
    except Exception:
        return None

    if dividends is None or dividends.empty:
        return None

    # yfinance>=1.3.0 returns .dividends as a Series; <=1.2.2 returns a 1-col
    # DataFrame whose .items() yields (col_name, Series) — not (date, value) —
    # crashing the loop below on float(Series). Squeeze a DataFrame to its single
    # column. No-op on a Series (ndim 1), so the result is identical on 1.3.0.
    if getattr(dividends, "ndim", 1) == 2:
        dividends = dividends.iloc[:, 0]

    rows = []
    for idx, value in dividends.items():
        dividend = float(value or 0)
        rows.append(
            {
                "date": idx.date().isoformat(),
                "label": idx.date().isoformat(),
                "adjDividend": dividend,
                "dividend": dividend,
            }
        )
    return {"symbol": symbol, "historical": rows, "data_source": "yfinance"}


def _yf_quote_profile(symbol: str) -> Optional[dict]:
    """Fetch quote/profile-like candidate data from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        ticker = yf.Ticker(symbol)
        fast_info = getattr(ticker, "fast_info", {}) or {}
        info = ticker.info or {}
    except Exception:
        return None

    price = fast_info.get("last_price") or info.get("regularMarketPrice") or info.get("currentPrice")
    if not price:
        return None
    return {
        "symbol": symbol,
        "price": float(price),
        "marketCap": fast_info.get("market_cap") or info.get("marketCap") or 0,
        "companyName": info.get("longName") or info.get("shortName") or symbol,
        "name": info.get("shortName") or info.get("longName") or symbol,
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", ""),
        "data_source": "yfinance",
    }


def _yf_price_history(symbol: str, days: int = 30) -> list[dict]:
    """Daily closes from yfinance in the FMP historical-price-eod row shape:
    ``{"date", "close", "data_source": "yfinance"}``, most-recent-first, at most
    ``days`` rows.

    Free-tier leg for ``get_historical_prices``: /stable/historical-price-eod/full
    is symbol-whitelist gated (402 on nearly every non-US name), which left the
    dividend screeners RSI-blind — every candidate died on "Insufficient price
    data" and every monthly run reported 0 qualified (WPP-20260906-007,
    2026-09-06). Raw closes (auto_adjust=False) to match FMP's ``close``; RSI is
    scale-invariant, so a uniformly pence-quoted LSE series needs no fold here
    (a MIXED-unit series would — yfinance quotes one unit per line). Fail-closed:
    ``[]`` on ImportError / fetch error / empty frame / no positive closes —
    never a synthetic, padded or provider-spliced series.
    """
    try:
        import yfinance as yf
    except ImportError:
        return []
    start = (datetime.now().date() - timedelta(days=days * 2 + 10)).isoformat()
    try:
        hist = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=False)
    except Exception:
        return []
    if hist is None or getattr(hist, "empty", True) or "Close" not in hist.columns:
        return []
    # Validity contract (codex plan r1 P1): finite positive raw closes only, one
    # row per exchange-local session (first wins), completed sessions only (a
    # bar dated today is the live partial session while any venue is open),
    # newest first, no filling, no splicing with FMP rows.
    today = datetime.now().date().isoformat()
    by_date: dict[str, float] = {}
    for idx, close in hist["Close"].items():
        try:
            value = float(close)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or value <= 0:  # NaN / ±inf / non-positive bar
            continue
        try:
            day = idx.date().isoformat()
        except AttributeError:
            continue
        if day >= today or day in by_date:
            continue
        by_date[day] = value
    rows = [{"date": d, "close": by_date[d], "data_source": "yfinance"} for d in sorted(by_date, reverse=True)]
    return rows[:days]


# yfinance annual-statement row labels -> the FMP field vocabulary the analyzer
# reads. Only fields the growth screen consumes; an absent label is an absent
# key (the analyzer's `.get(key, 0)` semantics for a sparse FMP row), never 0
# fabricated in. First label present wins where two map to one key.
_YF_STATEMENT_FIELDS = {
    "income_stmt": [
        ("Total Revenue", "revenue"),
        ("Diluted EPS", "eps"),
        ("Net Income", "netIncome"),
    ],
    "balance_sheet": [
        ("Total Debt", "totalDebt"),
        ("Stockholders Equity", "totalStockholdersEquity"),
        ("Current Assets", "totalCurrentAssets"),
        ("Current Liabilities", "totalCurrentLiabilities"),
    ],
    "cashflow": [
        ("Cash Dividends Paid", "dividendsPaid"),
        ("Common Stock Dividend Paid", "dividendsPaid"),
        ("Free Cash Flow", "freeCashFlow"),
        ("Operating Cash Flow", "operatingCashFlow"),
        ("Capital Expenditure", "capitalExpenditure"),
        ("Net Income From Continuing Operations", "netIncome"),
        ("Net Income", "netIncome"),
        ("Depreciation And Amortization", "depreciationAndAmortization"),
    ],
}


def _yf_statements(symbol: str, kind: str, limit: int = 5) -> list[dict]:
    """Annual statements from yfinance in the FMP list-of-dicts shape (newest
    first, ``date`` + mapped fields + ``data_source``). Free-tier leg for the
    post-RSI fundamentals gates — the FMP statement endpoints are whitelist
    gated, so after the price leg every non-US name still died on "Financial
    health concerns" with an EMPTY balance sheet (codex plan r1 P1,
    WPP-20260906-007). Fail-closed: ``[]`` on ImportError / error / empty frame."""
    try:
        import yfinance as yf
    except ImportError:
        return []
    try:
        frame = getattr(yf.Ticker(symbol), kind)
    except Exception:
        return []
    if frame is None or getattr(frame, "empty", True):
        return []
    rows: list[dict] = []
    for col in list(frame.columns)[:limit]:
        try:
            day = col.date().isoformat()
        except AttributeError:
            continue
        row: dict = {"date": day, "data_source": "yfinance"}
        for label, key in _YF_STATEMENT_FIELDS[kind]:
            if key in row or label not in frame.index:
                continue
            try:
                value = float(frame.loc[label, col])
            except (TypeError, ValueError, KeyError):
                continue
            if not math.isfinite(value):  # NaN / ±inf (nested gate r2 P2)
                continue
            row[key] = value
        rows.append(row)
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows


_BALANCE_SHEET_REQUIRED = ("totalDebt", "totalStockholdersEquity", "totalCurrentAssets", "totalCurrentLiabilities")


def _balance_sheet_complete(row: dict) -> bool:
    """Every field `analyze_financial_health` scores must be a finite number —
    a sparse row (yfinance omits Current Assets/Liabilities for financials)
    would otherwise default to None ratios and a free `financially_healthy`."""
    for key in _BALANCE_SHEET_REQUIRED:
        value = row.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            return False
    return True


def _yf_key_metrics(symbol: str) -> list[dict]:
    """Single key-metrics row from yfinance ``info`` in FMP vocabulary (decimal
    ratios, as FMP serves them). Fail-closed: ``[]``."""
    try:
        import yfinance as yf
    except ImportError:
        return []
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        return []
    row: dict = {"date": datetime.now().date().isoformat(), "data_source": "yfinance"}
    for src, key in (("trailingPE", "peRatio"), ("priceToBook", "pbRatio"),
                     ("returnOnEquity", "roe"), ("profitMargins", "netProfitMargin"),
                     ("payoutRatio", "payoutRatio")):
        value = info.get(src)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            row[key] = float(value)
    return [row] if len(row) > 2 else []


def _valid_ratio(value: Any) -> Optional[float]:
    """A P/E or P/B that can feed a verdict: finite and > 0, else None.

    Validity only — screening eligibility (pe_max / pb_max) is a separate
    question. A missing or non-positive ratio must never masquerade as a
    cheap one (WPP-20260603-031; codex plan review r1 P1/P2, 2026-09-06).
    """
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or x <= 0:
        return None
    return x


def _format_outcomes(analyzed: int, outcomes: dict, unavailable_reasons: dict, unanalyzed: int) -> str:
    """One machine-readable end-of-run line (see the loop's outcome accounting)."""
    reasons = ", ".join(f"{k}={v}" for k, v in sorted(unavailable_reasons.items()))
    return (
        f"Outcomes: analyzed {analyzed} · qualified {outcomes['qualified']} · "
        f"rejected_by_criteria {outcomes['rejected_by_criteria']} · "
        f"unavailable_input {outcomes['unavailable_input']}"
        + (f" [{reasons}]" if reasons else "")
        + f" · unanalyzed {unanalyzed}"
    )


def _report_coverage(selected: int, attempted: int, priced: list[str], unpriceable: list[str]) -> None:
    """One end-of-run stderr line so a thin result set is never mistaken for a
    thin universe: symbols never attempted (rate-limit break) are reported
    apart from symbols attempted and un-priceable (WPP-20260603-031)."""
    unattempted = selected - attempted
    print(
        f"Coverage: selected {selected} · attempted {attempted} · priced {len(priced)} · "
        f"un-priceable {len(unpriceable)}"
        + (f" [{', '.join(unpriceable)}]" if unpriceable else "")
        + f" · unattempted {unattempted}"
        + (" (rate-limit break)" if unattempted else ""),
        file=sys.stderr,
    )


class FINVIZClient:
    """Client for FINVIZ Elite API"""

    BASE_URL = "https://elite.finviz.com/export.ashx"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def screen_stocks(self) -> set[str]:
        """
        Screen stocks using FINVIZ Elite API with predefined criteria

        Criteria for dividend growth pullback opportunities (Balanced):
        - Market cap: Mid-cap or higher
        - Dividend yield: 0.5-3% (captures dividend growers without REITs/utilities)
        - Dividend growth (3Y): 10%+ (we'll verify 12%+ with FMP)
        - EPS growth (3Y): 5%+ (positive earnings momentum)
        - Sales growth (3Y): 5%+ (positive revenue momentum)
        - RSI (14): Under 40 (oversold/pullback)
        - Geography: USA

        Returns:
            Set of stock symbols
        """
        # Build filter string in FINVIZ format: key_value,key_value,...
        # Balanced criteria: Div Growth 10%+, EPS/Sales Growth 5%+ (30-40 candidates expected)
        filters = "cap_midover,fa_div_0.5to3,fa_divgrowth_3yo10,fa_eps3years_o5,fa_sales3years_o5,geo_usa,ta_rsi_os40"

        params = {
            "v": "151",  # View type
            "f": filters,  # Filter conditions
            "ft": "4",  # File type: CSV export
            "auth": self.api_key,
        }

        try:
            print("Fetching pre-screened stocks from FINVIZ Elite API...", file=sys.stderr)
            print(
                "FINVIZ Filters: Div Yield 0.5-3%, Div Growth 10%+, EPS Growth 5%+, Sales Growth 5%+, RSI <40",
                file=sys.stderr,
            )
            response = self.session.get(self.BASE_URL, params=params, timeout=30)

            if response.status_code == 200:
                # Parse CSV response
                csv_content = response.content.decode("utf-8")
                reader = csv.DictReader(io.StringIO(csv_content))

                symbols = set()
                for row in reader:
                    # FINVIZ CSV has 'Ticker' column
                    ticker = row.get("Ticker", "").strip()
                    if ticker:
                        symbols.add(ticker)

                print(f"✅ FINVIZ returned {len(symbols)} pre-screened stocks", file=sys.stderr)
                return symbols

            elif response.status_code == 401 or response.status_code == 403:
                print(
                    "ERROR: FINVIZ API authentication failed. Check your API key.", file=sys.stderr
                )
                print(f"Status code: {response.status_code}", file=sys.stderr)
                return set()
            else:
                print(f"ERROR: FINVIZ API request failed: {response.status_code}", file=sys.stderr)
                return set()

        except requests.exceptions.RequestException as e:
            print(f"ERROR: FINVIZ request exception: {e}", file=sys.stderr)
            return set()


# --- FMP historical endpoint: /stable only. ---
# FMP retired v3 for keys issued after 2025-08-31, and a v3 URL requested
# through fmp_compat is rewritten straight back to the equivalent /stable
# endpoint (`_V3_TO_STABLE` in the repo-root fmp_compat.py) — so the old "v3
# fallback" rung was never a distinct upstream, only a second rate-limited
# query of the SAME endpoint (WPP-20260831-004).
_FMP_HIST_ENDPOINTS = [
    "https://financialmodelingprep.com/stable/historical-price-eod/full",
]


class FMPClient:
    """Financial Modeling Prep API client with rate limiting."""

    STABLE_URL = "https://financialmodelingprep.com/stable"

    _ENDPOINT_FAILURE_THRESHOLD = 3
    # v3 path-style endpoints whose trailing symbol moves to a ?symbol= query
    # on /stable (e.g. v3 income-statement/AAPL -> stable income-statement?symbol=AAPL)
    _SYMBOL_QUERY_ENDPOINTS = (
        "income-statement",
        "balance-sheet-statement",
        "cash-flow-statement",
        "key-metrics",
        "ratios",
        "profile",
        "quote",
    )

    def __init__(self, api_key: str):
        # The caller-supplied key is applied per CALL (see `_request`), never
        # assigned to the environment here: a process-global assignment made the
        # last client constructed own every later call in the interpreter.
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"apikey": self.api_key})
        self.rate_limit_reached = False
        self.retry_count = 0
        self._endpoint_failures: dict[str, int] = {}
        self._disabled_endpoints: set[str] = set()

    def _raw_stable_get(self, url: str, params: dict) -> Optional[dict]:
        """Direct /stable GET — used only when fmp_compat is not importable.

        No key failover (that lives in fmp_get); the constructor's apikey
        session header is the only credential on this path.
        """
        try:
            resp = self.session.get(url, params=params, timeout=30)
        except Exception:
            return None
        if getattr(resp, "status_code", None) != 200:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    def _request(self, url: str, params: dict, quiet: bool = False) -> Optional[dict]:
        """Single rate-limited GET. Returns parsed JSON, or None on failure.

        `quiet=True` suppresses the WARNING line for callers that handle a
        miss themselves (the circuit-breakered historical fetch).

        `key_override` scopes this client's key to the call — fmp_get reads its
        credentials from the environment, and a process-global assignment made
        the last client constructed own every later call (codex gate P2).
        """
        if self.rate_limit_reached:
            return None

        if params is None:
            params = {}

        if fmp_get is not None:
            with key_override(self.api_key):
                result = fmp_get(url, params=params, timeout=30)
        else:
            result = self._raw_stable_get(url, params)
        time.sleep(0.3)  # Rate limiting: 0.3s between requests
        if result is None and not quiet:
            print(f"WARNING: FMP request failed or quota-limited: {url}", file=sys.stderr)
        return result

    def _stable_spec(self, endpoint: str, params: dict) -> Optional[tuple]:
        """Map a v3 path-style endpoint to its (stable_url, stable_params).

        Returns None when there is no known /stable equivalent.
        """
        p = dict(params or {})
        if endpoint == "stock-screener":
            return f"{self.STABLE_URL}/company-screener", p
        if endpoint.startswith("historical-price-full/stock_dividend/"):
            p["symbol"] = endpoint.rsplit("/", 1)[-1]
            return f"{self.STABLE_URL}/dividends", p
        head, _, sym = endpoint.partition("/")
        if sym and head in self._SYMBOL_QUERY_ENDPOINTS:
            p["symbol"] = sym
            return f"{self.STABLE_URL}/{head}", p
        return None

    @staticmethod
    def _normalize(endpoint: str, data):
        """Reshape /stable responses to match the v3 shapes callers expect."""
        if data is None:
            return None
        # Dividends: /stable returns a flat list; v3 returned {"historical": [...]}.
        if endpoint.startswith("historical-price-full/stock_dividend/"):
            return {"historical": data} if isinstance(data, list) else data
        # key-metrics: /stable renamed roe -> returnOnEquity (same ratio scale).
        if endpoint.startswith("key-metrics/") and isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict) and "roe" not in rec and "returnOnEquity" in rec:
                    rec["roe"] = rec["returnOnEquity"]
        # cash-flow: /stable renamed dividendsPaid -> netDividendsPaid (same
        # negative-outflow value; callers take abs()).
        if endpoint.startswith("cash-flow-statement/") and isinstance(data, list):
            for rec in data:
                if (
                    isinstance(rec, dict)
                    and "dividendsPaid" not in rec
                    and "netDividendsPaid" in rec
                ):
                    rec["dividendsPaid"] = rec["netDividendsPaid"]
        return data

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """GET an FMP /stable endpoint from a legacy v3 path-style `endpoint`.

        Callers still pass the legacy v3 path-style strings (e.g.
        "income-statement/AAPL"); this routes them to the /stable query-style
        equivalent and normalizes the response back to the v3 shape callers
        expect. There is no v3 rung: FMP retired v3 for keys issued after
        2025-08-31, and fmp_compat rewrites a v3 URL straight back to the same
        /stable endpoint, so the old fallback only spent a second rate-limited
        call on the SAME upstream (WPP-20260831-004).
        """
        if self.rate_limit_reached:
            return None
        params = params or {}
        spec = self._stable_spec(endpoint, params)
        if spec is None:
            print(
                f"WARNING: no /stable equivalent for FMP endpoint {endpoint!r}",
                file=sys.stderr,
            )
            return None
        url, req_params = spec
        data = self._request(url, req_params)
        if data is None:
            return None
        return self._normalize(endpoint, data)

    def screen_stocks(self, min_market_cap: int = 2000000000, exchange: str = None) -> list[dict]:
        """Screen stocks by market cap and exchange."""
        params = {"marketCapMoreThan": min_market_cap}
        if exchange:
            params["exchange"] = exchange

        result = self._get("stock-screener", params)
        return result if result else []

    def get_historical_prices(self, symbol: str, days: int = 30) -> Optional[list[dict]]:
        """Get historical daily prices, most-recent-first (/stable only).

        /stable/historical-price-eod/full returns a flat list of OHLCV bars and
        is bounded with from/to. Routed through `_request`, so on the fmp_compat
        path the payload arrives already reshaped to the legacy
        {"symbol", "historical"} dict; both shapes are accepted.
        """
        for base_url in _FMP_HIST_ENDPOINTS:
            if base_url in self._disabled_endpoints:
                continue
            today = datetime.now().date()
            params = {
                "symbol": symbol,
                # ~days trading days needs ~2x calendar days; +10 for slack.
                "from": (today - timedelta(days=days * 2 + 10)).isoformat(),
                "to": today.isoformat(),
            }
            try:
                data = self._request(base_url, params, quiet=True)
                # Flat list of bars (most-recent-first).
                if isinstance(data, list):
                    if data:
                        self._endpoint_failures[base_url] = 0
                        return data[:days]
                    self._record_endpoint_failure(base_url)
                    continue
                # fmp_compat-normalised shape: {"symbol": ..., "historical": [...]}.
                # An EMPTY `historical` is a miss, not a success.
                if isinstance(data, dict) and data.get("historical"):
                    self._endpoint_failures[base_url] = 0
                    return data["historical"][:days]
                if isinstance(data, dict) and data.get("historicalStockList"):
                    for entry in data["historicalStockList"]:
                        if entry.get("symbol", "").replace("-", ".") == symbol.replace("-", "."):
                            # An EMPTY matched entry is a miss too (nested gate r2 P2):
                            # it must fall through to the yfinance leg below.
                            if entry.get("historical"):
                                self._endpoint_failures[base_url] = 0
                                return entry["historical"][:days]
                            break
                self._record_endpoint_failure(base_url)
            except Exception:
                self._record_endpoint_failure(base_url)
        # Every FMP rung missed (free-tier 402 / breaker-disabled): yfinance
        # leg, same row shape, fail-closed (WPP-20260906-007).
        yf_rows = _yf_price_history(symbol, days)
        if yf_rows:
            return yf_rows
        return None

    def _record_endpoint_failure(self, base_url: str) -> None:
        """Track consecutive failures and disable endpoint after threshold."""
        failures = self._endpoint_failures.get(base_url, 0) + 1
        self._endpoint_failures[base_url] = failures
        if failures >= self._ENDPOINT_FAILURE_THRESHOLD:
            self._disabled_endpoints.add(base_url)

    def get_dividend_history(self, symbol: str) -> Optional[dict]:
        """Get historical dividend payments."""
        result = self._get(f"historical-price-full/stock_dividend/{symbol}")
        if result and len(result.get("historical", [])) >= 16:
            return result
        yf_result = _yf_dividend_history(symbol)
        return yf_result or result

    def get_income_statement(self, symbol: str, limit: int = 5) -> Optional[list[dict]]:
        """Get income statement data."""
        result = self._get(f"income-statement/{symbol}", {"limit": limit})
        return result if result else _yf_statements(symbol, "income_stmt", limit)

    def get_balance_sheet(self, symbol: str, limit: int = 5) -> Optional[list[dict]]:
        """Get balance sheet data."""
        result = self._get(f"balance-sheet-statement/{symbol}", {"limit": limit})
        return result if result else _yf_statements(symbol, "balance_sheet", limit)

    def get_cash_flow(self, symbol: str, limit: int = 5) -> Optional[list[dict]]:
        """Get cash flow statement data."""
        result = self._get(f"cash-flow-statement/{symbol}", {"limit": limit})
        return result if result else _yf_statements(symbol, "cashflow", limit)

    def get_key_metrics(self, symbol: str, limit: int = 5) -> Optional[list[dict]]:
        """Get key financial metrics."""
        result = self._get(f"key-metrics/{symbol}", {"limit": limit})
        return result if result else _yf_key_metrics(symbol)

    def get_company_profile(self, symbol: str) -> Optional[dict]:
        """Get company profile including sector information."""
        result = self._get(f"profile/{symbol}")
        if result and isinstance(result, list) and len(result) > 0:
            return result[0]
        return None

    def get_quote_with_profile(self, symbol: str) -> Optional[dict]:
        """
        Get quote data merged with profile data to include sector information.

        Returns:
            Dict with quote data + sector/companyName from profile, or None on error
        """
        # First get quote data
        quote = self._get(f"quote/{symbol}")
        if not quote or not isinstance(quote, list) or len(quote) == 0:
            return _yf_quote_profile(symbol)

        quote_data = quote[0].copy()

        # Then get profile for sector information
        profile = self.get_company_profile(symbol)
        if profile:
            # Merge profile data into quote (profile has more accurate sector/companyName)
            quote_data["sector"] = profile.get("sector", "Unknown")
            quote_data["companyName"] = profile.get("companyName", quote_data.get("name", ""))
            quote_data["industry"] = profile.get("industry", "")
        else:
            # Fallback if profile fetch fails
            quote_data["sector"] = quote_data.get("sector", "Unknown")
            quote_data["companyName"] = quote_data.get("name", quote_data.get("companyName", ""))

        return quote_data


def load_symbol_universe(path: str) -> list[str]:
    """Load symbols from a CSV or one-symbol-per-line text file."""
    universe_path = Path(path).expanduser()
    if not universe_path.exists():
        print(f"ERROR: Universe file not found: {universe_path}", file=sys.stderr)
        return []

    symbols: list[str] = []
    with universe_path.open(newline="", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        sample_lines = sample.splitlines()
        first_line = sample_lines[0].lower() if sample_lines else ""
        header_cells = {cell.strip() for cell in first_line.split(",")}
        has_header = bool({"ticker", "symbol"} & header_cells)
        if not has_header and len(sample_lines) > 1 and "," in first_line:
            has_header = csv.Sniffer().has_header(sample)
        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = (
                    row.get("Ticker")
                    or row.get("ticker")
                    or row.get("Symbol")
                    or row.get("symbol")
                    or next(iter(row.values()), "")
                )
                symbol = (symbol or "").strip()
                if symbol:
                    symbols.append(symbol)
        else:
            for line in f:
                symbol = line.strip().split(",")[0].strip()
                if symbol and not symbol.startswith("#"):
                    symbols.append(symbol)

    seen: set[str] = set()
    return [s for s in symbols if not (s in seen or seen.add(s))]


def build_candidates_from_universe(
    symbols: list[str], client: FMPClient, max_candidates: int | None = None
) -> list[dict]:
    """Fetch quote/profile data for a local symbol universe."""
    selected = symbols[:max_candidates] if max_candidates else symbols
    candidates: list[dict] = []
    print(f"Fetching quote and profile data for {len(selected)} local-universe symbols...", file=sys.stderr)
    attempted = 0
    unpriceable: list[str] = []
    for symbol in selected:
        attempted += 1
        stock_data = client.get_quote_with_profile(symbol)
        if stock_data:
            candidates.append(stock_data)
        else:
            unpriceable.append(symbol)
        if client.rate_limit_reached:
            break
    _report_coverage(len(selected), attempted, [c.get("symbol", "") for c in candidates], unpriceable)
    return candidates


class RSICalculator:
    """Calculate Relative Strength Index (RSI) from price data."""

    @staticmethod
    def calculate_rsi(prices: list[float], period: int = 14) -> Optional[float]:
        """
        Calculate RSI using standard formula.

        Args:
            prices: List of closing prices (oldest first)
            period: RSI period (default 14)

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(prices) < period + 1:
            return None

        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [change if change > 0 else 0 for change in changes]
        losses = [-change if change < 0 else 0 for change in changes]

        # Calculate initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Calculate smoothed averages for remaining periods
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        # Calculate RSI
        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return round(rsi, 2)


def _series_is_frozen_30d(prices: list[float], *, min_bars: int = 20, max_range_pct: float = 3.0) -> bool:
    """
    Detect a frozen/thinly-updated 30-calendar-day price series that would
    manufacture a meaningless RSI reading.

    RSI is scale-free, so a series that barely moves for weeks can still
    register as "oversold" purely from noise. Founding case: SDR.L printed
    RSI 33.5 on 2026-09-03 while sitting 0.92% off its 52-week high — its
    last twelve closes were 587.9/588.4/588.9/588.4/588.4/588.4/588.4/
    588.4/588.9/589.4/584.5/584.0, a 2.04% 4-month range. A true oversold
    read looks different: same day, IGG.L gapped 1706→1460→1371→1360→
    1350→1312 (a real -21% move, RSI 18) and must NOT be flagged frozen.

    This is the 30-day-window variant of this repo's producer-side 60-bar
    frozen-series predicate — this screener only fetches 30 calendar days
    of history (see `get_historical_prices(symbol, days=30)`), so the
    window and default thresholds are sized for that shorter series.

    Args:
        prices: Closing prices for the fetched window (any order).
        min_bars: Minimum number of bars required to classify the series;
            fewer bars are not classifiable and are treated as not frozen.
        max_range_pct: A (max-min)/max*100 range below this threshold is
            considered frozen.

    Returns:
        True if the series is frozen/thin, False otherwise (including when
        there are too few bars to classify).
    """
    if len(prices) < min_bars:
        return False

    hi = max(prices)
    lo = min(prices)
    if hi <= 0:
        return False

    range_pct = (hi - lo) / hi * 100
    return range_pct < max_range_pct


class StockAnalyzer:
    """Analyze stock fundamentals and dividend growth."""

    @staticmethod
    def calculate_cagr(start_value: float, end_value: float, years: int) -> Optional[float]:
        """Calculate Compound Annual Growth Rate."""
        if start_value <= 0 or end_value <= 0 or years <= 0:
            return None
        return round(((end_value / start_value) ** (1 / years) - 1) * 100, 2)

    @staticmethod
    def analyze_dividend_growth(
        dividend_history: list[dict],
    ) -> tuple[Optional[float], bool, Optional[float], int]:
        """
        Analyze dividend growth rate (3-year CAGR and consistency) and return latest annual dividend.

        Returns:
            Tuple of (CAGR%, consistent_growth, latest_annual_dividend, years_of_growth)
        """
        if not dividend_history or "historical" not in dividend_history:
            return None, False, None, 0

        dividends = dividend_history["historical"]
        if len(dividends) < 4:
            return None, False, None, 0

        # Sort by date and aggregate by year
        dividends = sorted(dividends, key=lambda x: x["date"])
        annual_dividends = {}
        for div in dividends:
            year = div["date"][:4]
            annual_dividends[year] = annual_dividends.get(year, 0) + div.get("dividend", 0)

        # Exclude current year because partial-year dividends distort CAGR calculations.
        current_year = str(date.today().year)
        annual_dividends.pop(current_year, None)

        if len(annual_dividends) < 4:
            return None, False, None, 0

        # Get all available years sorted (oldest first)
        all_years = sorted(annual_dividends.keys())
        all_div_values = [annual_dividends[y] for y in all_years]

        # Get last 4 years for CAGR calculation
        years = all_years[-4:]
        div_values = [annual_dividends[y] for y in years]

        # Calculate 3-year CAGR
        cagr = StockAnalyzer.calculate_cagr(div_values[0], div_values[-1], 3)

        # Check consistency (no significant cuts)
        consistent = all(
            div_values[i] >= div_values[i - 1] * 0.95 for i in range(1, len(div_values))
        )

        # Count consecutive years of growth (from most recent going back)
        years_of_growth = 0
        for i in range(len(all_div_values) - 1, 0, -1):
            if all_div_values[i] >= all_div_values[i - 1] * 0.95:  # Allow 5% tolerance
                years_of_growth += 1
            else:
                break

        # Latest annual dividend
        latest_annual_dividend = div_values[-1]

        return cagr, consistent, latest_annual_dividend, years_of_growth

    @staticmethod
    def is_reit(stock_data: dict) -> bool:
        """
        Determine if a stock is a REIT based on sector/industry.

        Args:
            stock_data: Dict containing sector and/or industry fields

        Returns:
            True if the stock is likely a REIT
        """
        sector = stock_data.get("sector", "").lower()
        industry = stock_data.get("industry", "").lower()

        # Check for Real Estate sector or REIT in industry
        if "real estate" in sector:
            return True
        if "reit" in industry:
            return True

        return False

    @staticmethod
    def calculate_ffo(cash_flows: list[dict]) -> Optional[float]:
        """
        Calculate Funds From Operations (FFO) for REITs.

        FFO = Net Income + Depreciation & Amortization
        (Simplified formula - does not include gains/losses on property sales)

        Args:
            cash_flows: List of cash flow statements (newest first)

        Returns:
            FFO value or None if data is missing
        """
        if not cash_flows:
            return None

        latest_cf = cash_flows[0]
        net_income = latest_cf.get("netIncome", 0)
        depreciation = latest_cf.get("depreciationAndAmortization", 0)

        if net_income == 0 and depreciation == 0:
            return None

        return net_income + depreciation

    @staticmethod
    def calculate_ffo_payout_ratio(cash_flows: list[dict]) -> Optional[float]:
        """
        Calculate FFO payout ratio for REITs.

        FFO Payout Ratio = Dividends Paid / FFO

        Args:
            cash_flows: List of cash flow statements (newest first)

        Returns:
            FFO payout ratio as percentage, or None if calculation fails
        """
        if not cash_flows:
            return None

        ffo = StockAnalyzer.calculate_ffo(cash_flows)
        if not ffo or ffo <= 0:
            return None

        latest_cf = cash_flows[0]
        dividends_paid = abs(latest_cf.get("dividendsPaid", 0))

        if dividends_paid <= 0:
            return None

        return round((dividends_paid / ffo) * 100, 1)

    @staticmethod
    def calculate_payout_ratios(
        income_stmts: list[dict], cash_flows: list[dict], is_reit: bool = False
    ) -> dict:
        """
        Calculate payout ratios using dividendsPaid from cash flow statement.

        For REITs, uses FFO-based payout ratio instead of net income-based.

        Args:
            income_stmts: List of income statements (newest first)
            cash_flows: List of cash flow statements (newest first)
            is_reit: Whether the stock is a REIT (uses FFO for payout calculation)

        Returns:
            Dict with payout_ratio and fcf_payout_ratio (as percentages)
        """
        result = {"payout_ratio": None, "fcf_payout_ratio": None}

        if not cash_flows:
            return result

        latest_cf = cash_flows[0]
        dividends_paid = abs(latest_cf.get("dividendsPaid", 0))
        fcf = latest_cf.get("freeCashFlow", 0)

        # For REITs, use FFO-based payout ratio
        if is_reit:
            result["payout_ratio"] = StockAnalyzer.calculate_ffo_payout_ratio(cash_flows)
        else:
            # For non-REITs, use traditional net income-based payout ratio
            if income_stmts:
                latest_income = income_stmts[0]
                net_income = latest_income.get("netIncome", 0)

                if net_income > 0 and dividends_paid > 0:
                    result["payout_ratio"] = round((dividends_paid / net_income) * 100, 1)

        # Calculate FCF payout ratio (same for both REIT and non-REIT)
        if fcf > 0 and dividends_paid > 0:
            result["fcf_payout_ratio"] = round((dividends_paid / fcf) * 100, 1)

        return result

    @staticmethod
    def get_payout_ratio_from_metrics(key_metrics: list[dict]) -> Optional[float]:
        """
        Get payout ratio directly from key_metrics as fallback.

        Args:
            key_metrics: List of key metrics (newest first)

        Returns:
            Payout ratio as percentage, or None if not available
        """
        if not key_metrics:
            return None

        latest = key_metrics[0]
        payout_ratio = latest.get("payoutRatio")

        if payout_ratio is not None:
            # payoutRatio from FMP is a decimal (e.g., 0.316 = 31.6%)
            return round(payout_ratio * 100, 1)

        return None

    @staticmethod
    def analyze_financial_health(balance_sheet: list[dict]) -> dict:
        """Analyze financial health metrics."""
        if not balance_sheet:
            return {}

        latest = balance_sheet[0]

        total_debt = latest.get("totalDebt", 0)
        total_equity = latest.get("totalStockholdersEquity", 0)
        current_assets = latest.get("totalCurrentAssets", 0)
        current_liabilities = latest.get("totalCurrentLiabilities", 0)

        debt_to_equity = round(total_debt / total_equity, 2) if total_equity > 0 else None
        current_ratio = (
            round(current_assets / current_liabilities, 2) if current_liabilities > 0 else None
        )

        financially_healthy = (debt_to_equity is None or debt_to_equity < 2.0) and (
            current_ratio is None or current_ratio > 1.0
        )

        return {
            "debt_to_equity": debt_to_equity,
            "current_ratio": current_ratio,
            "financially_healthy": financially_healthy,
        }

    @staticmethod
    def analyze_growth_metrics(income_stmts: list[dict]) -> dict:
        """Analyze revenue and EPS growth trends."""
        if not income_stmts or len(income_stmts) < 4:
            return {"revenue_cagr_3y": None, "eps_cagr_3y": None}

        # Sort by date (newest first from API)
        revenue_3y_ago = income_stmts[3].get("revenue", 0)
        revenue_latest = income_stmts[0].get("revenue", 0)

        eps_3y_ago = income_stmts[3].get("eps", 0)
        eps_latest = income_stmts[0].get("eps", 0)

        revenue_cagr = StockAnalyzer.calculate_cagr(revenue_3y_ago, revenue_latest, 3)
        eps_cagr = StockAnalyzer.calculate_cagr(eps_3y_ago, eps_latest, 3)

        return {"revenue_cagr_3y": revenue_cagr, "eps_cagr_3y": eps_cagr}

    @staticmethod
    def calculate_composite_score(stock_data: dict) -> float:
        """
        Calculate composite score (0-100) based on:
        - Dividend Growth (40%): Reward higher CAGR
        - Financial Quality (30%): ROE, profit margins, debt levels
        - Technical Setup (20%): Lower RSI = better entry opportunity
        - Valuation (10%): P/E and P/B for context
        """
        score = 0.0

        # Dividend Growth Score (40 points max)
        div_cagr = stock_data.get("dividend_cagr_3y", 0)
        if div_cagr >= 20:
            score += 40
        elif div_cagr >= 15:
            score += 35
        elif div_cagr >= 12:
            score += 30
        else:
            score += 20

        # Add bonus for consistency
        if stock_data.get("dividend_consistent", False):
            score += 5

        # Financial Quality Score (30 points max)
        roe = stock_data.get("roe", 0)
        profit_margin = stock_data.get("profit_margin", 0)
        debt_to_equity = stock_data.get("debt_to_equity", 999)

        if roe >= 20:
            score += 12
        elif roe >= 15:
            score += 10
        elif roe >= 10:
            score += 7
        else:
            score += 3

        if profit_margin >= 20:
            score += 10
        elif profit_margin >= 15:
            score += 8
        elif profit_margin >= 10:
            score += 6
        else:
            score += 3

        if debt_to_equity < 0.5:
            score += 8
        elif debt_to_equity < 1.0:
            score += 6
        elif debt_to_equity < 2.0:
            score += 3

        # Technical Setup Score (20 points max) - Lower RSI = Higher score
        rsi = stock_data.get("rsi", 50)
        if rsi <= 25:
            score += 20  # Extreme oversold
        elif rsi <= 30:
            score += 18
        elif rsi <= 35:
            score += 15
        elif rsi <= 40:
            score += 12
        else:
            score += 5

        # Valuation Score (10 points max) - Context only, not exclusionary.
        # A missing/invalid ratio earns 0 points: the old `0` default scored
        # an absent P/E as the cheapest possible one (codex plan review r1 P1).
        pe_ratio = _valid_ratio(stock_data.get("pe_ratio"))
        pb_ratio = _valid_ratio(stock_data.get("pb_ratio"))

        if pe_ratio is not None:
            if pe_ratio < 15:
                score += 5
            elif pe_ratio < 25:
                score += 3

        if pb_ratio is not None:
            if pb_ratio < 3:
                score += 5
            elif pb_ratio < 5:
                score += 3

        return round(min(score, 100), 1)


def screen_dividend_growth_pullbacks(
    api_key: str,
    min_yield: float = 1.5,
    min_div_growth: float = 12.0,
    rsi_max: float = 40.0,
    max_candidates: int = None,
    finviz_symbols: Optional[set[str]] = None,
    universe_symbols: Optional[list[str]] = None,
) -> list[dict]:
    """
    Main screening function.

    Args:
        api_key: FMP API key
        min_yield: Minimum dividend yield % (default 1.5%)
        min_div_growth: Minimum 3-year dividend CAGR % (default 12%)
        rsi_max: Maximum RSI value (default 40)
        max_candidates: Maximum number of candidates to analyze (None = all)
        finviz_symbols: Optional set of symbols from FINVIZ pre-screening

    Returns:
        List of qualified stocks with full analysis
    """
    client = FMPClient(api_key)
    analyzer = StockAnalyzer()
    rsi_calc = RSICalculator()

    print(f"\n{'=' * 80}", file=sys.stderr)
    print("Dividend Growth Pullback Screener", file=sys.stderr)
    print(f"{'=' * 80}", file=sys.stderr)
    print("\nCriteria:", file=sys.stderr)
    print(f"  - Dividend Yield ≥ {min_yield}%", file=sys.stderr)
    print(f"  - Dividend Growth (3Y CAGR) ≥ {min_div_growth}%", file=sys.stderr)
    print(f"  - RSI ≤ {rsi_max}", file=sys.stderr)
    print("  - Market Cap ≥ $2B", file=sys.stderr)
    print("  - Exchange: NYSE, NASDAQ", file=sys.stderr)
    print(f"\n{'=' * 80}\n", file=sys.stderr)

    # Step 1: Get candidate list
    if finviz_symbols:
        print(
            f"Step 1: Using FINVIZ pre-screened symbols ({len(finviz_symbols)} stocks)...",
            file=sys.stderr,
        )
        # Convert FINVIZ symbols to candidate format for FMP analysis
        # We'll fetch quote data with profile to get sector information
        candidates = []
        print("Fetching quote and profile data from FMP for FINVIZ symbols...", file=sys.stderr)
        for symbol in finviz_symbols:
            stock_data = client.get_quote_with_profile(symbol)
            if stock_data:
                candidates.append(stock_data)

            if client.rate_limit_reached:
                print(
                    f"⚠️  FMP rate limit reached while fetching quotes. Using {len(candidates)} symbols.",
                    file=sys.stderr,
                )
                break

        print(
            f"Retrieved quote and profile data for {len(candidates)} symbols from FMP",
            file=sys.stderr,
        )
    elif universe_symbols:
        print(
            f"Step 1: Using local universe ({len(universe_symbols)} symbols)...",
            file=sys.stderr,
        )
        candidates = build_candidates_from_universe(universe_symbols, client, max_candidates)
        print(f"Retrieved quote and profile data for {len(candidates)} local symbols", file=sys.stderr)
    else:
        print("Step 1: FMP Stock Screener is unavailable on this FMP tier.", file=sys.stderr)
        print("Use --use-finviz or --universe portfolio/watchlist.csv for screening.", file=sys.stderr)
        candidates = []

    if not candidates:
        print("ERROR: No candidates found or API error", file=sys.stderr)
        return []

    # Limit candidates if specified
    if max_candidates and not finviz_symbols:
        candidates = candidates[:max_candidates]
        print(f"Limiting analysis to first {max_candidates} candidates", file=sys.stderr)

    print("\nStep 2: Detailed analysis of candidates...", file=sys.stderr)
    print("Note: Analysis will continue until API rate limit is reached\n", file=sys.stderr)

    results = []
    frozen_series_skipped = 0
    # Outcome accounting (codex plan r1 P1 #4): quote coverage is not screening
    # coverage. Every candidate ends as QUALIFIED, REJECTED_BY_CRITERIA (a gate
    # evaluated on real data), UNAVAILABLE_INPUT (a required input could not be
    # fetched — the screen did NOT evaluate that name) or UNANALYZED (rate-limit
    # break). The end-of-run "Outcomes:" line lets a consumer tell "screen
    # completed, no matches" from "screen could not run".
    outcomes = {"qualified": 0, "rejected_by_criteria": 0, "unavailable_input": 0}
    unavailable_reasons: dict[str, int] = {}

    def _unavailable(reason: str) -> None:
        outcomes["unavailable_input"] += 1
        unavailable_reasons[reason] = unavailable_reasons.get(reason, 0) + 1

    def _rejected() -> None:
        outcomes["rejected_by_criteria"] += 1

    for i, stock in enumerate(candidates, 1):
        symbol = stock.get("symbol", "")
        company_name = stock.get("companyName", "")

        print(f"[{i}/{len(candidates)}] Analyzing {symbol} - {company_name}...", file=sys.stderr)

        # Check rate limit
        if client.rate_limit_reached:
            print(f"\n⚠️  API rate limit reached after analyzing {i - 1} stocks.", file=sys.stderr)
            print(
                f"Returning results collected so far: {len(results)} qualified stocks",
                file=sys.stderr,
            )
            break

        # Get current price
        current_price = stock.get("price", 0)
        if current_price <= 0:
            print("  ⚠️  No valid price data", file=sys.stderr)
            _unavailable("price")
            continue

        # Fetch dividend history
        dividend_history = client.get_dividend_history(symbol)
        if client.rate_limit_reached:
            break

        if not dividend_history:
            print("  ⚠️  No dividend history", file=sys.stderr)
            _unavailable("dividend_history")
            continue

        # Analyze dividend growth
        div_cagr, div_consistent, annual_dividend, div_years_of_growth = (
            analyzer.analyze_dividend_growth(dividend_history)
        )
        if not div_cagr:
            print("  ⚠️  Dividend CAGR not computable (history too short)", file=sys.stderr)
            _unavailable("dividend_history_short")
            continue
        if div_cagr < min_div_growth:
            print(f"  ⚠️  Dividend CAGR {div_cagr}% < {min_div_growth}%", file=sys.stderr)
            _rejected()
            continue

        if not annual_dividend:
            print("  ⚠️  Cannot determine annual dividend", file=sys.stderr)
            _unavailable("annual_dividend")
            continue

        # Calculate actual dividend yield
        actual_dividend_yield = (annual_dividend / current_price) * 100

        if actual_dividend_yield < min_yield:
            print(
                f"  ⚠️  Dividend yield {actual_dividend_yield:.2f}% < {min_yield}%", file=sys.stderr
            )
            _rejected()
            continue

        print(
            f"  ✓ Dividend: {actual_dividend_yield:.2f}% yield, {div_cagr}% CAGR", file=sys.stderr
        )

        # Fetch historical prices for RSI
        historical_prices = client.get_historical_prices(symbol, days=30)
        if client.rate_limit_reached:
            break

        if not historical_prices or len(historical_prices) < 20:
            print("  ⚠️  Insufficient price data for RSI calculation", file=sys.stderr)
            _unavailable("price_history")
            continue

        # Calculate RSI
        prices = [p["close"] for p in reversed(historical_prices)]  # Oldest first
        rsi = rsi_calc.calculate_rsi(prices, period=14)

        if rsi is None:
            print("  ⚠️  RSI calculation failed", file=sys.stderr)
            _unavailable("price_history")
            continue

        if _series_is_frozen_30d(prices):
            hi, lo = max(prices), min(prices)
            range_pct = (hi - lo) / hi * 100 if hi > 0 else 0.0
            print(
                f"  ⚠️  frozen/thin price series ({range_pct:.2f}% 30d range) — "
                f"RSI {rsi} not meaningful, skipped",
                file=sys.stderr,
            )
            frozen_series_skipped += 1
            _unavailable("frozen_series")
            continue

        if rsi > rsi_max:
            print(f"  ⚠️  RSI {rsi} > {rsi_max}", file=sys.stderr)
            _rejected()
            continue

        print(f"  ✓ RSI: {rsi} (oversold)", file=sys.stderr)

        # Fetch additional fundamental data
        income_stmts = client.get_income_statement(symbol, limit=5)
        if client.rate_limit_reached:
            break

        balance_sheet = client.get_balance_sheet(symbol, limit=5)
        if client.rate_limit_reached:
            break

        cash_flow = client.get_cash_flow(symbol, limit=5)
        if client.rate_limit_reached:
            break

        key_metrics = client.get_key_metrics(symbol, limit=1)
        if client.rate_limit_reached:
            break

        # Analyze growth metrics
        growth_metrics = analyzer.analyze_growth_metrics(income_stmts if income_stmts else [])

        # Check for positive revenue and EPS growth
        revenue_cagr = growth_metrics.get("revenue_cagr_3y")
        eps_cagr = growth_metrics.get("eps_cagr_3y")

        if revenue_cagr is not None and revenue_cagr < 0:
            print("  ⚠️  Negative revenue growth", file=sys.stderr)
            _rejected()
            continue

        if eps_cagr is not None and eps_cagr < 0:
            print("  ⚠️  Negative EPS growth", file=sys.stderr)
            _rejected()
            continue

        # Analyze financial health — an EMPTY or INCOMPLETE balance sheet is an
        # unavailable input, not a failed gate (codex plan r1 P1 #1; nested gate
        # r1 P1: the analyzer's missing-field defaults would read as healthy).
        if not balance_sheet or not _balance_sheet_complete(balance_sheet[0]):
            print("  ⚠️  Balance sheet unavailable/incomplete (financial health not evaluated)", file=sys.stderr)
            _unavailable("balance_sheet")
            continue
        health_metrics = analyzer.analyze_financial_health(balance_sheet)

        if not health_metrics.get("financially_healthy", False):
            print("  ⚠️  Financial health concerns", file=sys.stderr)
            _rejected()
            continue

        # Extract additional metrics
        income_stmts[0] if income_stmts else {}
        latest_metrics = key_metrics[0] if key_metrics else {}

        # Check if this is a REIT (uses different payout ratio calculation)
        is_reit = analyzer.is_reit(stock)

        # Calculate payout ratios using the new method
        payout_ratios = analyzer.calculate_payout_ratios(
            income_stmts if income_stmts else [], cash_flow if cash_flow else [], is_reit=is_reit
        )
        payout_ratio = payout_ratios["payout_ratio"]
        fcf_payout_ratio = payout_ratios["fcf_payout_ratio"]

        # Fallback to key_metrics if calculation failed (only for non-REITs)
        if payout_ratio is None and not is_reit:
            payout_ratio = analyzer.get_payout_ratio_from_metrics(
                key_metrics if key_metrics else []
            )

        # Determine dividend sustainability
        # Sustainable if payout ratio < 80% and FCF covers dividends
        dividend_sustainable = False
        if payout_ratio and fcf_payout_ratio:
            dividend_sustainable = payout_ratio < 80 and fcf_payout_ratio < 100
        elif payout_ratio:
            dividend_sustainable = payout_ratio < 80

        # Provenance (nested gate r2 P2): which leg supplied each evidence class.
        data_sources = {
            "quote": stock.get("data_source", "fmp"),
            "dividends": (dividend_history.get("data_source", "fmp")
                          if isinstance(dividend_history, dict) else "fmp"),
            "prices": historical_prices[0].get("data_source", "fmp"),
            "fundamentals": balance_sheet[0].get("data_source", "fmp"),
        }

        # Build result object
        result = {
            "symbol": symbol,
            "company_name": company_name,
            "data_sources": data_sources,
            "sector": stock.get("sector", "Unknown"),
            "market_cap": stock.get("marketCap", 0),
            "price": current_price,
            "dividend_yield": round(actual_dividend_yield, 2),
            "annual_dividend": round(annual_dividend, 2),
            "dividend_cagr_3y": div_cagr,
            "dividend_consistent": div_consistent,
            "rsi": rsi,
            "pe_ratio": _valid_ratio(latest_metrics.get("peRatio")),
            "pb_ratio": _valid_ratio(latest_metrics.get("pbRatio")),
            "revenue_cagr_3y": revenue_cagr,
            "eps_cagr_3y": eps_cagr,
            "payout_ratio": payout_ratio,
            "fcf_payout_ratio": fcf_payout_ratio,
            "dividend_sustainable": dividend_sustainable,
            "dividend_years_of_growth": div_years_of_growth,
            "debt_to_equity": health_metrics.get("debt_to_equity"),
            "current_ratio": health_metrics.get("current_ratio"),
            "financially_healthy": health_metrics.get("financially_healthy", False),
            # roe and netProfitMargin from FMP are decimals (e.g., 0.25 = 25%),
            # like payoutRatio above; convert to whole-number percents so they
            # match the composite-score thresholds and the report's "%" display.
            "roe": (latest_metrics.get("roe") or 0) * 100,
            "profit_margin": (latest_metrics.get("netProfitMargin") or 0) * 100,
        }

        # Calculate composite score
        result["composite_score"] = analyzer.calculate_composite_score(result)

        results.append(result)
        outcomes["qualified"] += 1
        print(f"  ✅ QUALIFIED - Score: {result['composite_score']}", file=sys.stderr)

    # Sort by composite score
    results.sort(key=lambda x: x["composite_score"], reverse=True)

    print(f"\n{'=' * 80}", file=sys.stderr)
    print("Screening Complete!", file=sys.stderr)
    print(f"Qualified Stocks: {len(results)}", file=sys.stderr)
    if frozen_series_skipped:
        print(f"Frozen/thin price series skipped: {frozen_series_skipped}", file=sys.stderr)
    # Every candidate that reached a terminal outcome is counted in exactly one
    # bucket; anything else (a rate-limit break at ANY fetch, mid-analysis
    # included — nested gate r2 P2) is unanalyzed.
    analyzed = sum(outcomes.values())
    print(_format_outcomes(analyzed, outcomes, unavailable_reasons, len(candidates) - analyzed),
          file=sys.stderr)
    print(f"{'=' * 80}\n", file=sys.stderr)

    return results


def _format_stock_sources(stock: dict) -> str:
    sources = stock.get("data_sources") or {}
    return ", ".join(f"{k}={v}" for k, v in sources.items()) or "fmp"


def _describe_data_sources(results: list[dict]) -> str:
    """Header provenance line: FMP-only, or FMP + the yfinance leg when any
    qualified stock carries yfinance evidence (per-stock detail in each
    section's `Data sources` row and the JSON `data_sources`)."""
    legs = {v for r in results for v in (r.get("data_sources") or {}).values()}
    if "yfinance" in legs:
        return "Financial Modeling Prep API + yfinance free-tier leg (see per-stock `Data sources`)"
    return "Financial Modeling Prep API"


def generate_markdown_report(results: list[dict], criteria: dict, output_path: str):
    """Generate human-readable markdown report."""

    report = f"""# Dividend Growth Pullback Screening Report

**Generated:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC
**Data Source:** {_describe_data_sources(results)}

## Executive Summary

**Total Qualified Stocks:** {len(results)}

### Screening Criteria

- **Dividend Yield:** ≥ {criteria["dividend_yield_min"]}%
- **Dividend Growth (3Y CAGR):** ≥ {criteria["dividend_cagr_min"]}%
- **RSI:** ≤ {criteria["rsi_max"]} (oversold/pullback)
- **Market Cap:** ≥ $2 billion
- **Financial Health:** Positive revenue/EPS growth, D/E < 2.0, Current Ratio > 1.0

---

"""

    if not results:
        report += """## No Stocks Qualified

**Possible Reasons:**
- Strong bull market with few oversold stocks
- Dividend growth criteria (12%+) is very selective
- RSI threshold may be too strict for current market conditions

**Recommendations:**
- Relax RSI threshold to ≤45 for early pullback phase
- Lower dividend growth to ≥10% for more candidates
- Check back during market corrections or sector rotations

"""
    else:
        for i, stock in enumerate(results, 1):
            rsi_interpretation = (
                "Extreme Oversold"
                if stock["rsi"] < 30
                else "Strong Oversold"
                if stock["rsi"] < 35
                else "Early Pullback"
            )

            report += f"""## {i}. {stock["symbol"]} - {stock["company_name"]}

**Sector:** {stock["sector"]}
**Market Cap:** ${stock["market_cap"] / 1e9:.1f}B
**Current Price:** ${stock["price"]:.2f}
**Composite Score:** {stock["composite_score"]}/100

### Dividend Growth Profile

| Metric | Value | Assessment |
|--------|-------|------------|
| Dividend Yield | **{stock["dividend_yield"]:.2f}%** | {
                "✓ Above 2%" if stock["dividend_yield"] >= 2 else "⚠ Below 2%"
            } |
| Annual Dividend | ${stock["annual_dividend"]:.2f} | |
| 3Y Dividend CAGR | **{stock["dividend_cagr_3y"]:.2f}%** | {
                "🔥 Exceptional"
                if stock["dividend_cagr_3y"] >= 20
                else "✓ Excellent"
                if stock["dividend_cagr_3y"] >= 15
                else "✓ Strong"
            } |
| Dividend Consistency | {"Yes" if stock["dividend_consistent"] else "No"} | {
                "✓" if stock["dividend_consistent"] else "⚠"
            } |
| Payout Ratio | {f"{stock['payout_ratio']:.1f}%" if stock["payout_ratio"] else "N/A"} | {
                "✓ Sustainable"
                if stock["payout_ratio"] and stock["payout_ratio"] < 70
                else "⚠ High"
                if stock["payout_ratio"] and stock["payout_ratio"] < 100
                else "❌ Risk"
                if stock["payout_ratio"]
                else "N/A"
            } |

### Technical Setup

| Metric | Value | Interpretation |
|--------|-------|----------------|
| RSI (14-period) | **{stock["rsi"]:.1f}** | {rsi_interpretation} |
| Data sources | {_format_stock_sources(stock)} | per evidence class (fmp = FMP /stable, yfinance = free-tier leg) |
| Entry Timing | {
                "Immediate - Scale in 50%"
                if stock["rsi"] < 30
                else "Good - Full position OK"
                if stock["rsi"] < 35
                else "Conservative - High conviction"
            } | |
| Stop Loss Suggestion | {
                f"{((stock['rsi'] - 30) / 2 + 3):.0f}% below entry"
                if stock["rsi"] >= 30
                else "8% below entry"
            } | |

**RSI Context:** {
                "Extreme oversold reading suggests panic selling or negative news. Wait for RSI to turn up (>30) before entry to confirm stabilization."
                if stock["rsi"] < 30
                else "Strong oversold in uptrend. Normal correction creating entry opportunity. Can initiate position with standard risk management."
                if stock["rsi"] < 35
                else "Early pullback in uptrend. Conservative entry point with lower risk of further decline. Suitable for high-conviction additions."
            }

### Business Fundamentals

| Metric | Value | Status |
|--------|-------|--------|
| Revenue CAGR (3Y) | {
                f"{stock['revenue_cagr_3y']:.2f}%" if stock["revenue_cagr_3y"] else "N/A"
            } | {"✓" if stock["revenue_cagr_3y"] and stock["revenue_cagr_3y"] > 0 else "⚠"} |
| EPS CAGR (3Y) | {f"{stock['eps_cagr_3y']:.2f}%" if stock["eps_cagr_3y"] else "N/A"} | {
                "✓" if stock["eps_cagr_3y"] and stock["eps_cagr_3y"] > 0 else "⚠"
            } |
| ROE | {f"{stock['roe']:.1f}%" if stock["roe"] else "N/A"} | {
                "✓ Excellent"
                if stock["roe"] and stock["roe"] >= 20
                else "✓ Good"
                if stock["roe"] and stock["roe"] >= 15
                else "⚠ Moderate"
                if stock["roe"]
                else "N/A"
            } |
| Net Profit Margin | {f"{stock['profit_margin']:.1f}%" if stock["profit_margin"] else "N/A"} | {
                "✓" if stock["profit_margin"] and stock["profit_margin"] >= 10 else "⚠"
            } |

### Financial Health

| Metric | Value | Status |
|--------|-------|--------|
| Debt-to-Equity | {
                f"{stock['debt_to_equity']:.2f}" if stock["debt_to_equity"] is not None else "N/A"
            } | {
                "✓ Very Low"
                if stock["debt_to_equity"] and stock["debt_to_equity"] < 0.5
                else "✓ Low"
                if stock["debt_to_equity"] and stock["debt_to_equity"] < 1.0
                else "⚠ Moderate"
                if stock["debt_to_equity"]
                else "N/A"
            } |
| Current Ratio | {f"{stock['current_ratio']:.2f}" if stock["current_ratio"] else "N/A"} | {
                "✓ Healthy"
                if stock["current_ratio"] and stock["current_ratio"] > 1.2
                else "⚠ Adequate"
                if stock["current_ratio"]
                else "N/A"
            } |

### Investment Thesis

**10-Year Dividend Projection ({stock["dividend_cagr_3y"]:.0f}% CAGR):**
- Current Yield on Cost: {stock["dividend_yield"]:.2f}%
- Year 5 Yield on Cost: {stock["dividend_yield"] * (1 + stock["dividend_cagr_3y"] / 100) ** 5:.2f}%
- Year 10 Yield on Cost: {stock["dividend_yield"] * (1 + stock["dividend_cagr_3y"] / 100) ** 10:.2f}%

**Entry Strategy:**
{f"- RSI {stock['rsi']:.0f} indicates {rsi_interpretation.lower()} condition"}
- {
                "Scale in with 50% position now, add remaining on RSI >30 confirmation"
                if stock["rsi"] < 30
                else f"Full position acceptable with stop loss {((stock['rsi'] - 30) / 2 + 3):.0f}% below entry"
                if stock["rsi"] < 35
                else "Conservative entry for high-conviction add with 3-5% stop loss"
            }
- Time horizon: 6-12 months minimum (long-term dividend growth play)

**Risk Factors:**
{
                f"- Payout ratio {stock['payout_ratio']:.0f}% limits dividend growth runway"
                if stock["payout_ratio"] and stock["payout_ratio"] > 70
                else "- Monitor payout ratio sustainability"
            }
{
                f"- Debt-to-equity {stock['debt_to_equity']:.1f} requires monitoring"
                if stock["debt_to_equity"] and stock["debt_to_equity"] > 1.0
                else ""
            }
- RSI can remain oversold in downtrends - watch for reversal confirmation
- Dividend growth may slow if business growth moderates

---

"""

    report += f"""
## Methodology

This screening combines fundamental dividend analysis with technical timing indicators:

1. **Fundamental Filter:** Dividend yield ≥{criteria["dividend_yield_min"]}%, dividend CAGR ≥{criteria["dividend_cagr_min"]}%, positive business growth
2. **Technical Filter:** RSI ≤{criteria["rsi_max"]} identifies temporary pullbacks in quality stocks
3. **Quality Filter:** Financial health checks (debt, liquidity, profitability)
4. **Ranking:** Composite score balancing dividend growth (40%), quality (30%), technical setup (20%), valuation (10%)

**Investment Philosophy:**
High dividend growth stocks (12%+ CAGR) compound wealth through rising dividends rather than high current yield. A 1.5% yielding stock growing dividends at 15%/year becomes a 4% yielder in 6 years and 9% yielder in 12 years - far superior to a 4% yielder growing at 3%/year. Buying during RSI oversold conditions (≤40) enhances returns by entering at technical support levels.

---

**Disclaimer:** This report is for informational purposes only. Past dividend growth does not guarantee future performance. RSI oversold conditions do not guarantee price reversals. Conduct thorough due diligence and consult a financial advisor before making investment decisions.

**Report Generated:** {datetime.utcnow().isoformat()}Z
"""

    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Markdown report saved: {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Screen dividend growth stocks with RSI oversold using FINVIZ + FMP API (two-stage approach)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Two-stage screening: FINVIZ pre-screen + FMP detailed analysis (RECOMMENDED)
  python3 screen_dividend_growth_rsi.py --use-finviz

  # FMP-only screening (original method)
  python3 screen_dividend_growth_rsi.py

  # Provide API keys as arguments
  python3 screen_dividend_growth_rsi.py --use-finviz --fmp-api-key YOUR_FMP_KEY --finviz-api-key YOUR_FINVIZ_KEY

  # Custom parameters
  python3 screen_dividend_growth_rsi.py --use-finviz --min-yield 2.0 --min-div-growth 15.0 --rsi-max 35

Environment Variables:
  FMP_API_KEY       - Financial Modeling Prep API key
  FINVIZ_API_KEY    - FINVIZ Elite API key (required for --use-finviz)
        """,
    )

    parser.add_argument(
        "--fmp-api-key", type=str, help="FMP API key (or set FMP_API_KEY environment variable)"
    )
    parser.add_argument(
        "--finviz-api-key",
        type=str,
        help="FINVIZ Elite API key (or set FINVIZ_API_KEY environment variable)",
    )
    parser.add_argument(
        "--use-finviz",
        action="store_true",
        help="Use FINVIZ Elite API for pre-screening (recommended to reduce FMP API calls)",
    )
    parser.add_argument(
        "--min-yield", type=float, default=1.5, help="Minimum dividend yield %% (default: 1.5)"
    )
    parser.add_argument(
        "--min-div-growth",
        type=float,
        default=12.0,
        help="Minimum 3-year dividend CAGR %% (default: 12.0)",
    )
    parser.add_argument(
        "--rsi-max", type=float, default=40.0, help="Maximum RSI value (default: 40.0)"
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum candidates to analyze (default: all)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        help="CSV or text file containing symbols to analyze instead of the unavailable FMP stock screener",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for JSON and Markdown outputs (default: TraderMonty logs directory)",
    )

    args = parser.parse_args()

    # Get FMP API key
    fmp_api_key = args.fmp_api_key or os.environ.get("FMP_API_KEY")
    if not fmp_api_key:
        print(
            "ERROR: FMP API key required. Provide via --fmp-api-key or FMP_API_KEY environment variable",
            file=sys.stderr,
        )
        sys.exit(1)

    # FINVIZ pre-screening (optional)
    finviz_symbols = None
    if args.use_finviz:
        finviz_api_key = args.finviz_api_key or os.environ.get("FINVIZ_API_KEY")
        if not finviz_api_key:
            print(
                "ERROR: FINVIZ API key required when using --use-finviz. Provide via --finviz-api-key or FINVIZ_API_KEY environment variable",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\n{'=' * 80}", file=sys.stderr)
        print("DIVIDEND GROWTH PULLBACK SCREENER (TWO-STAGE)", file=sys.stderr)
        print(f"{'=' * 80}\n", file=sys.stderr)

        finviz_client = FINVIZClient(finviz_api_key)
        finviz_symbols = finviz_client.screen_stocks()

        if not finviz_symbols:
            print("ERROR: No stocks found in FINVIZ pre-screening", file=sys.stderr)
            sys.exit(1)

        print(f"\n{'=' * 80}\n", file=sys.stderr)

    universe_symbols = load_symbol_universe(args.universe) if args.universe else None

    # Run screening
    results = screen_dividend_growth_pullbacks(
        api_key=fmp_api_key,
        min_yield=args.min_yield,
        min_div_growth=args.min_div_growth,
        rsi_max=args.rsi_max,
        max_candidates=args.max_candidates,
        finviz_symbols=finviz_symbols,
        universe_symbols=universe_symbols,
    )

    # Prepare metadata
    criteria = {
        "dividend_yield_min": args.min_yield,
        "dividend_cagr_min": args.min_div_growth,
        "rsi_max": args.rsi_max,
        "revenue_trend": "positive over 3 years",
        "eps_trend": "positive over 3 years",
    }

    # Generate outputs
    today = date.today().isoformat()

    # Determine output directory (project root logs/ folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from skills/dividend-growth-pullback-screener/scripts to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    logs_dir = args.output_dir or os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # JSON output
    json_output = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "criteria": criteria,
            "total_results": len(results),
        },
        "stocks": results,
    }

    json_path = os.path.join(logs_dir, f"dividend_growth_pullback_results_{today}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)

    print(f"✅ JSON results saved: {json_path}", file=sys.stderr)

    # Markdown report
    md_path = os.path.join(logs_dir, f"dividend_growth_pullback_screening_{today}.md")
    generate_markdown_report(results, criteria, md_path)

    print(f"\n{'=' * 80}", file=sys.stderr)
    print(f"Screening complete! Found {len(results)} qualified stocks.", file=sys.stderr)
    print(f"{'=' * 80}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
