"""WPP-20260906-007 (codex plan r1 P1 #1/#3/#4) — the growth screen on the
free tier, end to end.

Pins: the yfinance statement leg maps annual frames into the FMP vocabulary
(newest first, absent labels absent, NaN skipped); the key-metrics leg reads
``info``; the client falls through only on an FMP miss; the price leg drops
today's partial bar and duplicate sessions; and the screening loop, driven
by a fake client that has good dividend + RSI data but NO fundamentals,
reports that name as UNAVAILABLE_INPUT (balance_sheet) — never as a failed
"Financial health" gate — on a machine-readable ``Outcomes:`` line, while
a fully-served name qualifies.
"""
import os
import sys
import types
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import screen_dividend_growth_rsi as mod  # noqa: E402


def _frame(rows: dict, cols):
    idx = pd.DatetimeIndex(cols)
    return pd.DataFrame({c: [rows[label][i] for label in rows] for i, c in enumerate(idx)},
                        index=list(rows.keys()), columns=idx)


COLS = ["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"]


def _fake_yf(monkeypatch, *, income=None, balance=None, cashflow=None, info=None, history=None):
    class _T:
        def __init__(self, symbol):
            self.income_stmt = income if income is not None else pd.DataFrame()
            self.balance_sheet = balance if balance is not None else pd.DataFrame()
            self.cashflow = cashflow if cashflow is not None else pd.DataFrame()
            self.info = info or {}

        def history(self, **kw):
            return history if history is not None else pd.DataFrame()

    fake = types.ModuleType("yfinance")
    fake.Ticker = _T
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def test_statement_leg_maps_into_fmp_vocabulary_newest_first(monkeypatch):
    income = _frame({"Total Revenue": [400.0, 300.0, 200.0, 100.0],
                     "Diluted EPS": [4.0, float("nan"), 2.0, 1.0],
                     "Net Income": [40.0, 30.0, 20.0, 10.0]}, COLS)
    _fake_yf(monkeypatch, income=income)
    rows = mod._yf_statements("ALV.DE", "income_stmt", limit=5)
    assert [r["date"] for r in rows] == COLS
    assert rows[0] == {"date": "2025-12-31", "data_source": "yfinance", "revenue": 400.0, "eps": 4.0, "netIncome": 40.0}
    assert "eps" not in rows[1]  # NaN -> absent, never 0
    assert mod.StockAnalyzer.analyze_growth_metrics(rows)["revenue_cagr_3y"] is not None


def test_statement_leg_first_label_wins_and_empty_frame_is_empty(monkeypatch):
    cf = _frame({"Common Stock Dividend Paid": [-9.0], "Cash Dividends Paid": [-10.0],
                 "Free Cash Flow": [50.0]}, COLS[:1])
    _fake_yf(monkeypatch, cashflow=cf)
    row = mod._yf_statements("X", "cashflow")[0]
    assert row["dividendsPaid"] == -10.0 and row["freeCashFlow"] == 50.0
    assert mod._yf_statements("X", "balance_sheet") == []


def test_key_metrics_leg_reads_info_and_client_falls_through_only_on_fmp_miss(monkeypatch):
    _fake_yf(monkeypatch, info={"trailingPE": 14.2, "priceToBook": 2.1, "returnOnEquity": 0.18,
                                "profitMargins": 0.11, "payoutRatio": 0.45, "longName": "x"})
    assert mod._yf_key_metrics("X")[0]["peRatio"] == 14.2
    client = mod.FMPClient(api_key="dummy")
    monkeypatch.setattr(client, "_get", lambda *a, **k: [{"peRatio": 9.9}])
    assert client.get_key_metrics("X", limit=1) == [{"peRatio": 9.9}]  # FMP hit wins
    monkeypatch.setattr(client, "_get", lambda *a, **k: [])
    assert client.get_key_metrics("X", limit=1)[0]["roe"] == 0.18
    assert client.get_balance_sheet("X") == []  # empty frame stays empty


def test_price_leg_drops_todays_partial_bar_and_duplicate_sessions(monkeypatch):
    today = datetime.now().date()
    days = [today - timedelta(days=i) for i in range(40, 0, -1)] + [today - timedelta(days=1), today]
    closes = [100.0 + i for i in range(len(days))]
    frame = pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(days))
    _fake_yf(monkeypatch, history=frame)
    rows = mod._yf_price_history("X", days=30)
    assert rows[0]["date"] == (today - timedelta(days=1)).isoformat()
    assert rows[0]["close"] == 139.0  # first occurrence of the duplicated session wins
    assert len({r["date"] for r in rows}) == len(rows) == 30


def _dividends(years=6, start=1.0, growth=1.15):
    rows = []
    d = 2020
    for y in range(years):
        for q in (3, 6, 9, 12):
            rows.append({"date": f"{d + y}-{q:02d}-15", "adjDividend": round(start * growth ** y / 4, 4),
                         "dividend": round(start * growth ** y / 4, 4)})
    return {"symbol": "X", "historical": list(reversed(rows))}


class _FakeClient:
    """Good quote, dividends and an oversold price series for every symbol;
    fundamentals only for the symbol named in `served`."""
    rate_limit_reached = False

    def __init__(self, served):
        self.served = served

    def get_quote_with_profile(self, symbol):
        return {"symbol": symbol, "companyName": symbol, "price": 50.0, "sector": "Industrials"}

    def get_dividend_history(self, symbol):
        return _dividends()

    def get_historical_prices(self, symbol, days=30):
        closes = [42.0 + i * 0.6 for i in range(30)]  # newest first: 42 today, 59 a month ago -> decline, RSI << 40
        return [{"date": f"2026-08-{30 - i:02d}", "close": c} for i, c in enumerate(closes)]

    def _served(self, symbol, rows):
        return rows if symbol == self.served else []

    def get_income_statement(self, symbol, limit=5):
        return self._served(symbol, [{"date": c, "revenue": 130.0 - 10.0 * i, "eps": 2.0, "netIncome": 20.0}
                                     for i, c in enumerate(COLS)])

    def get_balance_sheet(self, symbol, limit=5):
        return self._served(symbol, [{"date": COLS[0], "totalDebt": 10.0, "totalStockholdersEquity": 100.0,
                                      "totalCurrentAssets": 30.0, "totalCurrentLiabilities": 10.0}])

    def get_cash_flow(self, symbol, limit=5):
        return self._served(symbol, [{"date": COLS[0], "dividendsPaid": -5.0, "freeCashFlow": 30.0}])

    def get_key_metrics(self, symbol, limit=5):
        return self._served(symbol, [{"peRatio": 12.0, "pbRatio": 1.5, "roe": 0.15, "netProfitMargin": 0.1}])


def test_loop_reports_missing_fundamentals_as_unavailable_not_as_a_failed_gate(monkeypatch, capsys):
    client = _FakeClient(served="GOOD")
    full_bs = client.get_balance_sheet

    def sparse_bs(symbol, limit=5):
        if symbol == "SPARSE":  # yfinance shape for a financial: no current assets/liabilities
            return [{"date": COLS[0], "totalDebt": 10.0, "totalStockholdersEquity": 100.0}]
        return full_bs(symbol, limit)
    client.get_balance_sheet = sparse_bs
    monkeypatch.setattr(mod, "FMPClient", lambda api_key: client)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
    results = mod.screen_dividend_growth_pullbacks("dummy", universe_symbols=["GOOD", "COLD", "SPARSE"])
    err = capsys.readouterr().err
    assert [r["symbol"] for r in results] == ["GOOD"]
    assert "Balance sheet unavailable" in err and "Financial health concerns" not in err
    line = next(l for l in err.splitlines() if l.startswith("Outcomes:"))
    assert line == "Outcomes: analyzed 3 · qualified 1 · rejected_by_criteria 0 · unavailable_input 2 [balance_sheet=2] · unanalyzed 0"


def test_balance_sheet_completeness_requires_every_scored_field_finite():
    ok = {"totalDebt": 1.0, "totalStockholdersEquity": 2.0, "totalCurrentAssets": 3.0, "totalCurrentLiabilities": 4.0}
    assert mod._balance_sheet_complete(ok)
    assert not mod._balance_sheet_complete({**ok, "totalCurrentAssets": float("nan")})
    assert not mod._balance_sheet_complete({k: v for k, v in ok.items() if k != "totalDebt"})
    assert not mod._balance_sheet_complete({**ok, "totalDebt": None})


def test_loop_counts_criteria_rejections_separately(monkeypatch, capsys):
    client = _FakeClient(served="GOOD")
    client.get_historical_prices = lambda symbol, days=30: [
        {"date": f"2026-08-{30 - i:02d}", "close": 60.0 - i * 0.6} for i in range(30)]  # newest 60, oldest 42.6 -> rally, RSI high
    monkeypatch.setattr(mod, "FMPClient", lambda api_key: client)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
    results = mod.screen_dividend_growth_pullbacks("dummy", universe_symbols=["GOOD"])
    err = capsys.readouterr().err
    assert results == []
    assert "Outcomes: analyzed 1 · qualified 0 · rejected_by_criteria 1 · unavailable_input 0 · unanalyzed 0" in err


# ── nested gate r2 P2 folds ────────────────────────────────────────────────────

def test_empty_matched_historical_entry_falls_through_to_yfinance(monkeypatch):
    client = mod.FMPClient(api_key="dummy")
    payload = {"historicalStockList": [{"symbol": "ALV.DE", "historical": []}]}
    monkeypatch.setattr(client, "_request", lambda *a, **k: payload)
    yf_rows = [{"date": "2026-09-04", "close": 1.0, "data_source": "yfinance"}]
    monkeypatch.setattr(mod, "_yf_price_history", lambda *a, **k: yf_rows)
    assert client.get_historical_prices("ALV.DE", days=30) == yf_rows


def test_fundamentals_legs_reject_non_finite_values(monkeypatch):
    income = _frame({"Total Revenue": [float("inf"), 300.0], "Net Income": [40.0, float("-inf")]}, COLS[:2])
    _fake_yf(monkeypatch, income=income, info={"trailingPE": float("inf"), "priceToBook": 2.0, "payoutRatio": True})
    rows = mod._yf_statements("X", "income_stmt")
    assert "revenue" not in rows[0] and "netIncome" not in rows[1]
    km = mod._yf_key_metrics("X")[0]
    assert "peRatio" not in km and km["pbRatio"] == 2.0 and "payoutRatio" not in km


def test_rate_limit_break_mid_analysis_counts_as_unanalyzed(monkeypatch, capsys):
    client = _FakeClient(served="GOOD")

    def limited_prices(symbol, days=30):
        client.rate_limit_reached = True  # limit hit AFTER the dividend fetch, before RSI
        return None
    client.get_historical_prices = limited_prices
    monkeypatch.setattr(mod, "FMPClient", lambda api_key: client)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
    assert mod.screen_dividend_growth_pullbacks("dummy", universe_symbols=["GOOD", "COLD"]) == []
    err = capsys.readouterr().err
    assert "Outcomes: analyzed 0 · qualified 0 · rejected_by_criteria 0 · unavailable_input 0 · unanalyzed 2" in err


def test_results_and_report_carry_provenance(monkeypatch, tmp_path, capsys):
    client = _FakeClient(served="GOOD")
    real_bs = client.get_balance_sheet
    client.get_balance_sheet = lambda symbol, limit=5: [dict(real_bs(symbol, limit)[0], data_source="yfinance")]
    monkeypatch.setattr(mod, "FMPClient", lambda api_key: client)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
    results = mod.screen_dividend_growth_pullbacks("dummy", universe_symbols=["GOOD"])
    assert results[0]["data_sources"] == {"quote": "fmp", "dividends": "fmp", "prices": "fmp", "fundamentals": "yfinance"}
    out = tmp_path / "r.md"
    mod.generate_markdown_report(results, {"dividend_yield_min": 1.5, "dividend_cagr_min": 12.0, "rsi_max": 40.0}, str(out))
    text = out.read_text()
    assert "yfinance free-tier leg" in text and "| Data sources | quote=fmp, dividends=fmp, prices=fmp, fundamentals=yfinance |" in text
    assert "## 1. GOOD - GOOD" in text  # the consumer's section shape is unchanged
