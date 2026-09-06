"""FMP /stable migration for theme-detector.

Keys issued after 2025-08-31 lose v3 access (403). Fixes:
- Quotes/history are fetched one symbol per request (FMP /stable does not
  support comma-batched symbols — that silently returns []).
- Historical uses /stable/historical-price-eod/full (the prior stable URL,
  historical-price-full, 404s) with timeseries->from/to and flat-list
  normalization back to the v3 {"symbol","historical"} shape.
- ETF holdings: /stable/etf/holdings?symbol= only. A v3 URL requested
  through fmp_compat is rewritten straight back to the identical /stable
  endpoint, so a "v3 fallback" was never a distinct upstream
  (WPP-20260831-004 / WPP-20260901-016) — TestEtfHoldings below pins
  `fmp_get = None` to exercise the standalone transport directly; the
  fmp_get path is covered at the real transport seam in
  scripts/tests/test_v3_rungs_gone_0901_016.py.
"""

from unittest.mock import MagicMock, patch

import pytest

import representative_stock_selector as rss
from etf_scanner import ETFScanner, _normalize_eod_flat_list, _stable_hist_url


class TestHistoricalNormalize:
    def test_flat_list_to_v3_dict(self):
        flat = [
            {"symbol": "AAPL", "date": "2026-05-19", "close": 298.97},
            {"symbol": "AAPL", "date": "2026-05-18", "close": 296.0},
        ]
        out = _normalize_eod_flat_list(flat, "AAPL")
        assert out["symbol"] == "AAPL"
        assert len(out["historical"]) == 2

    def test_no_match_returns_none(self):
        assert _normalize_eod_flat_list([{"symbol": "MSFT", "close": 1}], "AAPL") is None

    def test_dict_passthrough(self):
        d = {"symbol": "AAPL", "historical": [{"close": 1}]}
        assert _normalize_eod_flat_list(d, "AAPL") is d

    def test_stable_hist_url_converts_timeseries(self):
        base = "https://financialmodelingprep.com/stable/historical-price-eod/full"
        url, params = _stable_hist_url(base, "AAPL", {"timeseries": 20})
        assert url == base
        assert params["symbol"] == "AAPL"
        assert "from" in params and "to" in params
        assert "timeseries" not in params  # converted to a from/to range


class TestDeBatch:
    def test_batch_sizes_are_one(self):
        assert ETFScanner.FMP_QUOTE_BATCH_SIZE == 1
        assert ETFScanner.FMP_HIST_BATCH_SIZE == 1

    def test_quotes_fetched_one_symbol_per_request(self):
        scanner = ETFScanner(fmp_api_key="k")
        seen = []

        def fake_req(endpoint_key, symbols_str, extra_params=None):
            seen.append((endpoint_key, symbols_str))
            return [{"symbol": symbols_str, "price": 100.0}]

        scanner._fmp_request = fake_req
        scanner._fetch_fmp_quotes(["AAPL", "MSFT", "NVDA"])
        quote_syms = [s for e, s in seen if e == "quote"]
        assert all("," not in s for s in quote_syms)  # never comma-batched
        assert set(quote_syms) >= {"AAPL", "MSFT", "NVDA"}


class TestEtfHoldings:
    @pytest.fixture(autouse=True)
    def _direct_stable_transport(self, monkeypatch):
        """Exercise the standalone (no-fmp_compat) transport in this class.

        representative_stock_selector prefers `fmp_compat.fmp_get`; the tests
        below inject canned responses at `requests.get`, which is the direct
        /stable path used when the repo-root shim is not importable. Pinning
        `fmp_get = None` keeps that injection honest instead of silently
        reaching the network. The fmp_get path is covered at the real
        transport seam in scripts/tests/test_v3_rungs_gone_0901_016.py.
        """
        monkeypatch.setattr(rss, "fmp_get", None, raising=False)

    def _resp(self, status, payload):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = payload
        return r

    def test_stable_etf_holdings_first(self):
        sel = rss.RepresentativeStockSelector(fmp_api_key="k")
        sel._rate_limit = lambda: None
        with patch.object(rss, "requests") as mock_requests:
            mock_requests.get.return_value = self._resp(
                200, [{"asset": "NVDA", "marketValue": 1_000_000_000}]
            )
            holds = sel._fetch_etf_holdings("SPY", limit=5)
        assert holds[0]["symbol"] == "NVDA"
        assert mock_requests.get.call_args[0][0].endswith("/stable/etf/holdings")
        assert mock_requests.get.call_args[1]["params"] == {"symbol": "SPY"}

    def test_stable_failure_returns_empty_with_no_second_request(self):
        """A failed stable call makes no second (v3) request."""
        sel = rss.RepresentativeStockSelector(fmp_api_key="k")
        sel._rate_limit = lambda: None

        with patch.object(rss, "requests") as mock_requests:
            mock_requests.get.return_value = self._resp(403, {})
            holds = sel._fetch_etf_holdings("XLK", limit=5)

        assert holds == []
        assert mock_requests.get.call_count == 1
        url = mock_requests.get.call_args[0][0]
        assert url.endswith("/stable/etf/holdings")
        assert "api/v3" not in url
