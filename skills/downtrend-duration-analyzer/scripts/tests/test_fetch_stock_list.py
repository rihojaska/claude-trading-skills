"""FMP /stable migration: stock list uses /stable/company-screener only.

fetch_stock_list() used to try v3 /stock-screener as a fallback when the
stable call failed. FMP retired v3 for keys issued after 2025-08-31, and a v3
URL requested through fmp_compat is rewritten straight back to the identical
/stable endpoint, so that fallback was never a distinct upstream
(WPP-20260831-004 / WPP-20260901-016) — it has been deleted. These tests pin
`fmp_get = None` to exercise the standalone (no-fmp_compat) transport
directly; the fmp_get path is covered at the real transport seam in
scripts/tests/test_v3_rungs_gone_0901_016.py.
"""

from unittest.mock import MagicMock, patch

import analyze_downtrends
import pytest


@pytest.fixture(autouse=True)
def _direct_stable_transport(monkeypatch):
    monkeypatch.setattr(analyze_downtrends, "fmp_get", None, raising=False)


def _resp(status_code, payload):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


@patch("analyze_downtrends.requests.get")
def test_uses_stable_company_screener_first(mock_get):
    mock_get.return_value = _resp(
        200, [{"symbol": "XOM", "sector": "Energy", "marketCap": 500_000_000_000}]
    )
    stocks = analyze_downtrends.fetch_stock_list("key", sector="Energy")

    assert stocks[0]["symbol"] == "XOM"
    call = mock_get.call_args_list[0]
    assert call[0][0].endswith("/stable/company-screener")
    assert call[1]["params"]["sector"] == "Energy"
    assert call[1]["params"]["isActivelyTrading"] == "true"
    assert call[1]["params"]["limit"] == 500


@patch("analyze_downtrends.requests.get")
def test_stable_failure_returns_empty_with_no_second_request(mock_get):
    """A failed stable call makes no second (v3) request."""
    mock_get.return_value = _resp(403, {})

    stocks = analyze_downtrends.fetch_stock_list("key")

    assert stocks == []
    assert mock_get.call_count == 1
    url = mock_get.call_args_list[0][0][0]
    assert url.endswith("/stable/company-screener")
    assert "api/v3" not in url


@patch("analyze_downtrends.requests.get")
def test_returns_empty_when_all_fail(mock_get):
    mock_get.return_value = _resp(403, {})
    assert analyze_downtrends.fetch_stock_list("key") == []
