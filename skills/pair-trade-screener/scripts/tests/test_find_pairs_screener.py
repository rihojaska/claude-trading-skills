"""FMP /stable migration: sector screener uses /stable/company-screener only.

fetch_sector_stocks() used to try v3 /stock-screener as a fallback when the
stable call failed. FMP retired v3 for keys issued after 2025-08-31, and a v3
URL requested through fmp_compat is rewritten straight back to the identical
/stable endpoint, so that fallback was never a distinct upstream
(WPP-20260831-004 / WPP-20260901-016) — it has been deleted. These tests pin
`fmp_get = None` to exercise the standalone (no-fmp_compat) transport
directly; the fmp_get path is covered at the real transport seam in
scripts/tests/test_v3_rungs_gone_0901_016.py.
"""

from unittest.mock import MagicMock, patch

import find_pairs  # noqa: E402
import pytest
import statsmodels  # noqa: F401


@pytest.fixture(autouse=True)
def _direct_stable_transport(monkeypatch):
    monkeypatch.setattr(find_pairs, "fmp_get", None, raising=False)


def _resp(status_code, payload):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


@patch("find_pairs.requests.get")
def test_uses_stable_company_screener_first(mock_get):
    mock_get.return_value = _resp(
        200,
        [
            {
                "symbol": "NVDA",
                "companyName": "NVIDIA Corporation",
                "marketCap": 5_343_290_069_887,
                "sector": "Technology",
                "exchangeShortName": "NASDAQ",
                "isActivelyTrading": True,
            }
        ],
    )
    stocks = find_pairs.fetch_sector_stocks("Technology", "key")

    assert stocks[0]["symbol"] == "NVDA"
    assert stocks[0]["name"] == "NVIDIA Corporation"
    assert stocks[0]["exchange"] == "NASDAQ"
    assert stocks[0]["marketCap"] == 5_343_290_069_887

    call = mock_get.call_args_list[0]
    assert call[0][0].endswith("/stable/company-screener")
    assert call[1]["params"]["sector"] == "Technology"
    assert call[1]["params"]["marketCapMoreThan"] == 2_000_000_000


@patch("find_pairs.requests.get")
def test_stable_failure_raises_with_no_second_request(mock_get):
    """A failed stable call makes no second (v3) request and exits loudly."""
    mock_get.return_value = _resp(403, {})

    with pytest.raises(SystemExit):
        find_pairs.fetch_sector_stocks("Technology", "key")

    assert mock_get.call_count == 1
    url = mock_get.call_args_list[0][0][0]
    assert url.endswith("/stable/company-screener")
    assert "api/v3" not in url


@patch("find_pairs.requests.get")
def test_filters_inactive_symbols(mock_get):
    mock_get.return_value = _resp(
        200,
        [
            {"symbol": "ACTIVE", "isActivelyTrading": True},
            {"symbol": "DELISTED", "isActivelyTrading": False},
        ],
    )
    stocks = find_pairs.fetch_sector_stocks("Technology", "key")
    assert [s["symbol"] for s in stocks] == ["ACTIVE"]
