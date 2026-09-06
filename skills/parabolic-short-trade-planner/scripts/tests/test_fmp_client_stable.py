"""Regression tests for the FMP ``/stable`` endpoint migration (Issue #162).

Three ``fmp_client.py`` methods (``get_sp500_constituents``,
``get_earnings_calendar``, ``get_company_profile``) build explicit ``/stable``
URLs directly (S-FMPCLIENT-3, 2026-09-06 — the ``_fmp_compat.v3_to_stable()``
shim these used to route through is gone; every generated client now
delegates its FMP transport to the repo-root ``fmp_compat`` module instead).
These tests pin the corrected behaviour:

- ``/stable/sp500-constituent`` (not ``sp500_constituent``)
- ``/stable/earnings-calendar`` (not ``earning_calendar``)
- ``get_company_profile`` re-aliases ``marketCap`` → ``mktCap``

Offline: the network boundary (``_rate_limited_get``) is mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fmp_client import FMPClient  # noqa: E402


def _call_url_params(mock: MagicMock) -> tuple[str, dict | None]:
    """Extract (url, params) from a _rate_limited_get mock call, tolerating
    both positional and keyword ``params`` call styles."""
    call = mock.call_args
    url = call.args[0]
    if len(call.args) > 1:
        params = call.args[1]
    else:
        params = call.kwargs.get("params")
    return url, params


# --- Client methods build the corrected URLs ----------------------------------


def test_get_sp500_constituents_requests_hyphenated_url() -> None:
    client = FMPClient(api_key="test-key")
    client._rate_limited_get = MagicMock(return_value=[{"symbol": "AAPL"}])

    client.get_sp500_constituents()

    url, _ = _call_url_params(client._rate_limited_get)
    assert url == "https://financialmodelingprep.com/stable/sp500-constituent"


def test_get_earnings_calendar_requests_hyphenated_url_with_dates() -> None:
    client = FMPClient(api_key="test-key")
    client._rate_limited_get = MagicMock(return_value=[{"symbol": "AAPL"}])

    client.get_earnings_calendar("2026-05-20", "2026-06-15")

    url, params = _call_url_params(client._rate_limited_get)
    assert url == "https://financialmodelingprep.com/stable/earnings-calendar"
    assert params == {"from": "2026-05-20", "to": "2026-06-15"}


# --- Profile alias ------------------------------------------------------------


def test_get_company_profile_aliases_market_cap() -> None:
    client = FMPClient(api_key="test-key")
    # /stable/profile returns ``marketCap`` (no ``mktCap``).
    client._rate_limited_get = MagicMock(
        return_value=[{"symbol": "AAPL", "marketCap": 4_000_000_000_000}]
    )

    profile = client.get_company_profile("AAPL")

    assert profile is not None
    assert profile["mktCap"] == 4_000_000_000_000
    # The original key remains intact.
    assert profile["marketCap"] == 4_000_000_000_000


def test_get_company_profile_keeps_existing_mktcap() -> None:
    client = FMPClient(api_key="test-key")
    # If a row already carries ``mktCap`` it must not be overwritten.
    client._rate_limited_get = MagicMock(
        return_value=[{"symbol": "AAPL", "mktCap": 123, "marketCap": 456}]
    )

    profile = client.get_company_profile("AAPL")

    assert profile is not None
    assert profile["mktCap"] == 123
