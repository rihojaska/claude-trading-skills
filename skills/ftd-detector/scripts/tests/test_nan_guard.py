"""WPP-20260818-009 — non-finite market data must never read as a valid observation.

The 2026-08-18 incident: FMP was quota-blocked on both quote endpoints, the
yfinance fallback minted a NaN close, and `ftd_detector` printed `OK ($nan)`,
wrote NaN into three JSON fields, AND — the part the row missed — declared a
NASDAQ Follow-Through Day off that bar, because `NaN < FTD_GAIN_MINIMUM` is
False. That fabricated `dual_confirmation` and a 15-point score component.

The guard lives at ONE boundary: `FMPClient.get_historical_prices` /
`.get_quote`, which every provider and both batch helpers cross. These tests
pin that boundary, its date-ordering invariant, and the incident shape itself.
"""

import datetime as _dt
import math
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fmp_client as fc  # noqa: E402
from fmp_client import FMPClient  # noqa: E402

NAN = float("nan")


# ---------------------------------------------------------------------------
# yfinance fake: `_yf_history` calls `yf.Ticker(sym).history(...)` and then
# iterates `hist.tail(days).iloc[::-1].iterrows()`. Reproduce exactly that
# surface — a stub that skipped any link in the chain would let the row loop
# go unexercised, which is the mutation this fake exists to prevent.
# ---------------------------------------------------------------------------
class _FakeDF:
    def __init__(self, rows):
        self._rows = list(rows)

    @property
    def empty(self):
        return not self._rows

    def tail(self, n):
        return _FakeDF(self._rows[-n:]) if n else _FakeDF(self._rows)

    @property
    def iloc(self):
        return _FakeIloc(self._rows)

    def iterrows(self):
        return iter(self._rows)


class _FakeIloc:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, key):
        return _FakeDF(self._rows[key])


def _yf_rows(specs):
    """specs: list of (iso_date, close, volume) in CHRONOLOGICAL order."""
    out = []
    for iso, close, volume in specs:
        idx = _dt.datetime.fromisoformat(iso)
        out.append(
            (
                idx,
                {
                    "Open": close,
                    "High": close,
                    "Low": close,
                    "Close": close,
                    "Adj Close": close,
                    "Volume": volume,
                },
            )
        )
    return out


class _FakeYF:
    def __init__(self, rows):
        self._rows = rows

    def Ticker(self, symbol):  # noqa: N802 — mirrors the yfinance API
        outer = self

        class _T:
            def history(self, period=None, auto_adjust=None):
                return _FakeDF(outer._rows)

        return _T()


def _with_yf(rows):
    return patch.dict(sys.modules, {"yfinance": _FakeYF(rows)})


def _bars(specs):
    """FMP-shaped `historical` bars, MOST-RECENT-FIRST (the boundary's order)."""
    return [
        {
            "date": iso,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adjClose": close,
            "volume": volume,
        }
        for iso, close, volume in specs
    ]


def _client():
    with patch.dict(os.environ, {"FMP_API_KEY": "k"}):  # pragma: allowlist secret
        return FMPClient()


def _transport(*responses):
    """Patch the TRANSPORT, not `_request_with_fallback`.

    Content validation lives inside the endpoint loop so a shape-valid but
    untrustworthy stable response still gets v3 tried (codex gate r1). Patching
    the wrapper would skip the very loop under test; each positional argument
    here is one endpoint's parsed response, in stable-then-v3 order, and a single
    argument answers every endpoint.
    """
    queue = list(responses)

    def _get(self, url, params=None, quiet=False):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return patch.object(FMPClient, "_rate_limited_get", _get)


# ---------------------------------------------------------------------------
# T1 — `_yf_history` construction must not raise, and must not repair
# ---------------------------------------------------------------------------
class TestYfHistoryConstruction:
    def test_nan_volume_does_not_raise(self):
        """`int(row.get("Volume", 0) or 0)` raises on NaN — and the row loop sits
        OUTSIDE the fetch try/except, so today it escapes uncaught.

        MUTANT: restore the `int(... or 0)` cast during construction -> ValueError.
        """
        rows = _yf_rows([("2026-08-14", 100.0, 1_000_000), ("2026-08-17", 101.0, NAN)])
        with _with_yf(rows):
            out = fc._yf_history("QQQ", 2)
        assert out is not None, "a NaN volume must not destroy the whole fetch"
        assert len(out["historical"]) == 2

    def test_nan_close_is_carried_not_repaired(self):
        """`or 0` cannot catch NaN (bool(nan) is True). Construction must leave the
        value alone; judging validity is the boundary's job, and silently writing
        0.0 here would be P5's forbidden substitution.

        MUTANT: coerce non-finite -> 0.0 here -> the assert on isnan fails.
        """
        rows = _yf_rows([("2026-08-14", 100.0, 1_000_000), ("2026-08-17", NAN, 5_000)])
        with _with_yf(rows):
            out = fc._yf_history("QQQ", 2)
        newest = out["historical"][0]
        assert newest["date"] == "2026-08-17"
        assert math.isnan(newest["close"])


# ---------------------------------------------------------------------------
# T2 — the boundary: trim leading, reject interior, never cache a reject
# ---------------------------------------------------------------------------
class TestHistoryBoundary:
    def test_trims_leading_unsettled_bars(self):
        """`historical` is most-recent-first, so a leading invalid bar is an
        unsettled newest session. Removing it == running before that session
        existed, and everything older stays contiguous.
        """
        payload = {
            "symbol": "QQQ",
            "historical": _bars(
                [
                    ("2026-08-18", NAN, 1_000),
                    ("2026-08-17", 101.0, 1_000),
                    ("2026-08-14", 100.0, 1_000),
                ]
            ),
        }
        client = _client()
        with _transport(payload):
            out = client.get_historical_prices("QQQ", days=3)
        assert out is not None
        dates = [b["date"] for b in out["historical"]]
        assert dates == ["2026-08-17", "2026-08-14"]

    def test_rejects_on_interior_gap(self):
        """Dropping an INTERIOR bar makes two non-adjacent sessions adjacent.
        `rally_tracker` counts rally days positionally, so that silently
        compresses day numbering and can MANUFACTURE an FTD — the same class of
        wrong verdict this row exists to remove.

        MUTANT: trim interior bars instead of rejecting -> returns 2 bars.
        """
        payload = {
            "symbol": "QQQ",
            "historical": _bars(
                [
                    ("2026-08-18", 102.0, 1_000),
                    ("2026-08-17", NAN, 1_000),
                    ("2026-08-14", 100.0, 1_000),
                ]
            ),
        }
        client = _client()
        with _transport(payload), patch.object(fc, "_yf_history", return_value=None):
            assert client.get_historical_prices("QQQ", days=3) is None

    def test_reject_stamps_no_provenance_and_caches_nothing(self):
        """A rejected payload must not be cached (it would be re-served to every
        later caller) and must not claim a `data_sources` provenance.

        MUTANT: cache/stamp before validating -> both asserts fail.
        """
        payload = {"symbol": "QQQ", "historical": _bars([("2026-08-18", NAN, 1_000)])}
        client = _client()
        with _transport(payload), patch.object(fc, "_yf_history", return_value=None):
            assert client.get_historical_prices("QQQ", days=1) is None
        assert client.cache == {}
        assert "historical:QQQ" not in client.data_sources

    @pytest.mark.parametrize("bad", [0.0, -1.0, None, float("inf"), True])
    def test_non_positive_and_non_real_closes_are_invalid(self, bad):
        """A zero close is not a market observation, and it is also the `x / 0`
        route by which a *finite* input still produces a NaN change_pct.

        `True` is in the set deliberately: `isinstance(True, int)` holds and
        `math.isfinite(True) and True > 0` both pass, so without the explicit
        bool guard a payload decoding `"close": true` would be scored as a
        $1.00 close — a second, narrower route to the same fabrication.
        """
        payload = {
            "symbol": "QQQ",
            "historical": _bars([("2026-08-18", bad, 1_000), ("2026-08-17", 101.0, 1_000)]),
        }
        client = _client()
        with _transport(payload):
            out = client.get_historical_prices("QQQ", days=2)
        assert out is not None
        assert [b["date"] for b in out["historical"]] == ["2026-08-17"]

    def test_yfinance_path_crosses_the_same_boundary(self):
        """The fallback provider is validated by the same predicate, not a second copy."""
        client = _client()
        rows = _yf_rows([("2026-08-14", 100.0, 1_000), ("2026-08-17", 101.0, NAN)])
        with _transport(None), _with_yf(rows):
            out = client.get_historical_prices("QQQ", days=2)
        assert out is not None
        assert [b["date"] for b in out["historical"]] == ["2026-08-14"]


# ---------------------------------------------------------------------------
# T8 — the date-ordering invariant (the design's highest-risk assumption)
# ---------------------------------------------------------------------------
class TestDateOrderingInvariant:
    """Trim-leading is only safe if `historical` is most-recent-first. The v3
    passthrough shape cannot be proven offline, so the assumption is CHECKED.
    Without these, deleting the ordering check leaves every other test green
    while trim-leading silently deletes the OLDEST bars.
    """

    @pytest.mark.parametrize(
        "dates",
        [
            ["2026-08-14", "2026-08-17", "2026-08-18"],  # ascending
            ["2026-08-18", "2026-08-18", "2026-08-17"],  # duplicate
            ["2026-08-18", "not-a-date", "2026-08-17"],  # unparseable
            ["2026-08-18", "2026-08-17", None],  # missing
        ],
    )
    def test_non_descending_dates_reject_the_payload(self, dates):
        payload = {
            "symbol": "QQQ",
            "historical": [
                {
                    "date": d,
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "adjClose": 100.0,
                    "volume": 1_000,
                }
                for d in dates
            ],
        }
        client = _client()
        with _transport(payload), patch.object(fc, "_yf_history", return_value=None):
            assert client.get_historical_prices("QQQ", days=3) is None


# ---------------------------------------------------------------------------
# T4 + T9 — the quote half, both providers
# ---------------------------------------------------------------------------
class TestQuoteBoundary:
    def test_yf_quote_none_when_newest_raw_bar_invalid(self):
        """Returning the newest SURVIVING bar would publish a prior session's
        close under a `quote` provenance label. Returning None instead routes
        `ftd_detector` into its documented "Using historical close as current
        price" branch, which labels the source honestly.

        Defense-in-depth pin: monkeypatch the data, because once the boundary
        ships this case is unreachable through live flow and the test would
        otherwise pass vacuously.
        """
        rows = _yf_rows([("2026-08-14", 100.0, 1_000), ("2026-08-17", NAN, 1_000)])
        with _with_yf(rows):
            assert fc._yf_quote("QQQ") is None

    @pytest.mark.parametrize("bad", [NAN, None, 0.0, -3.0])
    def test_fmp_quote_with_bad_price_is_absent(self, bad):
        """`json.loads` accepts bare NaN, so an FMP quote can carry one. A bad
        price means no quote: fall through to yfinance, stamp no `quote:`
        provenance, cache nothing.

        MUTANT: validate history only -> this fails while everything else passes.
        """
        client = _client()
        with _transport([{"symbol": "QQQ", "price": bad}]), patch.object(fc, "_yf_quote", return_value=None):
            assert client.get_quote("QQQ") is None
        assert client.cache == {}
        assert "quote:QQQ" not in client.data_sources

    def test_batch_quotes_inherit_the_guard(self):
        """`get_batch_quotes` routes through `get_quote`, so it must inherit —
        not need — its own copy of the predicate (P7)."""
        client = _client()
        with (
            _transport([{"symbol": "QQQ", "price": NAN}, {"symbol": "SPY", "price": 500.0}]),
            patch.object(fc, "_yf_quote", return_value=None),
        ):
            out = client.get_batch_quotes(["QQQ", "SPY"])
        assert "QQQ" not in out
        assert out["SPY"]["price"] == 500.0


# ---------------------------------------------------------------------------
# T5 — the incident itself, end to end
# ---------------------------------------------------------------------------
class TestIncidentShape:
    """The 2026-08-18 artifact recorded `ftd_detected: true`, `gain_pct: NaN`,
    `gain_tier: "minimum"`, `dual_confirmation: true` and a `+15` component off
    a bar with no close.

    This is deliberately an INTEGRATION test through `FMPClient`: the fix does
    not change `detect_ftd`, so feeding that function NaN bars directly still
    fabricates an FTD afterwards. The claim under test is that the poisoned
    payload never reaches it.
    """

    def test_poisoned_payload_never_reaches_the_state_machine(self):
        import rally_tracker

        # A clean rally the tracker WILL score, with the incident's NaN bar as
        # the newest session — the exact 08-18 shape.
        chrono = [(f"2026-06-{d:02d}", 100.0 - d * 0.4, 1_000_000) for d in range(1, 29)]
        chrono += [(f"2026-07-{d:02d}", 88.0 + d * 0.9, 2_000_000) for d in range(1, 8)]
        specs = list(reversed(chrono))  # most-recent-first
        payload = {"symbol": "QQQ", "historical": _bars([("2026-07-08", NAN, 3_000_000)] + specs)}

        client = _client()
        with _transport(payload):
            out = client.get_historical_prices("QQQ", days=40)

        assert out is not None, "a trailing unsettled bar must not blank the symbol"
        assert all(
            math.isfinite(b["close"]) for b in out["historical"]
        ), "no non-finite close may survive the boundary"

        state = rally_tracker.analyze_single_index(
            list(reversed(out["historical"])), "NASDAQ"
        )
        ftd = state.get("ftd") or {}
        gain = ftd.get("gain_pct")
        assert gain is None or math.isfinite(gain), f"FTD scored off a non-finite gain: {gain!r}"
        assert state.get("current_price") is None or math.isfinite(state["current_price"])


# ---------------------------------------------------------------------------
# codex gate r1 — the two regressions the first implementation shipped
# ---------------------------------------------------------------------------
class TestOrderIsProvenBeforeTrimming:
    """A discarded PREFIX bar can be chronologically interior.

    Checking order only on the surviving tail passes an invalid `2026-08-16`
    sitting ahead of a valid `08-18 / 08-17 / 08-15`: the remainder descends
    cleanly, but the session dropped from between 08-17 and 08-15 is exactly the
    false adjacency the guard exists to prevent.
    """

    def test_invalid_prefix_bar_out_of_order_rejects_the_payload(self):
        payload = {
            "symbol": "QQQ",
            "historical": _bars(
                [
                    ("2026-08-16", NAN, 1_000),  # invalid AND chronologically interior
                    ("2026-08-18", 102.0, 1_000),
                    ("2026-08-17", 101.0, 1_000),
                    ("2026-08-15", 100.0, 1_000),
                ]
            ),
        }
        client = _client()
        with _transport(payload), patch.object(fc, "_yf_history", return_value=None):
            assert client.get_historical_prices("QQQ", days=4) is None


class TestContentFailureStillTriesTheNextEndpoint:
    """A shape-valid but untrustworthy stable response is a FAILED ENDPOINT.

    Validating in the public wrapper instead of the endpoint loop skipped v3 and
    went straight to yfinance — turning a recoverable response into missing
    history whenever yfinance was unavailable.
    """

    def test_bad_stable_history_falls_back_to_v3_not_yfinance(self):
        bad = {"symbol": "QQQ", "historical": _bars([("2026-08-18", NAN, 1_000)])}
        good = {
            "symbol": "QQQ",
            "historical": _bars([("2026-08-18", 102.0, 1_000), ("2026-08-17", 101.0, 1_000)]),
        }
        client = _client()
        with _transport(bad, good), patch.object(fc, "_yf_history", return_value=None):
            out = client.get_historical_prices("QQQ", days=2)
        assert out is not None, "v3 was never tried"
        assert [b["date"] for b in out["historical"]] == ["2026-08-18", "2026-08-17"]
        assert client.data_sources["historical:QQQ"] == "fmp"

    def test_bad_stable_quote_falls_back_to_v3_not_yfinance(self):
        client = _client()
        with (
            _transport([{"symbol": "QQQ", "price": NAN}], [{"symbol": "QQQ", "price": 450.0}]),
            patch.object(fc, "_yf_quote", return_value=None),
        ):
            out = client.get_quote("QQQ")
        assert out == [{"symbol": "QQQ", "price": 450.0}]
        assert client.data_sources["quote:QQQ"] == "fmp"


class TestPerFieldOhlcValidation:
    """A valid `close` does not make the bar valid.

    `_yf_history._cell` resolves each OHLC field independently, so a bar with a
    good Close and a NaN High/Low/Open is a realistic yfinance shape — and those
    fields are not decorative: `high`/`low` drive `track_rally_attempt`'s
    top-of-range Day-1 rule and `check_ftd_invalidation`'s `ftd_low`. Without
    this, deleting the per-field loop from `_bar_is_valid` leaves the suite green.
    """

    @pytest.mark.parametrize("field", ["open", "high", "low", "adjClose"])
    @pytest.mark.parametrize("bad", [NAN, 0.0, -1.0, None, True])
    def test_a_bad_non_close_field_invalidates_the_bar(self, field, bad):
        newest = {
            "date": "2026-08-18",
            "open": 102.0,
            "high": 102.0,
            "low": 102.0,
            "close": 102.0,
            "adjClose": 102.0,
            "volume": 1_000,
            field: bad,
        }
        payload = {
            "symbol": "QQQ",
            "historical": [newest] + _bars([("2026-08-17", 101.0, 1_000)]),
        }
        client = _client()
        with _transport(payload), patch.object(fc, "_yf_history", return_value=None):
            out = client.get_historical_prices("QQQ", days=2)
        assert out is not None
        assert [b["date"] for b in out["historical"]] == ["2026-08-17"]

    @pytest.mark.parametrize("bad", [NAN, -1.0, True])
    def test_a_bad_volume_invalidates_the_bar(self, bad):
        newest = {
            "date": "2026-08-18",
            "open": 102.0,
            "high": 102.0,
            "low": 102.0,
            "close": 102.0,
            "adjClose": 102.0,
            "volume": bad,
        }
        payload = {
            "symbol": "QQQ",
            "historical": [newest] + _bars([("2026-08-17", 101.0, 1_000)]),
        }
        client = _client()
        with _transport(payload), patch.object(fc, "_yf_history", return_value=None):
            out = client.get_historical_prices("QQQ", days=2)
        assert [b["date"] for b in out["historical"]] == ["2026-08-17"]
