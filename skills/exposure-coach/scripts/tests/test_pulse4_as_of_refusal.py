"""S-PULSE-4 — the coach refuses `--as-of` that names TODAY's cycle (WPP-20260907-001).

The 2026-09-07 live fire ran the coach with `--as-of "$(date +%F)"` (copied from the
orchestrator's Step 2 line). The coach then stemmed `exposure_replay_<date>_…`, which
the composite's `exposure_posture_<as_of>*` glob never reads, and the US region went
PARTIAL. `--as-of` is a historical-replay switch: a pinned date on today's LOCAL
calendar (the calendar the posture stem and the composite glob share since
S-PULSE-3) is refused before any input is loaded, and nothing is written.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calculate_exposure as ce  # noqa: E402


class _FrozenLocalPastMidnight(datetime):
    """00:20 local on 2026-09-02 == 21:20 UTC on 2026-09-01 (UTC+3)."""

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls(2026, 9, 2, 0, 20, 0)
        return cls(2026, 9, 1, 21, 20, 0, tzinfo=timezone.utc).astimezone(tz)


@pytest.fixture(autouse=True)
def _tallinn_clock():
    """Pin the process timezone so `astimezone()` (host-local) and the frozen
    `datetime.now()` agree on every host — a UTC CI box would otherwise keep
    21:30Z on 09-01 (codex plan review r1 P2). `tzset()` mutates C-runtime
    state, so the ORIGINAL value is restored explicitly and re-applied after
    the test (codex gate r2 P2) — monkeypatch's own TZ restore would run after
    this teardown, leaving the runtime on the pinned zone."""
    original = os.environ.get("TZ")
    os.environ["TZ"] = "Europe/Tallinn"
    time.tzset()
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()


def _pin_tz(zone: str) -> None:
    """Switch zones inside a test; the autouse fixture restores the original."""
    os.environ["TZ"] = zone
    time.tzset()


def _run(monkeypatch, out, as_of):
    monkeypatch.setattr(ce, "datetime", _FrozenLocalPastMidnight)
    monkeypatch.setattr(
        sys,
        "argv",
        ["calculate_exposure.py", "--as-of", as_of, "--output-dir", str(out), "--json-only"],
    )
    return ce.main()


def test_as_of_on_todays_local_date_is_refused_and_writes_nothing(tmp_path, monkeypatch, capsys):
    out = tmp_path / "reports"
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, out, "2026-09-02")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--as-of" in err and "2026-09-02" in err
    assert "WPP-20260907-001" in err
    assert "historical" in err.lower() and "without --as-of" in err.lower()
    assert not out.exists() or not list(out.iterdir())


def test_utc_instant_on_todays_local_date_is_refused(tmp_path, monkeypatch):
    # 21:30Z on 09-01 is 00:30 LOCAL on 09-02 — today's cycle in the calendar the
    # stem and the composite glob use, although still yesterday in UTC.
    out = tmp_path / "reports"
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, out, "2026-09-01T21:30:00Z")
    assert exc.value.code == 2
    assert not out.exists() or not list(out.iterdir())


def test_future_local_date_is_refused(tmp_path, monkeypatch):
    out = tmp_path / "reports"
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, out, "2026-09-03")
    assert exc.value.code == 2
    assert not out.exists() or not list(out.iterdir())


def test_date_only_today_is_refused_west_of_utc(tmp_path, monkeypatch):
    # A bare date is the calendar day the operator typed. Parsed as midnight UTC
    # and shifted into New York it would read as 09-01 and slip through as a
    # "past" replay (codex gate r1 P1) — the literal date is what is compared.
    _pin_tz("America/New_York")
    out = tmp_path / "reports"
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, out, "2026-09-02")
    assert exc.value.code == 2
    assert not out.exists() or not list(out.iterdir())


def test_date_only_yesterday_still_replays_west_of_utc(tmp_path, monkeypatch):
    _pin_tz("America/New_York")
    out = tmp_path / "reports"
    assert _run(monkeypatch, out, "2026-09-01") == 0
    assert [p.name for p in out.iterdir()] == ["exposure_replay_2026-09-01_2026-09-02_002000.json"]


def test_past_local_date_still_replays_to_the_replay_stem(tmp_path, monkeypatch):
    # Pins the S-PULSE-2 codex-gate r5 P1 behaviour the branch above sits on: a
    # PAST cycle replays under a distinct prefix carrying the pinned date, never
    # under the live `exposure_posture_` stem.
    out = tmp_path / "reports"
    assert _run(monkeypatch, out, "2026-09-01") == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["exposure_replay_2026-09-01_2026-09-02_002000.json"]
