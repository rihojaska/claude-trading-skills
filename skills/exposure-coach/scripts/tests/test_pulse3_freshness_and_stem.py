"""S-PULSE-3 pins on the exposure-coach side.

* WPP-20260902-001 — the `exposure_posture_` stem follows the LOCAL calendar
  (like every other producer in the Monday chain), so a 00:00–03:00 local run
  writes the file the composite's local-date glob looks for.
* WPP-20260901-018 — an FTD artifact carrying `metadata.latest_data_date`
  ages from that date, not from generated_at.
* WPP-20260901-019 — a `vix_term` leg in market-top's `data_freshness` enters
  the oldest-component min (weakest link): a stale VIX date stales top_risk.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calculate_exposure as ce  # noqa: E402


class _FrozenLocalPastMidnight(datetime):
    """00:20 local on 2026-09-02 == 21:20 UTC on 2026-09-01 (UTC+3)."""

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls(2026, 9, 2, 0, 20, 0)
        return cls(2026, 9, 1, 21, 20, 0, tzinfo=timezone.utc).astimezone(tz)


def test_posture_stem_uses_the_local_date(tmp_path, monkeypatch):
    out = tmp_path / "reports"
    monkeypatch.setattr(ce, "datetime", _FrozenLocalPastMidnight)
    monkeypatch.setattr(sys, "argv", ["calculate_exposure.py", "--output-dir", str(out), "--json-only"])
    assert ce.main() == 0
    files = sorted(out.glob("exposure_posture_*.json"))
    assert [f.name for f in files] == ["exposure_posture_2026-09-02_002000.json"]
    payload = json.loads(files[0].read_text())
    # generated_at keeps the real (UTC) clock — only the filename follows the local calendar
    stamped = json.dumps(payload)
    assert "2026-09-01T21:20" in stamped and "2026-09-02T00:20" not in stamped


def _now():
    return datetime(2026, 9, 7, 4, 36, tzinfo=timezone.utc)


def test_ftd_latest_data_date_governs_over_generated_at(tmp_path):
    data = {"metadata": {"generated_at": "2026-09-07 07:35:00", "latest_data_date": "2026-09-04"}}
    assert ce.extract_input_date(data) == datetime(2026, 9, 4, tzinfo=timezone.utc)
    p = tmp_path / "ftd.json"
    p.write_text(json.dumps(data))
    is_stale, age, reason = ce.classify_input_staleness("ftd", data, p, now=_now())
    assert not is_stale and 2.5 < age < 3.5 and reason is None


def test_stale_vix_term_leg_stales_top_risk(tmp_path):
    fresh = "2026-09-04"
    stale = (_now() - timedelta(days=12)).date().isoformat()
    data = {"metadata": {"generated_at": "2026-09-07 07:35:00",
                         "data_freshness": {"put_call": {"date": fresh, "factor": 1.0},
                                            "vix_term": {"date": stale, "factor": 0.70}}}}
    assert ce.extract_input_date(data).date().isoformat() == stale
    p = tmp_path / "top.json"
    p.write_text(json.dumps(data))
    is_stale, age, reason = ce.classify_input_staleness("top_risk", data, p, now=_now())
    assert is_stale and reason == "age"
    # and without the leg the same artifact is fresh — the leg is the only difference
    data["metadata"]["data_freshness"].pop("vix_term")
    assert not ce.classify_input_staleness("top_risk", data, p, now=_now())[0]
