"""WPP-20260818-009 — the writer invariant: strict JSON, atomic, and PAIR-atomic.

The 2026-08-18 run wrote `NaN` into three fields of
`ftd_detector_2026-08-18_091115.json`, which is not strict JSON: `json.loads`
with a `parse_constant` guard raises, and `JSON.parse` / Go's `encoding/json`
reject it outright. `scripts/regional_pulse/composite.py` already round-trips
strictly, so anything wired to that path fails hard on such a file.

These tests use an INDEPENDENT oracle — a `parse_constant`-rejecting reader —
rather than re-running `json.dumps`, which would only prove the serializer
agrees with itself.
"""

import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import report_generator as rg  # noqa: E402

NAN = float("nan")


def _strict_load(text):
    """Independent oracle: reject bare NaN/Infinity the way every other parser does."""

    def _reject(constant):
        raise ValueError(f"non-standard JSON constant: {constant}")

    return json.loads(text, parse_constant=_reject)


def _analysis(qqq=450.0, extra=None):
    payload = {
        "metadata": {"generated_at": "2026-08-27 09:00:00", "index_prices": {"sp500": 5000.0, "qqq": qqq}},
        "market_state": {"combined_state": "NO_SIGNAL", "dual_confirmation": False, "ftd_index": None},
        "sp500": {"state": "NO_SIGNAL", "current_price": 5000.0, "correction_depth_pct": None},
        "nasdaq": {"state": "NO_SIGNAL", "current_price": qqq, "correction_depth_pct": None},
        "quality_score": {"total_score": 0, "breakdown": {}, "signal": "No FTD"},
    }
    if extra:
        payload.update(extra)
    return payload


class TestStrictSerialization:
    def test_clean_payload_round_trips_through_a_strict_reader(self):
        text = rg.render_json_report(_analysis())
        assert _strict_load(text)["metadata"]["index_prices"]["qqq"] == 450.0

    def test_legitimate_nulls_are_not_an_error(self):
        """`null` is a normal value here — `correction_depth_pct` and
        `ftd_day_number` are null in real artifacts. Treating None as a failure
        would make the detector exit non-zero on its own healthy output."""
        text = rg.render_json_report(_analysis())
        assert '"correction_depth_pct": null' in text

    @pytest.mark.parametrize("bad", [NAN, float("inf"), float("-inf")])
    def test_non_finite_refuses_to_serialize(self, bad):
        """MUTANT: drop `allow_nan=False` -> `json.dumps` happily emits bare NaN."""
        with pytest.raises(ValueError):
            rg.render_json_report(_analysis(qqq=bad))

    def test_the_2026_08_18_artifact_shape_is_refused(self):
        """The exact incident payload: NaN in index_prices, current_price and gain_pct."""
        poisoned = _analysis(qqq=NAN)
        poisoned["nasdaq"]["ftd"] = {"ftd_detected": True, "gain_pct": NAN, "gain_tier": "minimum"}
        with pytest.raises(ValueError):
            rg.render_json_report(poisoned)


class TestPairAtomicity:
    def test_poisoned_payload_writes_neither_artifact(self, tmp_path):
        """The pair is the unit. Rendering JSON and writing it before the markdown
        renders would leave a current dated JSON from a failed run — and
        `promote_pulse_latest.py` globs dated sources, so it would be published.
        """
        j = tmp_path / "ftd_detector_2026-08-27_090000.json"
        m = tmp_path / "ftd_detector_2026-08-27_090000.md"
        with pytest.raises(ValueError):
            rg.write_reports(_analysis(qqq=NAN), str(j), str(m))
        assert not j.exists(), "a failed run must not leave a promotable JSON"
        assert not m.exists()

    def test_markdown_failure_also_writes_neither(self, tmp_path):
        """MUTANT: write the JSON before rendering the markdown -> the JSON survives."""
        j = tmp_path / "ftd_detector_2026-08-27_090000.json"
        m = tmp_path / "ftd_detector_2026-08-27_090000.md"
        with patch.object(rg, "render_markdown_report", side_effect=TypeError("unsupported format")):
            with pytest.raises(TypeError):
                rg.write_reports(_analysis(), str(j), str(m))
        assert not j.exists()
        assert not m.exists()

    def test_clean_payload_writes_both(self, tmp_path):
        j = tmp_path / "ftd_detector_2026-08-27_090000.json"
        m = tmp_path / "ftd_detector_2026-08-27_090000.md"
        rg.write_reports(_analysis(), str(j), str(m))
        assert _strict_load(j.read_text())["quality_score"]["total_score"] == 0
        assert m.read_text().strip()


class TestAtomicWrite:
    def test_failure_after_the_temp_write_leaves_nothing_promotable(self, tmp_path):
        """Fault-injected at `os.replace`, i.e. AFTER the temp file exists — a stub
        that raised before the side effect would hold trivially and would hold for
        a non-atomic writer too (the S-ALLOCATOR mutation).
        """
        target = tmp_path / "ftd_detector_2026-08-27_090000.json"
        with patch("report_generator.os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                rg._atomic_write_text(str(target), '{"a": 1}')
        assert not target.exists()
        leftovers = [p.name for p in tmp_path.iterdir()]
        assert leftovers == [], f"temp file left behind for the promoter to glob: {leftovers}"

    def test_write_is_a_replace_not_a_truncate(self, tmp_path):
        """A pre-existing file must never be observable as truncated: the writer
        builds a sibling temp file and `os.replace`s it into position."""
        target = tmp_path / "ftd_detector_2026-08-27_090000.json"
        target.write_text('{"previous": true}')
        with patch("report_generator.os.replace", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                rg._atomic_write_text(str(target), '{"new": true}')
        assert json.loads(target.read_text()) == {"previous": True}
