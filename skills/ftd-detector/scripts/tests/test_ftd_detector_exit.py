"""WPP-20260818-009 — `main()`'s failure contract at the process boundary.

Reaching the writer with a non-finite value means a provider path bypassed the
client's validated-bar boundary. The producer's job then is P5's: fail loud,
emit nothing, never substitute. The non-blocking half of P5 lives at the task
boundary (`.claude/scheduled-tasks/weekly-market-pulse-multiregion.md`), not
inside the producer — nothing runs this script under `subprocess(check=True)`.

Prior behaviour on this path: publish the JSON, then crash on the markdown.

Driving `main()` needs no refactor — this reuses the harness that
`test_fmp_client.py:TestCallerRegression` already established, including its
warning: patch `ftd_detector.FMPClient`, never this module's own import, because
conftest evicts and re-imports `fmp_client` between skill suites.
"""

import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

NAN = float("nan")


def _history(base, *, days=80, poison_latest=False):
    """Most-recent-first bars, optionally with the incident's unsettled latest bar."""
    bars = [
        {
            "date": f"2026-03-{20 - i:02d}",
            "open": base - i,
            "high": base - i + 5,
            "low": base - i - 5,
            "close": base - i,
            "adjClose": base - i,
            "volume": 3_000_000,
        }
        for i in range(days)
    ]
    if poison_latest:
        bars[0] = {**bars[0], "close": NAN, "open": NAN, "high": NAN, "low": NAN}
    return {"symbol": "X", "historical": bars}


def _run(tmp_path, hist_fn, quote=None):
    with (
        patch.dict(os.environ, {"FMP_API_KEY": "k"}),  # pragma: allowlist secret
        patch("sys.argv", ["ftd_detector.py", "--output-dir", str(tmp_path)]),
    ):
        import ftd_detector

        with (
            patch.object(ftd_detector.FMPClient, "get_historical_prices", side_effect=hist_fn),
            patch.object(ftd_detector.FMPClient, "get_quote", return_value=quote),
        ):
            ftd_detector.main()


def _artifacts(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir() if p.is_file())


class TestWriterFailureIsFatalAndSilent:
    def test_poisoned_payload_exits_two_and_emits_neither_artifact(self, tmp_path):
        """The boundary is patched OUT here on purpose — this is the residual case
        L3 exists for: a provider path that reached the writer unvalidated.

        MUTANT: swallow the ValueError and continue -> exit 0 with artifacts on
        disk, i.e. the false success this contract forbids.
        """

        def hist(symbol, days=365):
            return _history(5000.0 if symbol == "^GSPC" else 450.0,
                            poison_latest=(symbol == "QQQ"))

        with pytest.raises(SystemExit) as exc:
            _run(tmp_path, hist)
        assert exc.value.code == 2
        assert _artifacts(tmp_path) == [], "a failed run must publish nothing"


class TestDegradedRunStillPublishes:
    def test_absent_qqq_history_exits_zero_and_labels_itself_not_decision_grade(self, tmp_path):
        """The documented S&P-only degradation is a FEATURE, not a failure: it is
        honest about itself. What the boundary now does is route a poisoned QQQ
        payload HERE instead of into a fabricated Follow-Through Day.
        """

        def hist(symbol, days=365):
            return _history(5000.0) if symbol == "^GSPC" else None

        _run(tmp_path, hist)
        names = _artifacts(tmp_path)
        assert len(names) == 2, names
        payload = json.loads((tmp_path / [n for n in names if n.endswith(".json")][0]).read_text())
        coverage = payload["metadata"]["data_coverage"]
        assert coverage["decision_grade"] is False
        assert coverage["symbols"]["QQQ"]["status"] == "FAILED"

    def test_null_bearing_payload_does_not_trip_the_exit(self, tmp_path):
        """`null` is legitimate output (`correction_depth_pct`, `ftd_day_number`).
        Conflating None with non-finite would make the detector exit non-zero on
        its own healthy artifact.

        MUTANT: treat None as a fatal payload value -> this run exits 2.
        """

        def hist(symbol, days=365):
            return _history(5000.0 if symbol == "^GSPC" else 450.0)

        _run(tmp_path, hist)
        json_name = [n for n in _artifacts(tmp_path) if n.endswith(".json")][0]
        text = (tmp_path / json_name).read_text()
        assert "null" in text, "expected at least one legitimate null in a healthy artifact"
        assert "NaN" not in text
