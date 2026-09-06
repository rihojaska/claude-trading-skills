"""WPP-20260901-018 — the FTD artifact stamps `metadata.latest_data_date` =
the OLDEST latest-bar across the histories the computation used (S&P and QQQ),
so the exposure-coach ages it from data, not from generated_at. Reuses the
`main()` harness of test_ftd_detector_exit.py (patch ftd_detector.FMPClient).
"""
import glob
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _history(base, latest_day, *, days=80):
    bars = [
        {"date": f"2026-03-{latest_day - i:02d}" if latest_day - i > 0 else f"2026-02-{28 + (latest_day - i):02d}",
         "open": base - i, "high": base - i + 5, "low": base - i - 5, "close": base - i, "adjClose": base - i, "volume": 3_000_000}
        for i in range(days)
    ]
    return {"symbol": "X", "historical": bars}


def _run(tmp_path, hist_fn):
    with (
        patch.dict(os.environ, {"FMP_API_KEY": "k"}),  # pragma: allowlist secret
        patch("sys.argv", ["ftd_detector.py", "--output-dir", str(tmp_path)]),
    ):
        import ftd_detector

        with (
            patch.object(ftd_detector.FMPClient, "get_historical_prices", side_effect=hist_fn),
            patch.object(ftd_detector.FMPClient, "get_quote", return_value=None),
        ):
            try:
                ftd_detector.main()
            except SystemExit as e:  # main() may exit 0 explicitly
                assert e.code in (0, None), e.code
    files = glob.glob(os.path.join(str(tmp_path), "ftd_*.json")) or glob.glob(os.path.join(str(tmp_path), "*.json"))
    assert len(files) == 1, files
    return json.load(open(files[0]))["metadata"]


def test_oldest_latest_bar_across_sp500_and_qqq_governs(tmp_path):
    def hist(symbol, days=80):
        return _history(5000, 20) if symbol == "^GSPC" else _history(400, 18)

    md = _run(tmp_path, hist)
    assert md["latest_data_date"] == "2026-03-18"  # QQQ's older bar, not S&P's 03-20


def test_sp500_only_when_qqq_absent(tmp_path):
    def hist(symbol, days=80):
        return _history(5000, 20) if symbol == "^GSPC" else None

    md = _run(tmp_path, hist)
    assert md["latest_data_date"] == "2026-03-20"
    assert md["data_coverage"]["decision_grade"] is False
