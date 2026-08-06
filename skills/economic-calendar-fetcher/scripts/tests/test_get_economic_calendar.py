"""Tests for get_economic_calendar.py"""

import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path so we can import the script module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import get_economic_calendar as mod
from get_economic_calendar import (
    fetch_economic_calendar,
    format_event_output,
    get_api_key,
    validate_date_range,
)

# ---------------------------------------------------------------------------
# Sample fixtures
# ---------------------------------------------------------------------------

SAMPLE_EVENTS = [
    {
        "date": "2025-01-15 14:30:00",
        "country": "US",
        "event": "Consumer Price Index (CPI) YoY",
        "currency": "USD",
        "previous": 2.6,
        "estimate": 2.7,
        "actual": None,
        "change": None,
        "impact": "High",
        "changePercentage": None,
    },
    {
        "date": "2025-01-16 10:00:00",
        "country": "EU",
        "event": "ECB Interest Rate Decision",
        "currency": "EUR",
        "previous": 4.5,
        "estimate": 4.5,
        "actual": None,
        "change": None,
        "impact": "High",
        "changePercentage": None,
    },
]


# ---------------------------------------------------------------------------
# get_api_key tests
# ---------------------------------------------------------------------------


class TestGetApiKey:
    def test_returns_key_when_set(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test_key_123")
        assert get_api_key() == "test_key_123"

    def test_returns_none_when_not_set(self, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        assert get_api_key() is None


# ---------------------------------------------------------------------------
# fetch_economic_calendar tests
# ---------------------------------------------------------------------------


class TestFetchEconomicCalendar:
    """fetch_economic_calendar routes through fmp_compat.fmp_get.

    fmp_get swallows transport-level HTTP status (a 402-block, a 404 and a
    genuine empty all surface as None), so the urllib-level 402/404
    distinctions cannot be observed here. What must hold instead: any no-data
    outcome fails loud as ValueError — never silently "zero events".
    """

    def test_uses_singular_stable_endpoint(self, monkeypatch):
        captured = {}

        def fake_fmp_get(url, params=None, timeout=30, **_kw):
            captured["url"] = url
            captured["params"] = params
            return []

        monkeypatch.setattr(mod, "fmp_get", fake_fmp_get)
        fetch_economic_calendar("2025-01-01", "2025-01-07", "test_key")

        assert captured["url"] == "/stable/economic-calendar"
        assert "economics-calendar" not in captured["url"]
        assert captured["params"] == {"from": "2025-01-01", "to": "2025-01-07"}

    def test_success_returns_events(self, monkeypatch):
        monkeypatch.setattr(mod, "fmp_get", lambda *a, **k: list(SAMPLE_EVENTS))
        events = fetch_economic_calendar("2025-01-01", "2025-01-07", "test_key")
        assert events == SAMPLE_EVENTS

    def test_no_data_fails_loud(self, monkeypatch):
        # 402/404/quota/network all collapse to None in fmp_get — must raise,
        # never be treated as an empty calendar.
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        monkeypatch.setattr(mod, "fmp_get", lambda *a, **k: None)
        with pytest.raises(ValueError, match="failed|exhausted"):
            fetch_economic_calendar("2025-01-01", "2025-01-07", "test_key")

    def test_non_list_response_raises(self, monkeypatch):
        monkeypatch.setattr(mod, "fmp_get", lambda *a, **k: {"Error Message": "nope"})
        with pytest.raises(ValueError, match="Unexpected API response"):
            fetch_economic_calendar("2025-01-01", "2025-01-07", "test_key")

    def test_none_then_retry_with_explicit_key(self, monkeypatch):
        # First call fails with no env key; the explicit api_key argument is
        # exposed to fmp_compat for the retry, then the environment restored.
        calls = []

        def fake_fmp_get(url, params=None, timeout=30, **_kw):
            calls.append(os.environ.get("FMP_API_KEY"))
            return None if len(calls) == 1 else list(SAMPLE_EVENTS)

        monkeypatch.delenv("FMP_API_KEY", raising=False)
        monkeypatch.setattr(mod, "fmp_get", fake_fmp_get)
        events = fetch_economic_calendar("2025-01-01", "2025-01-07", "explicit_key")

        assert events == SAMPLE_EVENTS
        assert calls[1] == "explicit_key"
        assert os.environ.get("FMP_API_KEY") is None


# ---------------------------------------------------------------------------
# validate_date_range tests
# ---------------------------------------------------------------------------


class TestValidateDateRange:
    def test_valid_range(self):
        validate_date_range("2025-01-01", "2025-01-31")

    def test_same_day(self):
        validate_date_range("2025-06-15", "2025-06-15")

    def test_max_90_days(self):
        validate_date_range("2025-01-01", "2025-03-31")  # 89 days

    def test_exceeds_90_days(self):
        with pytest.raises(ValueError, match="exceeds maximum of 90 days"):
            validate_date_range("2025-01-01", "2025-06-01")

    def test_start_after_end(self):
        with pytest.raises(ValueError, match="after end date"):
            validate_date_range("2025-03-01", "2025-01-01")

    def test_invalid_date_format(self):
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_range("01-01-2025", "2025-01-31")

    def test_invalid_date_value(self):
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_range("2025-13-01", "2025-14-01")

    def test_past_dates_warns(self, capsys):
        past = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        past_end = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
        validate_date_range(past, past_end)
        captured = capsys.readouterr()
        assert "in the past" in captured.err


# ---------------------------------------------------------------------------
# format_event_output tests
# ---------------------------------------------------------------------------


class TestFormatEventOutput:
    def test_json_format_roundtrip(self):
        output = format_event_output(SAMPLE_EVENTS, "json")
        parsed = json.loads(output)
        assert len(parsed) == 2
        assert parsed[0]["event"] == "Consumer Price Index (CPI) YoY"

    def test_json_empty_list(self):
        output = format_event_output([], "json")
        assert json.loads(output) == []

    def test_text_format_header(self):
        output = format_event_output(SAMPLE_EVENTS, "text")
        assert "Total: 2" in output

    def test_text_format_contains_event_name(self):
        output = format_event_output(SAMPLE_EVENTS, "text")
        assert "Consumer Price Index (CPI) YoY" in output
        assert "ECB Interest Rate Decision" in output

    def test_text_format_shows_previous(self):
        output = format_event_output(SAMPLE_EVENTS, "text")
        assert "Previous: 2.6" in output

    def test_text_format_omits_none_actual(self):
        output = format_event_output(SAMPLE_EVENTS, "text")
        assert "Actual:" not in output

    def test_text_format_shows_actual_when_present(self):
        events = [
            {
                "date": "2025-01-10 14:30:00",
                "country": "US",
                "event": "NFP",
                "currency": "USD",
                "previous": 200,
                "estimate": 210,
                "actual": 256,
                "change": 56,
                "impact": "High",
                "changePercentage": 28.0,
            }
        ]
        output = format_event_output(events, "text")
        assert "Actual: 256" in output
        assert "Change: 56" in output
        assert "Change %: 28.0%" in output

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown output format"):
            format_event_output([], "csv")
