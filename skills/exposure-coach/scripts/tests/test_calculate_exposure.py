"""Tests for calculate_exposure.py."""

import json
from datetime import datetime, timedelta, timezone

from calculate_exposure import (
    CRITICAL_INPUTS,
    INPUT_MAX_AGE_DAYS,
    WEIGHTS,
    assess_input_staleness,
    calculate_composite_score,
    determine_bias,
    determine_confidence,
    determine_exposure_ceiling,
    determine_participation,
    determine_recommendation,
    extract_breadth_score,
    extract_ftd_score,
    extract_input_date,
    extract_regime_name,
    extract_regime_score,
    extract_top_risk_score,
    extract_uptrend_score,
    generate_markdown_report,
    generate_rationale,
    load_json_file,
)


def _iso_days_ago(days: float) -> str:
    """Internal-date fixture helper — always relative, never a hardcoded date
    (a pinned date silently crosses the max-age bound and reds the suite)."""
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


class TestExtractBreadthScore:
    """Tests for breadth score extraction."""

    def test_direct_breadth_score(self):
        data = {"breadth_score": 75}
        assert extract_breadth_score(data) == 75

    def test_composite_score_fallback(self):
        data = {"composite_score": 60}
        assert extract_breadth_score(data) == 60

    def test_ad_ratio_calculation_high(self):
        data = {"ad_ratio": 2.0, "nh_nl_ratio": 4.0}
        assert extract_breadth_score(data) == 90

    def test_ad_ratio_calculation_mid(self):
        data = {"ad_ratio": 1.2, "nh_nl_ratio": 1.5}
        assert extract_breadth_score(data) == 65

    def test_ad_ratio_calculation_low(self):
        data = {"ad_ratio": 0.5, "nh_nl_ratio": 0.3}
        assert extract_breadth_score(data) == 20

    def test_nested_composite_score(self):
        # market-breadth-analyzer nests its 0-100 health score under "composite"
        data = {"composite": {"composite_score": 72}}
        assert extract_breadth_score(data) == 72

    def test_flat_takes_priority_over_nested(self):
        data = {"breadth_score": 65, "composite": {"composite_score": 10}}
        assert extract_breadth_score(data) == 65

    def test_non_dict_composite_ignored(self):
        data = {"composite": "n/a", "ad_ratio": 2.0, "nh_nl_ratio": 4.0}
        assert extract_breadth_score(data) == 90

    def test_none_input(self):
        assert extract_breadth_score(None) is None

    def test_empty_dict(self):
        assert extract_breadth_score({}) is None


class TestExtractUptrendScore:
    """Tests for uptrend score extraction."""

    def test_direct_score(self):
        data = {"uptrend_score": 80}
        assert extract_uptrend_score(data) == 80

    def test_uptrend_pct_high(self):
        data = {"uptrend_pct": 60}
        score = extract_uptrend_score(data)
        assert score >= 75

    def test_uptrend_pct_mid(self):
        data = {"uptrend_pct": 40}
        score = extract_uptrend_score(data)
        assert 50 <= score <= 80

    def test_uptrend_pct_low(self):
        data = {"uptrend_pct": 15}
        score = extract_uptrend_score(data)
        assert score < 30

    def test_nested_composite_score(self):
        # uptrend-analyzer stores its score under "composite"
        data = {"composite": {"composite_score": 72}}
        assert extract_uptrend_score(data) == 72

    def test_nested_composite_uptrend_pct(self):
        data = {"composite": {"uptrend_pct": 60}}
        assert extract_uptrend_score(data) >= 75

    def test_flat_takes_priority_over_nested(self):
        data = {"uptrend_score": 80, "composite": {"composite_score": 10}}
        assert extract_uptrend_score(data) == 80

    def test_non_dict_composite_ignored(self):
        data = {"composite": "n/a", "uptrend_pct": 40}
        score = extract_uptrend_score(data)
        assert 50 <= score <= 80

    def test_none_input(self):
        assert extract_uptrend_score(None) is None

    def test_empty_dict(self):
        assert extract_uptrend_score({}) is None


class TestExtractRegimeScore:
    """Tests for regime score extraction."""

    def test_broadening_regime(self):
        data = {"regime": "Broadening"}
        assert extract_regime_score(data) == 80

    def test_contraction_regime(self):
        data = {"regime": "contraction"}
        assert extract_regime_score(data) == 20

    def test_current_regime_field(self):
        data = {"current_regime": "Transitional"}
        assert extract_regime_score(data) == 50

    def test_direct_regime_score(self):
        data = {"regime_score": 65}
        assert extract_regime_score(data) == 65

    def test_nested_regime_dict_current_regime(self):
        # macro-regime-detector emits regime as a nested object
        data = {"regime": {"current_regime": "Broadening"}}
        assert extract_regime_score(data) == 80

    def test_nested_regime_dict_unknown_defaults_50(self):
        data = {"regime": {"current_regime": "Sideways"}}
        assert extract_regime_score(data) == 50

    def test_nested_regime_dict_no_current_regime(self):
        data = {"regime": {"regime_label": "Risk-On"}}
        assert extract_regime_score(data) is None

    def test_none_input(self):
        assert extract_regime_score(None) is None

    def test_empty_dict(self):
        assert extract_regime_score({}) is None


class TestExtractRegimeName:
    """Tests for regime name extraction (incl. nested dict regression)."""

    def test_flat_string_regime(self):
        assert extract_regime_name({"regime": "broadening"}) == "Broadening"

    def test_flat_current_regime(self):
        assert extract_regime_name({"current_regime": "contraction"}) == "Contraction"

    def test_nested_label_preferred(self):
        data = {"regime": {"regime_label": "Risk-On", "current_regime": "broadening"}}
        assert extract_regime_name(data) == "Risk-on"

    def test_nested_current_regime_fallback(self):
        data = {"regime": {"current_regime": "transitional"}}
        assert extract_regime_name(data) == "Transitional"

    def test_nested_empty_dict_returns_unknown(self):
        assert extract_regime_name({"regime": {}}) == "Unknown"

    def test_dict_input_does_not_raise(self):
        # Regression: previously data["regime"].capitalize() raised on dict
        data = {"regime": {"current_regime": "broadening"}}
        result = extract_regime_name(data)
        assert isinstance(result, str)

    def test_none_input(self):
        assert extract_regime_name(None) == "Unknown"

    def test_empty_dict(self):
        assert extract_regime_name({}) == "Unknown"


class TestExtractTopRiskScore:
    """Tests for top risk score extraction."""

    def test_direct_score(self):
        data = {"top_risk_score": 30}
        assert extract_top_risk_score(data) == 30

    def test_top_probability_high(self):
        # High probability = low score (inverted)
        data = {"top_probability": 80}
        assert extract_top_risk_score(data) == 20

    def test_top_probability_low(self):
        # Low probability = high score
        data = {"top_probability": 10}
        assert extract_top_risk_score(data) == 90

    def test_distribution_days_few(self):
        data = {"distribution_days": 1}
        assert extract_top_risk_score(data) == 90

    def test_distribution_days_many(self):
        data = {"distribution_days": 8}
        assert extract_top_risk_score(data) == 15

    def test_nested_composite_inverted_high_risk(self):
        # market-top-detector composite=85 (Critical/Top Formation) -> low score
        data = {"composite": {"composite_score": 85}}
        assert extract_top_risk_score(data) == 15

    def test_nested_composite_inverted_low_risk(self):
        # composite=15 (Green/Normal) -> high (safe) score
        data = {"composite": {"composite_score": 15}}
        assert extract_top_risk_score(data) == 85

    def test_flat_takes_priority_over_nested(self):
        # explicit top_risk_score is already exposure-friendly; not inverted
        data = {"top_risk_score": 40, "composite": {"composite_score": 85}}
        assert extract_top_risk_score(data) == 40


class TestExtractFtdScore:
    """Tests for Follow-Through-Day score extraction (high = bullish, NOT inverted)."""

    def test_direct_ftd_score(self):
        assert extract_ftd_score({"ftd_score": 70}) == 70

    def test_nested_quality_score_strong(self):
        # ftd-detector real shape: strong FTD -> high score (bullish, direct)
        data = {"quality_score": {"total_score": 82, "signal": "Strong FTD"}}
        assert extract_ftd_score(data) == 82

    def test_nested_quality_score_no_ftd(self):
        data = {"quality_score": {"total_score": 0, "signal": "No FTD"}}
        assert extract_ftd_score(data) == 0

    def test_legacy_anomaly_level_still_supported(self):
        assert extract_ftd_score({"anomaly_level": "none"}) == 90

    def test_none_and_empty(self):
        assert extract_ftd_score(None) is None
        assert extract_ftd_score({}) is None


class TestRealUpstreamShapesAllCount:
    """Regression: the real upstream JSON shapes must all produce a score.

    Reproduces the reported bug where breadth/top_risk/ftd silently returned
    None (only regime + uptrend counted), forcing a missing-critical haircut
    and a CASH_PRIORITY / LOW-confidence verdict.
    """

    def test_all_five_inputs_extracted(self):
        breadth = {"composite": {"composite_score": 70}}  # market-breadth-analyzer
        uptrend = {"composite": {"composite_score": 65}}  # uptrend-analyzer
        regime = {"regime": {"current_regime": "broadening"}}  # macro-regime-detector
        top_risk = {"composite": {"composite_score": 20}}  # market-top-detector (low risk)
        ftd = {"quality_score": {"total_score": 75}}  # ftd-detector (strong FTD)

        scores = {
            "breadth": extract_breadth_score(breadth),
            "uptrend": extract_uptrend_score(uptrend),
            "regime": extract_regime_score(regime),
            "top_risk": extract_top_risk_score(top_risk),
            "ftd": extract_ftd_score(ftd),
        }
        # The bug: breadth/top_risk/ftd were None. All five must now resolve.
        assert all(v is not None for v in scores.values()), scores
        assert scores["breadth"] == 70
        assert scores["top_risk"] == 80  # inverted: 100 - 20
        assert scores["ftd"] == 75  # direct

        composite, provided, missing = calculate_composite_score(
            {**scores, "institutional": None, "sector": None, "theme": None}
        )
        # No critical input missing -> no haircut; healthy composite, not cash-priority
        assert set(missing).isdisjoint(CRITICAL_INPUTS)
        assert composite > 50


class TestCalculateCompositeScore:
    """Tests for composite score calculation."""

    def test_all_inputs_provided(self):
        scores = {
            "regime": 80,
            "top_risk": 70,
            "breadth": 65,
            "uptrend": 60,
            "institutional": 75,
            "sector": 70,
            "theme": 65,
            "ftd": 80,
        }
        composite, provided, missing = calculate_composite_score(scores)
        assert len(provided) == 8
        assert len(missing) == 0
        # Weighted average check
        expected = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
        assert abs(composite - expected) < 0.1

    def test_missing_critical_inputs(self):
        scores = {
            "regime": None,  # critical
            "top_risk": None,  # critical
            "breadth": 65,  # critical but present
            "uptrend": 60,
            "institutional": 75,
            "sector": 70,
            "theme": 65,
            "ftd": 80,
        }
        composite, provided, missing = calculate_composite_score(scores)
        assert "regime" in missing
        assert "top_risk" in missing
        # Haircut applied: 2 critical missing * 10 = 20
        assert len(provided) == 6

    def test_no_inputs(self):
        scores = {k: None for k in WEIGHTS}
        composite, provided, missing = calculate_composite_score(scores)
        assert composite == 50.0  # Default when no inputs
        assert len(provided) == 0
        assert len(missing) == 8


class TestDetermineExposureCeiling:
    """Tests for exposure ceiling mapping."""

    def test_high_composite(self):
        assert determine_exposure_ceiling(90) >= 90

    def test_mid_composite(self):
        ceiling = determine_exposure_ceiling(60)
        assert 50 <= ceiling <= 80

    def test_low_composite(self):
        ceiling = determine_exposure_ceiling(25)
        assert ceiling <= 30

    def test_very_low_composite(self):
        ceiling = determine_exposure_ceiling(10)
        assert ceiling <= 10


class TestDetermineRecommendation:
    """Tests for recommendation logic."""

    def test_cash_priority_low_composite(self):
        rec = determine_recommendation(25, 50, 0)
        assert rec == "CASH_PRIORITY"

    def test_cash_priority_low_top_risk(self):
        rec = determine_recommendation(60, 20, 0)
        assert rec == "CASH_PRIORITY"

    def test_reduce_only_mid_composite(self):
        rec = determine_recommendation(45, 50, 0)
        assert rec == "REDUCE_ONLY"

    def test_reduce_only_missing_critical(self):
        rec = determine_recommendation(60, 50, 2)
        assert rec == "REDUCE_ONLY"

    def test_new_entry_allowed(self):
        rec = determine_recommendation(70, 60, 0)
        assert rec == "NEW_ENTRY_ALLOWED"


class TestDetermineBias:
    """Tests for bias determination."""

    def test_inflationary_regime(self):
        bias = determine_bias("Inflationary", 50, None, None)
        assert bias == "VALUE"

    def test_contraction_regime(self):
        bias = determine_bias("Contraction", 50, None, None)
        assert bias == "DEFENSIVE"

    def test_broadening_with_strong_theme(self):
        bias = determine_bias("Broadening", 75, None, None)
        assert bias == "GROWTH"

    def test_sector_leadership_technology(self):
        sector_data = {"leadership": "Technology"}
        bias = determine_bias("Transitional", 50, sector_data, None)
        assert bias == "GROWTH"

    def test_sector_leadership_financials(self):
        sector_data = {"leadership": "Financials"}
        bias = determine_bias("Transitional", 50, sector_data, None)
        assert bias == "VALUE"

    def test_neutral_default(self):
        bias = determine_bias("Transitional", 50, None, None)
        assert bias == "NEUTRAL"


class TestDetermineParticipation:
    """Tests for participation assessment."""

    def test_broad_participation(self):
        part = determine_participation(70, 65, {"dispersion": 0.05})
        assert part == "BROAD"

    def test_narrow_participation(self):
        part = determine_participation(30, 35, {"dispersion": 0.25})
        assert part == "NARROW"

    def test_moderate_participation(self):
        part = determine_participation(55, 40, {"dispersion": 0.10})
        assert part == "MODERATE"


class TestDetermineConfidence:
    """Tests for confidence level."""

    def test_high_confidence(self):
        provided = list(WEIGHTS.keys())[:6]
        missing = list(WEIGHTS.keys())[6:]
        # Remove critical from missing
        missing = [m for m in missing if m not in CRITICAL_INPUTS]
        conf = determine_confidence(provided, missing)
        assert conf == "HIGH"

    def test_medium_confidence(self):
        provided = ["regime", "breadth", "uptrend", "sector"]
        missing = ["top_risk", "ftd", "theme", "institutional"]
        conf = determine_confidence(provided, missing)
        assert conf == "MEDIUM"

    def test_low_confidence(self):
        provided = ["sector", "theme"]
        missing = ["regime", "top_risk", "breadth", "uptrend", "ftd", "institutional"]
        conf = determine_confidence(provided, missing)
        assert conf == "LOW"


class TestGenerateRationale:
    """Tests for rationale generation."""

    def test_rationale_includes_participation(self):
        rationale = generate_rationale(
            70, "NEW_ENTRY_ALLOWED", "BROAD", "GROWTH", {"top_risk": 80, "regime": 75}, []
        )
        assert "Broad participation" in rationale

    def test_rationale_includes_missing_inputs(self):
        rationale = generate_rationale(
            60, "REDUCE_ONLY", "MODERATE", "NEUTRAL", {"breadth": 60}, ["regime", "top_risk"]
        )
        assert "Missing critical inputs" in rationale

    def test_rationale_cash_priority(self):
        rationale = generate_rationale(
            25, "CASH_PRIORITY", "NARROW", "DEFENSIVE", {"top_risk": 20}, []
        )
        assert "preservation" in rationale.lower()


class TestGenerateMarkdownReport:
    """Tests for markdown report generation."""

    def test_markdown_contains_exposure(self):
        result = {
            "generated_at": "2026-03-16T07:00:00Z",
            "confidence": "HIGH",
            "exposure_ceiling_pct": 75,
            "component_scores": {
                "breadth_score": 65,
                "regime_score": 80,
            },
            "recommendation": "NEW_ENTRY_ALLOWED",
            "bias": "GROWTH",
            "participation": "BROAD",
            "rationale": "Test rationale.",
            "inputs_missing": [],
        }
        md = generate_markdown_report(result)
        assert "75%" in md
        assert "NEW_ENTRY_ALLOWED" in md
        assert "GROWTH" in md

    def test_markdown_includes_missing(self):
        result = {
            "generated_at": "2026-03-16T07:00:00Z",
            "confidence": "MEDIUM",
            "exposure_ceiling_pct": 50,
            "component_scores": {"breadth_score": 60},
            "recommendation": "REDUCE_ONLY",
            "bias": "NEUTRAL",
            "participation": "NARROW",
            "rationale": "Caution advised.",
            "inputs_missing": ["regime", "top_risk"],
        }
        md = generate_markdown_report(result)
        assert "Missing Inputs" in md
        assert "regime" in md


class TestLoadJsonFile:
    """Tests for JSON file loading."""

    def test_load_valid_file(self, tmp_path):
        test_file = tmp_path / "test.json"
        test_data = {"key": "value"}
        test_file.write_text(json.dumps(test_data), encoding="utf-8")
        result = load_json_file(test_file)
        assert result == test_data

    def test_load_nonexistent_file(self, tmp_path):
        result = load_json_file(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_none_path(self):
        result = load_json_file(None)
        assert result is None

    def test_load_invalid_json(self, tmp_path):
        test_file = tmp_path / "invalid.json"
        test_file.write_text("not valid json", encoding="utf-8")
        result = load_json_file(test_file)
        assert result is None


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_with_all_inputs(self, tmp_path):
        """Test complete flow with all inputs provided."""
        import sys

        from calculate_exposure import main

        # Create mock input files
        breadth_file = tmp_path / "breadth.json"
        breadth_file.write_text(json.dumps({"breadth_score": 70}), encoding="utf-8")

        regime_file = tmp_path / "regime.json"
        regime_file.write_text(json.dumps({"regime": "Broadening"}), encoding="utf-8")

        top_risk_file = tmp_path / "top_risk.json"
        top_risk_file.write_text(json.dumps({"top_risk_score": 75}), encoding="utf-8")

        uptrend_file = tmp_path / "uptrend.json"
        uptrend_file.write_text(json.dumps({"uptrend_score": 65}), encoding="utf-8")

        output_dir = tmp_path / "reports"

        # Mock sys.argv
        original_argv = sys.argv
        sys.argv = [
            "calculate_exposure.py",
            "--breadth",
            str(breadth_file),
            "--regime",
            str(regime_file),
            "--top-risk",
            str(top_risk_file),
            "--uptrend",
            str(uptrend_file),
            "--output-dir",
            str(output_dir),
            "--json-only",
        ]

        try:
            result = main()
            assert result == 0

            # Check output files exist
            json_files = list(output_dir.glob("exposure_posture_*.json"))
            assert len(json_files) == 1

            # Validate JSON content
            with open(json_files[0]) as f:
                data = json.load(f)
            assert "exposure_ceiling_pct" in data
            assert "recommendation" in data
            assert data["confidence"] in ["HIGH", "MEDIUM", "LOW"]
        finally:
            sys.argv = original_argv

    def test_partial_inputs_reduce_confidence(self, tmp_path):
        """Test that missing critical inputs reduce confidence."""
        import sys

        from calculate_exposure import main

        # Create only one non-critical input. It carries a fresh internal date:
        # undated inputs are stale by contract and would be excluded from the
        # composite, which is a different scenario (covered below).
        sector_file = tmp_path / "sector.json"
        sector_file.write_text(
            json.dumps({"sector_score": 60, "generated_at": _iso_days_ago(1)}), encoding="utf-8"
        )

        output_dir = tmp_path / "reports"

        original_argv = sys.argv
        sys.argv = [
            "calculate_exposure.py",
            "--sector",
            str(sector_file),
            "--output-dir",
            str(output_dir),
            "--json-only",
        ]

        try:
            result = main()
            assert result == 0

            json_files = list(output_dir.glob("exposure_posture_*.json"))
            with open(json_files[0]) as f:
                data = json.load(f)

            # All critical inputs missing → LOW confidence
            assert data["confidence"] == "LOW"
            # Missing critical inputs triggers haircut → lower exposure
            assert data["exposure_ceiling_pct"] < 50
        finally:
            sys.argv = original_argv


class TestExtractInputDate:
    """Tests for the internal-date resolver (WPP-20260818-002)."""

    def test_top_level_generated_at(self):
        parsed = extract_input_date({"generated_at": "2026-08-24T07:42:00+00:00"})
        assert parsed == datetime(2026, 8, 24, 7, 42, tzinfo=timezone.utc)

    def test_metadata_generated_at_space_separated(self):
        # The live breadth/uptrend/top_risk/ftd sidecars use this exact shape.
        parsed = extract_input_date({"metadata": {"generated_at": "2026-08-24 07:42:00"}})
        assert parsed == datetime(2026, 8, 24, 7, 42, tzinfo=timezone.utc)

    def test_trailing_z_normalized(self):
        parsed = extract_input_date({"as_of": "2026-08-24T07:42:00Z"})
        assert parsed == datetime(2026, 8, 24, 7, 42, tzinfo=timezone.utc)

    def test_metadata_latest_data_date(self):
        parsed = extract_input_date({"metadata": {"latest_data_date": "2026-08-21"}})
        assert parsed == datetime(2026, 8, 21, tzinfo=timezone.utc)

    def test_no_date_anywhere(self):
        # The live regime/theme/sector/institutional stubs carry no date at all.
        assert extract_input_date({"regime": "broadening", "notes": "run 2026-06-09"}) is None

    def test_unparseable_date(self):
        assert extract_input_date({"generated_at": "last Tuesday"}) is None


class TestAssessInputStaleness:
    """Tests for the fail-closed staleness predicate."""

    def test_fresh_internal_date(self):
        data = {"generated_at": _iso_days_ago(1)}
        assert assess_input_staleness("breadth", data)[0] is False

    def test_over_max_age_is_stale(self):
        data = {"generated_at": _iso_days_ago(40)}
        is_stale, age = assess_input_staleness("regime", data)
        assert is_stale is True
        assert 39 < age < 41

    def test_undated_is_stale(self):
        is_stale, age = assess_input_staleness("regime", {"regime": "broadening"})
        assert is_stale is True
        assert age is None

    def test_absent_input_is_not_stale(self):
        # An absent input is 'missing', never 'stale' — the sets stay disjoint.
        assert assess_input_staleness("regime", None) == (False, None)

    def test_per_input_max_age_policy(self):
        data = {"generated_at": _iso_days_ago(20)}
        # 20d passes the 35d monthly bound but fails the 8d weekly one.
        assert assess_input_staleness("regime", data)[0] is False
        assert assess_input_staleness("breadth", data)[0] is True
        assert INPUT_MAX_AGE_DAYS["regime"] == 35
        assert INPUT_MAX_AGE_DAYS["breadth"] == 8

    def test_fresh_mtime_cannot_refresh_a_stale_internal_date(self, tmp_path):
        # These sidecars are git-tracked: a checkout resets mtime, so an
        # mtime read must never be able to freshen a stale internal date.
        path = tmp_path / "regime.json"
        path.write_text(json.dumps({"generated_at": _iso_days_ago(100)}), encoding="utf-8")
        is_stale, age = assess_input_staleness("regime", json.loads(path.read_text()), path)
        assert is_stale is True
        assert age > 99

    def test_stale_mtime_ages_a_fresh_internal_date(self, tmp_path):
        # mtime is a secondary bound in the fail-closed direction only.
        import os

        path = tmp_path / "regime.json"
        path.write_text(json.dumps({"generated_at": _iso_days_ago(1)}), encoding="utf-8")
        old = (datetime.now(timezone.utc) - timedelta(days=90)).timestamp()
        os.utime(path, (old, old))
        is_stale, age = assess_input_staleness("regime", json.loads(path.read_text()), path)
        assert is_stale is True
        assert age > 89


class TestCompositeStaleFilter:
    """Stale inputs are excluded from the composite and the weights renormalized."""

    def test_stale_input_excluded_and_renormalized(self):
        scores = {k: 60 for k in WEIGHTS}
        scores["theme"] = 100
        composite, provided, missing = calculate_composite_score(scores, stale=["theme"])
        assert "theme" not in provided
        assert "theme" not in missing  # stale is its own diagnostic bucket
        assert len(provided) == 7
        # Renormalized over the fresh set: the 100 never enters the average.
        assert abs(composite - 60.0) < 0.01

    def test_stale_matches_missing_for_the_math(self):
        scores = {k: 60 for k in WEIGHTS}
        stale_composite, _, _ = calculate_composite_score(scores, stale=["theme"])
        absent = dict(scores, theme=None)
        missing_composite, _, _ = calculate_composite_score(absent)
        assert stale_composite == missing_composite

    def test_stale_critical_applies_the_haircut(self):
        scores = {k: 60 for k in WEIGHTS}
        composite, _, _ = calculate_composite_score(scores, stale=["regime"])
        assert abs(composite - 50.0) < 0.01  # 60 renormalized, minus one 10pt haircut


class TestDetermineConfidenceStale:
    """Stale/undated inputs cap confidence (WPP-20260818-002)."""

    def test_stale_non_critical_caps_at_medium(self):
        provided = ["regime", "top_risk", "breadth", "uptrend", "ftd", "sector"]
        assert determine_confidence(provided, [], stale=["theme"]) == "MEDIUM"

    def test_stale_critical_caps_at_low(self):
        provided = ["top_risk", "breadth", "uptrend", "ftd", "sector", "institutional"]
        assert determine_confidence(provided, [], stale=["regime"]) == "LOW"

    def test_no_stale_argument_preserves_behaviour(self):
        provided = list(WEIGHTS.keys())[:6]
        missing = [m for m in list(WEIGHTS.keys())[6:] if m not in CRITICAL_INPUTS]
        assert determine_confidence(provided, missing) == "HIGH"


class TestGenerateRationaleCeiling:
    """The prose number must be the emitted ceiling, not the composite (-004)."""

    def test_prose_number_is_the_exposure_ceiling(self):
        rationale = generate_rationale(
            63.4,
            "NEW_ENTRY_ALLOWED",
            "BROAD",
            "GROWTH",
            {"top_risk": 80},
            [],
            exposure_ceiling=67,
        )
        assert "67% ceiling" in rationale
        assert "63%" not in rationale

    def test_ceiling_defaults_to_composite(self):
        rationale = generate_rationale(
            63.4, "NEW_ENTRY_ALLOWED", "BROAD", "GROWTH", {"top_risk": 80}, []
        )
        assert "63% ceiling" in rationale

    def test_stale_regime_clause_suppressed(self):
        rationale = generate_rationale(
            60,
            "NEW_ENTRY_ALLOWED",
            "BROAD",
            "GROWTH",
            {"top_risk": 80, "regime": 80},
            [],
            stale=["regime"],
        )
        assert "Favorable macro regime" not in rationale
        assert "regime" in rationale.lower()  # named as excluded instead


class TestMarkdownStaleLine:
    """The report discloses which inputs were dropped as stale."""

    def _result(self, **overrides):
        result = {
            "generated_at": "2026-08-25T07:00:00Z",
            "confidence": "LOW",
            "exposure_ceiling_pct": 63,
            "component_scores": {"breadth_score": 65},
            "recommendation": "NEW_ENTRY_ALLOWED",
            "bias": "NEUTRAL",
            "participation": "MODERATE",
            "rationale": "Test rationale.",
            "inputs_missing": [],
        }
        result.update(overrides)
        return result

    def test_stale_line_rendered(self):
        md = generate_markdown_report(
            self._result(
                inputs_stale=[
                    {"input": "regime", "age_days": None},
                    {"input": "theme", "age_days": 131.4},
                ],
                ceiling_decision_eligible=False,
            )
        )
        assert "⚠ Stale inputs" in md
        assert "regime (undated)" in md
        assert "theme (131.4d)" in md
        assert "advisory only" in md

    def test_absent_keys_do_not_break_legacy_results(self):
        md = generate_markdown_report(self._result())
        assert "Stale inputs" not in md
        assert "63%" in md


def _run_main(tmp_path, inputs):
    """Run main() over {key: payload} fixtures; returns the parsed result JSON."""
    import sys

    from calculate_exposure import main

    tmp_path.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "reports"
    argv = ["calculate_exposure.py", "--output-dir", str(output_dir), "--json-only"]
    for key, payload in inputs.items():
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        argv.extend([f"--{key.replace('_', '-')}", str(path)])

    original_argv = sys.argv
    sys.argv = argv
    try:
        assert main() == 0
    finally:
        sys.argv = original_argv
    files = sorted(output_dir.glob("exposure_posture_*.json"))
    assert len(files) == 1
    with open(files[0]) as f:
        return json.load(f)


class TestStaleRegimeCounterfactual:
    """A stale 25%-weight regime cannot move the verdict-bearing ceiling (P5)."""

    def _fresh_base(self):
        return {
            "breadth": {"breadth_score": 70, "generated_at": _iso_days_ago(1)},
            "uptrend": {"uptrend_score": 65, "generated_at": _iso_days_ago(1)},
            "top_risk": {"top_risk_score": 75, "generated_at": _iso_days_ago(1)},
        }

    def test_stale_regime_equals_absent_regime(self, tmp_path):
        stale_run = _run_main(
            tmp_path / "stale",
            dict(
                self._fresh_base(),
                regime={"regime": "Broadening", "generated_at": _iso_days_ago(100)},
            ),
        )
        absent_run = _run_main(tmp_path / "absent", self._fresh_base())
        assert stale_run["exposure_ceiling_pct"] == absent_run["exposure_ceiling_pct"]
        assert stale_run["recommendation"] == absent_run["recommendation"]
        assert stale_run["composite_score"] == absent_run["composite_score"]
        assert "regime_score" not in stale_run["component_scores"]
        # Diagnostics are exactly where the two runs differ.
        assert [s["input"] for s in stale_run["inputs_stale"]] == ["regime"]
        assert absent_run["inputs_stale"] == []
        assert stale_run["ceiling_decision_eligible"] is False
        assert absent_run["ceiling_decision_eligible"] is True

    def test_fresh_regime_does_move_the_ceiling(self, tmp_path):
        # Teeth check: the counterfactual above is only meaningful because a
        # FRESH regime of the same value changes the answer.
        fresh_run = _run_main(
            tmp_path / "fresh",
            dict(
                self._fresh_base(),
                regime={"regime": "Broadening", "generated_at": _iso_days_ago(1)},
            ),
        )
        absent_run = _run_main(tmp_path / "absent", self._fresh_base())
        assert fresh_run["exposure_ceiling_pct"] > absent_run["exposure_ceiling_pct"]
        assert fresh_run["ceiling_decision_eligible"] is True


class TestUndatedInputsIntegration:
    """The live reality: the regime/theme/sector/institutional stubs are undated."""

    def test_undated_critical_input_caps_confidence_and_eligibility(self, tmp_path):
        result = _run_main(
            tmp_path,
            {
                "breadth": {"breadth_score": 70, "generated_at": _iso_days_ago(1)},
                "uptrend": {"uptrend_score": 65, "generated_at": _iso_days_ago(1)},
                "top_risk": {"top_risk_score": 75, "generated_at": _iso_days_ago(1)},
                "regime": {"regime": "Broadening"},  # the live stub shape: no date
                "theme": {"theme_strength": "rotating"},
            },
        )
        assert result["confidence"] == "LOW"  # stale CRITICAL member
        assert result["ceiling_decision_eligible"] is False
        stale = {s["input"]: s["age_days"] for s in result["inputs_stale"]}
        assert stale == {"regime": None, "theme": None}
        assert "regime" not in result["inputs_provided"]
        assert "regime" not in result["inputs_missing"]

    def test_prose_ceiling_matches_the_emitted_ceiling(self, tmp_path):
        result = _run_main(
            tmp_path,
            {
                "breadth": {"breadth_score": 70, "generated_at": _iso_days_ago(1)},
                "uptrend": {"uptrend_score": 65, "generated_at": _iso_days_ago(1)},
                "top_risk": {"top_risk_score": 75, "generated_at": _iso_days_ago(1)},
            },
        )
        assert result["recommendation"] == "NEW_ENTRY_ALLOWED"
        assert f"{result['exposure_ceiling_pct']}% ceiling" in result["rationale"]

    def test_undated_non_critical_only_caps_at_medium(self, tmp_path):
        result = _run_main(
            tmp_path,
            {
                "breadth": {"breadth_score": 70, "generated_at": _iso_days_ago(1)},
                "uptrend": {"uptrend_score": 65, "generated_at": _iso_days_ago(1)},
                "top_risk": {"top_risk_score": 75, "generated_at": _iso_days_ago(1)},
                "regime": {"regime": "Broadening", "generated_at": _iso_days_ago(1)},
                "ftd": {"ftd_score": 70, "generated_at": _iso_days_ago(1)},
                "sector": {"sector_score": 60, "generated_at": _iso_days_ago(1)},
                "theme": {"theme_strength": "rotating"},  # undated
            },
        )
        assert result["confidence"] == "MEDIUM"
        assert result["ceiling_decision_eligible"] is True


import calculate_exposure


class TestInvalidDateEvidence:
    """Codex gate r2 P1: recognized-but-unusable data-date evidence fails
    closed instead of laundering through a fresh generated_at."""

    def test_na_latest_data_date_does_not_fall_back_to_generated_at(self):
        d = {"metadata": {"latest_data_date": "N/A", "generated_at": "2026-08-25 07:00:00"}}
        assert calculate_exposure.extract_input_date(d) is None

    def test_na_latest_data_date_is_stale(self):
        d = {"metadata": {"latest_data_date": "N/A", "generated_at": "2026-08-25 07:00:00"}}
        stale, age = calculate_exposure.assess_input_staleness(
            "uptrend", d, now=datetime(2026, 8, 25, 8, tzinfo=timezone.utc)
        )
        assert stale is True and age is None

    def test_market_top_per_component_dates_oldest_governs(self):
        d = {
            "metadata": {
                "generated_at": "2026-08-25 07:00:00",
                "data_freshness": {
                    "vix": {"date": "2026-08-24"},
                    "margin_debt": {"date": "2026-05-01"},
                },
            }
        }
        parsed = calculate_exposure.extract_input_date(d)
        assert parsed is not None and parsed.date().isoformat() == "2026-05-01"

    def test_generated_at_still_used_when_no_data_date_evidence_at_all(self):
        d = {"metadata": {"generated_at": "2026-08-25 07:00:00"}}
        parsed = calculate_exposure.extract_input_date(d)
        assert parsed is not None and parsed.year == 2026
