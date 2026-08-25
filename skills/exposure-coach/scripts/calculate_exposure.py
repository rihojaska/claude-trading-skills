#!/usr/bin/env python3
"""
Exposure Coach - Calculate market posture and exposure recommendation.

Synthesizes signals from multiple upstream skills to produce a unified
exposure ceiling, bias direction, and action recommendation.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Component weights for composite score
WEIGHTS = {
    "regime": 0.25,
    "top_risk": 0.20,
    "breadth": 0.15,
    "uptrend": 0.15,
    "institutional": 0.10,
    "sector": 0.05,
    "theme": 0.05,
    "ftd": 0.05,
}

# Critical inputs that reduce confidence when missing
CRITICAL_INPUTS = {"regime", "top_risk", "breadth"}

# Maximum age (days) of an input's own internal date before it is stale.
# Monthly-cadence detectors match the 35d market bound used by the portfolio
# staleness checker; the weekly CSV-backed ones get 8d (one cycle + a day).
INPUT_MAX_AGE_DAYS = {
    "regime": 35,
    "theme": 35,
    "sector": 35,
    "institutional": 35,
    "breadth": 8,
    "uptrend": 8,
    "top_risk": 8,
    "ftd": 8,
}

# Regime to baseline score mapping
REGIME_SCORES = {
    "broadening": 80,
    "concentration": 60,
    "transitional": 50,
    "inflationary": 40,
    "contraction": 20,
}


def load_json_file(path: Optional[Path]) -> Optional[dict]:
    """Load a JSON file if it exists and is valid."""
    if path is None or not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {path}: {e}", file=sys.stderr)
        return None


def _parse_timestamp(value) -> Optional[datetime]:
    """Parse an ISO-ish timestamp to aware UTC. None when it is not one."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        # fromisoformat rejects the military-zone suffix before 3.11.
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


# Internal-date keys, in precedence order: when the producer stamps run time it
# wins over the data date. Checked at the top level, then inside `metadata`.
# Actual DATA dates outrank report-generation times: a report regenerated
# from old market data carries a fresh generated_at alongside a stale
# latest_date, and preferring generated_at would launder that staleness
# (codex gate P1, 2026-08-25). generated_at is the LAST resort.
_DATA_DATE_KEYS = ("data_date", "as_of", "as_of_date", "date")
_GENERATED_KEYS = ("generated_at",)
_DATE_KEYS = _DATA_DATE_KEYS + _GENERATED_KEYS  # legacy alias (tests import it)


def extract_input_date(data: Optional[dict]) -> Optional[datetime]:
    """Resolve an input's own internal date, or None if it carries none.

    The live regime/theme/sector/institutional stubs carry no date at all;
    breadth/uptrend/top_risk/ftd stamp `metadata.generated_at`.
    """
    if not isinstance(data, dict):
        return None
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    freshness = (
        metadata.get("data_freshness") if isinstance(metadata.get("data_freshness"), dict) else {}
    )
    # Pass 1 — actual data dates (top-level, then metadata). A RECOGNIZED
    # data-date key holding an unusable value (the uptrend producer emits
    # latest_data_date: "N/A" when no latest row exists) is INVALID EVIDENCE,
    # not absent evidence — it must fail closed instead of laundering through
    # a fresh generated_at (codex gate r2 P1).
    invalid_evidence = False
    candidates = [data.get(k) for k in _DATA_DATE_KEYS]
    candidates += [metadata.get("latest_data_date"), freshness.get("latest_date")]
    candidates += [metadata.get(k) for k in _DATA_DATE_KEYS]
    # Per-component shapes: market-top stores metadata.data_freshness.
    # <component>.date; macro-regime stores components.<name>.current_date.
    # The OLDEST component date governs (weakest link, fail-closed).
    component_dates = []
    for v in freshness.values():
        if isinstance(v, dict):
            cd = _parse_timestamp(v.get("date"))
            if cd is not None:
                component_dates.append(cd)
            elif v.get("date") not in (None, ""):
                invalid_evidence = True
    components = data.get("components") if isinstance(data.get("components"), dict) else {}
    for v in components.values():
        if isinstance(v, dict):
            cd = _parse_timestamp(v.get("current_date"))
            if cd is not None:
                component_dates.append(cd)
            elif v.get("current_date") not in (None, ""):
                invalid_evidence = True
    for value in candidates:
        parsed = _parse_timestamp(value)
        if parsed is not None:
            return parsed
        if value not in (None, ""):
            invalid_evidence = True
    # invalid_evidence BEFORE the component return (codex r3 P1): a report
    # mixing one valid and one malformed component date is corrupt evidence —
    # market-top preserves malformed CLI date strings verbatim — and must
    # fail closed rather than ride the surviving valid date.
    if invalid_evidence:
        return None
    if component_dates:
        return min(component_dates)
    # Pass 2 — report-generation times, only when NO data-date evidence of any
    # kind (valid or invalid) exists.
    for value in (data.get("generated_at"), metadata.get("generated_at")):
        parsed = _parse_timestamp(value)
        if parsed is not None:
            return parsed
    return None


def assess_input_staleness(
    key: str,
    data: Optional[dict],
    path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> tuple[bool, Optional[float]]:
    """Age one loaded input against INPUT_MAX_AGE_DAYS. Returns (is_stale, age_days).

    Internal-date-first and fail-closed: an input carrying no resolvable date is
    stale with age None ("undated"). These sidecars are git-tracked, so a
    checkout resets mtime — an mtime read can only *age* an input here, never
    freshen one. An absent input is missing, not stale.
    """
    if data is None:
        return False, None
    now = now or datetime.now(timezone.utc)
    internal = extract_input_date(data)
    if internal is None:
        return True, None
    age_days = (now - internal).total_seconds() / 86400.0
    if age_days < -2:
        # A materially future-dated input is corrupt evidence, not fresh
        # evidence — beyond ~2d of clock skew it fails closed (codex gate P2).
        return True, age_days
    if path is not None:
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            mtime = None
        if mtime is not None:
            age_days = max(age_days, (now - mtime).total_seconds() / 86400.0)
    return age_days > INPUT_MAX_AGE_DAYS.get(key, 8), age_days


def extract_breadth_score(data: Optional[dict]) -> Optional[int]:
    """Extract breadth score from breadth analyzer output."""
    if data is None:
        return None
    # Support various field names from upstream skill
    if "breadth_score" in data:
        return int(data["breadth_score"])
    if "composite_score" in data:
        return int(data["composite_score"])
    # market-breadth-analyzer nests its 0-100 health score under "composite"
    # (100 = healthy). High = bullish, so used directly (no inversion);
    # MD-1 int(round()).
    composite = data.get("composite")
    if isinstance(composite, dict) and "composite_score" in composite:
        return int(round(composite["composite_score"]))
    if "ad_ratio" in data and "nh_nl_ratio" in data:
        ad = data["ad_ratio"]
        nh_nl = data["nh_nl_ratio"]
        if ad > 1.5 and nh_nl > 3.0:
            return 90
        elif ad >= 1.0 and nh_nl >= 1.0:
            return 65
        elif ad >= 0.7 and nh_nl >= 0.5:
            return 40
        else:
            return 20
    return None


def _uptrend_pct_to_score(pct: float) -> int:
    """Convert an uptrend participation percentage to a 0-100 score."""
    if pct > 50:
        return min(100, int(50 + pct))
    elif pct >= 35:
        return int(35 + pct)
    elif pct >= 20:
        return int(20 + pct)
    else:
        return int(pct)


def extract_uptrend_score(data: Optional[dict]) -> Optional[int]:
    """Extract uptrend participation score.

    Supports both the flat shape (``uptrend_score`` / ``uptrend_pct`` at the
    top level) and the nested shape emitted by uptrend-analyzer, which stores
    its result under a ``composite`` sub-object.
    """
    if data is None:
        return None
    if "uptrend_score" in data:
        return int(data["uptrend_score"])
    # uptrend-analyzer nests its score under "composite"
    composite = data.get("composite")
    if isinstance(composite, dict):
        if "composite_score" in composite:
            # MD-1 contract: int(round()), not truncation (57.7 -> 58)
            return int(round(composite["composite_score"]))
        if "uptrend_pct" in composite:
            return _uptrend_pct_to_score(composite["uptrend_pct"])
    if "uptrend_pct" in data:
        return _uptrend_pct_to_score(data["uptrend_pct"])
    return None


def extract_regime_score(data: Optional[dict]) -> Optional[int]:
    """Extract regime score from macro-regime-detector output.

    ``regime`` may be a flat string (legacy) or a nested object
    (``{"regime": {"current_regime": ...}}``) as emitted by
    macro-regime-detector.
    """
    if data is None:
        return None
    if "regime_score" in data:
        return int(data["regime_score"])
    regime = data.get("regime")
    if isinstance(regime, dict):
        name = regime.get("current_regime")
        if name:
            return REGIME_SCORES.get(str(name).lower().strip(), 50)
    elif isinstance(regime, str):
        return REGIME_SCORES.get(regime.lower().strip(), 50)
    if "current_regime" in data:
        return REGIME_SCORES.get(str(data["current_regime"]).lower().strip(), 50)
    return None


def extract_regime_name(data: Optional[dict]) -> str:
    """Extract regime name from data.

    Handles the legacy flat string form and the nested object form
    (``{"regime": {"regime_label": ..., "current_regime": ...}}``). For the
    nested form ``regime_label`` is preferred, falling back to
    ``current_regime``.
    """
    if data is None:
        return "Unknown"
    regime = data.get("regime")
    if isinstance(regime, dict):
        label = regime.get("regime_label") or regime.get("current_regime")
        return str(label).capitalize() if label else "Unknown"
    if isinstance(regime, str):
        return regime.capitalize()
    if "current_regime" in data:
        return str(data["current_regime"]).capitalize()
    return "Unknown"


def extract_top_risk_score(data: Optional[dict]) -> Optional[int]:
    """Extract top risk score (inverted - high risk = low score)."""
    if data is None:
        return None
    if "top_risk_score" in data:
        return int(data["top_risk_score"])
    # market-top-detector nests its 0-100 score under "composite", where HIGH =
    # higher top risk (>80 = Critical/Top Formation; verified 2026-08-11 against
    # the live sidecar: 32.6 -> "Yellow (Early Warning)", FTD logic reads
    # "composite < 40 = Green/Yellow"). Invert exactly once to the
    # exposure-friendly convention (high = safe to be exposed); MD-1 int(round()).
    composite = data.get("composite")
    if isinstance(composite, dict) and "composite_score" in composite:
        return max(0, min(100, int(round(100 - composite["composite_score"]))))
    if "top_probability" in data:
        prob = data["top_probability"]
        # Invert: high probability = low score
        return max(0, min(100, int(100 - prob)))
    if "distribution_days" in data:
        days = data["distribution_days"]
        if days <= 2:
            return 90
        elif days <= 4:
            return 65
        elif days <= 6:
            return 40
        else:
            return 15
    return None


def extract_ftd_score(data: Optional[dict]) -> Optional[int]:
    """Extract Follow-Through-Day score (high = strong FTD = bullish bottom).

    ftd-detector emits a 0-100 FTD quality score under ``quality_score``
    (``total_score``); a confirmed Follow-Through Day is a bullish
    bottom-confirmation signal, so the score is used directly (no inversion).
    """
    if data is None:
        return None
    if "ftd_score" in data:
        return int(data["ftd_score"])
    # ftd-detector real shape: {"quality_score": {"total_score": 0-100, ...}}
    quality = data.get("quality_score")
    if isinstance(quality, dict) and quality.get("total_score") is not None:
        return max(0, min(100, int(quality["total_score"])))
    if "anomaly_level" in data:
        level = data["anomaly_level"].lower()
        mapping = {"none": 90, "low": 80, "moderate": 55, "elevated": 35, "critical": 15}
        return mapping.get(level, 50)
    return None


def extract_theme_score(data: Optional[dict]) -> Optional[int]:
    """Extract theme strength score."""
    if data is None:
        return None
    if "theme_score" in data:
        return int(data["theme_score"])
    if "theme_strength" in data:
        strength = data["theme_strength"].lower()
        mapping = {"strong": 85, "stable": 65, "rotating": 40, "collapsing": 20}
        return mapping.get(strength, 50)
    return None


def extract_sector_score(data: Optional[dict]) -> Optional[int]:
    """Extract sector condition score."""
    if data is None:
        return None
    if "sector_score" in data:
        return int(data["sector_score"])
    if "dispersion" in data and "leadership" in data:
        disp = data["dispersion"]
        lead = data["leadership"].lower()
        if disp < 0.1 and lead in ["technology", "consumer discretionary"]:
            return 85
        elif disp < 0.2:
            return 65
        elif lead in ["utilities", "staples", "healthcare"]:
            return 35
        else:
            return 50
    return None


def extract_institutional_score(data: Optional[dict]) -> Optional[int]:
    """Extract institutional flow score."""
    if data is None:
        return None
    if "institutional_score" in data:
        return int(data["institutional_score"])
    if "net_flow" in data:
        flow = data["net_flow"]
        if flow > 0.5:
            return 90
        elif flow > 0:
            return 70
        elif flow > -0.5:
            return 40
        else:
            return 20
    if "flow_direction" in data:
        direction = data["flow_direction"].lower()
        mapping = {
            "strong_buying": 90,
            "buying": 70,
            "neutral": 50,
            "selling": 30,
            "strong_selling": 15,
        }
        return mapping.get(direction, 50)
    return None


def calculate_composite_score(
    scores: dict[str, Optional[int]],
    stale: Sequence[str] = (),
) -> tuple[float, list[str], list[str]]:
    """
    Calculate weighted composite score over the fresh inputs only.

    Stale inputs are excluded and the remaining weights renormalized: a stale
    score must never move the verdict-bearing ceiling. They are their own
    diagnostic bucket, so they appear in neither returned list.

    Returns:
        Tuple of (composite_score, inputs_provided, inputs_missing)
    """
    stale_set = set(stale)
    provided = []
    missing = []
    weighted_sum = 0.0
    total_weight = 0.0

    for key, weight in WEIGHTS.items():
        if key in stale_set:
            continue
        score = scores.get(key)
        if score is not None:
            weighted_sum += score * weight
            total_weight += weight
            provided.append(key)
        else:
            missing.append(key)

    if total_weight == 0:
        return 50.0, provided, missing

    composite = weighted_sum / total_weight

    # Apply haircut for critical inputs that did not reach the composite —
    # stale counts exactly like missing here.
    excluded_critical = (set(missing) | stale_set) & CRITICAL_INPUTS
    haircut = len(excluded_critical) * 10
    composite = max(0, composite - haircut)

    return composite, provided, missing


def determine_exposure_ceiling(composite: float) -> int:
    """Map composite score to exposure ceiling percentage."""
    if composite >= 80:
        return min(100, int(90 + (composite - 80)))
    elif composite >= 65:
        return int(70 + (composite - 65) * 1.3)
    elif composite >= 50:
        return int(50 + (composite - 50) * 1.3)
    elif composite >= 35:
        return int(30 + (composite - 35) * 1.3)
    elif composite >= 20:
        return int(10 + (composite - 20) * 1.3)
    else:
        return max(0, int(composite / 2))


def determine_recommendation(
    composite: float,
    top_risk_score: Optional[int],
    missing_critical: int,
    stale_critical: int = 0,
) -> str:
    """Determine action recommendation.

    stale_critical (keyword-with-default): ANY stale critical input caps the
    recommendation at REDUCE_ONLY (codex-gate r4 P1) — filtering a stale
    top_risk to None would otherwise DISABLE the <25/<40 risk guards and let
    bullish survivors emit NEW_ENTRY_ALLOWED on expired risk evidence. Cap,
    not CASH_PRIORITY: stale evidence means the risk is UNKNOWN, which blocks
    new entries but does not fabricate an active de-risking signal.
    """
    # CASH_PRIORITY conditions
    if composite < 30:
        return "CASH_PRIORITY"
    if top_risk_score is not None and top_risk_score < 25:
        return "CASH_PRIORITY"

    # REDUCE_ONLY conditions
    if composite < 50:
        return "REDUCE_ONLY"
    if top_risk_score is not None and top_risk_score < 40:
        return "REDUCE_ONLY"
    if missing_critical >= 2:
        return "REDUCE_ONLY"
    if stale_critical >= 1:
        return "REDUCE_ONLY"

    return "NEW_ENTRY_ALLOWED"


def determine_bias(
    regime_name: str,
    theme_score: Optional[int],
    sector_data: Optional[dict],
    institutional_data: Optional[dict],
) -> str:
    """Determine growth vs value bias."""
    regime_lower = regime_name.lower()

    # Strong regime signals
    if regime_lower == "inflationary":
        return "VALUE"
    if regime_lower == "contraction":
        return "DEFENSIVE"

    # Theme strength indicates growth momentum
    if theme_score is not None and theme_score > 60:
        if regime_lower in ["broadening", "concentration"]:
            return "GROWTH"

    # Sector leadership
    if sector_data and "leadership" in sector_data:
        lead = sector_data["leadership"].lower()
        if lead in ["technology", "consumer discretionary", "communications"]:
            return "GROWTH"
        if lead in ["financials", "energy", "materials", "industrials"]:
            return "VALUE"
        if lead in ["utilities", "staples", "healthcare"]:
            return "DEFENSIVE"

    # Institutional flow
    if institutional_data and "sector_flows" in institutional_data:
        flows = institutional_data["sector_flows"]
        if isinstance(flows, dict):
            growth_flow = sum(flows.get(s, 0) for s in ["Technology", "Consumer Discretionary"])
            value_flow = sum(flows.get(s, 0) for s in ["Financials", "Energy", "Industrials"])
            if growth_flow > value_flow + 0.2:
                return "GROWTH"
            if value_flow > growth_flow + 0.2:
                return "VALUE"

    return "NEUTRAL"


def determine_participation(
    uptrend_score: Optional[int], breadth_score: Optional[int], sector_data: Optional[dict]
) -> str:
    """Assess market participation breadth."""
    # Check uptrend and breadth scores
    uptrend_broad = uptrend_score is not None and uptrend_score >= 50
    breadth_broad = breadth_score is not None and breadth_score >= 50

    # Check sector dispersion if available
    low_dispersion = True
    if sector_data and "dispersion" in sector_data:
        low_dispersion = sector_data["dispersion"] < 0.15

    if uptrend_broad and breadth_broad and low_dispersion:
        return "BROAD"
    elif (uptrend_broad or breadth_broad) and low_dispersion:
        return "MODERATE"
    else:
        return "NARROW"


def determine_confidence(provided: list[str], missing: list[str], stale: Sequence[str] = ()) -> str:
    """Determine confidence level based on input completeness and freshness.

    Any stale (or undated) input caps confidence at MEDIUM; a stale critical
    input caps it at LOW — the posture is then an honest low-confidence read,
    not a HIGH rendered off months-old inputs.
    """
    stale_set = set(stale)
    if stale_set & CRITICAL_INPUTS:
        return "LOW"

    n_provided = len(provided)
    excluded_critical = len((set(missing) | stale_set) & CRITICAL_INPUTS)

    if n_provided >= 6 and excluded_critical == 0:
        level = "HIGH"
    elif n_provided >= 4 or excluded_critical <= 1:
        level = "MEDIUM"
    else:
        level = "LOW"

    return "MEDIUM" if stale_set and level == "HIGH" else level


def generate_rationale(
    composite: float,
    recommendation: str,
    participation: str,
    bias: str,
    scores: dict[str, Optional[int]],
    missing: list[str],
    exposure_ceiling: Optional[int] = None,
    stale: Sequence[str] = (),
) -> str:
    """Generate human-readable rationale."""
    stale_set = set(stale)
    parts = []

    # Participation assessment
    if participation == "BROAD":
        parts.append("Broad participation indicates healthy market internals.")
    elif participation == "NARROW":
        parts.append("Narrow participation suggests fragile market structure.")

    # Top risk assessment
    top_risk = scores.get("top_risk")
    if top_risk is not None:
        if top_risk >= 70:
            parts.append("Low distribution day count supports risk-on posture.")
        elif top_risk < 40:
            parts.append("Elevated top risk signals warrant caution.")

    # Regime context — suppressed when the regime input is stale; a months-old
    # regime read must not narrate a posture it was excluded from.
    regime = scores.get("regime")
    if regime is not None and "regime" not in stale_set:
        if regime >= 70:
            parts.append("Favorable macro regime supports elevated exposure.")
        elif regime < 40:
            parts.append("Challenging macro regime limits upside exposure.")

    # Stale inputs
    if stale_set:
        parts.append(
            f"Stale inputs ({', '.join(sorted(stale_set))}) were excluded from the composite."
        )

    # Missing inputs
    if missing:
        critical_missing = set(missing) & CRITICAL_INPUTS
        if critical_missing:
            parts.append(
                f"Missing critical inputs ({', '.join(critical_missing)}) reduce confidence."
            )

    # Recommendation context
    if recommendation == "CASH_PRIORITY":
        parts.append("Capital preservation is the priority.")
    elif recommendation == "REDUCE_ONLY":
        parts.append("New entries not recommended; consider trimming on strength.")
    else:
        # The sentence says "ceiling", so it must carry the emitted ceiling —
        # the composite is a different number (63 vs 67 on 2026-08-24).
        ceiling = int(composite) if exposure_ceiling is None else int(exposure_ceiling)
        parts.append(f"New positions allowed within the {ceiling}% ceiling.")

    return " ".join(parts)


def generate_markdown_report(result: dict) -> str:
    """Generate markdown report from result dict."""
    lines = [
        "# Market Posture Summary",
        f"**Date:** {result['generated_at'][:10]} | **Confidence:** {result['confidence']}",
        "",
        f"## Exposure Ceiling: {result['exposure_ceiling_pct']}%",
        "",
    ]

    # Freshness disclosure. Keys are read defensively: results written before
    # the staleness pass carry neither.
    stale_inputs = result.get("inputs_stale") or []
    if stale_inputs:
        rendered = []
        for entry in stale_inputs:
            age = entry.get("age_days")
            rendered.append(f"{entry['input']} ({'undated' if age is None else f'{age}d'})")
        lines.append(f"⚠ Stale inputs (excluded from the composite): {', '.join(rendered)}")
    if result.get("ceiling_decision_eligible") is False:
        lines.append("⚠ Ceiling is advisory only — a critical input is stale.")
    if stale_inputs:
        lines.append("")

    lines.extend(
        [
            "| Dimension | Score | Status |",
            "|-----------|-------|--------|",
        ]
    )

    # Add component scores
    status_map = {
        (70, 101): "Strong",
        (50, 70): "Healthy",
        (30, 50): "Cautious",
        (0, 30): "Weak",
    }

    for key in [
        "breadth",
        "uptrend",
        "regime",
        "top_risk",
        "ftd",
        "theme",
        "sector",
        "institutional",
    ]:
        score = result["component_scores"].get(f"{key}_score")
        if score is not None:
            status = "N/A"
            for (lo, hi), label in status_map.items():
                if lo <= score < hi:
                    status = label
                    break
            display_name = key.replace("_", " ").title()
            lines.append(f"| {display_name} | {score} | {status} |")

    lines.extend(
        [
            "",
            f"## Recommendation: {result['recommendation']}",
            "",
            f"**Bias:** {result['bias']}",
            f"**Participation:** {result['participation']}",
            "",
            "### Rationale",
            result["rationale"],
            "",
        ]
    )

    if result["inputs_missing"]:
        lines.extend(
            [
                "### Missing Inputs",
                ", ".join(result["inputs_missing"]),
                "",
            ]
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate market exposure posture from upstream skill outputs"
    )
    parser.add_argument("--breadth", type=Path, help="Path to breadth analyzer JSON")
    parser.add_argument("--uptrend", type=Path, help="Path to uptrend analyzer JSON")
    parser.add_argument("--regime", type=Path, help="Path to macro-regime-detector JSON")
    parser.add_argument("--top-risk", type=Path, help="Path to market-top-detector JSON")
    parser.add_argument("--ftd", type=Path, help="Path to ftd-detector JSON")
    parser.add_argument("--theme", type=Path, help="Path to theme-detector JSON")
    parser.add_argument("--sector", type=Path, help="Path to sector-analyst JSON")
    parser.add_argument(
        "--institutional", type=Path, help="Path to institutional-flow-tracker JSON"
    )
    parser.add_argument(
        "--as-of",
        help="Pin the freshness-evaluation clock (ISO date/datetime, assumed "
        "UTC) — required for deterministic historical replays; default: now",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for reports (default: reports/)",
    )
    parser.add_argument("--json-only", action="store_true", help="Output JSON only, skip markdown")

    args = parser.parse_args()
    if args.as_of:
        ts = args.as_of
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        try:
            args_now = datetime.fromisoformat(ts)
        except ValueError:
            parser.error(f"--as-of: not an ISO date/datetime: {args.as_of!r}")
        if args_now.tzinfo is None:
            args_now = args_now.replace(tzinfo=timezone.utc)
    else:
        args_now = datetime.now(timezone.utc)

    # Load all inputs
    breadth_data = load_json_file(args.breadth)
    uptrend_data = load_json_file(args.uptrend)
    regime_data = load_json_file(args.regime)
    top_risk_data = load_json_file(args.top_risk)
    ftd_data = load_json_file(args.ftd)
    theme_data = load_json_file(args.theme)
    sector_data = load_json_file(args.sector)
    institutional_data = load_json_file(args.institutional)

    # Extract scores
    scores: dict[str, Optional[int]] = {
        "breadth": extract_breadth_score(breadth_data),
        "uptrend": extract_uptrend_score(uptrend_data),
        "regime": extract_regime_score(regime_data),
        "top_risk": extract_top_risk_score(top_risk_data),
        "ftd": extract_ftd_score(ftd_data),
        "theme": extract_theme_score(theme_data),
        "sector": extract_sector_score(sector_data),
        "institutional": extract_institutional_score(institutional_data),
    }

    # Age every LOADED input — not only score-bearing ones: sector and
    # institutional raw payloads reach determine_bias even when their score
    # fails to extract, so a score-less-but-loaded input must still age out
    # (codex gate P2). A never-loaded input stays missing, not stale; a
    # loaded, score-less, aged-out input may appear in both buckets — that is
    # honest (it is stale AND unusable for the composite).
    # freshness_now (possibly --as-of-pinned) drives ONLY staleness; `now`
    # stays the real clock for generated_at and the output filename, so a
    # historical replay cannot overwrite or misorder live artifacts
    # (codex-gate r4 P2).
    freshness_now = args_now
    now = datetime.now(timezone.utc)
    input_paths = {
        "breadth": (breadth_data, args.breadth),
        "uptrend": (uptrend_data, args.uptrend),
        "regime": (regime_data, args.regime),
        "top_risk": (top_risk_data, args.top_risk),
        "ftd": (ftd_data, args.ftd),
        "theme": (theme_data, args.theme),
        "sector": (sector_data, args.sector),
        "institutional": (institutional_data, args.institutional),
    }
    stale_ages: dict[str, Optional[float]] = {}
    for key, (data, path) in input_paths.items():
        if data is None:
            continue
        is_stale, age_days = assess_input_staleness(key, data, path, now=freshness_now)
        if is_stale:
            stale_ages[key] = age_days
    stale = sorted(stale_ages)

    # A stale input is dropped from every downstream calculation exactly like a
    # missing one; it survives only in the diagnostics below.
    effective_scores = {k: (None if k in stale_ages else v) for k, v in scores.items()}
    effective_sector = None if "sector" in stale_ages else sector_data
    effective_institutional = None if "institutional" in stale_ages else institutional_data

    # Calculate composite
    composite, provided, missing = calculate_composite_score(scores, stale=stale)

    # Determine outputs
    exposure_ceiling = determine_exposure_ceiling(composite)
    excluded_critical = len((set(missing) | set(stale)) & CRITICAL_INPUTS)
    recommendation = determine_recommendation(
        composite,
        effective_scores["top_risk"],
        excluded_critical,
        stale_critical=len(set(stale) & CRITICAL_INPUTS),
    )

    regime_name = "Unknown" if "regime" in stale_ages else extract_regime_name(regime_data)
    bias = determine_bias(
        regime_name, effective_scores["theme"], effective_sector, effective_institutional
    )
    participation = determine_participation(
        effective_scores["uptrend"], effective_scores["breadth"], effective_sector
    )
    confidence = determine_confidence(provided, missing, stale=stale)

    rationale = generate_rationale(
        composite,
        recommendation,
        participation,
        bias,
        effective_scores,
        missing,
        exposure_ceiling=exposure_ceiling,
        stale=stale,
    )

    # Build result
    result = {
        "schema_version": "1.0",
        "generated_at": now.isoformat(),
        "exposure_ceiling_pct": exposure_ceiling,
        "bias": bias,
        "participation": participation,
        "recommendation": recommendation,
        "confidence": confidence,
        "composite_score": round(composite, 1),
        "component_scores": {f"{k}_score": v for k, v in effective_scores.items() if v is not None},
        "inputs_provided": provided,
        "inputs_missing": missing,
        "inputs_stale": [
            {
                "input": k,
                "age_days": None if stale_ages[k] is None else round(stale_ages[k], 1),
            }
            for k in stale
        ],
        # The ceiling stays rendered — blanking it is worse than an honest
        # low-confidence posture — but a stale CRITICAL input makes it
        # advisory-only for downstream consumers.
        "ceiling_decision_eligible": not (set(stale) & CRITICAL_INPUTS),
        "rationale": rationale,
    }

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = now.strftime("%Y-%m-%d_%H%M%S")

    # Write JSON
    json_path = args.output_dir / f"exposure_posture_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"JSON report: {json_path}")

    # Write markdown unless --json-only
    if not args.json_only:
        md_content = generate_markdown_report(result)
        md_path = args.output_dir / f"exposure_posture_{timestamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Markdown report: {md_path}")

    # Print summary to stdout
    print(f"\nExposure Ceiling: {exposure_ceiling}%")
    print(f"Recommendation: {recommendation}")
    print(f"Bias: {bias}")
    print(f"Confidence: {confidence}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
