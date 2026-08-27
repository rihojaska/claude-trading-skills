#!/usr/bin/env python3
"""
FTD Detector - Main Orchestrator

Detects Follow-Through Day (FTD) signals for market bottom confirmation
using William O'Neil's methodology with dual-index tracking.

Usage:
    python3 ftd_detector.py --api-key YOUR_KEY
    python3 ftd_detector.py  # uses FMP_API_KEY env var

Output:
    - JSON: ftd_detector_YYYY-MM-DD_HHMMSS.json
    - Markdown: ftd_detector_YYYY-MM-DD_HHMMSS.md
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from fmp_client import FMPClient
from post_ftd_monitor import assess_post_ftd_health
from rally_tracker import get_market_state
from report_generator import write_reports


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FTD Detector - Follow-Through Day Bottom Confirmation"
    )
    parser.add_argument(
        "--api-key", help="FMP API key (defaults to FMP_API_KEY environment variable)"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for reports (default: current directory)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("FTD Detector - Follow-Through Day Bottom Confirmation")
    print("O'Neil Rally Attempt + FTD State Machine (Dual Index)")
    print("=" * 70)
    print()

    # Initialize FMP client
    try:
        client = FMPClient(api_key=args.api_key)
        print("FMP API client initialized")
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ========================================================================
    # Step 1: Fetch Market Data (4 API calls)
    # ========================================================================
    print()
    print("Step 1: Fetching Market Data")
    print("-" * 70)

    # S&P 500 history (60 trading days)
    print("  Fetching S&P 500 history...", end=" ", flush=True)
    sp500_history_data = client.get_historical_prices("^GSPC", days=80)
    sp500_history = sp500_history_data.get("historical", []) if sp500_history_data else []
    if sp500_history:
        print(f"OK ({len(sp500_history)} days)")
    else:
        print("FAILED")
        print("ERROR: Cannot proceed without S&P 500 data", file=sys.stderr)
        sys.exit(1)

    # NASDAQ/QQQ history (60 trading days)
    print("  Fetching NASDAQ (QQQ) history...", end=" ", flush=True)
    qqq_history_data = client.get_historical_prices("QQQ", days=80)
    qqq_history = qqq_history_data.get("historical", []) if qqq_history_data else []
    if qqq_history:
        print(f"OK ({len(qqq_history)} days)")
    else:
        print("WARN - NASDAQ data unavailable, using S&P 500 only")

    # S&P 500 quote (for current price)
    print("  Fetching S&P 500 quote...", end=" ", flush=True)
    sp500_quote_list = client.get_quote("^GSPC")
    sp500_quote = sp500_quote_list[0] if sp500_quote_list else None
    if sp500_quote:
        print(f"OK (${sp500_quote.get('price', 0):.2f})")
    else:
        print("WARN - Using historical close as current price")

    # QQQ quote
    print("  Fetching QQQ quote...", end=" ", flush=True)
    qqq_quote_list = client.get_quote("QQQ")
    qqq_quote = qqq_quote_list[0] if qqq_quote_list else None
    if qqq_quote:
        print(f"OK (${qqq_quote.get('price', 0):.2f})")
    else:
        print("WARN - Using historical close as current price")

    print()

    data_sources = client.get_api_stats().get("data_sources", {})
    data_coverage = {
        "decision_grade": bool(sp500_history and qqq_history),
        "reason": "S&P 500 and QQQ data available"
        if sp500_history and qqq_history
        else "QQQ/NASDAQ confirmation unavailable; S&P-only FTD output is degraded",
        "symbols": {
            "^GSPC": {
                "history": data_sources.get("historical:^GSPC", "missing"),
                "quote": data_sources.get("quote:^GSPC", "missing") if sp500_quote else "historical_close",
                "status": "OK" if sp500_history else "FAILED",
            },
            "QQQ": {
                "history": data_sources.get("historical:QQQ", "missing"),
                "quote": data_sources.get("quote:QQQ", "missing") if qqq_quote else "historical_close",
                "status": "OK" if qqq_history else "FAILED",
            },
        },
    }

    # ========================================================================
    # Step 2: Run State Machine (Rally Tracker)
    # ========================================================================
    print("Step 2: Analyzing Market State")
    print("-" * 70)

    market_state = get_market_state(sp500_history, qqq_history)

    sp500_state = market_state["sp500"]["state"]
    nasdaq_state = market_state["nasdaq"]["state"]
    combined = market_state["combined_state"]

    print(f"  S&P 500 State: {sp500_state}")
    print(f"  NASDAQ State:  {nasdaq_state}")
    print(f"  Combined:      {combined}")

    # Print swing low info if found
    for label, idx_data in [("S&P 500", market_state["sp500"]), ("NASDAQ", market_state["nasdaq"])]:
        swing = idx_data.get("swing_low")
        if swing:
            print(
                f"  {label} Swing Low: {swing['swing_low_date']} "
                f"(${swing['swing_low_price']:.2f}, "
                f"{swing['decline_pct']:.1f}% decline)"
            )
        rally = idx_data.get("rally_attempt")
        if rally and rally.get("day1_date"):
            print(f"  {label} Rally Day 1: {rally['day1_date']} (Day {rally['current_day_count']})")

    print()

    # ========================================================================
    # Step 3: Post-FTD Health Assessment
    # ========================================================================
    print("Step 3: Post-FTD Health Assessment")
    print("-" * 70)

    # Convert to chronological for post-FTD analysis
    sp500_chrono = list(reversed(sp500_history))
    nasdaq_chrono = list(reversed(qqq_history)) if qqq_history else []

    market_state = assess_post_ftd_health(market_state, sp500_chrono, nasdaq_chrono)

    quality = market_state.get("quality_score", {})
    print(f"  Quality Score: {quality.get('total_score', 0)}/100")
    print(f"  Signal: {quality.get('signal', 'N/A')}")
    print(f"  Guidance: {quality.get('guidance', 'N/A')}")
    print(f"  Exposure Range: {quality.get('exposure_range', 'N/A')}")

    # Power trend
    pt = market_state.get("power_trend", {})
    if pt:
        print(
            f"  Power Trend: {'YES' if pt.get('power_trend') else 'No'} "
            f"({pt.get('conditions_met', 0)}/3 conditions)"
        )

    # Post-FTD distribution
    dist = market_state.get("post_ftd_distribution", {})
    if dist:
        print(
            f"  Post-FTD Distribution Days: {dist.get('distribution_count', 0)} "
            f"(monitored {dist.get('days_monitored', 0)} days)"
        )

    # Invalidation
    inv = market_state.get("ftd_invalidation", {})
    if inv and inv.get("invalidated"):
        print(
            f"  FTD INVALIDATED on {inv.get('invalidation_date')} "
            f"({inv.get('days_after_ftd')} days after FTD)"
        )

    print()

    # ========================================================================
    # Step 4: Generate Reports
    # ========================================================================
    print("Step 4: Generating Reports")
    print("-" * 70)

    analysis = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "api_calls": client.get_api_stats(),
            "data_coverage": data_coverage,
            "index_prices": {
                "sp500": sp500_quote.get("price", 0)
                if sp500_quote
                else (sp500_history[0].get("close", 0) if sp500_history else None),
                "qqq": qqq_quote.get("price", 0)
                if qqq_quote
                else (qqq_history[0].get("close", 0) if qqq_history else None),
            },
        },
        "market_state": {
            "combined_state": market_state["combined_state"],
            "dual_confirmation": market_state["dual_confirmation"],
            "ftd_index": market_state.get("ftd_index"),
        },
        "sp500": _serialize_index(market_state["sp500"]),
        "nasdaq": _serialize_index(market_state["nasdaq"]),
        "quality_score": quality,
        "post_ftd_distribution": market_state.get("post_ftd_distribution", {}),
        "ftd_invalidation": market_state.get("ftd_invalidation", {}),
        "power_trend": market_state.get("power_trend", {}),
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    json_file = os.path.join(args.output_dir, f"ftd_detector_{timestamp}.json")
    md_file = os.path.join(args.output_dir, f"ftd_detector_{timestamp}.md")

    try:
        write_reports(analysis, json_file, md_file)
    except (ValueError, TypeError, OSError) as exc:
        # WPP-20260818-009. Reaching here means a non-finite or unformattable
        # value survived the client's validated-bar boundary, so the analysis
        # cannot be published as evidence: emit NEITHER artifact and say so.
        #
        # Exiting non-zero is safe here in a way it was NOT for the allocator
        # sidecar (S-ALLOCATOR, 2026-08-27): nothing runs this script under
        # `subprocess(check=True)` — it is invoked by the weekly-market-pulse
        # task, whose sidecar carries the typed continuation contract. P5's
        # "context never blocks" belongs at that boundary; inside the producer,
        # P5's "fail loud, emit nothing, never substitute" is what binds. The
        # prior behaviour published the JSON and then crashed on the markdown.
        #
        # The cause differs by exception type, and the diagnostic must say which:
        # the task sidecar instructs the operator to copy this text verbatim into
        # the Run Integrity table and to file a WPP row against the named layer.
        # A single hard-coded "the boundary was bypassed" line would send every
        # markdown-formatting bug to the wrong place.
        if isinstance(exc, ValueError):
            cause = (
                "  A non-finite value reached the writer, which means a provider "
                "path bypassed the client's validated-bar boundary "
                "(fmp_client._validate_history) — fix it THERE, not here."
            )
        elif isinstance(exc, TypeError):
            cause = (
                "  The Markdown renderer could not format a value — typically a "
                "legitimate None at a format site with no guard. Fix the "
                "renderer (report_generator.render_markdown_report); the client "
                "boundary is not implicated."
            )
        else:
            cause = (
                "  The report pair could not be written to disk. This is an I/O "
                "failure, not a data defect: check the output directory, its "
                "filesystem and free space."
            )
        print()
        print("=" * 70, file=sys.stderr)
        print(f"FATAL: reports were not published: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  Neither the JSON nor the Markdown artifact was written.", file=sys.stderr)
        print(cause, file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(2)

    print()
    print("=" * 70)
    print("FTD Detection Complete")
    print("=" * 70)
    print(f"  Combined State: {market_state['combined_state']}")
    print(f"  Quality Score: {quality.get('total_score', 0)}/100 ({quality.get('signal', 'N/A')})")
    print(f"  JSON Report: {json_file}")
    print(f"  Markdown Report: {md_file}")
    print()

    stats = client.get_api_stats()
    print("API Usage:")
    print(f"  API calls made: {stats['api_calls_made']}")
    print(f"  Cache entries: {stats['cache_entries']}")
    print()


def _serialize_index(idx_data: dict) -> dict:
    """Serialize index analysis for JSON output, removing large rally_days lists."""
    result = {
        "state": idx_data.get("state"),
        "current_price": idx_data.get("current_price"),
        "lookback_high": idx_data.get("lookback_high"),
        "correction_depth_pct": idx_data.get("correction_depth_pct"),
    }

    swing = idx_data.get("swing_low")
    if swing:
        result["swing_low"] = {
            "date": swing.get("swing_low_date"),
            "price": swing.get("swing_low_price"),
            "decline_pct": swing.get("decline_pct"),
            "down_days": swing.get("down_days"),
            "recent_high_date": swing.get("recent_high_date"),
            "recent_high_price": swing.get("recent_high_price"),
        }

    rally = idx_data.get("rally_attempt")
    if rally:
        result["rally_attempt"] = {
            "day1_date": rally.get("day1_date"),
            "current_day_count": rally.get("current_day_count"),
            "invalidated": rally.get("invalidated"),
            "invalidation_reason": rally.get("invalidation_reason"),
        }

    ftd = idx_data.get("ftd")
    if ftd:
        result["ftd"] = {
            "ftd_detected": ftd.get("ftd_detected"),
            "ftd_date": ftd.get("ftd_date"),
            "ftd_day_number": ftd.get("ftd_day_number"),
            "ftd_low": ftd.get("ftd_low"),
            "gain_pct": ftd.get("gain_pct"),
            "gain_tier": ftd.get("gain_tier"),
            "volume_above_avg": ftd.get("volume_above_avg"),
        }

    return result


if __name__ == "__main__":
    main()
