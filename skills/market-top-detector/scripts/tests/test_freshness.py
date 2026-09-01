"""Tests for Data Freshness Management"""

from datetime import date, timedelta

from utils import count_business_days


def _expected_factor(biz_days: int) -> float:
    """Map business days to expected freshness factor."""
    if biz_days <= 1:
        return 1.0
    elif biz_days <= 3:
        return 0.95
    elif biz_days <= 7:
        return 0.85
    else:
        return 0.70


class TestDataFreshness:
    """Test data freshness computation."""

    def test_today_returns_1(self):
        """Data from today -> freshness factor 1.0."""
        from market_top_detector import compute_data_freshness

        today = date.today().isoformat()
        result = compute_data_freshness({"breadth_200dma_date": today})
        assert result["breadth_200dma"]["factor"] == 1.0

    def test_recent_data_factor(self):
        """Data from a few calendar days ago uses business day counting."""
        from market_top_detector import compute_data_freshness

        d = date.today() - timedelta(days=2)
        result = compute_data_freshness({"breadth_200dma_date": d.isoformat()})
        biz = count_business_days(d, date.today())
        assert result["breadth_200dma"]["factor"] == _expected_factor(biz)

    def test_week_old_data_factor(self):
        """Data from ~1 week ago uses business day counting."""
        from market_top_detector import compute_data_freshness

        d = date.today() - timedelta(days=5)
        result = compute_data_freshness({"breadth_200dma_date": d.isoformat()})
        biz = count_business_days(d, date.today())
        assert result["breadth_200dma"]["factor"] == _expected_factor(biz)

    def test_old_data_returns_070(self):
        """Data from many calendar days ago -> 0.70 (business days > 7)."""
        from market_top_detector import compute_data_freshness

        # 20 calendar days guarantees 14+ business days -> factor 0.70
        d = (date.today() - timedelta(days=20)).isoformat()
        result = compute_data_freshness({"breadth_200dma_date": d})
        assert result["breadth_200dma"]["factor"] == 0.70

    def test_no_date_returns_1(self):
        """No date provided -> assume fresh (1.0)."""
        from market_top_detector import compute_data_freshness

        result = compute_data_freshness({})
        assert result["overall_confidence"] == 1.0

    def test_no_value_returns_none(self):
        """Date given but no value -> entry should still compute."""
        from market_top_detector import compute_data_freshness

        d = date.today().isoformat()
        result = compute_data_freshness({"put_call_date": d})
        assert result["put_call"]["factor"] == 1.0

    def test_overall_confidence_is_min(self):
        """Overall confidence = min of all provided factors."""
        from market_top_detector import compute_data_freshness

        today = date.today().isoformat()
        # 20 calendar days -> 14+ business days -> factor 0.70
        old = (date.today() - timedelta(days=20)).isoformat()
        result = compute_data_freshness(
            {
                "breadth_200dma_date": today,
                "put_call_date": old,
            }
        )
        assert result["overall_confidence"] == 0.70

    def test_future_date_returns_070(self):
        """Future date should be treated as anomaly with factor 0.70."""
        from market_top_detector import compute_data_freshness

        future = (date.today() + timedelta(days=5)).isoformat()
        result = compute_data_freshness({"breadth_200dma_date": future})
        assert result["breadth_200dma"]["factor"] == 0.70
        assert result["breadth_200dma"]["age_days"] is None

    def test_weekend_tolerance(self):
        """Friday data should still be fresh on Monday (1 business day).

        Instead of mocking date, we test the underlying count_business_days
        that compute_data_freshness now uses.
        """
        from utils import count_business_days

        friday = date(2026, 3, 13)  # Friday
        monday = date(2026, 3, 16)  # Monday
        biz_days = count_business_days(friday, monday)
        # 1 business day -> factor would be 1.0 (<=1 threshold)
        assert biz_days == 1


class TestUndatedPresent:
    """compute_data_freshness records a typed 0.70 leg for value-without-date inputs."""

    def test_undated_present_emits_literal_none_date(self):
        from market_top_detector import compute_data_freshness

        result = compute_data_freshness({}, undated_present=("put_call",))
        assert result["put_call"] == {"date": None, "age_days": None, "factor": 0.70}
        assert result["overall_confidence"] == 0.70

    def test_undated_none_survives_json_round_trip(self):
        """The date must serialize as JSON null, never a string sentinel."""
        import json

        from market_top_detector import compute_data_freshness

        result = compute_data_freshness({}, undated_present=("margin_debt",))
        assert json.loads(json.dumps(result))["margin_debt"]["date"] is None

    def test_undated_present_folds_into_overall_confidence(self):
        from market_top_detector import compute_data_freshness

        today = date.today().isoformat()
        result = compute_data_freshness(
            {"breadth_200dma_date": today}, undated_present=("put_call",)
        )
        assert result["breadth_200dma"]["factor"] == 1.0
        assert result["overall_confidence"] == 0.70

    def test_absent_pair_stays_absent(self):
        """Neither value nor date -> no record at all, confidence 1.0."""
        from market_top_detector import compute_data_freshness

        result = compute_data_freshness({}, undated_present=())
        assert "put_call" not in result
        assert result["overall_confidence"] == 1.0

    def test_real_date_wins_over_undated_membership(self):
        """A label carrying a real date is dated; the undated leg must not overwrite it.

        This is the auto-breadth shape: the reconciler saw no CLI --breadth-200dma-date,
        but main() supplies the auto-fetched date, and the auto path still wins.
        """
        from market_top_detector import compute_data_freshness

        today = date.today().isoformat()
        result = compute_data_freshness(
            {"breadth_200dma_date": today}, undated_present=("breadth_200dma",)
        )
        assert result["breadth_200dma"] == {
            "date": today,
            "age_days": result["breadth_200dma"]["age_days"],
            "factor": 1.0,
        }
        assert result["overall_confidence"] == 1.0


class TestReconcileFlagPairs:
    """Orphaned value/date CLI flags are reconciled before anything consumes them."""

    @staticmethod
    def _args(**overrides):
        import argparse

        defaults = {
            "breadth_200dma": None,
            "breadth_200dma_date": None,
            "breadth_50dma": None,
            "breadth_50dma_date": None,
            "put_call": None,
            "put_call_date": None,
            "margin_debt_yoy": None,
            "margin_debt_date": None,
            "vix_term": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_orphan_date_is_dropped_with_warning(self, capsys):
        from market_top_detector import _reconcile_flag_pairs

        args = self._args(put_call_date="2026-08-01")
        undated = _reconcile_flag_pairs(args)

        assert args.put_call_date is None
        assert undated == set()
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "--put-call-date" in err
        assert "--put-call" in err

    def test_orphan_value_is_kept_and_labelled(self, capsys):
        from market_top_detector import _reconcile_flag_pairs

        args = self._args(put_call=0.67)
        undated = _reconcile_flag_pairs(args)

        assert args.put_call == 0.67
        assert args.put_call_date is None
        assert undated == {"put_call"}
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "--put-call" in err

    def test_margin_debt_pair_uses_the_yoy_value_dest(self, capsys):
        """--margin-debt-yoy is the value flag for the margin_debt date."""
        from market_top_detector import _reconcile_flag_pairs

        args = self._args(margin_debt_yoy=36.0)
        assert _reconcile_flag_pairs(args) == {"margin_debt"}

        args2 = self._args(margin_debt_date="2026-08-01")
        assert _reconcile_flag_pairs(args2) == set()
        assert args2.margin_debt_date is None

    def test_correct_pair_untouched_and_silent(self, capsys):
        from market_top_detector import _reconcile_flag_pairs

        args = self._args(put_call=0.67, put_call_date="2026-08-01")
        undated = _reconcile_flag_pairs(args)

        assert args.put_call == 0.67
        assert args.put_call_date == "2026-08-01"
        assert undated == set()
        assert capsys.readouterr().err == ""

    def test_absent_pair_untouched_and_silent(self, capsys):
        from market_top_detector import _reconcile_flag_pairs

        args = self._args()
        assert _reconcile_flag_pairs(args) == set()
        assert capsys.readouterr().err == ""

    def test_vix_term_is_exempt(self, capsys):
        """--vix-term has no date partner and must never be reconciled."""
        from market_top_detector import _reconcile_flag_pairs

        args = self._args(vix_term="contango")
        undated = _reconcile_flag_pairs(args)

        assert args.vix_term == "contango"
        assert "vix_term" not in undated
        assert undated == set()
        assert capsys.readouterr().err == ""

    def test_auto_breadth_override_still_wins_after_reconciliation(self, capsys):
        """An orphan --breadth-200dma-date is dropped, then the auto path supplies its own.

        Reproduces main()'s sequence: reconcile on raw args, then the auto-fetch branch
        (which only fires when args.breadth_200dma is None) sets breadth_auto_date.
        """
        from market_top_detector import _reconcile_flag_pairs, compute_data_freshness

        args = self._args(breadth_200dma_date="2020-01-02")
        undated = _reconcile_flag_pairs(args)
        assert args.breadth_200dma_date is None
        assert undated == set()

        # Auto path fires because args.breadth_200dma is still None.
        breadth_source = "auto"
        breadth_auto_date = date.today().isoformat()
        freshness_args = {
            "breadth_200dma_date": breadth_auto_date
            if breadth_source == "auto"
            else args.breadth_200dma_date,
        }
        result = compute_data_freshness(freshness_args, undated_present=undated)
        assert result["breadth_200dma"]["date"] == breadth_auto_date
        assert result["breadth_200dma"]["factor"] == 1.0

    def test_pairs_bind_to_real_argparse_dests(self):
        """Every _FLAG_PAIRS dest must exist on a real parsed Namespace."""
        import sys
        from unittest.mock import patch

        from market_top_detector import _FLAG_PAIRS, parse_arguments

        with patch.object(sys, "argv", ["market_top_detector.py"]):
            real_args = parse_arguments()

        for label, (value_dest, date_dest) in _FLAG_PAIRS.items():
            assert hasattr(real_args, value_dest), f"{label}: missing {value_dest}"
            assert hasattr(real_args, date_dest), f"{label}: missing {date_dest}"
        assert "vix_term" not in _FLAG_PAIRS
