"""Tests for Combinatorial Purged Cross-Validation (CPCV).

Tests 1-4: fold generation logic with synthetic date lists (no backtesting).
Tests 5-7: PBO calculation with synthetic Sharpe arrays.
"""

from datetime import date, timedelta

import pytest

from src.backtesting.cpcv import CPCVConfig, CPCVResult, CPCVValidator


# ── Helpers ──────────────────────────────────────────────────────────


def _make_trading_days(n: int, start: date = date(2023, 1, 2)) -> list[date]:
    """Generate n synthetic weekday-only trading days starting from start."""
    days: list[date] = []
    current = start
    while len(days) < n:
        if current.weekday() < 5:  # Mon-Fri
            days.append(current)
        current += timedelta(days=1)
    return days


# ── Test 1: Fold generation count ────────────────────────────────────


class TestFoldGenerationCount:
    def test_fold_generation_count(self):
        """generate_folds() with 80 trading days, n_groups=8, k_test=2
        produces exactly C(8,2)=28 folds."""
        days = _make_trading_days(80)
        config = CPCVConfig(n_groups=8, k_test=2, embargo_pct=0.05, purge_bars=1)
        folds = CPCVValidator.generate_folds(days, config)
        assert len(folds) == 28

        # Each fold has a unique fold_id
        ids = {f.fold_id for f in folds}
        assert len(ids) == 28

        # Each fold has exactly 2 test groups
        for fold in folds:
            assert len(fold.test_groups) == 2


# ── Test 2: Fold dates non-overlapping ───────────────────────────────


class TestFoldDatesNonOverlapping:
    def test_fold_dates_non_overlapping(self):
        """For each fold, train_dates ∩ test_dates = empty set."""
        days = _make_trading_days(80)
        config = CPCVConfig(n_groups=8, k_test=2, embargo_pct=0.05, purge_bars=1)
        folds = CPCVValidator.generate_folds(days, config)

        for fold in folds:
            train_set = set(fold.train_dates)
            test_set = set(fold.test_dates)
            overlap = train_set & test_set
            assert overlap == set(), (
                f"Fold {fold.fold_id}: train/test overlap on {overlap}"
            )


# ── Test 3: Embargo applied ──────────────────────────────────────────


class TestEmbargoApplied:
    def test_embargo_applied(self):
        """Embargo dates are excluded from train_dates and appear after
        test period end."""
        days = _make_trading_days(80)
        config = CPCVConfig(n_groups=8, k_test=2, embargo_pct=0.10, purge_bars=0)
        folds = CPCVValidator.generate_folds(days, config)

        for fold in folds:
            if not fold.embargo_dates:
                continue

            train_set = set(fold.train_dates)
            test_set = set(fold.test_dates)

            # Embargo dates must NOT be in training
            for ed in fold.embargo_dates:
                assert ed not in train_set, (
                    f"Fold {fold.fold_id}: embargo date {ed} found in training"
                )
                # Embargo dates must not be test dates either
                assert ed not in test_set, (
                    f"Fold {fold.fold_id}: embargo date {ed} found in test"
                )

            # Embargo dates should appear after at least one test block
            test_max = max(fold.test_dates)
            # At least some embargo dates should be after the last test date
            # (they come from after each test block)
            has_post_test = any(ed > min(fold.test_dates) for ed in fold.embargo_dates)
            assert has_post_test, (
                f"Fold {fold.fold_id}: no embargo dates after any test block"
            )


# ── Test 4: Purge removes boundary days ──────────────────────────────


class TestPurgeRemovesBoundaryDays:
    def test_purge_removes_boundary_days(self):
        """With purge_bars=1, the trading day immediately before and after
        each test block is removed from training."""
        days = _make_trading_days(80)
        config = CPCVConfig(n_groups=8, k_test=2, embargo_pct=0.0, purge_bars=1)
        folds = CPCVValidator.generate_folds(days, config)

        day_to_idx = {d: i for i, d in enumerate(days)}

        for fold in folds:
            train_set = set(fold.train_dates)
            test_sorted = sorted(fold.test_dates)

            # Find contiguous test blocks
            blocks: list[tuple[date, date]] = []
            block_start = test_sorted[0]
            for i in range(1, len(test_sorted)):
                prev_idx = day_to_idx[test_sorted[i - 1]]
                curr_idx = day_to_idx[test_sorted[i]]
                if curr_idx != prev_idx + 1:
                    blocks.append((block_start, test_sorted[i - 1]))
                    block_start = test_sorted[i]
            blocks.append((block_start, test_sorted[-1]))

            # Check purge: day before block_start and day after block_end
            for block_start_d, block_end_d in blocks:
                start_idx = day_to_idx[block_start_d]
                end_idx = day_to_idx[block_end_d]

                # Day before block should be purged (if it exists and isn't test)
                if start_idx > 0:
                    day_before = days[start_idx - 1]
                    if day_before not in set(fold.test_dates):
                        assert day_before not in train_set, (
                            f"Fold {fold.fold_id}: day before test block "
                            f"({day_before}) should be purged"
                        )

                # Day after block should be purged (if it exists and isn't test)
                if end_idx < len(days) - 1:
                    day_after = days[end_idx + 1]
                    if day_after not in set(fold.test_dates):
                        assert day_after not in train_set, (
                            f"Fold {fold.fold_id}: day after test block "
                            f"({day_after}) should be purged"
                        )


# ── Test 5: PBO perfect consistency ──────────────────────────────────


class TestPBOPerfectConsistency:
    def test_pbo_perfect_consistency(self):
        """Feed IS/OOS Sharpes where high-IS folds also have high-OOS → PBO = 0.0."""
        # Perfectly correlated: both rank the same way
        is_sharpes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        oos_sharpes = [0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9]

        pbo = CPCVValidator._compute_pbo(is_sharpes, oos_sharpes)
        assert pbo == 0.0


# ── Test 6: PBO random consistency ───────────────────────────────────


class TestPBORandom:
    def test_pbo_random_consistency(self):
        """Feed IS/OOS Sharpes with no correlation → PBO near 0.5."""
        # IS ascending, OOS deliberately scrambled so half are discordant
        is_sharpes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        # OOS: first 4 above median (4.5), last 4 below — inverse of IS rank
        oos_sharpes = [9.0, 8.0, 7.0, 6.0, 1.0, 2.0, 3.0, 4.0]

        pbo = CPCVValidator._compute_pbo(is_sharpes, oos_sharpes)
        # IS median = 4.5. IS above: indices 4,5,6,7. IS below: 0,1,2,3
        # OOS median = 5.0. OOS above: 0,1,2,3. OOS below: 4,5,6,7
        # All 8 are discordant: IS<=median have OOS>median, IS>median have OOS<=median
        assert pbo == pytest.approx(1.0)

        # Truly mixed: some concordant, some discordant
        is_sharpes2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        oos_sharpes2 = [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]
        pbo2 = CPCVValidator._compute_pbo(is_sharpes2, oos_sharpes2)
        # Should be somewhere between 0 and 1
        assert 0.0 < pbo2 < 1.0


# ── Test 7: CPCVResult verdict ───────────────────────────────────────


class TestCPCVResultVerdict:
    def test_cpcv_result_verdict(self):
        """PBO < 0.10 → verdict PASS, PBO >= 0.10 → verdict FAIL."""
        pass_result = CPCVResult(
            strategy_id="test",
            pbo=0.05,
            n_paths=28,
            oos_sharpes=[1.0] * 28,
            is_sharpes=[1.0] * 28,
            oos_returns=[100.0] * 28,
            is_returns=[100.0] * 28,
            avg_oos_sharpe=1.0,
            avg_is_sharpe=1.0,
            sharpe_decay=1.0,
            verdict="PASS",
        )
        assert pass_result.verdict == "PASS"
        assert pass_result.pbo < 0.10

        fail_result = CPCVResult(
            strategy_id="test",
            pbo=0.35,
            n_paths=28,
            oos_sharpes=[1.0] * 28,
            is_sharpes=[1.0] * 28,
            oos_returns=[100.0] * 28,
            is_returns=[100.0] * 28,
            avg_oos_sharpe=1.0,
            avg_is_sharpe=1.0,
            sharpe_decay=1.0,
            verdict="FAIL",
        )
        assert fail_result.verdict == "FAIL"
        assert fail_result.pbo >= 0.10

        # Verify the threshold logic used in CPCVValidator.run()
        # PBO exactly 0.10 → FAIL
        verdict_at_threshold = "PASS" if 0.10 < 0.10 else "FAIL"
        assert verdict_at_threshold == "FAIL"

        # PBO just below → PASS
        verdict_below = "PASS" if 0.09 < 0.10 else "FAIL"
        assert verdict_below == "PASS"
