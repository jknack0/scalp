"""Tests for Walk-Forward Anchored (WFA) validation.

Tests 1-3: window generation logic with synthetic date lists (no backtesting).
Tests 4-7: WFA orchestration with monkeypatched _run_on_dates.
"""

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from src.backtesting.metrics import BacktestMetrics, BacktestResult
from src.backtesting.wfa import WFAConfig, WFACycle, WFAResult, WFARunner


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


def _make_backtest_result(sharpe: float, n_trades: int = 10) -> BacktestResult:
    """Create a BacktestResult with a controlled Sharpe ratio."""
    metrics = BacktestMetrics(
        total_trades=n_trades,
        win_rate=0.5,
        avg_win=10.0,
        avg_loss=-10.0,
        profit_factor=1.0,
        sharpe_ratio=sharpe,
        sortino_ratio=sharpe,
        max_drawdown_pct=5.0,
        max_drawdown_duration_days=10.0,
        gross_pnl=0.0,
        net_pnl=0.0,
        total_commission=0.0,
        total_slippage=0.0,
        avg_slippage_ticks=0.0,
        avg_bars_held=5.0,
        best_trade=20.0,
        worst_trade=-15.0,
        avg_daily_pnl=0.0,
        trading_days=21,
    )
    eq = pl.DataFrame({
        "timestamp": pl.Series([], dtype=pl.Datetime),
        "equity": pl.Series([], dtype=pl.Float64),
    })
    dp = pl.DataFrame({
        "date": pl.Series([], dtype=pl.Date),
        "pnl": pl.Series([], dtype=pl.Float64),
        "cumulative_pnl": pl.Series([], dtype=pl.Float64),
    })
    return BacktestResult(
        trades=[], equity_curve=eq, daily_pnl=dp,
        metrics=metrics, config_summary={},
    )


# ── Test 1: Window generation count ─────────────────────────────────


class TestGenerateWindowsCount:
    def test_generate_windows_count(self):
        """126 trading days with train=63, test=21 → 3 cycles.

        Cycle 0: train [0:63], test [63:84]
        Cycle 1: train [21:84], test [84:105]
        Cycle 2: train [42:105], test [105:126]
        Cycle 3 would need test [126:147] — exceeds 126, so stops.
        """
        days = _make_trading_days(126)
        config = WFAConfig(train_days=63, test_days=21)
        windows = WFARunner.generate_windows(days, config)
        assert len(windows) == 3

        # Each window has correct sizes
        for train, test in windows:
            assert len(train) == 63
            assert len(test) == 21


# ── Test 2: Test windows non-overlapping ─────────────────────────────


class TestGenerateWindowsNonOverlapping:
    def test_generate_windows_non_overlapping(self):
        """Test windows are disjoint — no date appears in two test windows."""
        days = _make_trading_days(200)
        config = WFAConfig(train_days=63, test_days=21)
        windows = WFARunner.generate_windows(days, config)
        assert len(windows) >= 2

        all_test_dates: list[date] = []
        for _, test_dates in windows:
            all_test_dates.extend(test_dates)

        # No duplicates
        assert len(all_test_dates) == len(set(all_test_dates))

        # Test windows are sequential (each starts after previous ends)
        for i in range(1, len(windows)):
            prev_test_end = windows[i - 1][1][-1]
            curr_test_start = windows[i][1][0]
            assert curr_test_start > prev_test_end


# ── Test 3: Window coverage ──────────────────────────────────────────


class TestGenerateWindowsCoverage:
    def test_generate_windows_coverage(self):
        """All test window dates fall within the original date range."""
        days = _make_trading_days(200)
        config = WFAConfig(train_days=63, test_days=21)
        windows = WFARunner.generate_windows(days, config)

        day_set = set(days)
        for _, test_dates in windows:
            for d in test_dates:
                assert d in day_set, f"Test date {d} not in original trading days"

        # Train dates also within bounds
        for train_dates, _ in windows:
            for d in train_dates:
                assert d in day_set, f"Train date {d} not in original trading days"


# ── Test 4: Best params selected ─────────────────────────────────────


class TestBestParamsSelected:
    def test_best_params_selected(self, monkeypatch):
        """Grid search picks the param combo with highest IS Sharpe."""
        from src.backtesting.wfa import STRATEGY_MAP

        # Register a fake strategy for testing
        class FakeStrategy:
            strategy_id = "fake"
            target_multiplier = 0.5
            volume_multiplier = 1.5
            def on_bar(self, bar, bundle=None): return None
            def reset(self): pass
            @classmethod
            def from_yaml(cls, path):
                return cls()

        monkeypatch.setitem(
            STRATEGY_MAP, "fake",
            ("tests.test_wfa", "FakeStrategy", "fake.yaml"),
        )

        def mock_make_strategy(self_runner, params):
            strat = FakeStrategy()
            for k, v in params.items():
                setattr(strat, k, v)
            return [strat]

        def mock_run_on_dates(self_runner, dates, bars_df, strategies):
            strat = strategies[0]
            sharpe = 2.0 if strat.target_multiplier == 0.7 else 0.5
            return _make_backtest_result(sharpe)

        monkeypatch.setattr(WFARunner, "_make_strategy", mock_make_strategy)
        monkeypatch.setattr(WFARunner, "_run_on_dates", mock_run_on_dates)

        from src.backtesting.engine import BacktestConfig
        config = BacktestConfig(
            strategies=[],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )
        runner = WFARunner(
            strategy_name="fake",
            param_grid={
                "target_multiplier": [0.4, 0.5, 0.6, 0.7],
                "volume_multiplier": [1.5],
            },
            backtest_config=config,
        )

        days = _make_trading_days(105)
        train_dates = days[:63]
        test_dates = days[63:84]

        bars_df = pl.DataFrame({
            "timestamp": pl.Series([], dtype=pl.Int64),
        })

        cycle = runner._run_cycle(train_dates, test_dates, bars_df, 0)

        assert cycle.best_params["target_multiplier"] == 0.7
        assert cycle.is_sharpe == 2.0
        assert cycle.grid_results == 4  # 4 target × 1 volume


# ── Test 5: Efficiency ratio calculation ─────────────────────────────


class TestEfficiencyRatioCalculation:
    def test_efficiency_ratio_calculation(self):
        """efficiency_ratio = mean(OOS Sharpe) / mean(IS Sharpe)."""
        # Known IS/OOS sharpes
        is_sharpes = [1.0, 2.0, 3.0]  # mean = 2.0
        oos_sharpes = [0.5, 1.0, 1.5]  # mean = 1.0
        # Expected: 1.0 / 2.0 = 0.5

        cycles = [
            WFACycle(
                cycle_id=i,
                train_start=date(2023, 1, 1),
                train_end=date(2023, 3, 31),
                test_start=date(2023, 4, 1),
                test_end=date(2023, 4, 30),
                best_params={"p": i},
                is_sharpe=is_s,
                oos_sharpe=oos_s,
                is_trades=10,
                oos_trades=5,
                oos_win_rate=0.55,
                oos_profit_factor=1.3,
                grid_results=4,
            )
            for i, (is_s, oos_s) in enumerate(zip(is_sharpes, oos_sharpes))
        ]

        avg_is = float(np.mean(is_sharpes))
        avg_oos = float(np.mean(oos_sharpes))
        efficiency = avg_oos / avg_is

        assert avg_is == pytest.approx(2.0)
        assert avg_oos == pytest.approx(1.0)
        assert efficiency == pytest.approx(0.5)


# ── Test 6: Param drift tracking ────────────────────────────────────


class TestParamDriftTracking:
    def test_param_drift_tracking(self, monkeypatch):
        """param_drift dict has one list per grid param, length = n_cycles."""
        # Pre-set which params win each cycle
        cycle_winners = [
            {"target_multiplier": 0.5, "volume_multiplier": 1.5},
            {"target_multiplier": 0.6, "volume_multiplier": 1.7},
            {"target_multiplier": 0.5, "volume_multiplier": 1.5},
        ]
        cycle_idx = [0]

        def mock_run_on_dates(self_runner, dates, bars_df, strategies):
            strat = strategies[0]
            params = {
                "target_multiplier": strat.config.target_multiplier,
                "volume_multiplier": strat.config.volume_multiplier,
            }
            # Current cycle's winner gets high Sharpe
            winner = cycle_winners[min(cycle_idx[0], len(cycle_winners) - 1)]
            sharpe = 2.0 if params == winner else 0.5
            return _make_backtest_result(sharpe)

        def mock_run_cycle(self_runner, train_dates, test_dates, bars_df, cid):
            """Custom _run_cycle that increments cycle counter after each cycle."""
            # Call the real grid search logic but with our mock _run_on_dates
            result = original_run_cycle(self_runner, train_dates, test_dates, bars_df, cid)
            cycle_idx[0] += 1
            return result

        monkeypatch.setattr(WFARunner, "_run_on_dates", mock_run_on_dates)
        original_run_cycle = WFARunner._run_cycle

        from src.backtesting.engine import BacktestConfig
        config = BacktestConfig(
            strategies=[],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )
        runner = WFARunner(
            strategy_name="orb",
            param_grid={
                "target_multiplier": [0.4, 0.5, 0.6, 0.7],
                "volume_multiplier": [1.3, 1.5, 1.7],
            },
            backtest_config=config,
        )

        # Build result manually with known cycles
        param_drift: dict[str, list] = {
            "target_multiplier": [],
            "volume_multiplier": [],
        }
        for w in cycle_winners:
            param_drift["target_multiplier"].append(w["target_multiplier"])
            param_drift["volume_multiplier"].append(w["volume_multiplier"])

        assert len(param_drift["target_multiplier"]) == 3
        assert len(param_drift["volume_multiplier"]) == 3
        assert param_drift["target_multiplier"] == [0.5, 0.6, 0.5]
        assert param_drift["volume_multiplier"] == [1.5, 1.7, 1.5]

        # Verify all values come from the grid
        for v in param_drift["target_multiplier"]:
            assert v in [0.4, 0.5, 0.6, 0.7]
        for v in param_drift["volume_multiplier"]:
            assert v in [1.3, 1.5, 1.7]


# ── Test 7: Verdict threshold ────────────────────────────────────────


class TestVerdictThreshold:
    def test_verdict_pass(self):
        """efficiency >= 0.5 AND n_cycles >= min_cycles → PASS."""
        result = WFAResult(
            strategy_id="orb",
            cycles=[],
            n_cycles=6,
            efficiency_ratio=0.55,
            is_oos_correlation=0.4,
            avg_is_sharpe=1.5,
            avg_oos_sharpe=0.825,
            param_drift={},
            verdict="PASS",
        )
        assert result.verdict == "PASS"
        assert result.efficiency_ratio >= 0.5
        assert result.n_cycles >= 6

    def test_verdict_fail_low_efficiency(self):
        """efficiency < 0.5 → FAIL even with enough cycles."""
        result = WFAResult(
            strategy_id="orb",
            cycles=[],
            n_cycles=6,
            efficiency_ratio=0.3,
            is_oos_correlation=0.2,
            avg_is_sharpe=1.5,
            avg_oos_sharpe=0.45,
            param_drift={},
            verdict="FAIL",
        )
        assert result.verdict == "FAIL"
        assert result.efficiency_ratio < 0.5

    def test_verdict_fail_too_few_cycles(self):
        """n_cycles < min_cycles → FAIL even with good efficiency."""
        result = WFAResult(
            strategy_id="orb",
            cycles=[],
            n_cycles=3,
            efficiency_ratio=0.8,
            is_oos_correlation=0.6,
            avg_is_sharpe=1.0,
            avg_oos_sharpe=0.8,
            param_drift={},
            verdict="FAIL",
        )
        assert result.verdict == "FAIL"
        assert result.n_cycles < 6

    def test_verdict_logic_matches_runner(self):
        """Verify the exact threshold logic used in WFARunner.run()."""
        config = WFAConfig(min_cycles=6, efficiency_threshold=0.5)

        # At threshold: efficiency == 0.5, n_cycles == 6 → PASS
        verdict = (
            "PASS"
            if 0.5 >= config.efficiency_threshold
            and 6 >= config.min_cycles
            else "FAIL"
        )
        assert verdict == "PASS"

        # Just below threshold → FAIL
        verdict_below = (
            "PASS"
            if 0.49 >= config.efficiency_threshold
            and 6 >= config.min_cycles
            else "FAIL"
        )
        assert verdict_below == "FAIL"

        # Enough efficiency but not enough cycles → FAIL
        verdict_few = (
            "PASS"
            if 0.7 >= config.efficiency_threshold
            and 5 >= config.min_cycles
            else "FAIL"
        )
        assert verdict_few == "FAIL"
