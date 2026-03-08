"""Tests for Deflated Sharpe Ratio (DSR).

All tests use synthetic daily return arrays — no Parquet or backtesting needed.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.backtesting.dsr import DSRConfig, DeflatedSharpeCalculator
from src.backtesting.metrics import Trade
from src.strategies.base import Direction


# ── Test 1: PSR high Sharpe ─────────────────────────────────────────


class TestPSRHighSharpe:
    def test_psr_high_sharpe(self):
        """Large positive Sharpe with many observations → PSR near 1.0."""
        rng = np.random.default_rng(42)
        # Mean = 0.001 (strong daily return), std = 0.005 → SR ≈ 0.2 daily ≈ 3.17 annual
        returns = rng.normal(0.001, 0.005, size=1000)

        result = DeflatedSharpeCalculator.compute(
            returns, n_trials=1, strategy_id="high_sharpe"
        )
        assert result.observed_sharpe > 2.0
        assert result.psr > 0.99


# ── Test 2: PSR zero Sharpe ─────────────────────────────────────────


class TestPSRZeroSharpe:
    def test_psr_zero_sharpe(self):
        """Returns with mean ≈ 0 → PSR ≈ 0.5 (coin flip vs benchmark=0)."""
        rng = np.random.default_rng(123)
        returns = rng.normal(0.0, 0.01, size=500)

        result = DeflatedSharpeCalculator.compute(
            returns, n_trials=1, strategy_id="zero_sharpe"
        )
        # PSR should be near 0.5 (not exactly, due to sampling noise)
        assert 0.2 < result.psr < 0.8


# ── Test 3: Expected max increases with trials ──────────────────────


class TestExpectedMaxIncreasesWithTrials:
    def test_expected_max_increases_with_trials(self):
        """E[max(SR)] with N=1 < N=4 < N=20."""
        T = 500
        skew = 0.0
        kurt = 0.0  # excess kurtosis of normal

        e1 = DeflatedSharpeCalculator.expected_max_sharpe(1, T, skew, kurt)
        e4 = DeflatedSharpeCalculator.expected_max_sharpe(4, T, skew, kurt)
        e20 = DeflatedSharpeCalculator.expected_max_sharpe(20, T, skew, kurt)

        assert e1 == 0.0  # No correction for single trial
        assert e4 > e1
        assert e20 > e4


# ── Test 4: Variance accounts for kurtosis ──────────────────────────


class TestVarianceAccountsForKurtosis:
    def test_variance_accounts_for_kurtosis(self):
        """Var(SR) with normal kurtosis < Var(SR) with fat tails."""
        T = 500
        sharpe = 1.0
        skew = 0.0

        var_normal = DeflatedSharpeCalculator.variance_of_sharpe(
            sharpe, T, skew, kurtosis=0.0  # excess kurtosis = 0 for normal
        )
        var_fat = DeflatedSharpeCalculator.variance_of_sharpe(
            sharpe, T, skew, kurtosis=10.0  # heavy tails
        )
        assert var_fat > var_normal


# ── Test 5: DSR more conservative than PSR ──────────────────────────


class TestDSRMoreConservativeThanPSR:
    def test_dsr_more_conservative_than_psr(self):
        """For N>1 trials: DSR <= PSR always. With N=1: DSR == PSR."""
        rng = np.random.default_rng(99)
        returns = rng.normal(0.0005, 0.01, size=500)

        # N=1: DSR should equal PSR
        result_1 = DeflatedSharpeCalculator.compute(
            returns, n_trials=1, strategy_id="single"
        )
        assert result_1.dsr == pytest.approx(result_1.psr, abs=1e-10)

        # N=4: DSR should be <= PSR
        result_4 = DeflatedSharpeCalculator.compute(
            returns, n_trials=4, strategy_id="multi"
        )
        assert result_4.dsr <= result_4.psr + 1e-10  # small tolerance


# ── Test 6: DSR verdict threshold ───────────────────────────────────


class TestDSRVerdictThreshold:
    def test_dsr_verdict_threshold(self):
        """DSR >= 0.95 → PASS, DSR < 0.95 → FAIL."""
        rng = np.random.default_rng(42)

        # Strong signal → should PASS with n_trials=1
        strong_returns = rng.normal(0.002, 0.005, size=1000)
        result_pass = DeflatedSharpeCalculator.compute(
            strong_returns, n_trials=1, strategy_id="strong"
        )
        assert result_pass.verdict == "PASS"
        assert result_pass.dsr >= 0.95

        # Weak signal with many trials → should FAIL
        weak_returns = rng.normal(0.0001, 0.01, size=200)
        result_fail = DeflatedSharpeCalculator.compute(
            weak_returns,
            n_trials=20,
            strategy_id="weak",
        )
        assert result_fail.verdict == "FAIL"
        assert result_fail.dsr < 0.95


# ── Test 7: compute_from_trades integration ─────────────────────────


class TestComputeFromTradesIntegration:
    def test_compute_from_trades_integration(self):
        """Feed synthetic Trade objects → DSRResult has valid fields."""
        base_date = datetime(2023, 1, 3, 10, 0, 0)
        trades = []
        for i in range(50):
            entry_time = base_date + timedelta(days=i)
            exit_time = entry_time + timedelta(minutes=30)
            # Mix of wins and losses
            net_pnl = 25.0 if i % 3 != 0 else -20.0
            trades.append(
                Trade(
                    trade_id=f"t{i}",
                    strategy_id="test_strat",
                    direction=Direction.LONG,
                    entry_price=4500.0,
                    exit_price=4502.0 if net_pnl > 0 else 4498.0,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    size=1,
                    gross_pnl=net_pnl + 0.70,
                    slippage_cost=0.0,
                    commission=0.70,
                    net_pnl=net_pnl,
                    exit_reason="target" if net_pnl > 0 else "stop",
                    bars_held=30,
                    entry_slippage_ticks=0.0,
                    exit_slippage_ticks=0.0,
                )
            )

        result = DeflatedSharpeCalculator.compute_from_trades(
            trades,
            initial_capital=10_000.0,
            n_trials=4,
            strategy_id="test_strat",
        )

        assert result.strategy_id == "test_strat"
        assert result.n_trials == 4
        assert result.sample_size > 0
        assert result.observed_sharpe != 0.0
        assert 0.0 <= result.psr <= 1.0
        assert 0.0 <= result.dsr <= 1.0
        assert result.expected_max_sharpe > 0.0  # n_trials=4
        assert result.verdict in ("PASS", "FAIL")
