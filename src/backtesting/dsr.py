"""Deflated Sharpe Ratio — Bailey & Lopez de Prado (2014).

Corrects observed Sharpe ratios for multiple testing bias. Two key metrics:

- **PSR** (Probabilistic Sharpe Ratio): probability that the true Sharpe
  exceeds a benchmark, accounting for non-normal returns (skewness, kurtosis).

- **DSR** (Deflated Sharpe Ratio): PSR with the benchmark set to the expected
  maximum Sharpe under the null hypothesis (all strategies have zero true
  Sharpe). More strategies tested → higher haircut.

A strategy passes DSR if its deflated p-value >= 0.95.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import norm, skew as sp_skew

from src.backtesting.metrics import MetricsCalculator, Trade

# Euler-Mascheroni constant
_EULER_MASCHERONI = 0.5772156649015329


@dataclass
class DSRConfig:
    """Configuration for Deflated Sharpe Ratio computation."""

    benchmark_sharpe: float = 0.0
    significance_level: float = 0.95
    annualization_factor: int = 252


@dataclass(frozen=True)
class DSRResult:
    """Output of a DSR computation."""

    strategy_id: str
    observed_sharpe: float
    psr: float
    expected_max_sharpe: float
    dsr: float
    n_trials: int
    sample_size: int
    skewness: float
    kurtosis: float
    sr_std_error: float
    verdict: str


class DeflatedSharpeCalculator:
    """Static methods for PSR and DSR computation."""

    @staticmethod
    def variance_of_sharpe(
        sharpe: float, T: int, skew: float, kurtosis: float
    ) -> float:
        """Variance of the Sharpe ratio estimator (Lo 2002 / Bailey-LdP).

        Args:
            sharpe: Observed (annualized) Sharpe ratio.
            T: Sample size (number of return observations).
            skew: Sample skewness of returns.
            kurtosis: Sample excess kurtosis of returns.

        Returns:
            Var(SR_hat). Always non-negative.
        """
        if T <= 1:
            return 0.0
        var = (
            1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
        ) / (T - 1)
        return max(var, 0.0)

    @staticmethod
    def expected_max_sharpe(
        n_trials: int, T: int, skew: float, kurtosis: float
    ) -> float:
        """Expected maximum Sharpe under the null (all true SR = 0).

        Uses the approximation from Bailey & Lopez de Prado (2014):
        E[max(SR)] ≈ √Var(SR) * [(1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))]

        Args:
            n_trials: Number of strategies tested (N).
            T: Sample size.
            skew: Sample skewness.
            kurtosis: Sample excess kurtosis.

        Returns:
            Expected max Sharpe. 0.0 if n_trials <= 1.
        """
        if n_trials <= 1:
            return 0.0

        # Under the null (SR=0), variance simplifies
        var_sr = DeflatedSharpeCalculator.variance_of_sharpe(0.0, T, skew, kurtosis)
        std_sr = math.sqrt(var_sr) if var_sr > 0 else 0.0

        if std_sr < 1e-15:
            return 0.0

        N = float(n_trials)
        gamma = _EULER_MASCHERONI

        e_max = std_sr * (
            (1.0 - gamma) * norm.ppf(1.0 - 1.0 / N)
            + gamma * norm.ppf(1.0 - 1.0 / (N * math.e))
        )
        return float(e_max)

    @staticmethod
    def probabilistic_sharpe_ratio(
        observed_sr: float,
        benchmark_sr: float,
        T: int,
        skew: float,
        kurtosis: float,
    ) -> float:
        """Probabilistic Sharpe Ratio — P(true SR > benchmark).

        PSR = Φ((observed_sr - benchmark_sr) / √Var(SR))

        Args:
            observed_sr: Observed annualized Sharpe.
            benchmark_sr: Benchmark Sharpe to test against.
            T: Sample size.
            skew: Sample skewness.
            kurtosis: Sample excess kurtosis.

        Returns:
            Probability in [0, 1]. Returns 0.0 if T < 2 or variance ≈ 0.
        """
        if T < 2:
            return 0.0

        var_sr = DeflatedSharpeCalculator.variance_of_sharpe(
            observed_sr, T, skew, kurtosis
        )
        if var_sr < 1e-15:
            return 0.0

        z = (observed_sr - benchmark_sr) / math.sqrt(var_sr)
        return float(norm.cdf(z))

    @staticmethod
    def compute(
        daily_returns: np.ndarray,
        n_trials: int,
        config: DSRConfig | None = None,
        strategy_id: str = "strategy",
    ) -> DSRResult:
        """Compute DSR from daily returns.

        Args:
            daily_returns: Array of daily returns (not annualized).
            n_trials: Number of strategies tested.
            config: DSRConfig (uses defaults if None).
            strategy_id: Label for the strategy.

        Returns:
            DSRResult with all computed fields.
        """
        if config is None:
            config = DSRConfig()

        T = len(daily_returns)
        if T < 2:
            return DSRResult(
                strategy_id=strategy_id,
                observed_sharpe=0.0,
                psr=0.0,
                expected_max_sharpe=0.0,
                dsr=0.0,
                n_trials=n_trials,
                sample_size=T,
                skewness=0.0,
                kurtosis=0.0,
                sr_std_error=0.0,
                verdict="FAIL",
            )

        # Observed annualized Sharpe (for display)
        observed_sharpe = MetricsCalculator.sharpe(
            daily_returns, periods=config.annualization_factor
        )

        # Non-annualized (daily) Sharpe — used in the variance formula
        # The Bailey-LdP variance formula is defined for non-annualized SR
        std = float(np.std(daily_returns, ddof=1))
        daily_sr = float(np.mean(daily_returns)) / std if std > 1e-15 else 0.0

        # Higher moments
        skew_val = float(sp_skew(daily_returns))
        kurt_val = float(sp_kurtosis(daily_returns, fisher=True))  # excess kurtosis

        # PSR against benchmark (using daily SR; benchmark also daily)
        daily_benchmark = config.benchmark_sharpe / math.sqrt(
            config.annualization_factor
        )
        psr = DeflatedSharpeCalculator.probabilistic_sharpe_ratio(
            daily_sr, daily_benchmark, T, skew_val, kurt_val
        )

        # Expected max Sharpe (daily scale, multiple testing haircut)
        e_max_daily = DeflatedSharpeCalculator.expected_max_sharpe(
            n_trials, T, skew_val, kurt_val
        )

        # DSR = PSR against expected max Sharpe (daily scale)
        dsr = DeflatedSharpeCalculator.probabilistic_sharpe_ratio(
            daily_sr, e_max_daily, T, skew_val, kurt_val
        )

        # Annualize expected max for display
        e_max = e_max_daily * math.sqrt(config.annualization_factor)

        # SR standard error (annualized scale for display)
        var_sr = DeflatedSharpeCalculator.variance_of_sharpe(
            daily_sr, T, skew_val, kurt_val
        )
        sr_std_error = (
            math.sqrt(var_sr) * math.sqrt(config.annualization_factor)
            if var_sr > 0
            else 0.0
        )

        verdict = "PASS" if dsr >= config.significance_level else "FAIL"

        return DSRResult(
            strategy_id=strategy_id,
            observed_sharpe=observed_sharpe,
            psr=psr,
            expected_max_sharpe=e_max,
            dsr=dsr,
            n_trials=n_trials,
            sample_size=T,
            skewness=skew_val,
            kurtosis=kurt_val,
            sr_std_error=sr_std_error,
            verdict=verdict,
        )

    @staticmethod
    def compute_from_trades(
        trades: list[Trade],
        initial_capital: float,
        n_trials: int,
        config: DSRConfig | None = None,
        strategy_id: str = "strategy",
    ) -> DSRResult:
        """Compute DSR from a list of Trade objects.

        Builds daily returns from trades, then delegates to compute().

        Args:
            trades: List of completed trades.
            initial_capital: Starting capital for return calculation.
            n_trials: Number of strategies tested.
            config: DSRConfig (uses defaults if None).
            strategy_id: Label for the strategy.

        Returns:
            DSRResult.
        """
        if not trades:
            return DeflatedSharpeCalculator.compute(
                np.array([]), n_trials, config, strategy_id
            )

        # Use MetricsCalculator to get daily P&L
        _, _, daily_pnl_df = MetricsCalculator.from_trades(trades, initial_capital)

        if daily_pnl_df.is_empty():
            return DeflatedSharpeCalculator.compute(
                np.array([]), n_trials, config, strategy_id
            )

        # Convert daily P&L to returns
        daily_pnl = daily_pnl_df["pnl"].to_numpy()
        daily_returns = daily_pnl / initial_capital

        return DeflatedSharpeCalculator.compute(
            daily_returns, n_trials, config, strategy_id
        )
