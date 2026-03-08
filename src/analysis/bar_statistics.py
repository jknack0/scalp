"""Statistical validation for dollar bars vs time bars.

Tests whether dollar bars produce returns closer to IID (independent and
identically distributed) than fixed-interval time bars. Uses:
- Ljung-Box test for serial autocorrelation
- Augmented Dickey-Fuller test for stationarity
- Hurst exponent for long-range dependence
- Kolmogorov-Smirnov test for normality
"""

from dataclasses import dataclass

import numpy as np
from hurst import compute_Hc
from scipy.stats import kstest
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


# ── Result dataclasses ───────────────────────────────────────────────

@dataclass(frozen=True)
class AutocorrelationResult:
    """Ljung-Box test results across multiple lags."""
    lags: list[int]
    statistics: list[float]
    p_values: list[float]
    significant_lags: list[int]  # lags where p < 0.05

    @property
    def has_autocorrelation(self) -> bool:
        return len(self.significant_lags) > 0


@dataclass(frozen=True)
class StationarityResult:
    """Augmented Dickey-Fuller test result."""
    adf_statistic: float
    p_value: float
    is_stationary: bool  # True if p < 0.05


@dataclass(frozen=True)
class HurstResult:
    """Hurst exponent result."""
    H: float
    classification: str  # "mean-reverting", "random-walk", "trending"


@dataclass(frozen=True)
class BarComparisonReport:
    """Full statistical report for one bar type."""
    bar_type: str
    n_bars: int
    autocorrelation: AutocorrelationResult
    stationarity: StationarityResult
    hurst: HurstResult
    ks_normality_p: float


# ── Core functions ───────────────────────────────────────────────────

def compute_log_returns(closes: np.ndarray) -> np.ndarray:
    """Compute log returns from a close price series.

    Args:
        closes: 1-D array of close prices.

    Returns:
        Array of log(close[t] / close[t-1]), length = len(closes) - 1.
    """
    closes = np.asarray(closes, dtype=np.float64)
    return np.diff(np.log(closes))


def ljung_box_test(
    returns: np.ndarray,
    lags: list[int] | None = None,
) -> AutocorrelationResult:
    """Run the Ljung-Box Q-test for serial autocorrelation.

    Args:
        returns: 1-D array of returns.
        lags: List of lags to test. Defaults to [1, 2, 5, 10, 20, 50].

    Returns:
        AutocorrelationResult with test stats and significant lags.
    """
    if lags is None:
        lags = [1, 2, 5, 10, 20, 50]

    # Filter lags that exceed series length
    max_lag = len(returns) - 1
    lags = [lag for lag in lags if lag < max_lag]

    if not lags:
        return AutocorrelationResult(
            lags=[], statistics=[], p_values=[], significant_lags=[]
        )

    result = acorr_ljungbox(returns, lags=lags, return_df=True)
    statistics = result["lb_stat"].tolist()
    p_values = result["lb_pvalue"].tolist()
    significant = [lag for lag, p in zip(lags, p_values) if p < 0.05]

    return AutocorrelationResult(
        lags=lags,
        statistics=statistics,
        p_values=p_values,
        significant_lags=significant,
    )


def adf_test(series: np.ndarray) -> StationarityResult:
    """Run the Augmented Dickey-Fuller test for stationarity.

    Args:
        series: 1-D time series (e.g., returns).

    Returns:
        StationarityResult — stationary if p < 0.05.
    """
    stat, p_value, *_ = adfuller(series, autolag="AIC")
    return StationarityResult(
        adf_statistic=float(stat),
        p_value=float(p_value),
        is_stationary=bool(p_value < 0.05),
    )


def hurst_exponent(returns: np.ndarray) -> HurstResult:
    """Compute the Hurst exponent via R/S analysis.

    Args:
        returns: 1-D array of returns (must have length >= 20).

    Returns:
        HurstResult with H value and classification:
        - H < 0.45 → mean-reverting
        - 0.45 ≤ H ≤ 0.55 → random-walk
        - H > 0.55 → trending
    """
    H, _, _ = compute_Hc(returns, kind="change", simplified=True)
    H = float(H)

    if H < 0.45:
        classification = "mean-reverting"
    elif H > 0.55:
        classification = "trending"
    else:
        classification = "random-walk"

    return HurstResult(H=H, classification=classification)


def ks_normality_test(returns: np.ndarray) -> float:
    """Kolmogorov-Smirnov test for normality.

    Args:
        returns: 1-D array of returns.

    Returns:
        KS test p-value. Low p (< 0.05) → returns are NOT normal.
    """
    standardized = (returns - returns.mean()) / returns.std()
    _, p_value = kstest(standardized, "norm")
    return float(p_value)


# ── High-level wrappers ─────────────────────────────────────────────

def analyze_bars(closes: np.ndarray, bar_type: str) -> BarComparisonReport:
    """Run all statistical tests on a single bar set.

    Args:
        closes: 1-D array of close prices.
        bar_type: Label for this bar type (e.g., "dollar", "1m", "5m").

    Returns:
        BarComparisonReport with all test results.
    """
    returns = compute_log_returns(closes)

    return BarComparisonReport(
        bar_type=bar_type,
        n_bars=len(closes),
        autocorrelation=ljung_box_test(returns),
        stationarity=adf_test(returns),
        hurst=hurst_exponent(returns),
        ks_normality_p=ks_normality_test(returns),
    )


def compare_bar_types(bar_sets: dict[str, np.ndarray]) -> list[BarComparisonReport]:
    """Run statistical comparison across multiple bar types.

    Args:
        bar_sets: Mapping of bar_type label → close price array.

    Returns:
        List of BarComparisonReport, one per bar type.
    """
    return [analyze_bars(closes, label) for label, closes in bar_sets.items()]


def print_comparison_report(reports: list[BarComparisonReport]) -> None:
    """Print a formatted summary table of bar comparison results."""
    header = (
        f"{'Bar Type':<12} {'N Bars':>8} {'LB Sig Lags':>12} "
        f"{'ADF p':>8} {'Stationary':>11} {'Hurst':>7} "
        f"{'H Class':>15} {'KS p':>8}"
    )
    print("\n" + "=" * len(header))
    print("BAR TYPE STATISTICAL COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in reports:
        n_sig = len(r.autocorrelation.significant_lags)
        total = len(r.autocorrelation.lags)
        print(
            f"{r.bar_type:<12} {r.n_bars:>8,} {n_sig}/{total:>10} "
            f"{r.stationarity.p_value:>8.4f} {'YES' if r.stationarity.is_stationary else 'NO':>11} "
            f"{r.hurst.H:>7.3f} {r.hurst.classification:>15} "
            f"{r.ks_normality_p:>8.4f}"
        )

    print("-" * len(header))
    print(
        "LB Sig Lags = Ljung-Box significant lags at p<0.05 (fewer = less autocorrelation)"
    )
    print("ADF p < 0.05 = stationary returns (good)")
    print("Hurst ≈ 0.5 = random walk (IID-like)")
    print("KS p > 0.05 = returns consistent with normality")
    print()
