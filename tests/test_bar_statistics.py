"""Tests for bar statistical validation (src/analysis/bar_statistics.py)."""

import numpy as np

from src.analysis.bar_statistics import (
    BarComparisonReport,
    analyze_bars,
    adf_test,
    compute_log_returns,
    hurst_exponent,
    ljung_box_test,
)


def test_log_returns_computation():
    """Log returns are correctly computed from close prices."""
    closes = np.array([100.0, 105.0, 110.0])
    returns = compute_log_returns(closes)
    assert len(returns) == 2
    assert abs(returns[0] - np.log(105 / 100)) < 1e-10
    assert abs(returns[1] - np.log(110 / 105)) < 1e-10


def test_adf_stationary_series():
    """ADF test correctly identifies a stationary series (white noise)."""
    rng = np.random.default_rng(42)
    stationary = rng.normal(0, 1, 1000)
    result = adf_test(stationary)
    assert result.is_stationary is True
    assert result.p_value < 0.05


def test_adf_nonstationary_series():
    """ADF test correctly identifies a non-stationary series (random walk)."""
    rng = np.random.default_rng(42)
    random_walk = np.cumsum(rng.normal(0, 1, 1000))
    result = adf_test(random_walk)
    assert result.is_stationary is False
    assert result.p_value > 0.05


def test_ljung_box_iid():
    """Ljung-Box test finds no significant autocorrelation in IID noise."""
    rng = np.random.default_rng(42)
    iid = rng.normal(0, 1, 1000)
    result = ljung_box_test(iid, lags=[1, 5, 10])
    # IID noise should have few or no significant lags
    assert len(result.significant_lags) <= 1  # allow one spurious hit at 5%


def test_hurst_returns_valid_result():
    """Hurst exponent returns valid H in [0,1] with correct classification."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 1, 2000)
    result = hurst_exponent(returns)
    assert 0.0 <= result.H <= 1.0
    assert result.classification in ("mean-reverting", "random-walk", "trending")


def test_analyze_bars_structure():
    """analyze_bars returns a complete BarComparisonReport."""
    rng = np.random.default_rng(42)
    # Simulate price series: start at 5000, random walk
    prices = 5000.0 + np.cumsum(rng.normal(0, 0.5, 500))
    # Ensure all prices are positive
    prices = np.abs(prices)

    report = analyze_bars(prices, "test_bars")
    assert isinstance(report, BarComparisonReport)
    assert report.bar_type == "test_bars"
    assert report.n_bars == 500
    assert isinstance(report.autocorrelation.lags, list)
    assert isinstance(report.stationarity.is_stationary, bool)
    assert isinstance(report.hurst.H, float)
    assert isinstance(report.ks_normality_p, float)
