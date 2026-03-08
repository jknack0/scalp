"""Tests for mean reversion / momentum transition analysis."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from src.analysis.regime_characterization import (
    AR1Result,
    AutocorrelationProfile,
    CrossingPoint,
    HurstByWindow,
    compute_acf_profile,
    compute_ar1_by_timescale,
    find_crossing_point,
    segment_by_time_of_day,
    compute_hurst_by_window,
)


def _make_1s_bars(n: int, prices: np.ndarray, start: datetime | None = None) -> pl.DataFrame:
    """Build a 1s bar DataFrame from a price series."""
    if start is None:
        start = datetime(2024, 6, 3, 9, 30, 0)  # Monday 9:30 AM
    return pl.DataFrame({
        "timestamp": [start + timedelta(seconds=i) for i in range(n)],
        "open": prices.tolist(),
        "high": (prices + 0.25).tolist(),
        "low": (prices - 0.25).tolist(),
        "close": prices.tolist(),
        "volume": [10] * n,
        "vwap": prices.tolist(),
    })


def test_acf_profile_white_noise():
    """White noise returns have ACF approximately zero at all lags."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 1, 5000)
    result = compute_acf_profile(returns, max_lag=50)

    assert isinstance(result, AutocorrelationProfile)
    assert len(result.lags) == 50
    # White noise: very few lags should be significant (expect < 5% by chance)
    assert len(result.significant_lags) <= 5


def test_acf_profile_autocorrelated():
    """AR(1) process with strong positive autocorrelation shows significant lag-1."""
    rng = np.random.default_rng(42)
    n = 5000
    series = np.zeros(n)
    for i in range(1, n):
        series[i] = 0.8 * series[i - 1] + rng.normal(0, 1)

    result = compute_acf_profile(series, max_lag=20)
    # Lag 1 should be significant and close to 0.8
    assert 1 in result.significant_lags
    assert result.acf_values[0] > 0.7  # lag 1 ACF


def test_hurst_by_window_structure():
    """Hurst by window returns correct number of windows with valid values."""
    rng = np.random.default_rng(42)
    n = 36000  # 10 hours of 1s bars — enough for 30s window (1200 bars → 1199 returns)
    prices = 5000.0 + np.cumsum(rng.normal(0, 0.01, n))
    df = _make_1s_bars(n, prices)

    result = compute_hurst_by_window(df, windows=[30, 120])
    assert isinstance(result, HurstByWindow)
    assert len(result.window_seconds) == 2
    assert len(result.hurst_values) == 2
    for h in result.hurst_values:
        assert (0.0 <= h <= 1.0) or np.isnan(h)


def test_ar1_positive_for_reverting_returns():
    """Negatively autocorrelated 1s returns give positive mean reversion coefficient."""
    rng = np.random.default_rng(42)
    n = 10800  # 3 hours
    # Generate mean-reverting 1s returns: strong negative autocorrelation
    raw_returns = np.zeros(n)
    for i in range(1, n):
        raw_returns[i] = -0.7 * raw_returns[i - 1] + rng.normal(0, 1)
    prices = 5000.0 + np.cumsum(raw_returns * 0.001)
    df = _make_1s_bars(n, prices)

    # Test at 1s timescale (no resampling distortion)
    results = compute_ar1_by_timescale(df, timescales=[1])
    assert len(results) == 1
    # Negative AR(1) on returns → positive mean reversion coeff
    assert results[0].mean_reversion_coeff > 0


def test_ar1_negative_for_trending_returns():
    """Positively autocorrelated 1s returns give negative mean reversion coefficient."""
    rng = np.random.default_rng(42)
    n = 10800
    # Generate trending 1s returns: strong positive autocorrelation
    raw_returns = np.zeros(n)
    for i in range(1, n):
        raw_returns[i] = 0.7 * raw_returns[i - 1] + rng.normal(0, 1)
    prices = 5000.0 + np.cumsum(raw_returns * 0.001)
    df = _make_1s_bars(n, prices)

    results = compute_ar1_by_timescale(df, timescales=[1])
    assert len(results) == 1
    # Positive AR(1) on returns → negative mean reversion coeff
    assert results[0].mean_reversion_coeff < 0


def test_find_crossing_point():
    """Crossing point is interpolated between positive and negative AR1 results."""
    ar1_results = [
        AR1Result("30s", 30, mean_reversion_coeff=0.10, ar1_raw=-0.10, p_value=0.01),
        AR1Result("2m", 120, mean_reversion_coeff=0.05, ar1_raw=-0.05, p_value=0.05),
        AR1Result("5m", 300, mean_reversion_coeff=-0.03, ar1_raw=0.03, p_value=0.10),
        AR1Result("15m", 900, mean_reversion_coeff=-0.08, ar1_raw=0.08, p_value=0.02),
    ]
    cp = find_crossing_point(ar1_results)
    assert cp is not None
    assert isinstance(cp, CrossingPoint)
    # Crossing between 2m (120s, coeff=0.05) and 5m (300s, coeff=-0.03)
    # Linear interp: 120 + (300-120) * (0.05 / (0.05 + 0.03)) = 120 + 180 * 0.625 = 232.5
    assert abs(cp.timescale_seconds - 232.5) < 1.0
    assert cp.below_regime == "mean-reverting"
    assert cp.above_regime == "momentum"


def test_segment_by_time_of_day():
    """Time-of-day segmentation correctly filters by hour ranges."""
    start = datetime(2024, 6, 3, 9, 0, 0)
    n = 8 * 3600  # 8 hours (9AM-5PM)
    timestamps = [start + timedelta(seconds=i) for i in range(n)]
    df = pl.DataFrame({
        "timestamp": timestamps,
        "close": [5000.0] * n,
    })

    segments = segment_by_time_of_day(df)
    assert "open" in segments
    assert "midday" in segments
    assert "close" in segments

    # Open: 9:30-11:00 = 90 min = 5400s
    assert len(segments["open"]) == 5400
    # Midday: 11:00-14:00 = 180 min = 10800s
    assert len(segments["midday"]) == 10800
    # Close: 14:00-16:00 = 120 min = 7200s
    assert len(segments["close"]) == 7200
