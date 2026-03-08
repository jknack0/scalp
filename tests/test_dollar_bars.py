"""Tests for dollar bar construction (src/data/bars.py::build_dollar_bars)."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from src.data.bars import build_dollar_bars

MES_POINT_VALUE = 5.0


def _make_1s_bars(
    n: int,
    close: float = 5000.0,
    volume: int = 10,
    start: datetime | None = None,
) -> pl.DataFrame:
    """Create synthetic 1s bar data."""
    if start is None:
        start = datetime(2024, 1, 2, 9, 30, 0)
    return pl.DataFrame({
        "timestamp": [start + timedelta(seconds=i) for i in range(n)],
        "open": [close] * n,
        "high": [close + 0.25] * n,
        "low": [close - 0.25] * n,
        "close": [close] * n,
        "volume": [volume] * n,
        "vwap": [close] * n,
    })


def test_bar_emits_at_threshold():
    """A dollar bar is emitted when cumulative dollar volume reaches the threshold."""
    # Each 1s bar: close(5000) * volume(10) * 5 = 250,000
    # Threshold = 250,000 → one bar per 1s bar
    df = _make_1s_bars(3, close=5000.0, volume=10)
    threshold = 5000.0 * 10 * MES_POINT_VALUE  # 250,000
    result = build_dollar_bars(df, threshold)
    assert len(result) == 3


def test_multiple_bars_emitted():
    """Multiple dollar bars are produced from a longer series."""
    # Each bar: 5000 * 10 * 5 = 250k. With 10 rows and threshold 500k → 5 bars.
    df = _make_1s_bars(10, close=5000.0, volume=10)
    threshold = 500_000.0  # 2 rows per bar
    result = build_dollar_bars(df, threshold)
    assert len(result) == 5


def test_ohlc_correctness():
    """OHLC values are correctly computed across constituent 1s bars."""
    start = datetime(2024, 1, 2, 9, 30, 0)
    df = pl.DataFrame({
        "timestamp": [start + timedelta(seconds=i) for i in range(3)],
        "open": [100.0, 101.0, 102.0],
        "high": [100.5, 103.0, 102.5],
        "low": [99.0, 100.5, 101.0],
        "close": [100.25, 101.5, 102.0],
        "volume": [10, 10, 10],
        "vwap": [100.0, 101.0, 102.0],
    })
    # Dollar vol per row ≈ close * 10 * 5 ≈ 5k each
    # Set threshold so all 3 rows form one bar
    threshold = 100_000.0  # won't be reached
    result = build_dollar_bars(df, threshold)
    # Threshold not reached → no bars emitted (residual behavior)
    assert len(result) == 0

    # Now set threshold so first 2 rows form a bar
    # Row 0 dollar vol: 100.25 * 10 * 5 = 5012.5
    # Row 1 dollar vol: 101.5 * 10 * 5 = 5075.0
    # Cumulative after row 1: 10087.5
    threshold = 10_000.0
    result = build_dollar_bars(df, threshold)
    assert len(result) == 1
    bar = result.row(0, named=True)
    assert bar["open"] == 100.0   # open of first constituent
    assert bar["high"] == 103.0   # max high across rows 0-1
    assert bar["low"] == 99.0     # min low across rows 0-1
    assert bar["close"] == 101.5  # close of last constituent


def test_vwap_calculation():
    """VWAP is volume-weighted average of constituent VWAPs."""
    start = datetime(2024, 1, 2, 9, 30, 0)
    df = pl.DataFrame({
        "timestamp": [start + timedelta(seconds=i) for i in range(2)],
        "open": [100.0, 100.0],
        "high": [100.0, 100.0],
        "low": [100.0, 100.0],
        "close": [100.0, 100.0],
        "volume": [20, 30],
        "vwap": [99.0, 101.0],
    })
    # Dollar vol per row: 100 * vol * 5. Row 0: 10k, Row 1: 15k. Total: 25k.
    threshold = 20_000.0
    result = build_dollar_bars(df, threshold)
    assert len(result) == 1
    bar = result.row(0, named=True)
    # Expected VWAP: (99*20 + 101*30) / (20+30) = (1980+3030)/50 = 100.2
    assert abs(bar["vwap"] - 100.2) < 0.01


def test_bar_duration_s():
    """bar_duration_s equals the time span of the constituent 1s bars."""
    start = datetime(2024, 1, 2, 9, 30, 0)
    df = _make_1s_bars(4, close=5000.0, volume=10, start=start)
    # Each row: 5000 * 10 * 5 = 250k. Threshold 500k → 2 rows per bar.
    threshold = 500_000.0
    result = build_dollar_bars(df, threshold)
    assert len(result) == 2
    # First bar spans second 0 to second 1 → duration = 1.0s
    assert result["bar_duration_s"][0] == 1.0
    # Second bar spans second 2 to second 3 → duration = 1.0s
    assert result["bar_duration_s"][1] == 1.0


def test_empty_input():
    """Empty input returns an empty DataFrame with correct schema."""
    df = pl.DataFrame({
        "timestamp": pl.Series([], dtype=pl.Datetime),
        "open": pl.Series([], dtype=pl.Float64),
        "high": pl.Series([], dtype=pl.Float64),
        "low": pl.Series([], dtype=pl.Float64),
        "close": pl.Series([], dtype=pl.Float64),
        "volume": pl.Series([], dtype=pl.Int64),
        "vwap": pl.Series([], dtype=pl.Float64),
    })
    result = build_dollar_bars(df, 1000.0)
    assert len(result) == 0
    assert "dollar_volume" in result.columns
    assert "bar_duration_s" in result.columns


def test_threshold_not_reached():
    """When total dollar volume is below threshold, no bars are emitted."""
    df = _make_1s_bars(2, close=100.0, volume=1)
    # Dollar vol per row: 100 * 1 * 5 = 500. Total = 1000.
    threshold = 5000.0
    result = build_dollar_bars(df, threshold)
    assert len(result) == 0


def test_residual_not_emitted():
    """Partial accumulation at the end of data is NOT emitted as a bar."""
    df = _make_1s_bars(3, close=5000.0, volume=10)
    # Each row: 250k. Total: 750k. Threshold 500k → 1 bar, 1 row residual.
    threshold = 500_000.0
    result = build_dollar_bars(df, threshold)
    assert len(result) == 1
