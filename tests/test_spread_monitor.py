"""Tests for the spread mean reversion filter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.filters.spread_monitor import (
    SpreadConfig,
    SpreadMonitor,
    SpreadSnapshot,
)


def _snap(bid: float, ask: float) -> SpreadSnapshot:
    """Helper to create a snapshot with current UTC time."""
    return SpreadSnapshot(timestamp=datetime.now(timezone.utc), bid=bid, ask=ask)


async def test_z_score_computed_correctly():
    """Z-score should match manual numpy computation."""
    config = SpreadConfig(z_threshold=2.0, min_samples=5)
    monitor = SpreadMonitor(config=config)

    # Push known spreads: 0.25 repeated, then a spike
    spreads = [0.25] * 10 + [0.75]
    for s in spreads:
        state = await monitor.push(_snap(100.0, 100.0 + s))

    # Manual z-score for last push
    arr = np.array(spreads)
    expected_mean = float(np.mean(arr))
    expected_std = float(np.std(arr, ddof=1))
    expected_z = (0.75 - expected_mean) / expected_std

    assert abs(state.z_score - expected_z) < 1e-9
    assert abs(state.rolling_mean - expected_mean) < 1e-9
    assert abs(state.rolling_std - expected_std) < 1e-9


async def test_is_spread_normal_true_within_threshold():
    """Spread within z-threshold should be flagged as normal."""
    config = SpreadConfig(z_threshold=2.0, min_samples=5)
    monitor = SpreadMonitor(config=config)

    # Push stable spreads
    for _ in range(50):
        await monitor.push(_snap(100.0, 100.25))

    is_normal, reason = monitor.is_spread_normal()
    assert is_normal is True
    assert reason == ""


async def test_is_spread_normal_false_on_spike():
    """Spread spike beyond z-threshold should block trading."""
    config = SpreadConfig(z_threshold=2.0, min_samples=5)
    monitor = SpreadMonitor(config=config)

    # Build a stable baseline
    for _ in range(50):
        await monitor.push(_snap(100.0, 100.25))

    # Inject a massive spike
    await monitor.push(_snap(100.0, 105.0))

    is_normal, reason = monitor.is_spread_normal()
    assert is_normal is False
    assert "exceeds" in reason


async def test_no_filter_below_min_samples():
    """Should not filter when buffer has fewer than min_samples entries."""
    config = SpreadConfig(z_threshold=2.0, min_samples=30)
    monitor = SpreadMonitor(config=config)

    # Push only 10 observations — even a spike should pass
    for _ in range(9):
        await monitor.push(_snap(100.0, 100.25))
    await monitor.push(_snap(100.0, 110.0))  # huge spike

    is_normal, reason = monitor.is_spread_normal()
    assert is_normal is True  # not enough data to filter


async def test_zero_spread_flagged_abnormal():
    """Zero or negative spread should be flagged as data error."""
    config = SpreadConfig(min_samples=1)
    monitor = SpreadMonitor(config=config)

    # Push enough baseline
    for _ in range(5):
        await monitor.push(_snap(100.0, 100.25))

    # Push zero spread (bid == ask)
    await monitor.push(_snap(100.0, 100.0))

    is_normal, reason = monitor.is_spread_normal()
    assert is_normal is False
    assert "zero or negative" in reason


async def test_negative_spread_flagged_abnormal():
    """Negative spread (ask < bid) should be flagged as data error."""
    config = SpreadConfig(min_samples=1)
    monitor = SpreadMonitor(config=config)

    await monitor.push(_snap(100.25, 100.0))  # ask < bid

    is_normal, reason = monitor.is_spread_normal()
    assert is_normal is False
    assert "zero or negative" in reason


async def test_no_data_returns_false():
    """With no data pushed, is_spread_normal should return False."""
    monitor = SpreadMonitor()
    is_normal, reason = monitor.is_spread_normal()
    assert is_normal is False
    assert "no data" in reason


async def test_buffer_respects_maxlen():
    """Buffer should not exceed configured maxlen."""
    config = SpreadConfig(maxlen=10, min_samples=1)
    monitor = SpreadMonitor(config=config)

    for _ in range(50):
        await monitor.push(_snap(100.0, 100.25))

    assert monitor.buffer_size == 10


async def test_persist_flushes_to_parquet(tmp_path: Path):
    """Persistence should flush to Parquet every persist_every_n pushes."""
    config = SpreadConfig(
        persist_every_n=5, min_samples=1, parquet_dir=str(tmp_path)
    )
    monitor = SpreadMonitor(config=config, persist=True)

    for _ in range(12):
        await monitor.push(_snap(100.0, 100.25))

    # 12 pushes with persist_every_n=5 → flushes at push 5 and 10
    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(parquet_files) == 1  # same date → single file

    df = pl.read_parquet(parquet_files[0])
    assert len(df) == 10  # 5 rows at push 5 + 5 rows at push 10
    assert set(df.columns) == {
        "timestamp",
        "current_spread",
        "rolling_mean",
        "rolling_std",
        "z_score",
        "is_normal",
    }
