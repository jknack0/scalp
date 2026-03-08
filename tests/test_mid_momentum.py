"""Tests for the mid-price momentum signal module."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.filters.mid_momentum import (
    MidMomentumMonitor,
    MidSnapshot,
    MomentumConfig,
)


def _snap(bid: float, ask: float) -> MidSnapshot:
    """Helper to create a snapshot with current UTC time."""
    return MidSnapshot(timestamp=datetime.now(timezone.utc), bid=bid, ask=ask)


async def test_slope_positive_for_increasing_mids():
    """Slope should be positive when mid-prices are strictly increasing."""
    config = MomentumConfig(regression_window=10, neutral_threshold=0.1)
    monitor = MidMomentumMonitor(config=config)

    for i in range(20):
        signal = await monitor.push(_snap(100.0 + i * 0.25, 100.50 + i * 0.25))

    assert signal.slope > 0
    assert signal.direction == "up"


async def test_slope_negative_for_decreasing_mids():
    """Slope should be negative when mid-prices are strictly decreasing."""
    config = MomentumConfig(regression_window=10, neutral_threshold=0.1)
    monitor = MidMomentumMonitor(config=config)

    for i in range(20):
        signal = await monitor.push(_snap(110.0 - i * 0.25, 110.50 - i * 0.25))

    assert signal.slope < 0
    assert signal.direction == "down"


async def test_direction_neutral_below_threshold():
    """Direction should be neutral when drift_score is below threshold."""
    config = MomentumConfig(regression_window=10, neutral_threshold=5.0)
    monitor = MidMomentumMonitor(config=config)

    # Flat series with tiny noise — drift score will be near zero
    for _ in range(20):
        signal = await monitor.push(_snap(100.0, 100.50))

    assert signal.direction == "neutral"
    assert abs(signal.drift_score) < config.neutral_threshold


async def test_strength_clamped_to_unit():
    """Strength should be clamped to [0.0, 1.0]."""
    config = MomentumConfig(regression_window=5, neutral_threshold=0.01)
    monitor = MidMomentumMonitor(config=config)

    # Massive ramp to force drift_score > 1.0
    for i in range(10):
        signal = await monitor.push(_snap(100.0 + i * 10.0, 100.50 + i * 10.0))

    assert 0.0 <= signal.strength <= 1.0


async def test_handles_fewer_samples_than_window():
    """Should not crash when buffer has fewer samples than regression_window."""
    config = MomentumConfig(regression_window=20)
    monitor = MidMomentumMonitor(config=config)

    # Push only 3 snapshots (well under regression_window of 20)
    for i in range(3):
        signal = await monitor.push(_snap(100.0 + i, 100.50 + i))

    assert signal is not None
    assert signal.slope > 0  # still computes on available data


async def test_single_sample_returns_zero_slope():
    """A single sample should yield zero slope."""
    monitor = MidMomentumMonitor()

    signal = await monitor.push(_snap(100.0, 100.50))

    assert signal.slope == 0.0
    assert signal.drift_score == 0.0
    assert signal.direction == "neutral"


async def test_mid_property():
    """MidSnapshot.mid should be the average of bid and ask."""
    snap = MidSnapshot(
        timestamp=datetime.now(timezone.utc), bid=100.0, ask=100.50
    )
    assert snap.mid == 100.25


async def test_persist_flushes_to_parquet(tmp_path: Path):
    """Persistence should flush to Parquet every persist_every_n pushes."""
    config = MomentumConfig(
        persist_every_n=5, parquet_dir=str(tmp_path), regression_window=5
    )
    monitor = MidMomentumMonitor(config=config, persist=True)

    for i in range(12):
        await monitor.push(_snap(100.0 + i * 0.1, 100.50 + i * 0.1))

    # 12 pushes with persist_every_n=5 → flushes at push 5 and 10
    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(parquet_files) == 1  # same date → single file

    df = pl.read_parquet(parquet_files[0])
    assert len(df) == 10  # 5 rows at push 5 + 5 rows at push 10
    assert set(df.columns) == {
        "timestamp",
        "mid",
        "slope",
        "drift_score",
        "direction",
        "strength",
    }
