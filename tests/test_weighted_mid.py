"""Tests for the weighted mid-price module."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.filters.weighted_mid import (
    WeightedMidConfig,
    WeightedMidMonitor,
    WeightedMidSnapshot,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _snap(bid: float, ask: float, bid_size: int, ask_size: int) -> WeightedMidSnapshot:
    return WeightedMidSnapshot(
        timestamp=_now(), bid=bid, ask=ask, bid_size_top=bid_size, ask_size_top=ask_size
    )


async def test_weighted_mid_equals_raw_mid_when_sizes_equal():
    """Weighted mid should equal raw mid when bid and ask sizes are the same."""
    monitor = WeightedMidMonitor()

    signal = await monitor.push(_snap(100.0, 100.50, bid_size=50, ask_size=50))

    assert signal.weighted_mid == pytest.approx(signal.raw_mid)
    assert signal.raw_mid == pytest.approx(100.25)


async def test_weighted_mid_tilts_toward_bid_when_ask_size_larger():
    """Larger ask size → weighted mid tilts toward bid (less ask resistance means
    price leans up, but the formula weights by opposite side, pulling toward bid)."""
    monitor = WeightedMidMonitor()

    # ask_size (100) >> bid_size (10)
    # weighted = (100*100.0 + 10*100.50) / 110 = (10000 + 1005) / 110 = 100.045...
    signal = await monitor.push(_snap(100.0, 100.50, bid_size=10, ask_size=100))

    assert signal.weighted_mid < signal.raw_mid  # tilts toward bid
    assert signal.weighted_mid == pytest.approx(
        (100 * 100.0 + 10 * 100.50) / 110
    )


async def test_weighted_mid_tilts_toward_ask_when_bid_size_larger():
    """Larger bid size → weighted mid tilts toward ask."""
    monitor = WeightedMidMonitor()

    # bid_size (100) >> ask_size (10)
    # weighted = (10*100.0 + 100*100.50) / 110 = (1000 + 10050) / 110 = 100.454...
    signal = await monitor.push(_snap(100.0, 100.50, bid_size=100, ask_size=10))

    assert signal.weighted_mid > signal.raw_mid  # tilts toward ask
    assert signal.weighted_mid == pytest.approx(
        (10 * 100.0 + 100 * 100.50) / 110
    )


async def test_divergence_z_score_computed_correctly():
    """Z-score should match manual computation."""
    config = WeightedMidConfig(maxlen=100)
    monitor = WeightedMidMonitor(config=config)

    # Build baseline with equal sizes (divergence ≈ 0)
    divergences: list[float] = []
    for _ in range(10):
        signal = await monitor.push(_snap(100.0, 100.50, bid_size=50, ask_size=50))
        divergences.append(signal.divergence)

    # Now push an imbalanced quote
    signal = await monitor.push(_snap(100.0, 100.50, bid_size=10, ask_size=100))
    divergences.append(signal.divergence)

    # Manual z-score
    arr = np.array(divergences)
    expected_z = (divergences[-1] - float(np.mean(arr))) / float(np.std(arr, ddof=1))
    assert signal.divergence_z_score == pytest.approx(expected_z, abs=1e-9)


async def test_lean_up_when_z_score_above_threshold():
    """Lean should be 'up' when divergence z-score exceeds threshold."""
    config = WeightedMidConfig(lean_threshold=1.0)
    monitor = WeightedMidMonitor(config=config)

    # Build stable baseline (equal sizes)
    for _ in range(20):
        await monitor.push(_snap(100.0, 100.50, bid_size=50, ask_size=50))

    # Spike: heavy bid side → weighted_mid > raw_mid → positive divergence
    signal = await monitor.push(_snap(100.0, 100.50, bid_size=500, ask_size=1))

    assert signal.divergence > 0
    assert signal.lean == "up"


async def test_lean_down_when_z_score_below_negative_threshold():
    """Lean should be 'down' when divergence z-score is below -threshold."""
    config = WeightedMidConfig(lean_threshold=1.0)
    monitor = WeightedMidMonitor(config=config)

    # Build stable baseline
    for _ in range(20):
        await monitor.push(_snap(100.0, 100.50, bid_size=50, ask_size=50))

    # Spike: heavy ask side → weighted_mid < raw_mid → negative divergence
    signal = await monitor.push(_snap(100.0, 100.50, bid_size=1, ask_size=500))

    assert signal.divergence < 0
    assert signal.lean == "down"


async def test_lean_neutral_within_threshold():
    """Lean should be 'neutral' when z-score is within threshold."""
    config = WeightedMidConfig(lean_threshold=1.0)
    monitor = WeightedMidMonitor(config=config)

    for _ in range(20):
        signal = await monitor.push(_snap(100.0, 100.50, bid_size=50, ask_size=50))

    assert signal.lean == "neutral"


async def test_get_vwap_anchor_returns_weighted_when_significant():
    """get_vwap_anchor should return weighted_mid when z-score is significant."""
    config = WeightedMidConfig(lean_threshold=1.0)
    monitor = WeightedMidMonitor(config=config)

    # Build baseline
    for _ in range(20):
        await monitor.push(_snap(100.0, 100.50, bid_size=50, ask_size=50))

    # Spike to make z-score significant
    await monitor.push(_snap(100.0, 100.50, bid_size=500, ask_size=1))

    anchor = monitor.get_vwap_anchor()
    assert anchor == monitor.latest_signal.weighted_mid


async def test_get_vwap_anchor_returns_raw_when_not_significant():
    """get_vwap_anchor should return raw_mid when z-score is not significant."""
    config = WeightedMidConfig(lean_threshold=1.0)
    monitor = WeightedMidMonitor(config=config)

    # All equal sizes → z-score ≈ 0
    for _ in range(20):
        await monitor.push(_snap(100.0, 100.50, bid_size=50, ask_size=50))

    anchor = monitor.get_vwap_anchor()
    assert anchor == monitor.latest_signal.raw_mid


async def test_get_vwap_anchor_raises_with_no_data():
    """get_vwap_anchor should raise RuntimeError when no data pushed."""
    monitor = WeightedMidMonitor()

    with pytest.raises(RuntimeError, match="no data"):
        monitor.get_vwap_anchor()


async def test_zero_sizes_falls_back_to_raw_mid():
    """When both bid and ask sizes are zero, weighted mid should equal raw mid."""
    monitor = WeightedMidMonitor()

    signal = await monitor.push(_snap(100.0, 100.50, bid_size=0, ask_size=0))

    assert signal.weighted_mid == pytest.approx(signal.raw_mid)
    assert signal.weighted_mid == pytest.approx(100.25)


async def test_persist_flushes_to_parquet(tmp_path: Path):
    """Persistence should flush to Parquet every persist_every_n pushes."""
    config = WeightedMidConfig(persist_every_n=5, parquet_dir=str(tmp_path))
    monitor = WeightedMidMonitor(config=config, persist=True)

    for _ in range(12):
        await monitor.push(_snap(100.0, 100.50, bid_size=50, ask_size=50))

    # 12 pushes with persist_every_n=5 → flushes at push 5 and 10
    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(parquet_files) == 1

    df = pl.read_parquet(parquet_files[0])
    assert len(df) == 10
    assert set(df.columns) == {
        "timestamp",
        "raw_mid",
        "weighted_mid",
        "divergence",
        "divergence_z_score",
        "lean",
    }
