"""Tests for the depth deterioration signal module."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.filters.depth_monitor import (
    DepthConfig,
    DepthMonitor,
)
from src.filters import L2Snapshot


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _snap(
    bids: list[tuple[float, int]] | None = None,
    asks: list[tuple[float, int]] | None = None,
) -> L2Snapshot:
    return L2Snapshot(timestamp=_now(), bids=bids or [], asks=asks or [])


def _balanced_snap(bid_size: int = 100, ask_size: int = 100) -> L2Snapshot:
    """Helper: 5-level book with uniform sizes per side."""
    bids = [(100.0 - i * 0.25, bid_size) for i in range(5)]
    asks = [(100.25 + i * 0.25, ask_size) for i in range(5)]
    return _snap(bids=bids, asks=asks)


async def test_breakout_lean_up_when_ask_thins():
    """breakout_lean should be 'up' when only the ask side thins."""
    config = DepthConfig(depth_levels=5, thin_threshold=0.6, min_samples=10)
    monitor = DepthMonitor(config=config)

    # Build baseline with balanced book
    for _ in range(20):
        await monitor.push(_balanced_snap(bid_size=100, ask_size=100))

    # Ask side collapses to 20% of normal
    thin_asks = [(100.25 + i * 0.25, 20) for i in range(5)]
    normal_bids = [(100.0 - i * 0.25, 100) for i in range(5)]
    signal = await monitor.push(_snap(bids=normal_bids, asks=thin_asks))

    assert signal.ask_thinning is True
    assert signal.bid_thinning is False
    assert signal.breakout_lean == "up"
    assert monitor.is_thinning("ask") is True
    assert monitor.is_thinning("bid") is False


async def test_breakout_lean_down_when_bid_thins():
    """breakout_lean should be 'down' when only the bid side thins."""
    config = DepthConfig(depth_levels=5, thin_threshold=0.6, min_samples=10)
    monitor = DepthMonitor(config=config)

    for _ in range(20):
        await monitor.push(_balanced_snap(bid_size=100, ask_size=100))

    # Bid side collapses
    thin_bids = [(100.0 - i * 0.25, 20) for i in range(5)]
    normal_asks = [(100.25 + i * 0.25, 100) for i in range(5)]
    signal = await monitor.push(_snap(bids=thin_bids, asks=normal_asks))

    assert signal.bid_thinning is True
    assert signal.ask_thinning is False
    assert signal.breakout_lean == "down"


async def test_breakout_lean_none_when_both_thin():
    """breakout_lean should be 'none' when both sides thin equally."""
    config = DepthConfig(depth_levels=5, thin_threshold=0.6, min_samples=10)
    monitor = DepthMonitor(config=config)

    for _ in range(20):
        await monitor.push(_balanced_snap(bid_size=100, ask_size=100))

    # Both sides collapse
    thin_bids = [(100.0 - i * 0.25, 20) for i in range(5)]
    thin_asks = [(100.25 + i * 0.25, 20) for i in range(5)]
    signal = await monitor.push(_snap(bids=thin_bids, asks=thin_asks))

    assert signal.bid_thinning is True
    assert signal.ask_thinning is True
    assert signal.breakout_lean == "none"


async def test_breakout_lean_none_when_stable():
    """breakout_lean should be 'none' when both sides are healthy."""
    config = DepthConfig(depth_levels=5, thin_threshold=0.6, min_samples=10)
    monitor = DepthMonitor(config=config)

    for _ in range(20):
        signal = await monitor.push(_balanced_snap(bid_size=100, ask_size=100))

    assert signal.bid_thinning is False
    assert signal.ask_thinning is False
    assert signal.breakout_lean == "none"


async def test_no_signal_before_min_samples():
    """Thinning should never be flagged before min_samples is reached."""
    config = DepthConfig(depth_levels=5, thin_threshold=0.6, min_samples=60)
    monitor = DepthMonitor(config=config)

    # Push only 10 snapshots with collapsing depth — should NOT flag
    for _ in range(10):
        thin_bids = [(100.0 - i * 0.25, 5) for i in range(5)]
        normal_asks = [(100.25 + i * 0.25, 100) for i in range(5)]
        signal = await monitor.push(_snap(bids=thin_bids, asks=normal_asks))

    assert signal.bid_thinning is False
    assert signal.ask_thinning is False
    assert signal.breakout_lean == "none"
    assert monitor.is_thinning("bid") is False


async def test_depth_ratio_one_when_equal_to_mean():
    """depth_ratio should be ~1.0 when current depth equals rolling mean."""
    config = DepthConfig(depth_levels=5, min_samples=5)
    monitor = DepthMonitor(config=config)

    # Push uniform depth — ratio should converge to 1.0
    for _ in range(20):
        signal = await monitor.push(_balanced_snap(bid_size=100, ask_size=100))

    assert signal.bid_depth_ratio == pytest.approx(1.0)
    assert signal.ask_depth_ratio == pytest.approx(1.0)


async def test_depth_ratio_below_one_when_current_less_than_mean():
    """depth_ratio should be < 1.0 when current depth is below rolling mean."""
    config = DepthConfig(depth_levels=5, min_samples=5)
    monitor = DepthMonitor(config=config)

    # Build baseline at 100 per level
    for _ in range(20):
        await monitor.push(_balanced_snap(bid_size=100, ask_size=100))

    # Drop bid depth to 50
    low_bids = [(100.0 - i * 0.25, 50) for i in range(5)]
    normal_asks = [(100.25 + i * 0.25, 100) for i in range(5)]
    signal = await monitor.push(_snap(bids=low_bids, asks=normal_asks))

    assert signal.bid_depth_ratio < 1.0
    assert signal.ask_depth_ratio == pytest.approx(1.0, abs=0.05)


async def test_handles_empty_book():
    """Should handle empty bid/ask sides without crashing."""
    config = DepthConfig(depth_levels=5, min_samples=1)
    monitor = DepthMonitor(config=config)

    signal = await monitor.push(_snap(bids=[], asks=[]))

    assert signal.bid_depth == 0
    assert signal.ask_depth == 0
    assert signal.breakout_lean == "none"


async def test_is_thinning_returns_false_with_no_data():
    """is_thinning should return False when no data has been pushed."""
    monitor = DepthMonitor()
    assert monitor.is_thinning("bid") is False
    assert monitor.is_thinning("ask") is False


async def test_persist_flushes_to_parquet(tmp_path: Path):
    """Persistence should flush to Parquet every persist_every_n pushes."""
    config = DepthConfig(
        depth_levels=5, persist_every_n=5, parquet_dir=str(tmp_path), min_samples=1
    )
    monitor = DepthMonitor(config=config, persist=True)

    for _ in range(12):
        await monitor.push(_balanced_snap())

    # 12 pushes with persist_every_n=5 → flushes at push 5 and 10
    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(parquet_files) == 1

    df = pl.read_parquet(parquet_files[0])
    assert len(df) == 10
    assert set(df.columns) == {
        "timestamp",
        "bid_depth",
        "ask_depth",
        "bid_depth_ratio",
        "ask_depth_ratio",
        "bid_thinning",
        "ask_thinning",
        "breakout_lean",
    }
