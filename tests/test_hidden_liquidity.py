"""Tests for the hidden liquidity map module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from src.filters.hidden_liquidity import (
    HiddenLiquidityConfig,
    HiddenLiquidityDetector,
    HiddenLiquidityMap,
)
from src.filters import L2Snapshot


def _ts(offset_days: float = 0.0) -> datetime:
    """Return a UTC timestamp with optional offset in days."""
    return datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        days=offset_days
    )


def _snap(
    bids: list[tuple[float, int]] | None = None,
    asks: list[tuple[float, int]] | None = None,
) -> L2Snapshot:
    return L2Snapshot(timestamp=_ts(), bids=bids or [], asks=asks or [])


# ------------------------------------------------------------------
# HiddenLiquidityDetector
# ------------------------------------------------------------------


async def test_hidden_event_recorded_at_invisible_level():
    """Trade at a level with zero visible depth should produce a hidden event."""
    config = HiddenLiquidityConfig(visibility_threshold=10)
    detector = HiddenLiquidityDetector(config=config)

    # Book has no depth at 100.50
    detector.update_book(
        _snap(
            bids=[(100.0, 50), (99.75, 50)],
            asks=[(100.25, 50), (100.75, 50)],
        )
    )

    event = detector.check_trade(price=100.50, size=5, timestamp=_ts())
    assert event is not None
    assert event.price == 100.50
    assert event.size == 5


async def test_hidden_event_recorded_at_near_zero_level():
    """Trade at a level with size below threshold should produce a hidden event."""
    config = HiddenLiquidityConfig(visibility_threshold=10)
    detector = HiddenLiquidityDetector(config=config)

    # 100.25 has only 3 contracts visible (< threshold of 10)
    detector.update_book(
        _snap(
            bids=[(100.0, 50)],
            asks=[(100.25, 3), (100.50, 50)],
        )
    )

    event = detector.check_trade(price=100.25, size=10, timestamp=_ts())
    assert event is not None


async def test_no_hidden_event_at_visible_level():
    """Trade at a level with sufficient visible depth should not produce event."""
    config = HiddenLiquidityConfig(visibility_threshold=10)
    detector = HiddenLiquidityDetector(config=config)

    # 100.25 has 50 contracts — well above threshold
    detector.update_book(
        _snap(
            bids=[(100.0, 50)],
            asks=[(100.25, 50), (100.50, 50)],
        )
    )

    event = detector.check_trade(price=100.25, size=10, timestamp=_ts())
    assert event is None


async def test_no_event_without_snapshot():
    """No hidden event when no L2 snapshot has been provided yet."""
    detector = HiddenLiquidityDetector()
    event = detector.check_trade(price=100.25, size=10, timestamp=_ts())
    assert event is None


# ------------------------------------------------------------------
# HiddenLiquidityMap — strength
# ------------------------------------------------------------------


async def test_strength_zero_for_unknown_price():
    """get_strength should return 0.0 for a price with no history."""
    hmap = HiddenLiquidityMap()
    assert hmap.get_strength(100.25, now=_ts()) == 0.0


async def test_strength_positive_for_recent_events():
    """Recent events should produce a positive strength score."""
    hmap = HiddenLiquidityMap()
    now = _ts()

    for i in range(5):
        hmap.record_hidden_event(100.25, 10, now - timedelta(hours=i))

    strength = hmap.get_strength(100.25, now=now)
    assert strength > 0.0
    assert strength <= 1.0


async def test_recency_decay_reduces_old_event_strength():
    """Events far in the past should contribute less to strength."""
    config = HiddenLiquidityConfig(decay_days=30.0)
    hmap = HiddenLiquidityMap(config=config)
    now = _ts()

    # Record 5 events 60 days ago
    old_time = now - timedelta(days=60)
    for _ in range(5):
        hmap.record_hidden_event(100.25, 10, old_time)

    # Record 5 events today
    hmap_fresh = HiddenLiquidityMap(config=config)
    for _ in range(5):
        hmap_fresh.record_hidden_event(100.25, 10, now)

    old_strength = hmap.get_strength(100.25, now=now)
    fresh_strength = hmap_fresh.get_strength(100.25, now=now)

    assert fresh_strength > old_strength


async def test_strength_clamped_to_one():
    """Strength should never exceed 1.0 even with many events."""
    hmap = HiddenLiquidityMap()
    now = _ts()

    for i in range(100):
        hmap.record_hidden_event(100.25, 10, now - timedelta(minutes=i))

    assert hmap.get_strength(100.25, now=now) == 1.0


async def test_strength_deterministic():
    """Same input data should always produce the same strength."""
    config = HiddenLiquidityConfig(decay_days=30.0)
    now = _ts()

    results = []
    for _ in range(3):
        hmap = HiddenLiquidityMap(config=config)
        for i in range(5):
            hmap.record_hidden_event(100.25, 10, now - timedelta(hours=i))
        results.append(hmap.get_strength(100.25, now=now))

    assert results[0] == results[1] == results[2]


# ------------------------------------------------------------------
# HiddenLiquidityMap — spatial queries
# ------------------------------------------------------------------


async def test_get_nearby_levels_within_radius():
    """get_nearby_levels should return levels within the tick radius."""
    config = HiddenLiquidityConfig(tick_size=0.25, radius_ticks=4)
    hmap = HiddenLiquidityMap(config=config)
    now = _ts()

    # Levels at 100.00, 100.25, 100.50, 102.00
    for price in [100.0, 100.25, 100.50, 102.0]:
        for _ in range(5):
            hmap.record_hidden_event(price, 10, now)

    # Query around 100.25 with radius 4 ticks (1.0 point)
    nearby = hmap.get_nearby_levels(100.25, radius_ticks=4, now=now)

    prices = {lvl.price for lvl in nearby}
    assert 100.0 in prices
    assert 100.25 in prices
    assert 100.50 in prices
    assert 102.0 not in prices  # outside radius


async def test_get_nearby_levels_sorted_by_strength():
    """Nearby levels should be sorted by strength descending."""
    hmap = HiddenLiquidityMap()
    now = _ts()

    # More events at 100.25 than 100.00
    for _ in range(10):
        hmap.record_hidden_event(100.25, 10, now)
    for _ in range(2):
        hmap.record_hidden_event(100.0, 10, now)

    nearby = hmap.get_nearby_levels(100.0, radius_ticks=4, now=now)
    assert len(nearby) >= 2
    assert nearby[0].strength >= nearby[1].strength


async def test_get_nearby_levels_empty_for_no_history():
    """get_nearby_levels should return empty list when no data exists."""
    hmap = HiddenLiquidityMap()
    nearby = hmap.get_nearby_levels(100.0, now=_ts())
    assert nearby == []


# ------------------------------------------------------------------
# Strategy integration
# ------------------------------------------------------------------


async def test_is_near_hidden_support_long():
    """Long direction should find hidden support below current price."""
    config = HiddenLiquidityConfig(
        min_strength=0.3, radius_ticks=4, tick_size=0.25
    )
    hmap = HiddenLiquidityMap(config=config)
    now = _ts()

    # Strong hidden level at 99.75 (below current 100.25)
    for _ in range(10):
        hmap.record_hidden_event(99.75, 20, now)

    assert hmap.is_near_hidden_support(100.25, "long", now=now) is True


async def test_is_near_hidden_support_short():
    """Short direction should find hidden resistance above current price."""
    config = HiddenLiquidityConfig(
        min_strength=0.3, radius_ticks=4, tick_size=0.25
    )
    hmap = HiddenLiquidityMap(config=config)
    now = _ts()

    # Strong hidden level at 100.75 (above current 100.25)
    for _ in range(10):
        hmap.record_hidden_event(100.75, 20, now)

    assert hmap.is_near_hidden_support(100.25, "short", now=now) is True


async def test_is_near_hidden_support_false_when_weak():
    """Should return False when hidden levels exist but are too weak."""
    config = HiddenLiquidityConfig(
        min_strength=0.5, radius_ticks=4, tick_size=0.25, decay_days=30.0
    )
    hmap = HiddenLiquidityMap(config=config)
    now = _ts()

    # Single old event — weak strength
    hmap.record_hidden_event(99.75, 10, now - timedelta(days=90))

    assert hmap.is_near_hidden_support(100.25, "long", now=now) is False


async def test_is_near_hidden_support_false_when_no_levels():
    """Should return False when no hidden levels exist."""
    hmap = HiddenLiquidityMap()
    assert hmap.is_near_hidden_support(100.25, "long", now=_ts()) is False


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------


async def test_persist_and_reload(tmp_path: Path):
    """Events should round-trip through Parquet correctly."""
    config = HiddenLiquidityConfig(parquet_dir=str(tmp_path))
    now = _ts()

    # Write events
    hmap = HiddenLiquidityMap(config=config, persist=True)
    for i in range(5):
        hmap.record_hidden_event(100.25, 10 + i, now)
    hmap.flush_parquet()

    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(parquet_files) == 1

    df = pl.read_parquet(parquet_files[0])
    assert len(df) == 5
    assert set(df.columns) == {"timestamp", "price", "size"}

    # Reload into a fresh map
    hmap2 = HiddenLiquidityMap(config=config)
    loaded = hmap2.load_from_parquet()
    assert loaded == 5
    assert hmap2.total_events == 5

    # Strength should match
    s1 = hmap.get_strength(100.25, now=now)
    s2 = hmap2.get_strength(100.25, now=now)
    assert s1 == pytest.approx(s2)
