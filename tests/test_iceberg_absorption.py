"""Tests for the iceberg and absorption detection module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from src.filters.iceberg_absorption import (
    AbsorptionDetector,
    AbsorptionSignal,
    IcebergConfig,
    IcebergDetector,
    IcebergSignal,
    TradeEvent,
    infer_aggressor,
)
from src.filters import L2Snapshot


def _ts(offset_s: float = 0.0) -> datetime:
    """Return a UTC timestamp with optional offset in seconds."""
    return datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=offset_s
    )


def _snap(
    ts: datetime,
    bids: list[tuple[float, int]] | None = None,
    asks: list[tuple[float, int]] | None = None,
) -> L2Snapshot:
    return L2Snapshot(timestamp=ts, bids=bids or [], asks=asks or [])


def _trade(
    ts: datetime, price: float, size: int, aggressor: str = "buy"
) -> TradeEvent:
    return TradeEvent(timestamp=ts, price=price, size=size, aggressor=aggressor)


# ------------------------------------------------------------------
# Tick rule
# ------------------------------------------------------------------


async def test_tick_rule_uptick_is_buy():
    """Uptick should infer buy aggressor."""
    assert infer_aggressor(100.25, 100.00) == "buy"


async def test_tick_rule_downtick_is_sell():
    """Downtick should infer sell aggressor."""
    assert infer_aggressor(99.75, 100.00) == "sell"


async def test_tick_rule_zero_tick_is_buy():
    """Zero-tick (same price) should infer buy aggressor."""
    assert infer_aggressor(100.00, 100.00) == "buy"


async def test_tick_rule_no_prior_is_buy():
    """No prior trade should default to buy aggressor."""
    assert infer_aggressor(100.00, None) == "buy"


# ------------------------------------------------------------------
# Iceberg Detector
# ------------------------------------------------------------------


async def test_iceberg_flagged_after_min_appearances_with_consumption():
    """Iceberg should fire after sustained presence + consumption."""
    config = IcebergConfig(
        min_appearances=5, consumed_threshold=20, top_n_levels=3
    )
    detector = IcebergDetector(config=config)

    ask_level = 100.25
    bids = [(100.0, 50), (99.75, 50), (99.50, 50)]
    asks = [(ask_level, 30), (100.50, 50), (100.75, 50)]

    # Interleave L2 snapshots and trades — level persists despite consumption
    signals: list[IcebergSignal] = []
    for i in range(6):
        sigs = detector.push_l2(_snap(_ts(i), bids=bids, asks=asks))
        signals.extend(sigs)
        # Trades consuming the ask level after it's registered
        if i < 3:
            detector.push_trade(_trade(_ts(i + 0.5), ask_level, 10, "buy"))

    # Should detect iceberg at ask_level
    iceberg_signals = [s for s in signals if s.price == ask_level]
    assert len(iceberg_signals) >= 1
    assert iceberg_signals[0].side == "ask"
    assert iceberg_signals[0].estimated_total_size >= 30  # consumed + visible
    assert 0.0 <= iceberg_signals[0].confidence <= 1.0


async def test_no_iceberg_for_brief_appearance():
    """No false positive when level appears briefly without trades."""
    config = IcebergConfig(
        min_appearances=10, consumed_threshold=50, top_n_levels=3
    )
    detector = IcebergDetector(config=config)

    bids = [(100.0, 50), (99.75, 50), (99.50, 50)]
    asks = [(100.25, 30), (100.50, 50), (100.75, 50)]

    # Only 3 snapshots — well below min_appearances
    all_signals: list[IcebergSignal] = []
    for i in range(3):
        sigs = detector.push_l2(_snap(_ts(i), bids=bids, asks=asks))
        all_signals.extend(sigs)

    assert len(all_signals) == 0


async def test_no_iceberg_without_consumption():
    """No iceberg when level persists but has zero consumption."""
    config = IcebergConfig(
        min_appearances=5, consumed_threshold=50, top_n_levels=3
    )
    detector = IcebergDetector(config=config)

    bids = [(100.0, 50), (99.75, 50), (99.50, 50)]
    asks = [(100.25, 30), (100.50, 50), (100.75, 50)]

    # No trades pushed — level persists but no consumption
    all_signals: list[IcebergSignal] = []
    for i in range(10):
        sigs = detector.push_l2(_snap(_ts(i), bids=bids, asks=asks))
        all_signals.extend(sigs)

    assert len(all_signals) == 0


async def test_iceberg_trade_updates_consumed():
    """push_trade should increment total_consumed on the correct level."""
    config = IcebergConfig(top_n_levels=3)
    detector = IcebergDetector(config=config)

    # Register the level first
    bids = [(100.0, 50)]
    asks = [(100.25, 50)]
    detector.push_l2(_snap(_ts(0), bids=bids, asks=asks))

    # Buy aggressor consumes ask at 100.25
    detector.push_trade(_trade(_ts(1), 100.25, 15, "buy"))

    key = (100.25, "ask")
    assert key in detector._levels
    assert detector._levels[key].total_consumed == 15


# ------------------------------------------------------------------
# Absorption Detector
# ------------------------------------------------------------------


async def test_absorption_signal_after_repeated_tests():
    """Absorbing signal should fire after test_threshold tests within window."""
    config = IcebergConfig(
        test_threshold=3, window_seconds=30.0, top_n_levels=3
    )
    detector = AbsorptionDetector(config=config)

    ask_level = 100.25
    bids = [(100.0, 50), (99.75, 50), (99.50, 50)]
    asks = [(ask_level, 100), (100.50, 50), (100.75, 50)]

    # Establish level in book
    detector.push_l2(_snap(_ts(0), bids=bids, asks=asks))

    # Push trades testing the ask level
    for i in range(4):
        detector.push_trade(_trade(_ts(i + 1), ask_level, 10, "buy"))

    # Push another L2 — level still present
    signals = detector.push_l2(_snap(_ts(5), bids=bids, asks=asks))

    absorbing = [s for s in signals if s.status == "absorbing"]
    assert len(absorbing) == 1
    assert absorbing[0].price == ask_level
    assert absorbing[0].test_count >= 3
    assert absorbing[0].absorbed_size == 40


async def test_absorption_broken_when_level_disappears():
    """Broken signal should fire when an absorbed level leaves the book."""
    config = IcebergConfig(
        test_threshold=3, window_seconds=30.0, top_n_levels=3
    )
    detector = AbsorptionDetector(config=config)

    ask_level = 100.25
    bids = [(100.0, 50), (99.75, 50), (99.50, 50)]
    asks_with = [(ask_level, 100), (100.50, 50), (100.75, 50)]
    asks_without = [(100.50, 50), (100.75, 50), (101.0, 50)]

    # Establish level
    detector.push_l2(_snap(_ts(0), bids=bids, asks=asks_with))

    # Test the level enough times
    for i in range(4):
        detector.push_trade(_trade(_ts(i + 1), ask_level, 10, "buy"))

    # Trigger absorbing signal
    detector.push_l2(_snap(_ts(5), bids=bids, asks=asks_with))

    # Now the level disappears
    signals = detector.push_l2(_snap(_ts(6), bids=bids, asks=asks_without))

    broken = [s for s in signals if s.status == "broken"]
    assert len(broken) == 1
    assert broken[0].price == ask_level
    assert broken[0].side == "ask"


async def test_no_absorption_without_enough_tests():
    """No absorption signal when test count is below threshold."""
    config = IcebergConfig(
        test_threshold=5, window_seconds=30.0, top_n_levels=3
    )
    detector = AbsorptionDetector(config=config)

    ask_level = 100.25
    bids = [(100.0, 50), (99.75, 50), (99.50, 50)]
    asks = [(ask_level, 100), (100.50, 50), (100.75, 50)]

    detector.push_l2(_snap(_ts(0), bids=bids, asks=asks))

    # Only 2 tests — below threshold of 5
    for i in range(2):
        detector.push_trade(_trade(_ts(i + 1), ask_level, 10, "buy"))

    signals = detector.push_l2(_snap(_ts(3), bids=bids, asks=asks))
    assert len(signals) == 0


async def test_absorption_tests_expire_outside_window():
    """Tests outside the time window should not count toward threshold."""
    config = IcebergConfig(
        test_threshold=3, window_seconds=10.0, top_n_levels=3
    )
    detector = AbsorptionDetector(config=config)

    ask_level = 100.25
    bids = [(100.0, 50), (99.75, 50), (99.50, 50)]
    asks = [(ask_level, 100), (100.50, 50), (100.75, 50)]

    detector.push_l2(_snap(_ts(0), bids=bids, asks=asks))

    # 3 trades at t=0,1,2 — within window
    for i in range(3):
        detector.push_trade(_trade(_ts(i), ask_level, 10, "buy"))

    # But check at t=20 — all tests are outside the 10s window
    detector.push_trade(_trade(_ts(20), ask_level, 10, "buy"))
    signals = detector.push_l2(_snap(_ts(20), bids=bids, asks=asks))

    # Only 1 test within window (the one at t=20), below threshold
    absorbing = [s for s in signals if s.status == "absorbing"]
    assert len(absorbing) == 0


# ------------------------------------------------------------------
# Parquet persistence
# ------------------------------------------------------------------


async def test_iceberg_persist_to_parquet(tmp_path: Path):
    """Iceberg signals should flush to Parquet when persist=True."""
    config = IcebergConfig(
        min_appearances=3,
        consumed_threshold=10,
        top_n_levels=3,
        persist=True,
        parquet_dir_iceberg=str(tmp_path / "iceberg"),
    )
    detector = IcebergDetector(config=config)

    bids = [(100.0, 50), (99.75, 50), (99.50, 50)]
    asks = [(100.25, 30), (100.50, 50), (100.75, 50)]

    # Interleave L2 and trades so the level is registered before trades hit it
    for i in range(5):
        detector.push_l2(_snap(_ts(i), bids=bids, asks=asks))
        if i < 2:
            detector.push_trade(_trade(_ts(i + 0.5), 100.25, 10, "buy"))

    parquet_files = list((tmp_path / "iceberg").glob("*.parquet"))
    assert len(parquet_files) == 1

    df = pl.read_parquet(parquet_files[0])
    assert len(df) >= 1
    assert "price" in df.columns
    assert "confidence" in df.columns


async def test_absorption_persist_to_parquet(tmp_path: Path):
    """Absorption signals should flush to Parquet when persist=True."""
    config = IcebergConfig(
        test_threshold=3,
        window_seconds=30.0,
        top_n_levels=3,
        persist=True,
        parquet_dir_absorption=str(tmp_path / "absorption"),
    )
    detector = AbsorptionDetector(config=config)

    bids = [(100.0, 50), (99.75, 50), (99.50, 50)]
    asks = [(100.25, 100), (100.50, 50), (100.75, 50)]

    detector.push_l2(_snap(_ts(0), bids=bids, asks=asks))

    for i in range(4):
        detector.push_trade(_trade(_ts(i + 1), 100.25, 10, "buy"))

    detector.push_l2(_snap(_ts(5), bids=bids, asks=asks))

    parquet_files = list((tmp_path / "absorption").glob("*.parquet"))
    assert len(parquet_files) == 1

    df = pl.read_parquet(parquet_files[0])
    assert len(df) >= 1
    assert set(df.columns) == {
        "timestamp",
        "price",
        "side",
        "test_count",
        "absorbed_size",
        "status",
    }
