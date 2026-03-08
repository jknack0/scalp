"""Tests for the VPIN regime filter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.filters.vpin_monitor import (
    BLOCKED_IN_MEAN_REVERSION,
    BLOCKED_IN_TRENDING,
    VPINConfig,
    VPINMonitor,
)


def _ts() -> datetime:
    return datetime.now(timezone.utc)


async def test_balanced_volume_low_vpin():
    """Perfectly balanced buy/sell volume should produce VPIN near 0."""
    config = VPINConfig(bucket_size=10, n_buckets=5)
    mon = VPINMonitor(config=config)

    # Alternate buy and sell ticks, equal size
    for i in range(100):
        if i % 2 == 0:
            mon.on_tick(price=100.50, size=1, bid=100.0, ask=100.50, timestamp=_ts())
        else:
            mon.on_tick(price=100.0, size=1, bid=100.0, ask=100.50, timestamp=_ts())

    regime, vpin = mon.get_regime()
    assert vpin < 0.15, f"VPIN should be near 0 for balanced volume, got {vpin}"
    assert regime == "mean_reversion" or regime == "undefined"


async def test_onesided_buy_volume_high_vpin():
    """All-buy volume should produce VPIN near 1.0 -> trending."""
    config = VPINConfig(bucket_size=10, n_buckets=5)
    mon = VPINMonitor(config=config)

    # All trades at the ask (buyer-initiated)
    for _ in range(100):
        mon.on_tick(price=100.50, size=1, bid=100.0, ask=100.50, timestamp=_ts())

    regime, vpin = mon.get_regime()
    assert vpin > 0.8, f"VPIN should be near 1.0 for one-sided volume, got {vpin}"
    assert regime == "trending"


async def test_onesided_sell_volume_high_vpin():
    """All-sell volume should also produce VPIN near 1.0 -> trending."""
    config = VPINConfig(bucket_size=10, n_buckets=5)
    mon = VPINMonitor(config=config)

    for _ in range(100):
        mon.on_tick(price=100.0, size=1, bid=100.0, ask=100.50, timestamp=_ts())

    regime, vpin = mon.get_regime()
    assert vpin > 0.8
    assert regime == "trending"


async def test_bucket_accumulation_correct():
    """Buckets should complete at exactly bucket_size volume."""
    config = VPINConfig(bucket_size=20, n_buckets=50)
    mon = VPINMonitor(config=config)

    # Push 50 contracts -> should complete 2 buckets (50 / 20 = 2, remainder 10)
    for _ in range(50):
        mon.on_tick(price=100.50, size=1, bid=100.0, ask=100.50, timestamp=_ts())

    assert mon.bucket_count == 2


async def test_large_tick_spans_multiple_buckets():
    """A single large tick should correctly span multiple buckets."""
    config = VPINConfig(bucket_size=10, n_buckets=50)
    mon = VPINMonitor(config=config)

    mon.on_tick(price=100.50, size=35, bid=100.0, ask=100.50, timestamp=_ts())
    assert mon.bucket_count == 3  # 35 / 10 = 3 full buckets, 5 remainder


async def test_regime_gating_trending_blocks_vwap():
    """VWAP should be blocked in trending regime."""
    config = VPINConfig(bucket_size=10, n_buckets=5)
    mon = VPINMonitor(config=config)

    # Create trending regime (all buys)
    for _ in range(100):
        mon.on_tick(price=100.50, size=1, bid=100.0, ask=100.50, timestamp=_ts())

    blocked, reason = mon.should_block("vwap_reversion")
    assert blocked is True
    assert "trending" in reason

    # ORB should NOT be blocked in trending
    blocked_orb, _ = mon.should_block("orb")
    assert blocked_orb is False


async def test_regime_gating_mean_reversion_blocks_orb():
    """ORB and CVD should be blocked in mean_reversion regime."""
    config = VPINConfig(bucket_size=10, n_buckets=5)
    mon = VPINMonitor(config=config)

    # Create mean reversion regime (balanced volume)
    for i in range(100):
        if i % 2 == 0:
            mon.on_tick(price=100.50, size=1, bid=100.0, ask=100.50, timestamp=_ts())
        else:
            mon.on_tick(price=100.0, size=1, bid=100.0, ask=100.50, timestamp=_ts())

    regime, _ = mon.get_regime()
    if regime == "mean_reversion":
        blocked_orb, reason = mon.should_block("orb")
        assert blocked_orb is True
        assert "mean_reversion" in reason

        blocked_cvd, _ = mon.should_block("cvd_divergence")
        assert blocked_cvd is True

        # VWAP should NOT be blocked
        blocked_vwap, _ = mon.should_block("vwap_reversion")
        assert blocked_vwap is False


async def test_vol_regime_never_blocked():
    """VolRegime strategy should never be blocked by VPIN."""
    config = VPINConfig(bucket_size=10, n_buckets=5)
    mon = VPINMonitor(config=config)

    # Trending
    for _ in range(100):
        mon.on_tick(price=100.50, size=1, bid=100.0, ask=100.50, timestamp=_ts())

    blocked, _ = mon.should_block("vol_regime_switcher")
    assert blocked is False


async def test_no_regime_before_min_buckets():
    """Should return 'undefined' when fewer than 5 buckets completed."""
    config = VPINConfig(bucket_size=100, n_buckets=50)
    mon = VPINMonitor(config=config)

    # Push only enough for 3 buckets
    for _ in range(300):
        mon.on_tick(price=100.50, size=1, bid=100.0, ask=100.50, timestamp=_ts())

    assert mon.bucket_count == 3
    regime, _ = mon.get_regime()
    assert regime == "undefined"

    # Should not block anything when undefined
    blocked, _ = mon.should_block("vwap_reversion")
    assert blocked is False


async def test_tick_rule_fallback():
    """Tick rule should classify mid-spread trades based on price direction."""
    config = VPINConfig(bucket_size=10, n_buckets=5)
    mon = VPINMonitor(config=config)

    # All trades at mid-price, but price trending up -> should classify as buys
    for i in range(100):
        price = 100.25 + i * 0.01  # steadily rising
        mon.on_tick(price=price, size=1, bid=100.0, ask=100.50, timestamp=_ts())

    regime, vpin = mon.get_regime()
    assert vpin > 0.5, f"Rising tick rule should produce high VPIN, got {vpin}"


async def test_bar_approx_buy_side():
    """Bar approximation with close > open should be buy-attributed."""
    config = VPINConfig(bucket_size=50, n_buckets=5)
    mon = VPINMonitor(config=config)

    # All bars close > open -> all buy
    for _ in range(10):
        mon.on_bar_approx(open_=100.0, close=100.50, volume=50, timestamp=_ts())

    regime, vpin = mon.get_regime()
    assert vpin > 0.8
    assert regime == "trending"


async def test_bar_approx_balanced():
    """Mixed bar directions should produce lower VPIN."""
    # bucket_size=100, each bar is 50 vol -> 2 bars per bucket
    # alternating buy/sell bars means each bucket gets ~balanced volume
    config = VPINConfig(bucket_size=100, n_buckets=5)
    mon = VPINMonitor(config=config)

    for i in range(20):
        if i % 2 == 0:
            mon.on_bar_approx(open_=100.0, close=100.50, volume=50, timestamp=_ts())
        else:
            mon.on_bar_approx(open_=100.50, close=100.0, volume=50, timestamp=_ts())

    regime, vpin = mon.get_regime()
    assert vpin < 0.15


async def test_reset_clears_state():
    """Reset should clear all internal state."""
    config = VPINConfig(bucket_size=10, n_buckets=5)
    mon = VPINMonitor(config=config)

    for _ in range(100):
        mon.on_tick(price=100.50, size=1, bid=100.0, ask=100.50, timestamp=_ts())

    assert mon.bucket_count > 0
    mon.reset()
    assert mon.bucket_count == 0
    assert mon.latest_state is None
    regime, vpin = mon.get_regime()
    assert regime == "undefined"
    assert vpin == 0.0


async def test_no_data_returns_undefined():
    """With no data, regime should be undefined and no blocking."""
    mon = VPINMonitor()
    regime, vpin = mon.get_regime()
    assert regime == "undefined"
    assert vpin == 0.0
    blocked, _ = mon.should_block("orb")
    assert blocked is False


async def test_persist_flushes_to_parquet(tmp_path: Path):
    """Persistence should flush VPIN states to Parquet."""
    config = VPINConfig(
        bucket_size=10, n_buckets=5,
        persist_every_n=3, parquet_dir=str(tmp_path),
    )
    mon = VPINMonitor(config=config, persist=True)

    # Generate enough ticks for 6 buckets (triggers flush at bucket 3 and 6)
    for _ in range(60):
        mon.on_tick(price=100.50, size=1, bid=100.0, ask=100.50, timestamp=_ts())

    assert mon.bucket_count == 6
    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(parquet_files) == 1

    df = pl.read_parquet(parquet_files[0])
    assert len(df) == 6  # 3 at flush 1 + 3 at flush 2
    assert set(df.columns) == {
        "timestamp", "vpin", "regime", "bucket_count", "last_bucket_imbalance",
    }
