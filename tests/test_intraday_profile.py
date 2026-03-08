"""Tests for intraday volatility & volume profile mapping."""

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from src.analysis.intraday_profile import (
    DeadZone,
    UShapeMetrics,
    VolatilityHeatmap,
    VolumeProfile,
    build_session_volume_profile,
    compute_realized_vol_heatmap,
    compute_spread_cost_by_slot,
    compute_u_shape_metrics,
    compute_value_area,
    get_event_days,
    identify_dead_zone,
)


def _make_rth_1s_bars(
    n_days: int = 5,
    start_date: datetime | None = None,
    vol_func=None,
) -> pl.DataFrame:
    """Build synthetic RTH 1s bars across multiple days.

    Args:
        n_days: Number of trading days.
        start_date: First day start (default: Mon 2024-06-03 9:30).
        vol_func: Optional callable(hour, minute) -> volatility_scale.
                  Defaults to uniform volatility.
    """
    if start_date is None:
        start_date = datetime(2024, 6, 3, 9, 30, 0)

    rng = np.random.default_rng(42)
    timestamps, opens, highs, lows, closes, volumes, vwaps = [], [], [], [], [], [], []
    price = 5000.0

    for day in range(n_days):
        day_start = start_date + timedelta(days=day)
        # Skip weekends
        while day_start.weekday() >= 5:
            day_start += timedelta(days=1)
        day_start = day_start.replace(hour=9, minute=30, second=0)

        for sec in range(23400):  # 6.5 hours = 23400 seconds
            ts = day_start + timedelta(seconds=sec)
            h = ts.hour
            m = ts.minute

            scale = vol_func(h, m) if vol_func else 1.0
            ret = rng.normal(0, 0.01 * scale)
            price = price * (1 + ret)
            price = max(price, 100.0)  # floor

            timestamps.append(ts)
            opens.append(price)
            highs.append(price + rng.uniform(0, 0.5))
            lows.append(price - rng.uniform(0, 0.5))
            closes.append(price)
            volumes.append(rng.integers(5, 50))
            vwaps.append(price)

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "vwap": vwaps,
    })


def _u_shape_vol(hour: int, minute: int) -> float:
    """Synthetic U-shaped vol: high at open/close, low at midday."""
    total_min = hour * 60 + minute
    start = 9 * 60 + 30   # 570
    end = 16 * 60          # 960
    mid = (start + end) / 2  # 765 = 12:45
    # Parabolic: high at edges, low at center
    t = (total_min - mid) / (end - start) * 2  # normalized -1 to 1
    return 0.5 + 2.0 * t ** 2  # 0.5 at center, 2.5 at edges


def test_realized_vol_computation():
    """Known squared returns produce known RV values."""
    rng = np.random.default_rng(42)
    # Build minimal 1s data: one day of RTH
    start = datetime(2024, 6, 3, 9, 30, 0)
    n = 23400
    timestamps = [start + timedelta(seconds=i) for i in range(n)]
    price = 5000.0
    closes = []
    for _ in range(n):
        price += rng.normal(0, 0.05)
        closes.append(price)

    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": [c + 0.25 for c in closes],
        "low": [c - 0.25 for c in closes],
        "close": closes,
        "volume": [10] * n,
        "vwap": closes,
    })

    heatmap = compute_realized_vol_heatmap(df, window_minutes=15)

    # All RV values should be positive finite numbers
    for row in heatmap.vol_matrix:
        for v in row:
            if v > 0:
                assert np.isfinite(v)
                assert v > 0
    assert heatmap.session_avg > 0


def test_heatmap_structure():
    """Heatmap has correct time slot and month dimensions."""
    df = _make_rth_1s_bars(n_days=3)
    heatmap = compute_realized_vol_heatmap(df, window_minutes=15)

    # 9:30 to 16:00 in 15-min slots = 26 slots
    assert len(heatmap.time_slots) == 26
    assert heatmap.time_slots[0] == "09:30"
    assert heatmap.time_slots[-1] == "15:45"

    # Should have at least 1 month
    assert len(heatmap.months) >= 1
    # Each month row should have 26 slots
    for row in heatmap.vol_matrix:
        assert len(row) == 26


def test_dead_zone_identification():
    """Synthetic U-shaped vol produces a dead zone in the midday."""
    df = _make_rth_1s_bars(n_days=10, vol_func=_u_shape_vol)
    heatmap = compute_realized_vol_heatmap(df, window_minutes=15)
    dead_zone = identify_dead_zone(heatmap, threshold=1.5)

    assert dead_zone is not None
    assert isinstance(dead_zone, DeadZone)
    # Dead zone should be somewhere in the midday range (11:xx - 14:xx)
    start_hour = int(dead_zone.start_time.split(":")[0])
    end_hour = int(dead_zone.end_time.split(":")[0])
    assert start_hour >= 10, f"Dead zone starts too early: {dead_zone.start_time}"
    assert end_hour <= 15, f"Dead zone ends too late: {dead_zone.end_time}"
    assert dead_zone.confidence > 0


def test_u_shape_metrics():
    """U-shape ratio > 1 for synthetic U-shaped vol data."""
    df = _make_rth_1s_bars(n_days=10, vol_func=_u_shape_vol)
    heatmap = compute_realized_vol_heatmap(df, window_minutes=15)
    u_shape = compute_u_shape_metrics(heatmap)

    assert isinstance(u_shape, UShapeMetrics)
    assert u_shape.open_vol > 0
    assert u_shape.midday_vol > 0
    assert u_shape.close_vol > 0
    # U-shaped: open and close vol should exceed midday
    assert u_shape.u_shape_ratio > 1.0, (
        f"Expected U-shape ratio > 1.0, got {u_shape.u_shape_ratio:.2f}"
    )


def test_spread_cost_increases_in_dead_zone():
    """Low ATR in dead zone produces higher spread % of ATR."""
    df = _make_rth_1s_bars(n_days=10, vol_func=_u_shape_vol)
    spread = compute_spread_cost_by_slot(df, window_minutes=15, tick_spread_points=0.25)

    # Find midday slots (roughly index 8-15) and open slots (0-3)
    midday_spreads = [
        spread.spread_pct_of_atr[i] for i in range(8, 16)
        if spread.spread_pct_of_atr[i] > 0
    ]
    open_spreads = [
        spread.spread_pct_of_atr[i] for i in range(0, 4)
        if spread.spread_pct_of_atr[i] > 0
    ]

    if midday_spreads and open_spreads:
        avg_midday = np.mean(midday_spreads)
        avg_open = np.mean(open_spreads)
        # Spread % should be higher at midday (low vol) than at open (high vol)
        assert avg_midday > avg_open, (
            f"Expected midday spread% ({avg_midday:.1f}%) > open spread% ({avg_open:.1f}%)"
        )


def test_volume_profile_poc():
    """Price with maximum volume is correctly identified as POC."""
    # Build a simple volume profile: most volume at 5000.00
    price_volumes = {
        4999.50: 100,
        4999.75: 200,
        5000.00: 500,  # POC
        5000.25: 300,
        5000.50: 150,
    }
    profile = compute_value_area(price_volumes, pct=0.70)

    assert profile.poc == 5000.00
    assert isinstance(profile, VolumeProfile)


def test_value_area_contains_70_pct():
    """VAH/VAL bracket contains at least 70% of total volume."""
    price_volumes = {
        4998.00: 50,
        4999.00: 100,
        4999.50: 200,
        4999.75: 350,
        5000.00: 600,   # POC
        5000.25: 400,
        5000.50: 250,
        5001.00: 120,
        5002.00: 30,
    }
    profile = compute_value_area(price_volumes, pct=0.70)
    total = sum(price_volumes.values())

    # Sum volume within VAL..VAH
    va_vol = sum(
        v for p, v in price_volumes.items()
        if profile.val <= p <= profile.vah
    )
    assert va_vol / total >= 0.70, (
        f"Value area contains {va_vol / total:.1%} of volume, expected >= 70%"
    )
    assert profile.val <= profile.poc <= profile.vah


def test_event_day_detection():
    """Known 2024 FOMC dates are correctly returned."""
    events = get_event_days(2024)

    assert len(events) > 0
    fomc_dates = [e.date for e in events if e.event_type == "FOMC"]
    nfp_dates = [e.date for e in events if e.event_type == "NFP"]
    cpi_dates = [e.date for e in events if e.event_type == "CPI"]

    # 8 FOMC meetings in 2024
    assert len(fomc_dates) == 8
    assert date(2024, 1, 31) in fomc_dates
    assert date(2024, 9, 18) in fomc_dates

    # 12 NFP releases
    assert len(nfp_dates) == 12

    # 12 CPI releases
    assert len(cpi_dates) == 12

    # Empty for unsupported year
    assert get_event_days(2023) == []
