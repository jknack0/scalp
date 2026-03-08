"""Intraday volatility & volume profile mapping for MES.

Quantifies the U-shaped intraday vol pattern, identifies dead zone boundaries,
maps spread cost by time slot, builds volume profiles with POC/value area,
and flags event days. Output feeds Phase 3 strategy entry filters.

Key analyses:
- 15-minute realized volatility heatmap (time slot × month)
- Dead zone identification via vol threshold crossing
- Spread cost as % of ATR by time slot
- Session volume profiles with POC, VAH, VAL
- FOMC/NFP/CPI event day detection
"""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from src.analysis.bar_statistics import compute_log_returns
from src.data.bars import resample_bars


# ── Result dataclasses ───────────────────────────────────────────────

@dataclass(frozen=True)
class VolatilityHeatmap:
    """15-min realized vol heatmap across time slots and months."""
    time_slots: list[str]        # e.g. ["09:30", "09:45", ...]
    months: list[int]            # e.g. [1, 2, ..., 12]
    vol_matrix: list[list[float]]  # [month_idx][slot_idx] annualized RV
    session_avg: float           # grand mean across all slots/months


@dataclass(frozen=True)
class DeadZone:
    """Time window where vol drops below threshold × session average."""
    start_time: str   # e.g. "11:15"
    end_time: str     # e.g. "13:45"
    threshold_multiplier: float
    confidence: float  # fraction of months where this zone holds


@dataclass(frozen=True)
class UShapeMetrics:
    """U-shaped intraday vol pattern summary."""
    open_vol: float     # avg RV in first hour (9:30-10:30)
    midday_vol: float   # avg RV in midday (11:30-13:30)
    close_vol: float    # avg RV in last hour (15:00-16:00)
    u_shape_ratio: float  # (open + close) / (2 * midday)
    session_avg: float


@dataclass(frozen=True)
class SpreadCostProfile:
    """Spread as % of ATR by 15-min time slot."""
    time_slots: list[str]
    atr_values: list[float]
    spread_pct_of_atr: list[float]  # spread / ATR × 100


@dataclass(frozen=True)
class VolumeProfile:
    """Volume at each price level for one trading session."""
    date: date
    price_levels: list[float]
    volumes: list[int]
    poc: float           # price of control (max volume)
    vah: float           # value area high
    val: float           # value area low


@dataclass(frozen=True)
class EventDay:
    """Known macro event day."""
    date: date
    event_type: str      # "FOMC", "NFP", "CPI"


# ── Time slot helpers ────────────────────────────────────────────────

def _time_slot_label(hour: int, minute: int) -> str:
    """Format a time slot label like '09:30'."""
    return f"{hour:02d}:{minute:02d}"


def _rth_15min_slots() -> list[str]:
    """Generate 15-min slot labels for RTH (9:30-16:00)."""
    slots = []
    h, m = 9, 30
    while (h, m) < (16, 0):
        slots.append(_time_slot_label(h, m))
        m += 15
        if m >= 60:
            h += 1
            m -= 60
    return slots


def _slot_index(hour: int, minute: int) -> int:
    """Convert hour:minute to 15-min slot index (0-based from 9:30)."""
    total_minutes = hour * 60 + minute
    start_minutes = 9 * 60 + 30
    return (total_minutes - start_minutes) // 15


# ── Core functions ───────────────────────────────────────────────────

def compute_realized_vol_heatmap(
    df_1s: pl.DataFrame,
    window_minutes: int = 15,
) -> VolatilityHeatmap:
    """Compute realized vol heatmap by 15-min time slot and month.

    For each window_minutes bucket, computes RV from squared 1-min log returns
    (annualized). Groups by (time_slot, month).

    Args:
        df_1s: 1-second bar DataFrame with timestamp and close columns.
        window_minutes: Bucket size in minutes (default 15).

    Returns:
        VolatilityHeatmap with time_slots × months matrix of annualized RV.
    """
    # Resample to 1-minute bars
    df_1m = resample_bars(df_1s, "1m")

    # Add time-of-day and month columns (cast to Int64 to avoid Int8 overflow)
    df_1m = df_1m.with_columns([
        pl.col("timestamp").dt.hour().cast(pl.Int64).alias("hour"),
        pl.col("timestamp").dt.minute().cast(pl.Int64).alias("minute"),
        pl.col("timestamp").dt.month().alias("month"),
    ])

    # Filter to RTH only (9:30-16:00)
    total_min = pl.col("hour") * 60 + pl.col("minute")
    df_1m = df_1m.filter((total_min >= 9 * 60 + 30) & (total_min < 16 * 60))

    # Compute log returns
    df_1m = df_1m.sort("timestamp").with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return")
    ).drop_nulls("log_return")

    # Assign 15-min slot
    df_1m = df_1m.with_columns(
        ((pl.col("hour") * 60 + pl.col("minute") - 9 * 60 - 30) // window_minutes)
        .alias("slot_idx")
    )

    # Compute RV per (slot, month): sqrt(sum of squared returns) * annualization
    # Annualize: sqrt(252 trading days * slots_per_day)
    slots_per_day = (6 * 60 + 30) // window_minutes  # 9:30-16:00 = 390 min
    annualization = np.sqrt(252 * slots_per_day)

    rv_by_slot_month = (
        df_1m.group_by(["slot_idx", "month"])
        .agg([
            (pl.col("log_return").pow(2).mean().sqrt() * annualization).alias("rv"),
        ])
        .sort(["month", "slot_idx"])
    )

    # Build the matrix
    time_slots = _rth_15min_slots()
    n_slots = len(time_slots)
    months_present = sorted(rv_by_slot_month["month"].unique().to_list())

    vol_matrix: list[list[float]] = []
    all_vals: list[float] = []

    for month in months_present:
        month_data = rv_by_slot_month.filter(pl.col("month") == month)
        row = [0.0] * n_slots
        for r in month_data.iter_rows(named=True):
            idx = int(r["slot_idx"])
            if 0 <= idx < n_slots:
                row[idx] = float(r["rv"])
                all_vals.append(float(r["rv"]))
        vol_matrix.append(row)

    session_avg = float(np.mean(all_vals)) if all_vals else 0.0

    return VolatilityHeatmap(
        time_slots=time_slots,
        months=months_present,
        vol_matrix=vol_matrix,
        session_avg=session_avg,
    )


def identify_dead_zone(
    heatmap: VolatilityHeatmap,
    threshold: float = 1.5,
) -> DeadZone | None:
    """Find the contiguous block where vol < session_avg / threshold.

    Identifies the longest stretch of 15-min slots where average vol
    across months falls below (session_avg / threshold).

    Args:
        heatmap: VolatilityHeatmap from compute_realized_vol_heatmap.
        threshold: Multiplier — dead zone is where vol < session_avg / threshold.

    Returns:
        DeadZone with start/end times, or None if no dead zone found.
    """
    if not heatmap.vol_matrix or not heatmap.time_slots:
        return None

    # Average vol across months for each slot
    n_slots = len(heatmap.time_slots)
    avg_by_slot = []
    for s in range(n_slots):
        vals = [heatmap.vol_matrix[m][s] for m in range(len(heatmap.months))
                if heatmap.vol_matrix[m][s] > 0]
        avg_by_slot.append(float(np.mean(vals)) if vals else 0.0)

    cutoff = heatmap.session_avg / threshold

    # Find longest contiguous block below cutoff
    best_start = -1
    best_len = 0
    cur_start = -1
    cur_len = 0
    # Count months where each slot is below cutoff (for confidence)
    months_below: list[int] = []

    for i, v in enumerate(avg_by_slot):
        if v > 0 and v < cutoff:
            if cur_start < 0:
                cur_start = i
                cur_len = 0
            cur_len += 1
            if cur_len > best_len:
                best_start = cur_start
                best_len = cur_len
        else:
            cur_start = -1
            cur_len = 0

    if best_len == 0:
        return None

    # Compute confidence: fraction of months where each slot in the zone is below cutoff
    n_months = len(heatmap.months)
    confident_count = 0
    total_checks = 0
    for s in range(best_start, best_start + best_len):
        for m in range(n_months):
            v = heatmap.vol_matrix[m][s]
            if v > 0:
                total_checks += 1
                if v < cutoff:
                    confident_count += 1

    confidence = confident_count / total_checks if total_checks > 0 else 0.0

    # End slot: the slot after the last dead zone slot
    end_idx = best_start + best_len
    end_time = heatmap.time_slots[end_idx] if end_idx < n_slots else "16:00"

    return DeadZone(
        start_time=heatmap.time_slots[best_start],
        end_time=end_time,
        threshold_multiplier=threshold,
        confidence=confidence,
    )


def compute_u_shape_metrics(heatmap: VolatilityHeatmap) -> UShapeMetrics:
    """Compute open/midday/close vol averages and U-shape ratio.

    Segments:
    - Open: 9:30-10:30 (slots 0-3)
    - Midday: 11:30-13:30 (slots 8-15)
    - Close: 15:00-16:00 (slots 22-25)

    Args:
        heatmap: VolatilityHeatmap from compute_realized_vol_heatmap.

    Returns:
        UShapeMetrics with segment averages and ratio.
    """
    n_slots = len(heatmap.time_slots)

    def _avg_slots(start_idx: int, end_idx: int) -> float:
        vals = []
        for m in range(len(heatmap.months)):
            for s in range(start_idx, min(end_idx, n_slots)):
                v = heatmap.vol_matrix[m][s]
                if v > 0:
                    vals.append(v)
        return float(np.mean(vals)) if vals else 0.0

    # Slot indices: 9:30=0, 9:45=1, 10:00=2, 10:15=3 → open = 0..3
    # 11:30=8, 11:45=9, ..., 13:15=15 → midday = 8..15
    # 15:00=22, 15:15=23, 15:30=24, 15:45=25 → close = 22..25
    open_vol = _avg_slots(0, 4)
    midday_vol = _avg_slots(8, 16)
    close_vol = _avg_slots(22, 26)

    u_shape_ratio = (
        (open_vol + close_vol) / (2.0 * midday_vol)
        if midday_vol > 0
        else float("inf")
    )

    return UShapeMetrics(
        open_vol=open_vol,
        midday_vol=midday_vol,
        close_vol=close_vol,
        u_shape_ratio=u_shape_ratio,
        session_avg=heatmap.session_avg,
    )


def compute_spread_cost_by_slot(
    df_1s: pl.DataFrame,
    window_minutes: int = 15,
    tick_spread_points: float = 0.25,
) -> SpreadCostProfile:
    """Compute ATR per 15-min slot and spread as % of ATR.

    Args:
        df_1s: 1-second bar DataFrame.
        window_minutes: Bucket size in minutes (default 15).
        tick_spread_points: Assumed constant spread in index points (default 0.25).

    Returns:
        SpreadCostProfile with ATR and spread % per time slot.
    """
    # Resample to 1-minute bars for ATR computation
    df_1m = resample_bars(df_1s, "1m")

    df_1m = df_1m.with_columns([
        pl.col("timestamp").dt.hour().cast(pl.Int64).alias("hour"),
        pl.col("timestamp").dt.minute().cast(pl.Int64).alias("minute"),
    ])

    # Filter RTH
    total_min = pl.col("hour") * 60 + pl.col("minute")
    df_1m = df_1m.filter((total_min >= 9 * 60 + 30) & (total_min < 16 * 60))

    # True Range per 1-min bar
    df_1m = df_1m.sort("timestamp").with_columns(
        pl.col("close").shift(1).alias("prev_close")
    ).drop_nulls("prev_close").with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("prev_close")).abs(),
            (pl.col("low") - pl.col("prev_close")).abs(),
        ).alias("true_range")
    )

    # Assign 15-min slot
    df_1m = df_1m.with_columns(
        ((pl.col("hour") * 60 + pl.col("minute") - 9 * 60 - 30) // window_minutes)
        .alias("slot_idx")
    )

    # ATR per slot: mean true range of all 1-min bars in that slot
    atr_by_slot = (
        df_1m.group_by("slot_idx")
        .agg(pl.col("true_range").mean().alias("atr"))
        .sort("slot_idx")
    )

    time_slots = _rth_15min_slots()
    n_slots = len(time_slots)
    atr_values = [0.0] * n_slots
    spread_pct = [0.0] * n_slots

    for row in atr_by_slot.iter_rows(named=True):
        idx = int(row["slot_idx"])
        if 0 <= idx < n_slots:
            atr = float(row["atr"])
            atr_values[idx] = atr
            spread_pct[idx] = (tick_spread_points / atr * 100.0) if atr > 0 else 0.0

    return SpreadCostProfile(
        time_slots=time_slots,
        atr_values=atr_values,
        spread_pct_of_atr=spread_pct,
    )


def build_session_volume_profile(
    df_1s: pl.DataFrame,
    tick_size: float = 0.25,
) -> dict[float, int]:
    """Aggregate volume at each price tick level for one session.

    Args:
        df_1s: 1-second bar DataFrame for a single trading day.
        tick_size: Price tick size (default 0.25 for MES).

    Returns:
        Dict of price_level → total volume.
    """
    # Round close prices to tick size and aggregate volume
    df = df_1s.with_columns(
        (pl.col("close") / tick_size).round(0).cast(pl.Float64)
        .mul(tick_size)
        .alias("price_level")
    )

    vol_by_price = (
        df.group_by("price_level")
        .agg(pl.col("volume").sum().alias("total_volume"))
        .sort("price_level")
    )

    return {
        float(row["price_level"]): int(row["total_volume"])
        for row in vol_by_price.iter_rows(named=True)
    }


def compute_value_area(
    price_volumes: dict[float, int],
    pct: float = 0.70,
) -> VolumeProfile:
    """Compute POC, VAH, and VAL from a price→volume map.

    POC = price with highest volume.
    Value area = tightest range around POC containing ≥pct of total volume.

    Args:
        price_volumes: Dict of price_level → volume.
        pct: Target fraction of total volume for value area (default 0.70).

    Returns:
        VolumeProfile with POC, VAH, VAL (date set to date.min as placeholder).
    """
    if not price_volumes:
        return VolumeProfile(
            date=date.min,
            price_levels=[],
            volumes=[],
            poc=0.0,
            vah=0.0,
            val=0.0,
        )

    prices = sorted(price_volumes.keys())
    volumes = [price_volumes[p] for p in prices]
    total_vol = sum(volumes)

    # POC
    poc_idx = int(np.argmax(volumes))
    poc = prices[poc_idx]

    # Expand outward from POC to capture pct of volume
    target = total_vol * pct
    lo = poc_idx
    hi = poc_idx
    accumulated = volumes[poc_idx]

    while accumulated < target and (lo > 0 or hi < len(prices) - 1):
        # Compare volume one step below vs one step above
        vol_below = volumes[lo - 1] if lo > 0 else -1
        vol_above = volumes[hi + 1] if hi < len(prices) - 1 else -1

        if vol_below >= vol_above:
            lo -= 1
            accumulated += volumes[lo]
        else:
            hi += 1
            accumulated += volumes[hi]

    return VolumeProfile(
        date=date.min,
        price_levels=prices,
        volumes=volumes,
        poc=poc,
        vah=prices[hi],
        val=prices[lo],
    )


def build_daily_profiles(
    df_1s: pl.DataFrame,
    tick_size: float = 0.25,
) -> list[VolumeProfile]:
    """Build volume profiles for each trading day.

    Args:
        df_1s: 1-second bar DataFrame spanning multiple days.
        tick_size: Price tick size (default 0.25 for MES).

    Returns:
        List of VolumeProfile, one per trading day.
    """
    df = df_1s.with_columns(pl.col("timestamp").dt.date().alias("trade_date"))
    dates = sorted(df["trade_date"].unique().to_list())

    profiles = []
    for d in dates:
        day_df = df.filter(pl.col("trade_date") == d)
        pv = build_session_volume_profile(day_df, tick_size=tick_size)
        if not pv:
            continue
        profile = compute_value_area(pv)
        # Replace placeholder date with actual date
        profiles.append(VolumeProfile(
            date=d,
            price_levels=profile.price_levels,
            volumes=profile.volumes,
            poc=profile.poc,
            vah=profile.vah,
            val=profile.val,
        ))

    return profiles


def get_event_days(year: int = 2024) -> list[EventDay]:
    """Return hardcoded FOMC/NFP/CPI event dates for a given year.

    Args:
        year: Calendar year (currently only 2024 is populated).

    Returns:
        List of EventDay sorted by date.
    """
    events_2024 = [
        # FOMC rate decisions (statement release dates)
        EventDay(date(2024, 1, 31), "FOMC"),
        EventDay(date(2024, 3, 20), "FOMC"),
        EventDay(date(2024, 5, 1), "FOMC"),
        EventDay(date(2024, 6, 12), "FOMC"),
        EventDay(date(2024, 7, 31), "FOMC"),
        EventDay(date(2024, 9, 18), "FOMC"),
        EventDay(date(2024, 11, 7), "FOMC"),
        EventDay(date(2024, 12, 18), "FOMC"),
        # NFP (first Friday of each month)
        EventDay(date(2024, 1, 5), "NFP"),
        EventDay(date(2024, 2, 2), "NFP"),
        EventDay(date(2024, 3, 8), "NFP"),
        EventDay(date(2024, 4, 5), "NFP"),
        EventDay(date(2024, 5, 3), "NFP"),
        EventDay(date(2024, 6, 7), "NFP"),
        EventDay(date(2024, 7, 5), "NFP"),
        EventDay(date(2024, 8, 2), "NFP"),
        EventDay(date(2024, 9, 6), "NFP"),
        EventDay(date(2024, 10, 4), "NFP"),
        EventDay(date(2024, 11, 1), "NFP"),
        EventDay(date(2024, 12, 6), "NFP"),
        # CPI release dates
        EventDay(date(2024, 1, 11), "CPI"),
        EventDay(date(2024, 2, 13), "CPI"),
        EventDay(date(2024, 3, 12), "CPI"),
        EventDay(date(2024, 4, 10), "CPI"),
        EventDay(date(2024, 5, 15), "CPI"),
        EventDay(date(2024, 6, 12), "CPI"),
        EventDay(date(2024, 7, 11), "CPI"),
        EventDay(date(2024, 8, 14), "CPI"),
        EventDay(date(2024, 9, 11), "CPI"),
        EventDay(date(2024, 10, 10), "CPI"),
        EventDay(date(2024, 11, 13), "CPI"),
        EventDay(date(2024, 12, 11), "CPI"),
    ]

    if year == 2024:
        return sorted(events_2024, key=lambda e: e.date)

    return []


# ── Reporting ────────────────────────────────────────────────────────

def print_volatility_summary(
    heatmap: VolatilityHeatmap,
    dead_zone: DeadZone | None,
    u_shape: UShapeMetrics,
    spread: SpreadCostProfile,
) -> None:
    """Print formatted summary of intraday volatility analysis."""
    print(f"\n{'=' * 70}")
    print("INTRADAY VOLATILITY & VOLUME PROFILE SUMMARY")
    print(f"{'=' * 70}")

    # U-shape metrics
    print(f"\n  U-SHAPE METRICS")
    print(f"  {'─' * 40}")
    print(f"  Open vol (9:30-10:30):    {u_shape.open_vol:.4f}")
    print(f"  Midday vol (11:30-13:30): {u_shape.midday_vol:.4f}")
    print(f"  Close vol (15:00-16:00):  {u_shape.close_vol:.4f}")
    print(f"  U-shape ratio:            {u_shape.u_shape_ratio:.2f}x")
    print(f"  Session average:          {u_shape.session_avg:.4f}")

    # Dead zone
    print(f"\n  DEAD ZONE")
    print(f"  {'─' * 40}")
    if dead_zone:
        print(f"  Window:     {dead_zone.start_time} – {dead_zone.end_time}")
        print(f"  Threshold:  vol < session_avg / {dead_zone.threshold_multiplier}")
        print(f"  Confidence: {dead_zone.confidence:.1%} of months")
    else:
        print(f"  No dead zone identified at current threshold")

    # Spread cost — show slots with highest spread %
    print(f"\n  SPREAD COST (top 5 worst slots)")
    print(f"  {'─' * 40}")
    print(f"  {'Slot':<8} {'ATR':>8} {'Spread%':>10}")

    # Sort slots by spread_pct descending, take top 5
    slot_data = [
        (s, a, p) for s, a, p in
        zip(spread.time_slots, spread.atr_values, spread.spread_pct_of_atr)
        if a > 0
    ]
    slot_data.sort(key=lambda x: x[2], reverse=True)
    for slot, atr, pct in slot_data[:5]:
        print(f"  {slot:<8} {atr:>8.4f} {pct:>9.1f}%")

    # Spread cost — show slots with lowest spread %
    print(f"\n  SPREAD COST (top 5 best slots)")
    print(f"  {'─' * 40}")
    print(f"  {'Slot':<8} {'ATR':>8} {'Spread%':>10}")
    slot_data.sort(key=lambda x: x[2])
    for slot, atr, pct in slot_data[:5]:
        print(f"  {slot:<8} {atr:>8.4f} {pct:>9.1f}%")

    print(f"\n{'=' * 70}\n")
