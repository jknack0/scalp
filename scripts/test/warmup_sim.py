#!/usr/bin/env python3
"""Simulate live warmup using 5m parquet bars.

Loads a day of 5m bars, feeds them one-at-a-time through the SignalEngine
(same as live bot does), and prints signal snapshots to verify sanity.

Usage:
    python -u scripts/test/warmup_sim.py
    python -u scripts/test/warmup_sim.py --date 2024-06-15
"""

import argparse
import os
import sys
import time as _time
from datetime import date, datetime
from datetime import time as dt_time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import polars as pl

from src.core.events import BarEvent
from src.signals.signal_bundle import SignalEngine


def load_5m_bars(target_date: date) -> pl.DataFrame:
    """Load 5m bars for a specific date from parquet."""
    year = target_date.year
    path = f"data/parquet_5m/year={year}/data.parquet"
    df = pl.read_parquet(path)

    # Filter to target date (RTH: 9:30-16:00 ET)
    df = df.with_columns(
        pl.col("timestamp")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("US/Eastern")
            .alias("_et_ts"),
    )
    df = df.filter(
        (pl.col("_et_ts").dt.date() == target_date)
        & (pl.col("_et_ts").dt.time() >= dt_time(9, 30))
        & (pl.col("_et_ts").dt.time() < dt_time(16, 0))
    )
    return df.sort("timestamp")


def df_row_to_bar(row: dict) -> BarEvent:
    """Convert a parquet row to BarEvent."""
    ts = row["timestamp"]
    if hasattr(ts, "timestamp"):
        ts_ns = int(ts.timestamp() * 1e9)
    else:
        ts_ns = int(ts) * 1_000_000_000

    return BarEvent(
        symbol="MESH5",
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=int(row["volume"]),
        bar_type="5m",
        timestamp_ns=ts_ns,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2024-03-15",
                        help="Target date (YYYY-MM-DD)")
    parser.add_argument("--warmup-bars", type=int, default=80,
                        help="Number of bars before target date to use as warmup")
    args = parser.parse_args()

    target = date.fromisoformat(args.date)

    # Load bars: warmup from prior days + target date
    year = target.year
    path = f"data/parquet_5m/year={year}/data.parquet"
    df = pl.read_parquet(path)
    # Timestamps are already naive ET in the 5m parquet
    df = df.filter(
        (pl.col("timestamp").dt.time() >= dt_time(9, 30))
        & (pl.col("timestamp").dt.time() < dt_time(16, 0))
    )
    df = df.sort("timestamp")

    # Find target date rows
    df_indexed = df.with_row_index()
    target_rows = df_indexed.filter(pl.col("timestamp").dt.date() == target)
    if target_rows.is_empty():
        print(f"  No bars found for {target}")
        sys.exit(1)
    target_start_idx = int(target_rows["index"][0])

    # Warmup = N bars before target date
    warmup_start = max(0, target_start_idx - args.warmup_bars)
    warmup_df = df.slice(warmup_start, target_start_idx - warmup_start)
    target_df = df.slice(target_start_idx, len(target_rows))

    print(f"\n{'='*70}")
    print(f"  WARMUP SIMULATION: {target}")
    print(f"  Warmup bars: {len(warmup_df)} (from prior days)")
    print(f"  Target date bars: {len(target_df)}")
    print(f"{'='*70}\n")

    # Build merged signal list (same as main.py)
    all_signals = ["vwap_session", "adx", "atr", "relative_volume",
                   "session_time", "regime_v2", "donchian_channel"]
    signal_configs = {
        "donchian_channel": {"entry_period": 20, "exit_period": 10},
        "adx": {"period": 14, "threshold": 30.0},
        "regime_v2": {"model_path": "models/regime_v2", "pass_states": ["RANGING"]},
    }

    print("  Building SignalEngine...")
    engine = SignalEngine(all_signals, signal_configs)
    print(f"  Signals: {engine.signal_names}\n")

    # Phase 1: Feed warmup bars (simulate warmup_from_databento)
    bar_window: list[BarEvent] = []
    print(f"  --- WARMUP PHASE ({len(warmup_df)} bars) ---")
    t0 = _time.perf_counter()
    for row in warmup_df.iter_rows(named=True):
        bar = df_row_to_bar(row)
        bar_window.append(bar)

    # Compute once after warmup (like warmup_from_databento does)
    bundle = engine.compute(bar_window)
    dt = _time.perf_counter() - t0
    print(f"  Warmup computed in {dt:.2f}s ({len(bar_window)} bars)")

    # Print warmup signal state
    _print_snapshot(bundle, bar_window[-1], len(bar_window), "WARMUP")

    # Phase 2: Feed target date bars one at a time (simulate live)
    print(f"\n  --- LIVE SIMULATION ({len(target_df)} bars) ---")
    print(f"  {'bar':>4} {'time':>8} {'close':>8} {'atr':>7} {'adx':>5} {'rvol':>5} "
          f"{'vwap_sd':>7} {'regime':>8} {'conf':>5} {'dc_pass':>7} {'dc_dir':>6} {'dc_wid':>6}")
    print(f"  {'-'*95}")

    for i, row in enumerate(target_df.iter_rows(named=True)):
        bar = df_row_to_bar(row)
        bar_window.append(bar)
        if len(bar_window) > 500:
            bar_window = bar_window[-500:]

        t0 = _time.perf_counter()
        bundle = engine.compute(bar_window)
        dt = _time.perf_counter() - t0

        # Extract signal values
        atr_r = bundle.get("atr")
        adx_r = bundle.get("adx")
        rvol_r = bundle.get("relative_volume")
        vwap_r = bundle.get("vwap_session")
        regime_r = bundle.get("regime_v2")
        dc_r = bundle.get("donchian_channel")
        st_r = bundle.get("session_time")

        atr_val = atr_r.metadata.get("atr_raw", 0) if atr_r else 0
        adx_val = adx_r.value if adx_r else 0
        rvol_val = rvol_r.value if rvol_r else 0
        vwap_sd = vwap_r.metadata.get("deviation_sd", 0) if vwap_r else 0
        regime_name = regime_r.metadata.get("regime", "?") if regime_r and regime_r.metadata else "?"
        regime_conf = regime_r.metadata.get("confidence", 0) if regime_r and regime_r.metadata else 0
        dc_passes = dc_r.passes if dc_r else False
        dc_dir = dc_r.direction if dc_r else "none"
        dc_width = dc_r.metadata.get("width", 0) if dc_r else 0

        ts = row["timestamp"]
        time_str = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)

        slow = " SLOW" if dt > 0.1 else ""
        print(f"  {i+1:>4} {time_str:>8} {bar.close:>8.2f} {atr_val:>7.2f} {adx_val:>5.1f} "
              f"{rvol_val:>5.2f} {vwap_sd:>7.2f} {regime_name:>8} {regime_conf:>5.2f} "
              f"{'YES' if dc_passes else 'no':>7} {dc_dir:>6} {dc_width:>6.1f}{slow}")

    # Wait for any pending regime computation
    print(f"\n  Waiting for pending regime_v2 computation...")
    import time
    time.sleep(2)
    bundle = engine.compute(bar_window)
    regime_r = bundle.get("regime_v2")
    if regime_r:
        print(f"  Final regime: {regime_r.metadata}")

    print(f"\n  Done.\n")


def _print_snapshot(bundle, last_bar, n_bars, label):
    """Print a signal snapshot."""
    atr_r = bundle.get("atr")
    adx_r = bundle.get("adx")
    rvol_r = bundle.get("relative_volume")
    vwap_r = bundle.get("vwap_session")
    regime_r = bundle.get("regime_v2")

    print(f"\n  [{label}] bar_count={n_bars}, close={last_bar.close:.2f}")
    if atr_r:
        print(f"    ATR: raw={atr_r.metadata.get('atr_raw', 0):.4f}, "
              f"ticks={atr_r.value:.1f}, "
              f"vol_regime={atr_r.metadata.get('vol_regime', '?')}")
    if adx_r:
        print(f"    ADX: {adx_r.value:.1f}, +DI={adx_r.metadata.get('plus_di', 0):.1f}, "
              f"-DI={adx_r.metadata.get('minus_di', 0):.1f}, dir={adx_r.direction}")
    if rvol_r:
        print(f"    RVOL: {rvol_r.value:.2f}, passes={rvol_r.passes}")
    if vwap_r:
        print(f"    VWAP: dev_sd={vwap_r.metadata.get('deviation_sd', 0):.2f}, "
              f"slope={vwap_r.metadata.get('slope', 0):.4f}, "
              f"age={vwap_r.metadata.get('session_age_bars', 0)}")
    if regime_r:
        print(f"    Regime: {regime_r.metadata}")


if __name__ == "__main__":
    main()
