#!/usr/bin/env python3
"""Diagnose why CVD divergence strategy generates almost no trades."""

import os
import sys
from datetime import date, datetime, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
from zoneinfo import ZoneInfo

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.core.events import BarEvent
from src.features.feature_hub import FeatureHub
from src.strategies.cvd_divergence_strategy import (
    CVDDivergenceConfig,
    CVDDivergenceStrategy,
    DivergenceDetector,
    SwingDetector,
)

_ET = ZoneInfo("US/Eastern")


def count_pipeline(bars_df, lookback=5, div_thresh=0.15, poc_ticks=6):
    """Count swings, divergences, POC passes for given params."""
    swings = 0
    divs = 0
    poc_pass = 0
    poc_dists = []
    bar_idx = 0
    prev_date = None
    # We need a FeatureHub to get CVD and POC
    hub = FeatureHub()
    sd = SwingDetector(lookback=lookback)
    dd = DivergenceDetector(threshold_pct=div_thresh)

    for row in bars_df.iter_rows(named=True):
        bar_date = row["_et_ts"].date()
        if prev_date is not None and bar_date != prev_date:
            hub = FeatureHub()
            sd = SwingDetector(lookback=lookback)
            dd = DivergenceDetector(threshold_pct=div_thresh)
            bar_idx = 0
        prev_date = bar_date

        ts_ns = row["_timestamp_ns"]
        hub.on_bar(
            timestamp_ns=ts_ns,
            open_=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            volume=int(row["volume"]),
        )
        bar_idx += 1

        cvd_val = hub.cvd.cvd
        new_sw = sd.on_bar(bar_idx, float(row["close"]), cvd_val, ts_ns)
        if new_sw:
            swings += len(new_sw)
            d = dd.check_bearish(sd.recent_highs)
            if d is None:
                d = dd.check_bullish(sd.recent_lows)
            if d is not None:
                divs += 1
                poc_dist = hub.volume_profile.poc_distance_ticks
                poc_dists.append(poc_dist)
                if poc_dist <= poc_ticks:
                    poc_pass += 1

    return swings, divs, poc_pass, poc_dists


def main():
    start = date(2025, 3, 1)
    end = date(2025, 6, 1)

    # Load bars
    engine = BacktestEngine()
    bt_config = BacktestConfig(
        strategies=[],
        start_date=start, end_date=end,
        l1_parquet_dir="data/l1", l1_bar_seconds=5,
    )
    bars_df = engine._load_bars(bt_config)
    bars_df = bars_df.with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC").dt.convert_time_zone("US/Eastern").alias("_et_ts"),
        (pl.col("timestamp").cast(pl.Int64) * 1).alias("_timestamp_ns"),
    )
    bars_df = bars_df.filter(
        (pl.col("_et_ts").dt.time() >= time(9, 30))
        & (pl.col("_et_ts").dt.time() < time(16, 0))
    )

    cfg = CVDDivergenceConfig(require_hmm_states=[], min_confidence=0.3)
    print(f"\n{'=' * 60}")
    print("CVD Divergence Strategy Diagnostic — 3 months L1 (5s bars)")
    print(f"{'=' * 60}")
    print(f"  Total RTH bars: {len(bars_df):,}")

    # Default config
    print(f"\n  --- Default config (lookback={cfg.swing_lookback_bars}, thresh={cfg.divergence_threshold_pct}, poc={cfg.poc_proximity_ticks}) ---")
    sw, dv, poc, poc_dists = count_pipeline(bars_df, cfg.swing_lookback_bars, cfg.divergence_threshold_pct, cfg.poc_proximity_ticks)
    print(f"  Swings: {sw:,}  Divergences: {dv:,}  POC pass: {poc}")

    if poc_dists:
        arr = np.array(poc_dists)
        print(f"\n  POC distance at divergence (ticks):")
        print(f"    Mean:  {np.mean(arr):.1f}   Median: {np.median(arr):.1f}   Min: {np.min(arr):.1f}")
        print(f"    <= 6:  {np.sum(arr <= 6)} ({np.sum(arr <= 6)/len(arr)*100:.1f}%)")
        print(f"    <= 12: {np.sum(arr <= 12)} ({np.sum(arr <= 12)/len(arr)*100:.1f}%)")
        print(f"    <= 20: {np.sum(arr <= 20)} ({np.sum(arr <= 20)/len(arr)*100:.1f}%)")
        print(f"    <= 40: {np.sum(arr <= 40)} ({np.sum(arr <= 40)/len(arr)*100:.1f}%)")

    # Sweep lookback
    print(f"\n  --- Lookback sweep (div_thresh={cfg.divergence_threshold_pct}, poc={cfg.poc_proximity_ticks}) ---")
    for lb in [5, 10, 20, 50, 100, 200]:
        sw, dv, poc, _ = count_pipeline(bars_df, lb, cfg.divergence_threshold_pct, cfg.poc_proximity_ticks)
        window_sec = (2 * lb + 1) * 5
        print(f"  lookback={lb:3d} ({window_sec:5d}s):  swings={sw:>6,}  divs={dv:>4,}  POC pass={poc:>3}")

    # Sweep POC with best lookback candidates
    print(f"\n  --- POC sweep (lookback=50, div_thresh={cfg.divergence_threshold_pct}) ---")
    for poc_ticks in [6, 12, 20, 40, 80, 999]:
        sw, dv, poc, _ = count_pipeline(bars_df, 50, cfg.divergence_threshold_pct, poc_ticks)
        print(f"  poc_ticks={poc_ticks:3d}:  swings={sw:>6,}  divs={dv:>4,}  POC pass={poc:>4}")

    # Sweep div threshold
    print(f"\n  --- Divergence threshold sweep (lookback=50, poc=40) ---")
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.30]:
        sw, dv, poc, _ = count_pipeline(bars_df, 50, thresh, 40)
        print(f"  thresh={thresh:.2f}:  swings={sw:>6,}  divs={dv:>4,}  POC pass={poc:>4}")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
