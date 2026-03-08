#!/usr/bin/env python3
"""Diagnose why VWAP strategy generates 0 trades on real data.

Counts how many bars hit each gate in the signal pipeline.
"""

import os
import sys
from datetime import date, datetime, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zoneinfo import ZoneInfo

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.core.events import BarEvent
from src.features.feature_hub import FeatureHub
from src.strategies.base import Direction
from src.strategies.vwap_strategy import VWAPConfig, VWAPMode, VWAPStrategy

_ET = ZoneInfo("US/Eastern")


def main():
    hub = FeatureHub()
    cfg = VWAPConfig(reversion_hmm_states=[], pullback_hmm_states=[], require_reversal_candle=False)
    strat = VWAPStrategy(cfg, hub)

    # Load bars using engine's loader
    engine = BacktestEngine()
    bt_config = BacktestConfig(
        strategies=[strat],
        start_date=date(2025, 3, 1),
        end_date=date(2025, 6, 1),
        l1_parquet_dir="data/l1",
        l1_bar_seconds=5,
    )
    bars_df = engine._load_bars(bt_config)

    import polars as pl

    bars_df = bars_df.with_columns(
        pl.col("timestamp")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("US/Eastern")
        .alias("_et_ts"),
        (pl.col("timestamp").cast(pl.Int64) * 1).alias("_timestamp_ns"),
    )
    bars_df = bars_df.filter(
        (pl.col("_et_ts").dt.time() >= time(9, 30))
        & (pl.col("_et_ts").dt.time() < time(16, 0))
    )

    # Counters
    total_bars = 0
    in_session = 0
    past_cooldown = 0
    past_session_age = 0
    in_reversion = 0
    in_pullback = 0
    in_neutral = 0
    rev_past_sd = 0
    rev_past_reversal = 0
    pb_past_devsd = 0
    pb_past_candle = 0
    mode_changes = 0
    max_dev_sd = 0.0
    dev_sd_samples = []
    slope_samples = []
    sd_samples = []

    prev_date = None

    for row in bars_df.iter_rows(named=True):
        bar_time = row["_et_ts"]
        bar_date = bar_time.date()

        if prev_date is not None and bar_date != prev_date:
            strat.reset()
        prev_date = bar_date

        bar_event = BarEvent(
            symbol="MESM6",
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
            bar_type="5s",
            timestamp_ns=row["_timestamp_ns"],
            avg_bid_size=float(row.get("avg_bid_size", 0.0) or 0.0),
            avg_ask_size=float(row.get("avg_ask_size", 0.0) or 0.0),
            aggressive_buy_vol=float(row.get("aggressive_buy_vol", 0.0) or 0.0),
            aggressive_sell_vol=float(row.get("aggressive_sell_vol", 0.0) or 0.0),
        )

        old_mode = strat._mode
        strat._base_on_bar(bar_event)
        strat._update_mode()
        if strat._mode != old_mode:
            mode_changes += 1
        strat._bars_since_mode_change += 1
        strat._prev_bar_close = bar_event.close

        total_bars += 1

        # Check session
        bar_dt = strat._bar_et_datetime(bar_event)
        if not strat.is_active_session(bar_dt):
            continue
        in_session += 1

        # Check cooldown
        if strat._bars_since_mode_change <= cfg.mode_cooldown_bars:
            continue
        past_cooldown += 1

        # Check session age
        if strat._session_age_minutes(bar_event) < cfg.min_session_age_minutes:
            continue
        past_session_age += 1

        # Track mode distribution
        if strat._mode == VWAPMode.REVERSION:
            in_reversion += 1
        elif strat._mode == VWAPMode.PULLBACK:
            in_pullback += 1
        else:
            in_neutral += 1

        # Track VWAP stats
        vwap_calc = hub.vwap
        dev_sd = vwap_calc.deviation_sd
        slope = vwap_calc.slope_20bar
        sd = vwap_calc._sd

        dev_sd_samples.append(abs(dev_sd))
        slope_samples.append(abs(slope))
        sd_samples.append(sd)

        if abs(dev_sd) > max_dev_sd:
            max_dev_sd = abs(dev_sd)

        # Reversion check
        if strat._mode == VWAPMode.REVERSION:
            if abs(dev_sd) >= cfg.entry_sd_reversion:
                rev_past_sd += 1
                # Check reversal candle
                direction = Direction.LONG if dev_sd < 0 else Direction.SHORT
                if not cfg.require_reversal_candle or strat._is_reversal_candle(bar_event, direction):
                    rev_past_reversal += 1

        # Pullback check
        if strat._mode == VWAPMode.PULLBACK:
            if abs(dev_sd) <= cfg.pullback_entry_sd:
                pb_past_devsd += 1
                # Check continuation candle
                direction = Direction.LONG if slope_samples[-1] > 0 else Direction.SHORT
                prev = strat._prev_bar_close
                if prev > 0:
                    if direction == Direction.LONG and bar_event.close > prev:
                        pb_past_candle += 1
                    elif direction == Direction.SHORT and bar_event.close < prev:
                        pb_past_candle += 1

    # --- PASS 2: Run actual on_bar() flow and count signals ---
    print("\n  Running pass 2 (actual on_bar flow)...")
    hub2 = FeatureHub()
    strat2 = VWAPStrategy(
        VWAPConfig(reversion_hmm_states=[], pullback_hmm_states=[], require_reversal_candle=False),
        hub2,
    )
    prev_date2 = None
    actual_signals = 0
    first_signal_bar = None

    for row in bars_df.iter_rows(named=True):
        bar_time = row["_et_ts"]
        bar_date = bar_time.date()

        if prev_date2 is not None and bar_date != prev_date2:
            strat2.reset()
        prev_date2 = bar_date

        bar_event = BarEvent(
            symbol="MESM6",
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
            bar_type="5s",
            timestamp_ns=row["_timestamp_ns"],
            avg_bid_size=float(row.get("avg_bid_size", 0.0) or 0.0),
            avg_ask_size=float(row.get("avg_ask_size", 0.0) or 0.0),
            aggressive_buy_vol=float(row.get("aggressive_buy_vol", 0.0) or 0.0),
            aggressive_sell_vol=float(row.get("aggressive_sell_vol", 0.0) or 0.0),
        )

        prev_count = len(strat2._signals_generated)
        strat2.on_bar(bar_event)
        new_count = len(strat2._signals_generated)

        if new_count > prev_count:
            actual_signals += new_count - prev_count
            if first_signal_bar is None:
                sig = strat2._signals_generated[-1]
                first_signal_bar = f"{bar_time} close={bar_event.close} dir={sig.direction} conf={sig.confidence:.3f}"

    # Also trace one day in detail
    hub3 = FeatureHub()
    strat3 = VWAPStrategy(
        VWAPConfig(reversion_hmm_states=[], pullback_hmm_states=[], require_reversal_candle=False),
        hub3,
    )
    # Pick first full day
    first_day = bars_df.select(pl.col("_et_ts").dt.date().alias("d")).unique().sort("d").row(0)[0]
    day_bars = bars_df.filter(pl.col("_et_ts").dt.date() == first_day)
    trace_lines = []
    for row in day_bars.iter_rows(named=True):
        bar_event = BarEvent(
            symbol="MESM6",
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
            bar_type="5s",
            timestamp_ns=row["_timestamp_ns"],
            avg_bid_size=float(row.get("avg_bid_size", 0.0) or 0.0),
            avg_ask_size=float(row.get("avg_ask_size", 0.0) or 0.0),
            aggressive_buy_vol=float(row.get("aggressive_buy_vol", 0.0) or 0.0),
            aggressive_sell_vol=float(row.get("aggressive_sell_vol", 0.0) or 0.0),
        )
        strat3.on_bar(bar_event)
        bar_dt = strat3._bar_et_datetime(bar_event)
        vwap_calc = hub3.vwap
        dev_sd = vwap_calc.deviation_sd
        mode = strat3._mode.value
        cooldown = strat3._bars_since_mode_change
        active = strat3.is_active_session(bar_dt)
        session_age = strat3._session_age_minutes(bar_event)
        can_sig = strat3.can_generate_signal(bar_dt)

        if abs(dev_sd) >= 1.5:
            trace_lines.append(
                f"    {bar_dt.strftime('%H:%M:%S')} mode={mode:10s} cd={cooldown:3d} "
                f"|dev_sd|={abs(dev_sd):.2f} active={active} age={session_age:.0f}m can_sig={can_sig}"
            )

    import numpy as np

    print(f"\n{'=' * 60}")
    print("VWAP Strategy Diagnostic — 3 months L1 data (5s bars)")
    print(f"{'=' * 60}")
    print(f"\n  Total bars (RTH):       {total_bars:,}")
    print(f"  In active session:      {in_session:,}")
    print(f"  Past mode cooldown:     {past_cooldown:,}")
    print(f"  Past session age (15m): {past_session_age:,}")
    print(f"\n  Mode distribution (past all gates):")
    print(f"    REVERSION:  {in_reversion:,} ({in_reversion/max(past_session_age,1)*100:.1f}%)")
    print(f"    PULLBACK:   {in_pullback:,} ({in_pullback/max(past_session_age,1)*100:.1f}%)")
    print(f"    NEUTRAL:    {in_neutral:,} ({in_neutral/max(past_session_age,1)*100:.1f}%)")
    print(f"    Mode changes: {mode_changes:,}")

    print(f"\n  VWAP deviation_sd stats:")
    if dev_sd_samples:
        arr = np.array(dev_sd_samples)
        print(f"    Mean |dev_sd|:  {np.mean(arr):.4f}")
        print(f"    Max |dev_sd|:   {np.max(arr):.4f}")
        print(f"    P95 |dev_sd|:   {np.percentile(arr, 95):.4f}")
        print(f"    P99 |dev_sd|:   {np.percentile(arr, 99):.4f}")
        print(f"    Bars >= 2.0 SD: {np.sum(arr >= 2.0):,} ({np.sum(arr >= 2.0)/len(arr)*100:.2f}%)")
        print(f"    Bars >= 1.5 SD: {np.sum(arr >= 1.5):,} ({np.sum(arr >= 1.5)/len(arr)*100:.2f}%)")
        print(f"    Bars >= 1.0 SD: {np.sum(arr >= 1.0):,} ({np.sum(arr >= 1.0)/len(arr)*100:.2f}%)")

    print(f"\n  VWAP slope stats:")
    if slope_samples:
        arr = np.array(slope_samples)
        print(f"    Mean |slope|:   {np.mean(arr):.6f}")
        print(f"    P50 |slope|:    {np.percentile(arr, 50):.6f}")
        print(f"    Bars < 0.002 (REVERSION): {np.sum(arr < 0.002):,} ({np.sum(arr < 0.002)/len(arr)*100:.1f}%)")
        print(f"    Bars > 0.005 (PULLBACK):  {np.sum(arr > 0.005):,} ({np.sum(arr > 0.005)/len(arr)*100:.1f}%)")
        print(f"    Bars 0.002-0.005 (NEUTRAL): {np.sum((arr >= 0.002) & (arr <= 0.005)):,}")

    print(f"\n  VWAP SD (raw dollar value):")
    if sd_samples:
        arr = np.array(sd_samples)
        print(f"    Mean SD:  ${np.mean(arr):.4f}")
        print(f"    P50 SD:   ${np.percentile(arr, 50):.4f}")
        print(f"    P95 SD:   ${np.percentile(arr, 95):.4f}")

    print(f"\n  Reversion signal pipeline:")
    print(f"    In REVERSION mode:      {in_reversion:,}")
    print(f"    Past |dev_sd| >= 2.0:   {rev_past_sd:,}")
    print(f"    Past reversal candle:   {rev_past_reversal:,}")

    print(f"\n  Pullback signal pipeline:")
    print(f"    In PULLBACK mode:       {in_pullback:,}")
    print(f"    Past |dev_sd| <= 0.25:  {pb_past_devsd:,}")
    print(f"    Past continuation:      {pb_past_candle:,}")

    print(f"\n  PASS 2 — Actual on_bar() flow:")
    print(f"    Signals generated: {actual_signals}")
    if first_signal_bar:
        print(f"    First signal:      {first_signal_bar}")

    print(f"\n  Day trace ({first_day}) — bars with |dev_sd| >= 1.5:")
    for line in trace_lines[:30]:
        print(line)
    if len(trace_lines) > 30:
        print(f"    ... ({len(trace_lines)} total)")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
