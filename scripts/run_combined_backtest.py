#!/usr/bin/env python3
"""Run both surviving strategies (ORB + VWAP) over the past year on enriched 1s bars.

ORB:  tgt=1.0, exp=60m, spread(z=2.0)
VWAP: entry_sd=1.5, stop_sd=2.0, exp=30m, no filters
"""

import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.features.feature_hub import FeatureHub
from src.filters.spread_monitor import SpreadConfig, SpreadMonitor
from src.strategies.orb_strategy import ORBConfig, ORBStrategy
from src.strategies.vwap_strategy import VWAPConfig, VWAPStrategy


def make_orb():
    hub = FeatureHub()
    cfg = ORBConfig(
        require_hmm_states=[], min_confidence=0.0,
        target_multiplier=1.0,
        volume_multiplier=1.5,
        max_signal_time="10:30",
        expiry_minutes=60,
        require_vwap_alignment=True,
        max_signals_per_day=1,
    )
    return ORBStrategy(cfg, hub)


def make_vwap():
    hub = FeatureHub()
    cfg = VWAPConfig(
        reversion_hmm_states=[], pullback_hmm_states=[], min_confidence=0.3,
        entry_sd_reversion=1.5, stop_sd=2.0,
        pullback_entry_sd=0.5, mode_cooldown_bars=5,
        max_signals_per_day=4, expiry_minutes=30,
    )
    return VWAPStrategy(cfg, hub)


def run_backtest(name, strategies, start, end, spread_monitor=None, bars=None):
    engine = BacktestEngine()
    config = BacktestConfig(
        strategies=strategies,
        start_date=start, end_date=end,
        parquet_dir="data/parquet_1s_enriched",
        prebuilt_bars=bars,
        spread_monitor=spread_monitor,
    )
    if bars is None:
        bars = engine._load_bars(config)
        config = BacktestConfig(
            strategies=strategies,
            start_date=start, end_date=end,
            prebuilt_bars=bars,
            spread_monitor=spread_monitor,
        )

    r = engine.run(config)
    m = r.metrics
    print(f"\n  {name}")
    print(f"  {'=' * 60}")
    print(f"  Trades:       {m.total_trades}")
    print(f"  Win Rate:     {m.win_rate:.1%}")
    print(f"  Net PnL:      ${m.net_pnl:,.2f}")
    print(f"  Sharpe:       {m.sharpe_ratio:.3f}")
    print(f"  Sortino:      {m.sortino_ratio:.3f}")
    print(f"  Profit Factor:{m.profit_factor:.2f}")
    print(f"  Max DD:       {m.max_drawdown_pct:.1f}%")
    print(f"  Avg Win:      ${m.avg_win:.2f}")
    print(f"  Avg Loss:     ${m.avg_loss:.2f}")
    return bars, m


def main():
    start = date(2025, 3, 8)
    end = date(2026, 3, 8)

    print(f"\n{'=' * 70}")
    print(f"COMBINED BACKTEST: ORB + VWAP")
    print(f"Data: {start} to {end} (enriched 1s bars)")
    print(f"{'=' * 70}")

    # Load bars once
    engine = BacktestEngine()
    config = BacktestConfig(
        strategies=[], start_date=start, end_date=end,
        parquet_dir="data/parquet_1s_enriched",
    )
    print("\n  Loading enriched 1s bars...")
    bars = engine._load_bars(config)
    print(f"  Loaded {len(bars)} bars")

    # ORB with spread filter
    print("\n  Running ORB (tgt=1.0, exp=60m, spread z=2.0)...")
    sys.stdout.flush()
    sm = SpreadMonitor(config=SpreadConfig(z_threshold=2.0), persist=False)
    _, orb_m = run_backtest("ORB", [make_orb()], start, end, spread_monitor=sm, bars=bars)

    # VWAP no filters
    print("\n  Running VWAP (entry=1.5, stop=2.0, exp=30m, no filters)...")
    sys.stdout.flush()
    _, vwap_m = run_backtest("VWAP", [make_vwap()], start, end, bars=bars)

    # Combined (both strategies running simultaneously)
    print("\n  Running COMBINED (ORB + VWAP together)...")
    sys.stdout.flush()
    sm2 = SpreadMonitor(config=SpreadConfig(z_threshold=2.0), persist=False)
    _, combined_m = run_backtest("COMBINED (ORB + VWAP)", [make_orb(), make_vwap()], start, end, spread_monitor=sm2, bars=bars)

    # Summary
    print(f"\n\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'':20s} {'Trades':>7s} {'WR':>7s} {'PnL':>12s} {'Sharpe':>8s} {'MaxDD':>7s}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*12} {'-'*8} {'-'*7}")
    for name, m in [("ORB", orb_m), ("VWAP", vwap_m), ("COMBINED", combined_m)]:
        print(f"  {name:20s} {m.total_trades:7d} {m.win_rate:6.1%} ${m.net_pnl:>10,.2f} {m.sharpe_ratio:8.3f} {m.max_drawdown_pct:6.1f}%")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
