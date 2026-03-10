#!/usr/bin/env python3
"""Run both strategies (ORB + VWAP) on 10 years of 5s RTH bars.

ORB:  tgt=1.0, exp=60m (no spread filter — no bid/ask in raw 1s bars)
VWAP: entry_sd=1.5, stop_sd=2.0, exp=30m, no filters
"""

import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.disable(logging.CRITICAL)
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy


def make_orb():
    return ORBStrategy(config={
        "strategy": {"strategy_id": "orb", "max_signals_per_day": 1},
        "orb": {"target_multiplier": 1.0, "volume_multiplier": 1.5,
                "max_signal_time": "10:30", "expiry_minutes": 60,
                "require_vwap_alignment": True},
        "filter": {"min_score": 0, "require_direction_agreement": False, "signals": {}},
        "exit": {"time_stop_minutes": 60},
    })


def make_vwap():
    return VWAPReversionStrategy(config={
        "strategy": {"strategy_id": "vwap_reversion", "max_signals_per_day": 4},
        "vwap": {"entry_sd_reversion": 1.5, "stop_sd": 2.0,
                 "pullback_entry_sd": 0.5, "mode_cooldown_bars": 5,
                 "expiry_minutes": 30},
        "filter": {"min_score": 0, "require_direction_agreement": False, "signals": {}},
        "exit": {"target": "vwap", "stop_ticks": 8, "time_stop_minutes": 30},
    })


def run_one(name, strategies, start, end, bars):
    engine = BacktestEngine()
    config = BacktestConfig(
        strategies=strategies,
        start_date=start, end_date=end,
        prebuilt_bars=bars,
    )
    r = engine.run(config)
    m = r.metrics
    print(f"\n  {name}")
    print(f"  {'=' * 60}")
    print(f"  Trades:        {m.total_trades}")
    print(f"  Win Rate:      {m.win_rate:.1%}")
    print(f"  Net PnL:       ${m.net_pnl:,.2f}")
    print(f"  Sharpe:        {m.sharpe_ratio:.3f}")
    print(f"  Sortino:       {m.sortino_ratio:.3f}")
    print(f"  Profit Factor: {m.profit_factor:.2f}")
    print(f"  Max DD:        {m.max_drawdown_pct:.1f}%")
    print(f"  Avg Win:       ${m.avg_win:.2f}")
    print(f"  Avg Loss:      ${m.avg_loss:.2f}")
    sys.stdout.flush()
    return m


def main():
    start = date(2016, 1, 1)
    end = date(2025, 12, 31)

    print(f"\n{'=' * 70}")
    print(f"10-YEAR BACKTEST: ORB + VWAP on 5s RTH bars")
    print(f"Data: {start} to {end}")
    print(f"{'=' * 70}")

    # Load bars
    engine = BacktestEngine()
    config = BacktestConfig(
        strategies=[], start_date=start, end_date=end,
        parquet_dir="data/parquet_5s_rth",
    )
    print("\n  Loading 5s RTH bars...")
    sys.stdout.flush()
    bars = engine._load_bars(config)
    print(f"  Loaded {len(bars):,} bars")
    sys.stdout.flush()

    # ORB
    print("\n  Running ORB (tgt=1.0, exp=60m, no spread filter)...")
    sys.stdout.flush()
    orb_m = run_one("ORB", [make_orb()], start, end, bars)

    # VWAP
    print("\n  Running VWAP (entry=1.5, stop=2.0, exp=30m)...")
    sys.stdout.flush()
    vwap_m = run_one("VWAP", [make_vwap()], start, end, bars)

    # Combined
    print("\n  Running COMBINED...")
    sys.stdout.flush()
    combined_m = run_one("COMBINED (ORB + VWAP)", [make_orb(), make_vwap()], start, end, bars)

    # Summary
    print(f"\n\n{'=' * 70}")
    print(f"10-YEAR SUMMARY (2016-2025)")
    print(f"{'=' * 70}")
    print(f"  {'':20s} {'Trades':>7s} {'WR':>7s} {'PnL':>12s} {'Sharpe':>8s} {'Sortino':>8s} {'PF':>6s} {'MaxDD':>7s}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*12} {'-'*8} {'-'*8} {'-'*6} {'-'*7}")
    for name, m in [("ORB", orb_m), ("VWAP", vwap_m), ("COMBINED", combined_m)]:
        print(f"  {name:20s} {m.total_trades:7d} {m.win_rate:6.1%} ${m.net_pnl:>10,.2f} {m.sharpe_ratio:8.3f} {m.sortino_ratio:8.3f} {m.profit_factor:6.2f} {m.max_drawdown_pct:6.1f}%")
    print(f"\n  Per-year averages:")
    for name, m in [("ORB", orb_m), ("VWAP", vwap_m), ("COMBINED", combined_m)]:
        print(f"  {name:20s} {m.total_trades/10:.0f} trades/yr  ${m.net_pnl/10:,.2f}/yr")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
