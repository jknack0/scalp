#!/usr/bin/env python3
"""Diagnose why spread filter hurts VWAP strategy.

Compares trades blocked by spread filter vs kept — are blocked trades actually winners?
"""

import os
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.metrics import BacktestMetrics, MetricsCalculator
from src.features.feature_hub import FeatureHub
from src.filters.spread_monitor import SpreadConfig, SpreadMonitor
from src.filters.vpin_monitor import VPINConfig, VPINMonitor
from src.strategies.vwap_strategy import VWAPConfig, VWAPStrategy


def main():
    start = date(2025, 1, 1)
    end = date(2025, 12, 31)

    # Run WITHOUT spread filter to get all trades
    hub1 = FeatureHub()
    strat1 = VWAPStrategy(VWAPConfig(reversion_hmm_states=[], pullback_hmm_states=[], min_confidence=0.3), hub1)
    config1 = BacktestConfig(
        strategies=[strat1], start_date=start, end_date=end,
        l1_parquet_dir="data/l1", l1_bar_seconds=5,
    )
    r_no_filter = BacktestEngine().run(config1)

    # Run WITH spread filter
    hub2 = FeatureHub()
    strat2 = VWAPStrategy(VWAPConfig(reversion_hmm_states=[], pullback_hmm_states=[], min_confidence=0.3), hub2)
    sm = SpreadMonitor(config=SpreadConfig(), persist=False)
    config2 = BacktestConfig(
        strategies=[strat2], start_date=start, end_date=end,
        l1_parquet_dir="data/l1", l1_bar_seconds=5,
        spread_monitor=sm,
    )
    r_spread = BacktestEngine().run(config2)

    # Compare trade lists
    trades_all = r_no_filter.trades
    trades_kept = r_spread.trades

    # Match trades by entry time to find which were blocked
    kept_times = {t.entry_time for t in trades_kept}
    blocked = [t for t in trades_all if t.entry_time not in kept_times]
    kept = [t for t in trades_all if t.entry_time in kept_times]

    print(f"\n{'=' * 60}")
    print("VWAP + Spread Filter Diagnosis")
    print(f"{'=' * 60}")
    print(f"  Total trades (no filter):  {len(trades_all)}")
    print(f"  Kept by spread filter:     {len(kept)}")
    print(f"  Blocked by spread filter:  {len(blocked)}")

    if blocked:
        blocked_wins = [t for t in blocked if t.net_pnl > 0]
        blocked_losses = [t for t in blocked if t.net_pnl <= 0]
        blocked_pnl = sum(t.net_pnl for t in blocked)
        blocked_avg = blocked_pnl / len(blocked)
        print(f"\n  --- BLOCKED TRADES ---")
        print(f"  Winners blocked:  {len(blocked_wins)}")
        print(f"  Losers blocked:   {len(blocked_losses)}")
        print(f"  Blocked WR:       {len(blocked_wins)/len(blocked):.1%}")
        print(f"  Blocked total PnL: ${blocked_pnl:,.2f}")
        print(f"  Blocked avg PnL:   ${blocked_avg:,.2f}")

    if kept:
        kept_wins = [t for t in kept if t.net_pnl > 0]
        kept_pnl = sum(t.net_pnl for t in kept)
        kept_avg = kept_pnl / len(kept)
        print(f"\n  --- KEPT TRADES ---")
        print(f"  Winners kept:     {len(kept_wins)}")
        print(f"  Losers kept:      {len(kept) - len(kept_wins)}")
        print(f"  Kept WR:          {len(kept_wins)/len(kept):.1%}")
        print(f"  Kept total PnL:   ${kept_pnl:,.2f}")
        print(f"  Kept avg PnL:     ${kept_avg:,.2f}")

    # Show individual blocked trades
    if blocked:
        print(f"\n  --- BLOCKED TRADE DETAILS (top 20 by |PnL|) ---")
        blocked_sorted = sorted(blocked, key=lambda t: abs(t.net_pnl), reverse=True)
        for t in blocked_sorted[:20]:
            result = "WIN " if t.net_pnl > 0 else "LOSS"
            print(f"    {t.entry_time}  {t.direction:5s}  {result}  PnL=${t.net_pnl:>8.2f}  entry={t.entry_price:.2f}")

    # Now test VPIN-only for comparison
    print(f"\n{'=' * 60}")
    print("VWAP + VPIN-Only (recommended config)")
    print(f"{'=' * 60}")

    hub3 = FeatureHub()
    strat3 = VWAPStrategy(VWAPConfig(reversion_hmm_states=[], pullback_hmm_states=[], min_confidence=0.3), hub3)
    vm = VPINMonitor(config=VPINConfig(), persist=False)
    config3 = BacktestConfig(
        strategies=[strat3], start_date=start, end_date=end,
        l1_parquet_dir="data/l1", l1_bar_seconds=5,
        vpin_monitor=vm,
    )
    r_vpin = BacktestEngine().run(config3)
    m = r_vpin.metrics
    print(f"  Trades:       {m.total_trades}")
    print(f"  Win rate:     {m.win_rate:.1%}")
    print(f"  Net P&L:      ${m.net_pnl:,.2f}")
    print(f"  Sharpe:       {m.sharpe_ratio:.3f}")
    print(f"  Sortino:      {m.sortino_ratio:.3f}")
    print(f"  Profit factor: {m.profit_factor:.2f}")
    print(f"  Max drawdown: {m.max_drawdown_pct:.2f}%")
    print(f"  Avg win:      ${m.avg_win:,.2f}")
    print(f"  Avg loss:     ${m.avg_loss:,.2f}")

    # VPIN regime stats
    regime, vpin_val = vm.get_regime()
    print(f"\n  VPIN final:   {regime} (vpin={vpin_val:.3f})")
    print(f"  Buckets:      {vm.bucket_count}")

    # Blocked by VPIN
    vpin_trades = r_vpin.trades
    vpin_times = {t.entry_time for t in vpin_trades}
    vpin_blocked = [t for t in trades_all if t.entry_time not in vpin_times]
    if vpin_blocked:
        vpin_blocked_wins = [t for t in vpin_blocked if t.net_pnl > 0]
        vpin_blocked_pnl = sum(t.net_pnl for t in vpin_blocked)
        print(f"\n  --- VPIN-BLOCKED TRADES ---")
        print(f"  Blocked:     {len(vpin_blocked)}")
        print(f"  Blocked WR:  {len(vpin_blocked_wins)/len(vpin_blocked):.1%}")
        print(f"  Blocked PnL: ${vpin_blocked_pnl:,.2f}")
        print(f"  (VPIN correctly blocks ${-vpin_blocked_pnl:,.2f} of losses)" if vpin_blocked_pnl < 0 else f"  (VPIN incorrectly blocks ${vpin_blocked_pnl:,.2f} of profits)")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
