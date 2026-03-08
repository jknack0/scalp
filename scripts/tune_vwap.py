#!/usr/bin/env python3
"""Tune VWAP strategy parameters with VPIN filter on full year L1 data.

Sweeps key parameters and reports results in a comparison table.
"""

import os
import sys
from datetime import date
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.metrics import BacktestMetrics
from src.features.feature_hub import FeatureHub
from src.filters.vpin_monitor import VPINConfig, VPINMonitor
from src.strategies.vwap_strategy import VWAPConfig, VWAPStrategy


def run_one(start, end, entry_sd, stop_sd, pullback_sd, mode_cooldown, max_signals, expiry_min):
    hub = FeatureHub()
    cfg = VWAPConfig(
        reversion_hmm_states=[],
        pullback_hmm_states=[],
        min_confidence=0.3,
        entry_sd_reversion=entry_sd,
        stop_sd=stop_sd,
        pullback_entry_sd=pullback_sd,
        mode_cooldown_bars=mode_cooldown,
        max_signals_per_day=max_signals,
        expiry_minutes=expiry_min,
    )
    strat = VWAPStrategy(cfg, hub)
    vm = VPINMonitor(config=VPINConfig(), persist=False)

    config = BacktestConfig(
        strategies=[strat],
        start_date=start,
        end_date=end,
        l1_parquet_dir="data/l1",
        l1_bar_seconds=5,
        vpin_monitor=vm,
    )
    engine = BacktestEngine()
    return engine.run(config)


def main():
    start = date(2025, 1, 1)
    end = date(2025, 12, 31)

    # First: analyze current trade exit reasons
    print("\n  Analyzing current VWAP trades (VPIN-only, default config)...\n")
    hub = FeatureHub()
    strat = VWAPStrategy(VWAPConfig(reversion_hmm_states=[], pullback_hmm_states=[], min_confidence=0.3), hub)
    vm = VPINMonitor(config=VPINConfig(), persist=False)
    config = BacktestConfig(
        strategies=[strat], start_date=start, end_date=end,
        l1_parquet_dir="data/l1", l1_bar_seconds=5, vpin_monitor=vm,
    )
    r = BacktestEngine().run(config)

    # Trade analysis
    trades = r.trades
    rev_trades = [t for t in trades if t.metadata.get("mode") == "REVERSION"]
    pb_trades = [t for t in trades if t.metadata.get("mode") == "PULLBACK"]

    print(f"  Total: {len(trades)} trades")
    print(f"  REVERSION: {len(rev_trades)}  PULLBACK: {len(pb_trades)}")

    for label, subset in [("REVERSION", rev_trades), ("PULLBACK", pb_trades), ("ALL", trades)]:
        if not subset:
            continue
        wins = [t for t in subset if t.net_pnl > 0]
        pnl = sum(t.net_pnl for t in subset)
        exits = {}
        for t in subset:
            exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
        avg_bars = sum(t.bars_held for t in subset) / len(subset)
        avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0
        losses = [t for t in subset if t.net_pnl <= 0]
        avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0

        print(f"\n  --- {label} ({len(subset)} trades) ---")
        print(f"  WR: {len(wins)/len(subset):.1%}  PnL: ${pnl:,.2f}  Avg win: ${avg_win:,.2f}  Avg loss: ${avg_loss:,.2f}")
        print(f"  Avg bars held: {avg_bars:.0f}  Exits: {exits}")

        # Dev SD at entry distribution
        dev_sds = [abs(t.metadata.get("deviation_sd", 0)) for t in subset]
        if dev_sds:
            import numpy as np
            arr = np.array(dev_sds)
            print(f"  Entry |dev_sd|:  mean={np.mean(arr):.2f}  P50={np.median(arr):.2f}  P90={np.percentile(arr, 90):.2f}")

    # Parameter sweep
    print(f"\n{'=' * 90}")
    print("VWAP Parameter Sweep (with VPIN filter, full year)")
    print(f"{'=' * 90}")

    results = []
    sweep = list(product(
        [1.0, 1.5, 2.0],          # entry_sd
        [2.0, 2.5, 3.0, 3.5],     # stop_sd
        [0.5],                      # pullback_sd (keep fixed)
        [5],                        # mode_cooldown (keep fixed)
        [4],                        # max_signals (keep fixed)
        [30, 60],                   # expiry_minutes
    ))

    print(f"\n  Running {len(sweep)} configurations...\n")

    for i, (entry_sd, stop_sd, pb_sd, cooldown, max_sig, expiry) in enumerate(sweep):
        r = run_one(start, end, entry_sd, stop_sd, pb_sd, cooldown, max_sig, expiry)
        m = r.metrics
        results.append({
            "entry_sd": entry_sd,
            "stop_sd": stop_sd,
            "expiry": expiry,
            "trades": m.total_trades,
            "wr": m.win_rate,
            "pnl": m.net_pnl,
            "sharpe": m.sharpe_ratio,
            "sortino": m.sortino_ratio,
            "pf": m.profit_factor,
            "dd": m.max_drawdown_pct,
            "avg_win": m.avg_win,
            "avg_loss": m.avg_loss,
        })
        status = "+" if m.sharpe_ratio > 0 else "-"
        print(f"  [{i+1:2d}/{len(sweep)}] {status} entry={entry_sd:.1f} stop={stop_sd:.1f} exp={expiry:2d}m  "
              f"trades={m.total_trades:3d} WR={m.win_rate:.1%} Sharpe={m.sharpe_ratio:>7.3f} PnL=${m.net_pnl:>9,.2f} DD={m.max_drawdown_pct:.1f}%")

    # Sort by Sharpe and show top 10
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"\n{'=' * 90}")
    print("TOP 10 CONFIGURATIONS (by Sharpe)")
    print(f"{'=' * 90}")
    print(f"  {'entry':>5s} {'stop':>5s} {'exp':>4s} {'trades':>6s} {'WR':>6s} {'Sharpe':>8s} {'Sortino':>8s} {'PF':>6s} {'PnL':>10s} {'DD%':>6s} {'AvgW':>8s} {'AvgL':>8s}")
    print(f"  {'-' * 85}")
    for r in results[:10]:
        print(f"  {r['entry_sd']:>5.1f} {r['stop_sd']:>5.1f} {r['expiry']:>4d} {r['trades']:>6d} {r['wr']:>5.1%} "
              f"{r['sharpe']:>8.3f} {r['sortino']:>8.3f} {r['pf']:>6.2f} "
              f"{'${:,.2f}'.format(r['pnl']):>10s} {r['dd']:>5.1f}% "
              f"{'${:,.0f}'.format(r['avg_win']):>8s} {'${:,.0f}'.format(r['avg_loss']):>8s}")

    print(f"\n{'=' * 90}\n")


if __name__ == "__main__":
    main()
