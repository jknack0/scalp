#!/usr/bin/env python3
"""Tune Vol Regime strategy using VPIN regime instead of ATR percentile."""

import os
import sys
from datetime import date
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.disable(logging.CRITICAL)
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.features.feature_hub import FeatureHub
from src.filters.vpin_monitor import VPINConfig, VPINMonitor
from src.strategies.vol_regime_strategy import VolRegimeConfig, VolRegimeStrategy


def run_one(start, end, hi_tgt, hi_stop, lo_tgt, lo_stop, lo_entry_sd, pb_bars, max_sig):
    hub = FeatureHub()
    vm = VPINMonitor(config=VPINConfig(), persist=False)
    cfg = VolRegimeConfig(
        high_vol_hmm_states=[], low_vol_hmm_states=[], min_confidence=0.0,
        high_vol_target_ticks=hi_tgt, high_vol_stop_ticks=hi_stop,
        low_vol_target_ticks=lo_tgt, low_vol_stop_ticks=lo_stop,
        low_vol_entry_sd=lo_entry_sd, pullback_bars=pb_bars,
        max_signals_per_day=max_sig,
        use_vpin_regime=True,
    )
    strat = VolRegimeStrategy(cfg, hub, vpin_monitor=vm)
    config = BacktestConfig(
        strategies=[strat], start_date=start, end_date=end,
        l1_parquet_dir="data/l1", l1_bar_seconds=5,
        vpin_monitor=vm,
    )
    return BacktestEngine().run(config)


def main():
    # Use 3 months for fast iteration
    start = date(2025, 3, 1)
    end = date(2025, 6, 1)

    # Diagnostic
    print("  Running diagnostic (default config)...")
    r = run_one(start, end, 8, 2, 2, 30, 1.5, 3, 4)
    trades = r.trades
    hi_trades = [t for t in trades if t.metadata.get("regime") == "HIGH_VOL"]
    lo_trades = [t for t in trades if t.metadata.get("regime") == "LOW_VOL"]

    for label, subset in [("HIGH_VOL", hi_trades), ("LOW_VOL", lo_trades), ("ALL", trades)]:
        if not subset:
            print(f"  {label}: 0 trades")
            continue
        wins = [t for t in subset if t.net_pnl > 0]
        pnl = sum(t.net_pnl for t in subset)
        exits = {}
        for t in subset:
            exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
        avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0
        losses = [t for t in subset if t.net_pnl <= 0]
        avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0
        print(f"  {label} ({len(subset)}): WR={len(wins)/len(subset):.1%} PnL=${pnl:,.2f} AvgW=${avg_win:,.2f} AvgL=${avg_loss:,.2f} Exits={exits}")
    sys.stdout.flush()

    # HIGH_VOL sweep (disable LOW_VOL with entry_sd=99)
    print(f"\n{'=' * 80}")
    print("HIGH_VOL only sweep (3 months)")
    print(f"{'=' * 80}\n")

    results = []
    sweep = list(product([8, 12, 16], [2, 4, 6], [2, 3, 4]))
    for i, (hi_tgt, hi_stop, pb) in enumerate(sweep):
        r = run_one(start, end, hi_tgt, hi_stop, 2, 30, 99.0, pb, 4)
        m = r.metrics
        results.append({"hi_tgt": hi_tgt, "hi_stop": hi_stop, "pb": pb,
            "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
            "sharpe": m.sharpe_ratio, "pf": m.profit_factor, "dd": m.max_drawdown_pct})
        s = "+" if m.sharpe_ratio > 0 else "-"
        print(f"  [{i+1:2d}/{len(sweep)}] {s} tgt={hi_tgt:2d} stop={hi_stop} pb={pb} trades={m.total_trades:3d} WR={m.win_rate:.1%} Sharpe={m.sharpe_ratio:>7.3f} PnL=${m.net_pnl:>9,.2f}")
        sys.stdout.flush()

    # LOW_VOL sweep (disable HIGH_VOL with pullback_bars=999)
    print(f"\n{'=' * 80}")
    print("LOW_VOL only sweep (3 months)")
    print(f"{'=' * 80}\n")

    lo_results = []
    lo_sweep = list(product([4, 8, 12], [4, 8, 12], [1.5, 2.0, 2.5]))
    for i, (lo_tgt, lo_stop, lo_sd) in enumerate(lo_sweep):
        r = run_one(start, end, 8, 2, lo_tgt, lo_stop, lo_sd, 999, 4)
        m = r.metrics
        lo_results.append({"lo_tgt": lo_tgt, "lo_stop": lo_stop, "lo_sd": lo_sd,
            "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
            "sharpe": m.sharpe_ratio, "pf": m.profit_factor, "dd": m.max_drawdown_pct})
        s = "+" if m.sharpe_ratio > 0 else "-"
        print(f"  [{i+1:2d}/{len(lo_sweep)}] {s} tgt={lo_tgt:2d} stop={lo_stop:2d} sd={lo_sd:.1f} trades={m.total_trades:3d} WR={m.win_rate:.1%} Sharpe={m.sharpe_ratio:>7.3f} PnL=${m.net_pnl:>9,.2f}")
        sys.stdout.flush()

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    lo_results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n{'=' * 80}")
    print("TOP 5 HIGH_VOL:")
    for r in results[:5]:
        print(f"  tgt={r['hi_tgt']:2d} stop={r['hi_stop']} pb={r['pb']} trades={r['trades']:3d} WR={r['wr']:.1%} Sharpe={r['sharpe']:.3f} PnL=${r['pnl']:,.2f} DD={r['dd']:.1f}%")
    print(f"\nTOP 5 LOW_VOL:")
    for r in lo_results[:5]:
        print(f"  tgt={r['lo_tgt']:2d} stop={r['lo_stop']:2d} sd={r['lo_sd']:.1f} trades={r['trades']:3d} WR={r['wr']:.1%} Sharpe={r['sharpe']:.3f} PnL=${r['pnl']:,.2f} DD={r['dd']:.1f}%")
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
