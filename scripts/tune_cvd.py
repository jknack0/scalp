#!/usr/bin/env python3
"""Tune CVD divergence strategy parameters."""

import os
import sys
from datetime import date
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.features.feature_hub import FeatureHub
from src.filters.vpin_monitor import VPINConfig, VPINMonitor
from src.strategies.cvd_divergence_strategy import CVDDivergenceConfig, CVDDivergenceStrategy


def run_one(start, end, lookback, target_ticks, stop_buffer, max_hold, div_thresh, poc_ticks, use_vpin=False):
    hub = FeatureHub()
    cfg = CVDDivergenceConfig(
        require_hmm_states=[],
        min_confidence=0.0,
        swing_lookback_bars=lookback,
        target_ticks=target_ticks,
        stop_buffer_ticks=stop_buffer,
        max_hold_bars=max_hold,
        divergence_threshold_pct=div_thresh,
        poc_proximity_ticks=poc_ticks,
        max_signals_per_day=6,
    )
    strat = CVDDivergenceStrategy(cfg, hub)
    vm = VPINMonitor(config=VPINConfig(), persist=False) if use_vpin else None

    config = BacktestConfig(
        strategies=[strat],
        start_date=start, end_date=end,
        l1_parquet_dir="data/l1", l1_bar_seconds=5,
        vpin_monitor=vm,
    )
    return BacktestEngine().run(config)


def main():
    start = date(2025, 1, 1)
    end = date(2025, 12, 31)

    print(f"\n{'=' * 90}")
    print("CVD Divergence Parameter Sweep (full year, no VPIN)")
    print(f"{'=' * 90}")

    results = []
    sweep = list(product(
        [10, 20, 50],              # lookback
        [4, 8, 12, 16],            # target_ticks
        [2, 4, 6],                 # stop_buffer
        [0.10],                    # div_thresh
        [999],                     # poc_ticks (disabled — always 0 anyway)
    ))

    print(f"  Running {len(sweep)} configurations...\n")

    for i, (lb, tgt, stop, thresh, poc) in enumerate(sweep):
        r = run_one(start, end, lb, tgt, stop, 4, thresh, poc)
        m = r.metrics
        results.append({
            "lb": lb, "tgt": tgt, "stop": stop,
            "trades": m.total_trades, "wr": m.win_rate,
            "pnl": m.net_pnl, "sharpe": m.sharpe_ratio,
            "sortino": m.sortino_ratio, "pf": m.profit_factor,
            "dd": m.max_drawdown_pct,
            "avg_win": m.avg_win, "avg_loss": m.avg_loss,
        })
        status = "+" if m.sharpe_ratio > 0 else "-"
        print(f"  [{i+1:2d}/{len(sweep)}] {status} lb={lb:2d} tgt={tgt:2d} stop={stop} "
              f"trades={m.total_trades:3d} WR={m.win_rate:.1%} Sharpe={m.sharpe_ratio:>7.3f} PnL=${m.net_pnl:>9,.2f}")

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"\n{'=' * 90}")
    print("TOP 10 (by Sharpe)")
    print(f"{'=' * 90}")
    print(f"  {'lb':>3s} {'tgt':>4s} {'stop':>4s} {'trades':>6s} {'WR':>6s} {'Sharpe':>8s} {'PF':>6s} {'PnL':>10s} {'DD%':>6s}")
    print(f"  {'-' * 60}")
    for r in results[:10]:
        print(f"  {r['lb']:>3d} {r['tgt']:>4d} {r['stop']:>4d} {r['trades']:>6d} {r['wr']:>5.1%} "
              f"{r['sharpe']:>8.3f} {r['pf']:>6.2f} {'${:,.2f}'.format(r['pnl']):>10s} {r['dd']:>5.1f}%")

    # Run best config with VPIN
    if results and results[0]["sharpe"] > 0:
        best = results[0]
        print(f"\n  Running best config WITH VPIN filter...")
        r = run_one(start, end, best["lb"], best["tgt"], best["stop"], 4, 0.10, 999, use_vpin=True)
        m = r.metrics
        print(f"  VPIN: trades={m.total_trades} WR={m.win_rate:.1%} Sharpe={m.sharpe_ratio:.3f} PnL=${m.net_pnl:,.2f} DD={m.max_drawdown_pct:.1f}%")

    print(f"\n{'=' * 90}\n")


if __name__ == "__main__":
    main()
