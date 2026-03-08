#!/usr/bin/env python3
"""Tune ORB strategy parameters with VPIN filter on full year L1 data.

Sweeps key parameters and reports results in a comparison table.
"""

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
from src.strategies.orb_strategy import ORBConfig, ORBStrategy


def run_one(start, end, tgt_mult, vol_mult, max_time, expiry, vwap_align, max_sig):
    hub = FeatureHub()
    vm = VPINMonitor(config=VPINConfig(), persist=False)
    cfg = ORBConfig(
        require_hmm_states=[], min_confidence=0.0,
        target_multiplier=tgt_mult,
        volume_multiplier=vol_mult,
        max_signal_time=max_time,
        expiry_minutes=expiry,
        require_vwap_alignment=vwap_align,
        max_signals_per_day=max_sig,
    )
    strat = ORBStrategy(cfg, hub)
    config = BacktestConfig(
        strategies=[strat], start_date=start, end_date=end,
        l1_parquet_dir="data/l1", l1_bar_seconds=5,
        vpin_monitor=vm,
    )
    return BacktestEngine().run(config)


def main():
    start = date(2025, 1, 1)
    end = date(2025, 12, 31)

    # Diagnostic: default config trade analysis
    print("\n  Analyzing ORB trades (default config, full year)...\n")
    r = run_one(start, end, 0.5, 1.5, "11:00", 90, True, 1)
    trades = r.trades
    m = r.metrics

    if trades:
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        exits = {}
        for t in trades:
            exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
        avg_bars = sum(t.bars_held for t in trades) / len(trades)
        avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0

        print(f"  Trades: {len(trades)}  WR: {len(wins)/len(trades):.1%}  PnL: ${m.net_pnl:,.2f}")
        print(f"  Sharpe: {m.sharpe_ratio:.3f}  Sortino: {m.sortino_ratio:.3f}  PF: {m.profit_factor:.2f}")
        print(f"  Avg win: ${avg_win:,.2f}  Avg loss: ${avg_loss:,.2f}  Avg bars: {avg_bars:.0f}")
        print(f"  Exits: {exits}")

        # ORB width distribution
        widths = [t.metadata.get("orb_width_ticks", 0) for t in trades]
        vol_ratios = [t.metadata.get("volume_ratio", 0) for t in trades]
        times = [t.metadata.get("time_since_open_minutes", 0) for t in trades]
        if widths:
            import numpy as np
            print(f"\n  ORB width (ticks): mean={np.mean(widths):.1f}  P25={np.percentile(widths, 25):.1f}  P50={np.median(widths):.1f}  P75={np.percentile(widths, 75):.1f}")
            print(f"  Volume ratio: mean={np.mean(vol_ratios):.2f}  P50={np.median(vol_ratios):.2f}")
            print(f"  Time since open (min): mean={np.mean(times):.0f}  P50={np.median(times):.0f}  P90={np.percentile(times, 90):.0f}")

        # Win rate by direction
        longs = [t for t in trades if t.direction == "LONG"]
        shorts = [t for t in trades if t.direction == "SHORT"]
        for label, subset in [("LONG", longs), ("SHORT", shorts)]:
            if subset:
                w = len([t for t in subset if t.net_pnl > 0])
                p = sum(t.net_pnl for t in subset)
                print(f"  {label}: {len(subset)} trades  WR={w/len(subset):.1%}  PnL=${p:,.2f}")
    else:
        print("  No trades generated with default config!")

    sys.stdout.flush()

    # Main sweep
    print(f"\n{'=' * 90}")
    print("ORB Parameter Sweep (with VPIN filter, full year)")
    print(f"{'=' * 90}")

    results = []
    sweep = list(product(
        [0.3, 0.5, 0.75, 1.0],    # target_multiplier
        [1.0, 1.5, 2.0],           # volume_multiplier
        ["10:30", "11:00", "11:30"],  # max_signal_time
        [30, 60, 90],              # expiry_minutes
        [True, False],             # require_vwap_alignment
    ))

    print(f"\n  Running {len(sweep)} configurations...\n")
    sys.stdout.flush()

    for i, (tgt, vol, max_t, exp, vwap) in enumerate(sweep):
        r = run_one(start, end, tgt, vol, max_t, exp, vwap, 1)
        m = r.metrics
        results.append({
            "tgt": tgt, "vol": vol, "max_t": max_t, "exp": exp, "vwap": vwap,
            "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
            "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
            "pf": m.profit_factor, "dd": m.max_drawdown_pct,
            "avg_win": m.avg_win, "avg_loss": m.avg_loss,
        })
        s = "+" if m.sharpe_ratio > 0 else "-"
        print(f"  [{i+1:3d}/{len(sweep)}] {s} tgt={tgt:.2f} vol={vol:.1f} max={max_t} exp={exp:2d}m vwap={str(vwap):5s}  "
              f"trades={m.total_trades:3d} WR={m.win_rate:.1%} Sharpe={m.sharpe_ratio:>7.3f} PnL=${m.net_pnl:>9,.2f}")
        sys.stdout.flush()

    # Sort by Sharpe and show top 15
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"\n{'=' * 90}")
    print("TOP 15 CONFIGURATIONS (by Sharpe)")
    print(f"{'=' * 90}")
    print(f"  {'tgt':>5s} {'vol':>4s} {'max_t':>6s} {'exp':>4s} {'vwap':>5s} {'trades':>6s} {'WR':>6s} {'Sharpe':>8s} {'Sortino':>8s} {'PF':>6s} {'PnL':>10s} {'DD%':>6s}")
    print(f"  {'-' * 85}")
    for r in results[:15]:
        print(f"  {r['tgt']:>5.2f} {r['vol']:>4.1f} {r['max_t']:>6s} {r['exp']:>4d} {str(r['vwap']):>5s} {r['trades']:>6d} {r['wr']:>5.1%} "
              f"{r['sharpe']:>8.3f} {r['sortino']:>8.3f} {r['pf']:>6.2f} "
              f"{'${:,.2f}'.format(r['pnl']):>10s} {r['dd']:>5.1f}%")

    # Also show: how many configs are profitable?
    profitable = [r for r in results if r["pnl"] > 0]
    positive_sharpe = [r for r in results if r["sharpe"] > 0]
    print(f"\n  {len(profitable)}/{len(results)} configs profitable, {len(positive_sharpe)}/{len(results)} positive Sharpe")

    # max_signals_per_day sweep with best config
    if results and results[0]["sharpe"] > 0:
        best = results[0]
        print(f"\n{'=' * 90}")
        print(f"Max signals/day sweep with best config (tgt={best['tgt']}, vol={best['vol']}, max_t={best['max_t']}, exp={best['exp']}, vwap={best['vwap']})")
        print(f"{'=' * 90}\n")
        for max_sig in [1, 2, 3]:
            r = run_one(start, end, best["tgt"], best["vol"], best["max_t"], best["exp"], best["vwap"], max_sig)
            m = r.metrics
            print(f"  max_sig={max_sig}  trades={m.total_trades:3d} WR={m.win_rate:.1%} Sharpe={m.sharpe_ratio:.3f} PnL=${m.net_pnl:>9,.2f} DD={m.max_drawdown_pct:.1f}%")
        sys.stdout.flush()

    print(f"\n{'=' * 90}\n")


if __name__ == "__main__":
    main()
