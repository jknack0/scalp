#!/usr/bin/env python3
"""Validate spread + VPIN filters combined on any strategy over 1 year.

Runs 4 configurations:
1. No filters (baseline)
2. Spread filter only
3. VPIN filter only
4. Both filters

Usage:
    python scripts/validate_combined_filters.py
    python scripts/validate_combined_filters.py --strategy vwap
    python scripts/validate_combined_filters.py --strategy cvd --start 2025-01-01 --end 2025-12-31
    python scripts/validate_combined_filters.py --strategy all
"""

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.metrics import BacktestMetrics
from src.features.feature_hub import FeatureHub
from src.filters.spread_monitor import SpreadConfig, SpreadMonitor
from src.filters.vpin_monitor import VPINConfig, VPINMonitor
from src.strategies.orb_strategy import ORBConfig, ORBStrategy
from src.strategies.vwap_strategy import VWAPConfig, VWAPStrategy

STRATEGIES = ["orb", "vwap"]


def _parse_args():
    parser = argparse.ArgumentParser(description="Combined filter validation")
    parser.add_argument("--strategy", type=str, default="orb", choices=STRATEGIES + ["all"])
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--bar-interval", type=int, default=5)
    parser.add_argument("--capital", type=float, default=10_000.0)
    return parser.parse_args()


def _make_strategy(name: str):
    """Create a strategy instance configured to run without HMM."""
    hub = FeatureHub()
    if name == "orb":
        return ORBStrategy(ORBConfig(require_hmm_states=[], min_confidence=0.3), hub)
    elif name == "vwap":
        return VWAPStrategy(
            VWAPConfig(
                reversion_hmm_states=[],
                pullback_hmm_states=[],
                min_confidence=0.3,
            ),
            hub,
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")


def _run(strategy_name, start, end, bar_seconds, capital, spread=False, vpin=False, label=""):
    print(f"\n  [{label}]...")
    strategy = _make_strategy(strategy_name)

    sm = SpreadMonitor(config=SpreadConfig(), persist=False) if spread else None
    vm = VPINMonitor(config=VPINConfig(), persist=False) if vpin else None

    config = BacktestConfig(
        strategies=[strategy],
        start_date=start,
        end_date=end,
        initial_capital=capital,
        l1_parquet_dir="data/l1",
        l1_bar_seconds=bar_seconds,
        spread_monitor=sm,
        vpin_monitor=vm,
    )

    engine = BacktestEngine()
    return engine.run(config)


def _fmt(m: BacktestMetrics) -> dict:
    return {
        "trades": m.total_trades,
        "wr": m.win_rate,
        "pnl": m.net_pnl,
        "sharpe": m.sharpe_ratio,
        "sortino": m.sortino_ratio,
        "pf": m.profit_factor,
        "dd": m.max_drawdown_pct,
        "avg_win": m.avg_win,
        "avg_loss": m.avg_loss,
        "best": m.best_trade,
        "worst": m.worst_trade,
        "days": m.trading_days,
    }


def _print_results(strategy_name: str, results: dict, start, end, bar_interval, capital):
    keys = list(results.keys())

    print(f"\n{'=' * 72}")
    print(f"{strategy_name.upper()} Strategy — Combined Filter Validation (Spread + VPIN)")
    print(f"{'=' * 72}")
    print(f"  Date range:   {start} -> {end}")
    print(f"  Bar interval: {bar_interval}s")
    print(f"  Capital:      ${capital:,.0f}")

    print(f"\n{'':30s} {'No filter':>10s} {'Spread':>10s} {'VPIN':>10s} {'Both':>10s}")
    print(f"{'-' * 72}")

    rows = [
        ("Trades", "trades", "d"),
        ("Win rate", "wr", ".1%"),
        ("Net P&L", "pnl", ",.2f"),
        ("Sharpe", "sharpe", ".3f"),
        ("Sortino", "sortino", ".3f"),
        ("Profit factor", "pf", ".2f"),
        ("Max drawdown %", "dd", ".2f"),
        ("Avg win", "avg_win", ",.2f"),
        ("Avg loss", "avg_loss", ",.2f"),
        ("Best trade", "best", ",.2f"),
        ("Worst trade", "worst", ",.2f"),
        ("Trading days", "days", "d"),
    ]

    for row_label, key, fmt in rows:
        vals = [results[k][key] for k in keys]
        if fmt == "d":
            line = f"  {row_label:28s} {vals[0]:>10d} {vals[1]:>10d} {vals[2]:>10d} {vals[3]:>10d}"
        elif fmt == ".1%":
            line = f"  {row_label:28s} {vals[0]:>9.1%} {vals[1]:>9.1%} {vals[2]:>9.1%} {vals[3]:>9.1%}"
        elif fmt == ",.2f":
            line = f"  {row_label:28s} {'${:,.2f}'.format(vals[0]):>10s} {'${:,.2f}'.format(vals[1]):>10s} {'${:,.2f}'.format(vals[2]):>10s} {'${:,.2f}'.format(vals[3]):>10s}"
        elif fmt == ".3f":
            line = f"  {row_label:28s} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f} {vals[3]:>10.3f}"
        elif fmt == ".2f":
            line = f"  {row_label:28s} {vals[0]:>10.2f} {vals[1]:>10.2f} {vals[2]:>10.2f} {vals[3]:>10.2f}"
        else:
            line = f"  {row_label:28s} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}"
        print(line)

    # Improvement summary
    base = results[keys[0]]
    both = results[keys[3]]
    print(f"\n{'-' * 72}")
    print(f"  COMBINED FILTER IMPACT (Both vs No filter):")
    trade_reduction = (1 - both["trades"] / base["trades"]) * 100 if base["trades"] > 0 else 0
    print(f"    Trades reduced:  {base['trades']} -> {both['trades']} ({trade_reduction:.1f}% reduction)")
    print(f"    Sharpe:          {base['sharpe']:.3f} -> {both['sharpe']:.3f} ({both['sharpe'] - base['sharpe']:+.3f})")
    print(f"    Win rate:        {base['wr']:.1%} -> {both['wr']:.1%} ({both['wr'] - base['wr']:+.1%})")
    print(f"    Net P&L:         ${base['pnl']:,.2f} -> ${both['pnl']:,.2f} (${both['pnl'] - base['pnl']:+,.2f})")
    print(f"    Max drawdown:    {base['dd']:.2f}% -> {both['dd']:.2f}%")
    print(f"    Worst trade:     ${base['worst']:,.2f} -> ${both['worst']:,.2f}")

    print(f"\n{'=' * 72}\n")


def _run_strategy(strategy_name: str, start, end, bar_interval, capital):
    """Run all 4 filter configurations for a single strategy."""
    results = {}
    for label, sp, vp in [
        ("1. No filters", False, False),
        ("2. Spread only", True, False),
        ("3. VPIN only", False, True),
        ("4. Spread + VPIN", True, True),
    ]:
        r = _run(strategy_name, start, end, bar_interval, capital, spread=sp, vpin=vp, label=f"{strategy_name.upper()} {label}")
        results[label] = _fmt(r.metrics)

    _print_results(strategy_name, results, start, end, bar_interval, capital)
    return results


def main():
    args = _parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if args.strategy == "all":
        strategies = STRATEGIES
    else:
        strategies = [args.strategy]

    all_results = {}
    for strat in strategies:
        all_results[strat] = _run_strategy(strat, start, end, args.bar_interval, args.capital)

    # Cross-strategy summary if running all
    if len(strategies) > 1:
        print(f"\n{'=' * 72}")
        print("CROSS-STRATEGY SUMMARY — Combined Filters (Spread + VPIN)")
        print(f"{'=' * 72}")
        print(f"\n  {'Strategy':15s} {'Trades':>8s} {'-> Both':>8s} {'WR':>7s} {'-> Both':>7s} {'Sharpe':>8s} {'-> Both':>8s} {'P&L':>10s} {'-> Both':>10s}")
        print(f"  {'-' * 80}")
        for strat in strategies:
            r = all_results[strat]
            base = r["1. No filters"]
            both = r["4. Spread + VPIN"]
            print(
                f"  {strat:15s} {base['trades']:>8d} {both['trades']:>8d}"
                f" {base['wr']:>6.1%} {both['wr']:>6.1%}"
                f" {base['sharpe']:>8.3f} {both['sharpe']:>8.3f}"
                f" {'${:,.0f}'.format(base['pnl']):>10s} {'${:,.0f}'.format(both['pnl']):>10s}"
            )
        print(f"\n{'=' * 72}\n")


if __name__ == "__main__":
    main()
