#!/usr/bin/env python3
"""Phase 7.1 — Validate spread filter impact on ORB strategy.

Runs ORB backtest twice (with and without spread filter) on L1 data,
then compares trade counts, Sharpe, win rate, and net P&L.

Usage:
    python scripts/validate_spread_filter.py
    python scripts/validate_spread_filter.py --start 2025-03-01 --end 2025-06-01
    python scripts/validate_spread_filter.py --bar-interval 5
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
from src.strategies.orb_strategy import ORBConfig, ORBStrategy


def _parse_args():
    parser = argparse.ArgumentParser(description="Validate spread filter impact on ORB")
    parser.add_argument("--start", type=str, default="2025-03-01")
    parser.add_argument("--end", type=str, default="2025-06-01")
    parser.add_argument("--bar-interval", type=int, default=5, help="L1 bar seconds")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--z-threshold", type=float, default=2.0)
    return parser.parse_args()


def _run_backtest(
    start: date,
    end: date,
    bar_seconds: int,
    capital: float,
    spread_monitor: SpreadMonitor | None = None,
    label: str = "",
):
    print(f"\n  Running ORB backtest ({label})...")
    hub = FeatureHub()
    # Disable HMM gating (no model loaded in backtest) by passing empty require_hmm_states
    strategy = ORBStrategy(ORBConfig(require_hmm_states=[]), hub)

    config = BacktestConfig(
        strategies=[strategy],
        start_date=start,
        end_date=end,
        initial_capital=capital,
        l1_parquet_dir="data/l1",
        l1_bar_seconds=bar_seconds,
        spread_monitor=spread_monitor,
    )

    engine = BacktestEngine()
    result = engine.run(config)
    return result


def _print_metrics(m: BacktestMetrics, label: str):
    print(f"\n  --- {label} ---")
    print(f"  Trades:         {m.total_trades}")
    print(f"  Win rate:       {m.win_rate:.1%}")
    print(f"  Net P&L:        ${m.net_pnl:,.2f}")
    print(f"  Sharpe:         {m.sharpe_ratio:.3f}")
    print(f"  Sortino:        {m.sortino_ratio:.3f}")
    print(f"  Profit factor:  {m.profit_factor:.2f}")
    print(f"  Max drawdown:   {m.max_drawdown_pct:.2f}%")
    print(f"  Avg win:        ${m.avg_win:,.2f}")
    print(f"  Avg loss:       ${m.avg_loss:,.2f}")
    print(f"  Best trade:     ${m.best_trade:,.2f}")
    print(f"  Worst trade:    ${m.worst_trade:,.2f}")
    print(f"  Trading days:   {m.trading_days}")


def main():
    args = _parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"\n{'=' * 60}")
    print("Phase 7.1 -- Spread Filter Impact on ORB Strategy")
    print(f"{'=' * 60}")
    print(f"  Date range:     {start} -> {end}")
    print(f"  Bar interval:   {args.bar_interval}s")
    print(f"  Z-threshold:    {args.z_threshold}")
    print(f"  Capital:        ${args.capital:,.0f}")

    # Run WITHOUT spread filter
    result_no_filter = _run_backtest(
        start, end, args.bar_interval, args.capital,
        spread_monitor=None,
        label="NO spread filter",
    )

    # Run WITH spread filter
    spread_mon = SpreadMonitor(
        config=SpreadConfig(z_threshold=args.z_threshold),
        persist=False,
    )
    result_with_filter = _run_backtest(
        start, end, args.bar_interval, args.capital,
        spread_monitor=spread_mon,
        label="WITH spread filter (z={})".format(args.z_threshold),
    )

    # Print results
    m_no = result_no_filter.metrics
    m_yes = result_with_filter.metrics

    _print_metrics(m_no, "WITHOUT spread filter")
    _print_metrics(m_yes, f"WITH spread filter (z={args.z_threshold})")

    # Comparison
    print(f"\n  --- COMPARISON ---")
    trade_diff = m_yes.total_trades - m_no.total_trades
    trade_pct = (trade_diff / m_no.total_trades * 100) if m_no.total_trades > 0 else 0
    print(f"  Trades blocked: {abs(trade_diff)} ({abs(trade_pct):.1f}% reduction)")

    sharpe_diff = m_yes.sharpe_ratio - m_no.sharpe_ratio
    print(f"  Sharpe change:  {sharpe_diff:+.3f} ({m_no.sharpe_ratio:.3f} -> {m_yes.sharpe_ratio:.3f})")

    wr_diff = m_yes.win_rate - m_no.win_rate
    print(f"  Win rate change: {wr_diff:+.1%} ({m_no.win_rate:.1%} -> {m_yes.win_rate:.1%})")

    pnl_diff = m_yes.net_pnl - m_no.net_pnl
    print(f"  Net P&L change: ${pnl_diff:+,.2f} (${m_no.net_pnl:,.2f} -> ${m_yes.net_pnl:,.2f})")

    # Verdict
    print(f"\n  --- VERDICT ---")
    if m_yes.sharpe_ratio > m_no.sharpe_ratio and trade_pct < -5:
        print(f"  POSITIVE: Spread filter improves risk-adjusted returns")
        print(f"  Blocked {abs(trade_pct):.1f}% of trades, Sharpe improved by {sharpe_diff:+.3f}")
    elif m_yes.sharpe_ratio >= m_no.sharpe_ratio:
        print(f"  NEUTRAL: Spread filter has minimal impact")
    else:
        print(f"  NEGATIVE: Spread filter reduces Sharpe (may be too aggressive)")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
