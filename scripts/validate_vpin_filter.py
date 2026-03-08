#!/usr/bin/env python3
"""Phase 7.2 — Validate VPIN regime filter impact on a strategy.

Runs backtest twice (with and without VPIN gate) on L1 data,
then compares trade counts, Sharpe, win rate, and net P&L.

Usage:
    python scripts/validate_vpin_filter.py --strategy orb
    python scripts/validate_vpin_filter.py --strategy vwap
    python scripts/validate_vpin_filter.py --strategy vwap --start 2025-03-01 --end 2025-06-01
"""

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.metrics import BacktestMetrics
from src.features.feature_hub import FeatureHub
from src.filters.vpin_monitor import VPINConfig, VPINMonitor
from src.strategies.orb_strategy import ORBConfig, ORBStrategy
from src.strategies.vwap_strategy import VWAPConfig, VWAPStrategy


def _parse_args():
    parser = argparse.ArgumentParser(description="Validate VPIN filter impact")
    parser.add_argument("--strategy", type=str, default="orb", choices=["orb", "vwap"])
    parser.add_argument("--start", type=str, default="2025-03-01")
    parser.add_argument("--end", type=str, default="2025-06-01")
    parser.add_argument("--bar-interval", type=int, default=5)
    parser.add_argument("--capital", type=float, default=10_000.0)
    return parser.parse_args()


def _make_strategy(name: str):
    hub = FeatureHub()
    if name == "vwap":
        return VWAPStrategy(
            VWAPConfig(
                reversion_hmm_states=[],
                pullback_hmm_states=[],
                min_confidence=0.3,
            ),
            hub,
        )
    return ORBStrategy(ORBConfig(require_hmm_states=[]), hub)


def _run_backtest(
    start: date,
    end: date,
    bar_seconds: int,
    capital: float,
    strategy_name: str = "orb",
    vpin_monitor: VPINMonitor | None = None,
    label: str = "",
):
    print(f"\n  Running {strategy_name.upper()} backtest ({label})...")
    strategy = _make_strategy(strategy_name)

    config = BacktestConfig(
        strategies=[strategy],
        start_date=start,
        end_date=end,
        initial_capital=capital,
        l1_parquet_dir="data/l1",
        l1_bar_seconds=bar_seconds,
        vpin_monitor=vpin_monitor,
    )

    engine = BacktestEngine()
    return engine.run(config)


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

    strat = args.strategy
    blocked_in = "mean_reversion" if strat == "orb" else "trending"

    print(f"\n{'=' * 60}")
    print(f"Phase 7.2 -- VPIN Regime Filter Impact on {strat.upper()} Strategy")
    print(f"{'=' * 60}")
    print(f"  Strategy:       {strat.upper()}")
    print(f"  Date range:     {start} -> {end}")
    print(f"  Bar interval:   {args.bar_interval}s")
    print(f"  VPIN config:    bucket=100, n_buckets=50")
    print(f"  Thresholds:     trending>0.45, mean_rev<0.25")
    print(f"  {strat.upper()} blocked in: {blocked_in} regime")
    print(f"  Capital:        ${args.capital:,.0f}")

    # Run WITHOUT VPIN
    result_no = _run_backtest(
        start, end, args.bar_interval, args.capital,
        strategy_name=strat,
        vpin_monitor=None,
        label="NO VPIN filter",
    )

    # Run WITH VPIN
    vpin = VPINMonitor(config=VPINConfig(), persist=False)
    result_yes = _run_backtest(
        start, end, args.bar_interval, args.capital,
        strategy_name=strat,
        vpin_monitor=vpin,
        label="WITH VPIN filter",
    )

    m_no = result_no.metrics
    m_yes = result_yes.metrics

    _print_metrics(m_no, "WITHOUT VPIN filter")
    _print_metrics(m_yes, "WITH VPIN filter")

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

    # VPIN stats
    if vpin.latest_state:
        print(f"\n  --- VPIN Stats ---")
        print(f"  Buckets completed: {vpin.bucket_count}")
        regime, vpin_val = vpin.get_regime()
        print(f"  Final regime:      {regime} (vpin={vpin_val:.3f})")

    # Verdict
    print(f"\n  --- VERDICT ---")
    if m_yes.sharpe_ratio > m_no.sharpe_ratio and trade_pct < -5:
        print(f"  POSITIVE: VPIN filter improves risk-adjusted returns")
        print(f"  Blocked {abs(trade_pct):.1f}% of trades, Sharpe improved by {sharpe_diff:+.3f}")
    elif abs(trade_diff) == 0:
        print(f"  NO EFFECT: VPIN did not block any trades")
        print(f"  Strategy may not trigger during blocked regime")
    elif m_yes.sharpe_ratio >= m_no.sharpe_ratio:
        print(f"  NEUTRAL: VPIN filter has minimal impact")
    else:
        print(f"  NEGATIVE: VPIN filter reduces Sharpe (may be too aggressive)")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
