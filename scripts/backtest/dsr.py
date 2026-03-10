#!/usr/bin/env python3
"""Run Deflated Sharpe Ratio validation on a strategy.

Usage:
    python scripts/backtest/dsr.py --strategy orb --start 2020-01-01 --end 2024-01-01
    python scripts/backtest/dsr.py --strategy all --start 2020-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy

from src.backtesting.dsr import DSRConfig, DeflatedSharpeCalculator
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.metrics import BacktestResult
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy


STRATEGY_NAMES = ["orb", "vwap"]

_ORB_DEFAULT_CONFIG = {
    "strategy": {"strategy_id": "orb", "max_signals_per_day": 1},
    "orb": {"target_multiplier": 0.5, "volume_multiplier": 1.5, "max_signal_time": "10:30",
            "expiry_minutes": 60, "require_vwap_alignment": True},
    "filter": {"min_score": 0, "require_direction_agreement": False, "signals": {}},
    "exit": {"time_stop_minutes": 60},
}

_VWAP_DEFAULT_CONFIG = {
    "strategy": {"strategy_id": "vwap_reversion", "max_signals_per_day": 4},
    "vwap": {"entry_sd_reversion": 1.5, "stop_sd": 2.0, "pullback_entry_sd": 0.5,
             "mode_cooldown_bars": 5, "expiry_minutes": 30},
    "filter": {"min_score": 0, "require_direction_agreement": False, "signals": {}},
    "exit": {"target": "vwap", "stop_ticks": 8, "time_stop_minutes": 30},
}


def make_strategy(name: str) -> list:
    """Create a fresh strategy instance."""
    if name == "orb":
        return [ORBStrategy(config=copy.deepcopy(_ORB_DEFAULT_CONFIG))]
    elif name == "vwap":
        return [VWAPReversionStrategy(config=copy.deepcopy(_VWAP_DEFAULT_CONFIG))]
    raise ValueError(f"Unknown strategy: {name}")


def run_backtest(
    strategy_name: str,
    start: date,
    end: date,
    capital: float,
    parquet_dir: str,
) -> BacktestResult:
    """Run a backtest for a single strategy."""
    strategies = make_strategy(strategy_name)
    config = BacktestConfig(
        strategies=strategies,
        start_date=start,
        end_date=end,
        initial_capital=capital,
        parquet_dir=parquet_dir,
    )
    engine = BacktestEngine()
    return engine.run(config)


def print_dsr_result(result, start: date, end: date) -> None:
    """Print formatted DSR result."""
    print("=" * 60)
    print(f"  Strategy:          {result.strategy_id}")
    print(f"  Observed Sharpe:   {result.observed_sharpe:.3f}")
    print(f"  PSR (vs 0):        {result.psr:.3f}")
    print(f"  E[max(SR)]:        {result.expected_max_sharpe:.3f}")
    print(f"  DSR (vs haircut):  {result.dsr:.3f}")
    verdict_suffix = "" if result.verdict == "PASS" else " (< 0.95)"
    print(f"  Verdict:           {result.verdict}{verdict_suffix}")
    print()
    print(f"  Sample size:       {result.sample_size} days")
    print(f"  Skewness:          {result.skewness:.2f}")
    print(f"  Excess kurtosis:   {result.kurtosis:.2f}")
    print(f"  SR std error:      {result.sr_std_error:.3f}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Deflated Sharpe Ratio validation on a strategy"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=STRATEGY_NAMES + ["all"],
        help="Strategy to validate (or 'all' for all strategies)",
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--n-trials", type=int, default=4, help="Number of strategies tested (default 4)"
    )
    parser.add_argument(
        "--capital", type=float, default=10_000.0, help="Initial capital (default 10000)"
    )
    parser.add_argument(
        "--significance", type=float, default=0.95, help="Significance level (default 0.95)"
    )
    parser.add_argument(
        "--parquet-dir", default="data/parquet", help="Parquet data directory"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    dsr_config = DSRConfig(
        benchmark_sharpe=0.0,
        significance_level=args.significance,
    )

    strategy_names = (
        STRATEGY_NAMES if args.strategy == "all" else [args.strategy]
    )
    n_trials = len(strategy_names) if args.strategy == "all" else args.n_trials

    print(f"DSR Validation: {args.strategy}")
    print(f"  Date range: {start} → {end}")
    print(f"  Trials: {n_trials}")
    print()

    any_fail = False
    for name in strategy_names:
        print(f"Running backtest: {name} ...")
        bt_result = run_backtest(name, start, end, args.capital, args.parquet_dir)

        result = DeflatedSharpeCalculator.compute_from_trades(
            bt_result.trades,
            initial_capital=args.capital,
            n_trials=n_trials,
            config=dsr_config,
            strategy_id=name,
        )

        print()
        print_dsr_result(result, start, end)
        print()

        if result.verdict == "FAIL":
            any_fail = True

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
