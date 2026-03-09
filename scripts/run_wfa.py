#!/usr/bin/env python3
"""Run Walk-Forward Anchored (WFA) validation on a strategy.

Usage:
    python scripts/run_wfa.py --strategy orb --start 2020-01-01 --end 2024-01-01
    python scripts/run_wfa.py --strategy vwap --start 2020-01-01 --end 2024-01-01 --train-days 63 --test-days 21
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestConfig
from src.backtesting.wfa import WFAConfig, WFARunner
from src.features.feature_hub import FeatureHub
from src.strategies.base import StrategyBase
from src.strategies.orb_strategy import ORBConfig, ORBStrategy
from src.strategies.vwap_strategy import VWAPConfig, VWAPStrategy


STRATEGY_MAP = {
    "orb": (ORBConfig, ORBStrategy),
    "vwap": (VWAPConfig, VWAPStrategy),
}

PARAM_GRIDS = {
    "orb": {
        "target_multiplier": [0.4, 0.5, 0.6, 0.7],
        "volume_multiplier": [1.3, 1.5, 1.7, 2.0],
    },
    "vwap": {
        "entry_sd_reversion": [1.8, 2.0, 2.2, 2.5],
        "flat_slope_threshold": [0.001, 0.002, 0.003, 0.005],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Walk-Forward Anchored validation on a strategy"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=list(STRATEGY_MAP.keys()),
        help="Strategy to validate",
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--train-days", type=int, default=63, help="Training window in trading days (default 63)"
    )
    parser.add_argument(
        "--test-days", type=int, default=21, help="Test window in trading days (default 21)"
    )
    parser.add_argument(
        "--capital", type=float, default=10_000.0, help="Initial capital (default 10000)"
    )
    parser.add_argument(
        "--parquet-dir", default="data/parquet", help="Parquet data directory"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Max parallel workers (default: auto)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    strategy_name = args.strategy

    print(f"WFA Validation: {strategy_name}")
    print(f"  Date range: {start} → {end}")
    print(f"  Train: {args.train_days} days, Test: {args.test_days} days")
    print()

    param_grid = PARAM_GRIDS[strategy_name]
    n_combos = 1
    for vals in param_grid.values():
        n_combos *= len(vals)
    print(f"  Param grid: {n_combos} combinations per cycle")
    for name, vals in param_grid.items():
        print(f"    {name}: {vals}")
    print()

    # Template backtest config
    config_cls, strategy_cls = STRATEGY_MAP[strategy_name]
    hub = FeatureHub()
    cfg = config_cls()
    cfg.require_hmm_states = []
    template_strategies = [strategy_cls(cfg, hub)]

    template = BacktestConfig(
        strategies=template_strategies,
        start_date=start,
        end_date=end,
        initial_capital=args.capital,
        parquet_dir=args.parquet_dir,
    )

    wfa_config = WFAConfig(
        train_days=args.train_days,
        test_days=args.test_days,
        max_workers=args.workers,
    )

    runner = WFARunner(strategy_name, param_grid, template)
    result = runner.run(wfa_config)

    # Print summary
    print("=" * 60)
    print(f"  Strategy:           {result.strategy_id}")
    print(f"  Cycles:             {result.n_cycles}")
    print(f"  Efficiency Ratio:   {result.efficiency_ratio:.3f}")
    print(f"  IS/OOS Correlation: {result.is_oos_correlation:.2f}")
    print(f"  Avg IS Sharpe:      {result.avg_is_sharpe:.3f}")
    print(f"  Avg OOS Sharpe:     {result.avg_oos_sharpe:.3f}")

    threshold = wfa_config.efficiency_threshold
    if result.verdict == "PASS":
        print(f"  Verdict:            PASS (>= {threshold})")
    else:
        print(f"  Verdict:            FAIL (< {threshold})")

    # Parameter drift
    if result.param_drift:
        print()
        print("  Parameter Drift:")
        for param_name, values in result.param_drift.items():
            print(f"    {param_name}:  {values}")

    print("=" * 60)
    print()

    # Per-cycle table
    if result.cycles:
        print(
            f"{'Cycle':>5}  {'Train Start':>11}  {'Train End':>11}  "
            f"{'Test Start':>11}  {'Test End':>11}  "
            f"{'IS Sharpe':>10}  {'OOS Sharpe':>11}  Best Params"
        )
        print("-" * 120)
        for cycle in result.cycles:
            params_str = ", ".join(
                f"{k}={v}" for k, v in cycle.best_params.items()
            )
            print(
                f"{cycle.cycle_id:5d}  "
                f"{cycle.train_start!s:>11}  {cycle.train_end!s:>11}  "
                f"{cycle.test_start!s:>11}  {cycle.test_end!s:>11}  "
                f"{cycle.is_sharpe:10.3f}  {cycle.oos_sharpe:11.3f}  "
                f"{params_str}"
            )

    sys.exit(0 if result.verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
