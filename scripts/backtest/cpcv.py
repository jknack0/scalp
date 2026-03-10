#!/usr/bin/env python3
"""Run CPCV (Combinatorial Purged Cross-Validation) on a strategy.

Usage:
    python scripts/backtest/cpcv.py --strategy orb --start 2020-01-01 --end 2024-01-01
    python scripts/backtest/cpcv.py --strategy vwap --start 2020-01-01 --end 2024-01-01 --n-groups 6
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtesting.cpcv import CPCVConfig, CPCVValidator
from src.backtesting.engine import BacktestConfig
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy


STRATEGY_NAMES = ["orb", "vwap"]

_ORB_DEFAULT_CONFIG = {
    "strategy": {"strategy_id": "orb", "max_signals_per_day": 1},
    "orb": {"target_multiplier": 0.5, "stop_multiplier": 1.0, "volume_multiplier": 1.5,
            "min_range_pct": 0.002, "orb_start": "09:30", "orb_end": "09:45",
            "max_signal_time": "10:30", "require_vwap_alignment": True, "expiry_minutes": 60},
    "filter": {"min_score": 0, "require_direction_agreement": False, "signals": {}},
    "exit": {"time_stop_minutes": 60},
}

_VWAP_DEFAULT_CONFIG = {
    "strategy": {"strategy_id": "vwap_reversion", "max_signals_per_day": 4},
    "vwap": {"entry_sd_reversion": 1.5, "stop_sd": 2.0, "flat_slope_threshold": 0.003,
             "trending_slope_threshold": 0.008, "first_kiss_lookback_bars": 6,
             "min_session_age_minutes": 15, "require_reversal_candle": False,
             "pullback_entry_sd": 0.5, "mode_cooldown_bars": 5, "expiry_minutes": 30},
    "filter": {"min_score": 0, "require_direction_agreement": False, "signals": {}},
    "exit": {"target": "vwap", "stop_ticks": 8, "time_stop_minutes": 30},
}


def make_strategy_factory(name: str):
    """Return a callable that creates fresh strategy instances."""
    import copy

    def factory():
        if name == "orb":
            return [ORBStrategy(config=copy.deepcopy(_ORB_DEFAULT_CONFIG))]
        elif name == "vwap":
            return [VWAPReversionStrategy(config=copy.deepcopy(_VWAP_DEFAULT_CONFIG))]
        raise ValueError(f"Unknown strategy: {name}")

    return factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPCV validation on a strategy")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=STRATEGY_NAMES,
        help="Strategy to validate",
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--n-groups", type=int, default=8, help="Number of groups (default 8)")
    parser.add_argument("--k-test", type=int, default=2, help="Test groups per fold (default 2)")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital")
    parser.add_argument("--parquet-dir", default="data/parquet", help="Parquet data directory")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers (default: auto)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"CPCV Validation: {args.strategy}")
    print(f"  Date range: {start} -> {end}")
    print(f"  Groups: {args.n_groups}, Test groups: {args.k_test}")
    print()

    factory = make_strategy_factory(args.strategy)

    # Template backtest config (strategies/dates overridden per fold)
    template = BacktestConfig(
        strategies=factory(),
        start_date=start,
        end_date=end,
        initial_capital=args.capital,
        parquet_dir=args.parquet_dir,
    )

    strat_cls_map = {"orb": ORBStrategy, "vwap": VWAPReversionStrategy}
    strat_cls = strat_cls_map[args.strategy]
    strat_path = f"{strat_cls.__module__}.{strat_cls.__name__}"

    cpcv_config = CPCVConfig(
        n_groups=args.n_groups,
        k_test=args.k_test,
        max_workers=args.workers,
    )

    validator = CPCVValidator(
        factory, template,
        config_cls_path=strat_path,
        strategy_cls_path=strat_path,
    )
    result = validator.run(cpcv_config)

    # Print results
    print("=" * 60)
    print(f"  Strategy:       {result.strategy_id}")
    print(f"  PBO:            {result.pbo:.4f}")
    print(f"  Verdict:        {result.verdict}")
    print(f"  Folds:          {result.n_paths}")
    print(f"  Avg IS Sharpe:  {result.avg_is_sharpe:.3f}")
    print(f"  Avg OOS Sharpe: {result.avg_oos_sharpe:.3f}")
    print(f"  Sharpe Decay:   {result.sharpe_decay:.3f}")
    print("=" * 60)
    print()

    # Per-fold summary
    print(f"{'Fold':>4}  {'IS Sharpe':>10}  {'OOS Sharpe':>11}  {'IS Return':>10}  {'OOS Return':>11}")
    print("-" * 52)
    for i in range(result.n_paths):
        print(
            f"{i:4d}  "
            f"{result.is_sharpes[i]:10.3f}  "
            f"{result.oos_sharpes[i]:11.3f}  "
            f"{result.is_returns[i]:10.2f}  "
            f"{result.oos_returns[i]:11.2f}"
        )

    sys.exit(0 if result.verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
