#!/usr/bin/env python3
"""Smoke test: verify each strategy generates trades without HMM on 3 months of L1 data."""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy


def make_strategy(name):
    if name == "orb":
        return ORBStrategy(config={
            "strategy": {"strategy_id": "orb", "max_signals_per_day": 1},
            "orb": {},
            "filter": {"min_score": 0, "require_direction_agreement": False, "signals": {}},
            "exit": {"time_stop_minutes": 60},
        })
    elif name == "vwap":
        return VWAPReversionStrategy(config={
            "strategy": {"strategy_id": "vwap_reversion", "max_signals_per_day": 4},
            "vwap": {},
            "filter": {"min_score": 0, "require_direction_agreement": False, "signals": {}},
            "exit": {"target": "vwap", "stop_ticks": 8, "time_stop_minutes": 30},
        })


def main():
    start = date(2025, 3, 1)
    end = date(2025, 6, 1)

    for name in ["orb", "vwap"]:
        strat = make_strategy(name)
        config = BacktestConfig(
            strategies=[strat],
            start_date=start,
            end_date=end,
            l1_parquet_dir="data/l1",
            l1_bar_seconds=5,
        )
        engine = BacktestEngine()
        r = engine.run(config)
        m = r.metrics
        status = "OK" if m.total_trades > 0 else "NO TRADES"
        print(f"  {name:15s}  {status:10s}  trades={m.total_trades:4d}  WR={m.win_rate:.1%}  Sharpe={m.sharpe_ratio:.3f}  PnL=${m.net_pnl:,.2f}")


if __name__ == "__main__":
    main()
