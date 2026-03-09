#!/usr/bin/env python3
"""Smoke test: verify each strategy generates trades without HMM on 3 months of L1 data."""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.features.feature_hub import FeatureHub
from src.strategies.orb_strategy import ORBConfig, ORBStrategy
from src.strategies.vwap_strategy import VWAPConfig, VWAPStrategy


def make_strategy(name):
    hub = FeatureHub()
    if name == "orb":
        return ORBStrategy(ORBConfig(require_hmm_states=[], min_confidence=0.3), hub)
    elif name == "vwap":
        return VWAPStrategy(VWAPConfig(reversion_hmm_states=[], pullback_hmm_states=[], min_confidence=0.3), hub)


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
