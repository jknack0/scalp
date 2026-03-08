#!/usr/bin/env python3
"""Quick CVD divergence flow trace through actual strategy."""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.features.feature_hub import FeatureHub
from src.strategies.cvd_divergence_strategy import CVDDivergenceConfig, CVDDivergenceStrategy


def main():
    # Run with low confidence to see all trades
    for conf in [0.3, 0.1, 0.0]:
        hub = FeatureHub()
        cfg = CVDDivergenceConfig(require_hmm_states=[], min_confidence=conf)
        strat = CVDDivergenceStrategy(cfg, hub)

        config = BacktestConfig(
            strategies=[strat],
            start_date=date(2025, 3, 1),
            end_date=date(2025, 6, 1),
            l1_parquet_dir="data/l1",
            l1_bar_seconds=5,
        )
        r = BacktestEngine().run(config)
        m = r.metrics

        # Check confidence values of trades
        confidences = [t.metadata.get("magnitude", 0) for t in r.trades]
        print(f"  min_conf={conf:.1f}:  trades={m.total_trades:3d}  WR={m.win_rate:.1%}  Sharpe={m.sharpe_ratio:.3f}  PnL=${m.net_pnl:,.2f}")
        if r.trades:
            confs = [t.metadata.get("magnitude", 0) for t in r.trades[:5]]
            poc_dists = [t.metadata.get("poc_distance_ticks", 0) for t in r.trades[:5]]
            print(f"           first 5 magnitudes: {confs}")
            print(f"           first 5 poc_dists:  {poc_dists}")

    # Also try with bigger lookback
    print("\n  --- Lookback sweep ---")
    for lb in [5, 20, 50]:
        hub = FeatureHub()
        cfg = CVDDivergenceConfig(require_hmm_states=[], min_confidence=0.0, swing_lookback_bars=lb)
        strat = CVDDivergenceStrategy(cfg, hub)
        config = BacktestConfig(
            strategies=[strat],
            start_date=date(2025, 3, 1),
            end_date=date(2025, 6, 1),
            l1_parquet_dir="data/l1",
            l1_bar_seconds=5,
        )
        r = BacktestEngine().run(config)
        m = r.metrics
        print(f"  lookback={lb:3d}:  trades={m.total_trades:3d}  WR={m.win_rate:.1%}  Sharpe={m.sharpe_ratio:.3f}  PnL=${m.net_pnl:,.2f}")


if __name__ == "__main__":
    main()
