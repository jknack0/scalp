#!/usr/bin/env python3
"""Run VWAP backtest with L2 filters on September 2025 data.

Hardcoded params: entry_sd=1.5, stop_sd=2.0, expiry=30m
Sweeps L2 filter combos: depth, weighted_mid, mid_momentum.
"""

import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

import polars as pl

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.features.feature_hub import FeatureHub
from src.filters.depth_monitor import DepthConfig, DepthMonitor
from src.filters.mid_momentum import MidMomentumMonitor, MomentumConfig
from src.filters.spread_monitor import SpreadConfig, SpreadMonitor
from src.filters.weighted_mid import WeightedMidConfig, WeightedMidMonitor
from src.strategies.vwap_strategy import VWAPConfig, VWAPStrategy


def make_vwap():
    hub = FeatureHub()
    cfg = VWAPConfig(
        reversion_hmm_states=[], pullback_hmm_states=[], min_confidence=0.3,
        entry_sd_reversion=1.5, stop_sd=2.0,
        pullback_entry_sd=0.5, mode_cooldown_bars=5,
        max_signals_per_day=4, expiry_minutes=30,
    )
    return VWAPStrategy(cfg, hub)


def load_l2_sampled(path, sample_rate=10):
    """Load L2 RTH parquet, sample every Nth row. Engine uses ts_event directly."""
    print(f"  Loading L2 from {path} (sample 1/{sample_rate})...")
    df = pl.read_parquet(path)
    df = df.gather_every(sample_rate)
    print(f"  Loaded {len(df)} L2 snapshots")
    return df


def main():
    start = date(2025, 9, 1)
    end = date(2025, 9, 30)

    engine = BacktestEngine()

    # Load enriched 1s bars for September
    print("  Loading enriched 1s bars...")
    bars_config = BacktestConfig(
        strategies=[], start_date=start, end_date=end,
        parquet_dir="data/parquet_1s_enriched",
    )
    bars = engine._load_bars(bars_config)
    print(f"  Loaded {len(bars)} bars\n")

    # Load L2 data
    l2_path = "data/l2_rth/glbx-mdp3-20250901-20250930.mbp-10_rth.parquet"
    l2_df = load_l2_sampled(l2_path, sample_rate=10)

    # Filter combos
    combos = [
        {"name": "no_filters", "spread": None, "depth": None, "wmid": None, "momentum": None},
        {"name": "spread_z2.0", "spread": SpreadConfig(z_threshold=2.0), "depth": None, "wmid": None, "momentum": None},
    ]

    # Depth sweep
    for thin in [0.4, 0.6, 0.8]:
        combos.append({"name": f"depth_t{thin}", "spread": None, "depth": DepthConfig(thin_threshold=thin), "wmid": None, "momentum": None})

    # Weighted mid sweep
    for lean in [0.5, 1.0, 1.5]:
        combos.append({"name": f"wmid_l{lean}", "spread": None, "depth": None, "wmid": WeightedMidConfig(lean_threshold=lean), "momentum": None})

    # Momentum sweep
    for neut in [0.2, 0.3, 0.5]:
        combos.append({"name": f"mom_n{neut}", "spread": None, "depth": None, "wmid": None, "momentum": MomentumConfig(neutral_threshold=neut)})

    # Combos
    combos.append({"name": "depth+wmid", "spread": None, "depth": DepthConfig(), "wmid": WeightedMidConfig(), "momentum": None})
    combos.append({"name": "depth+mom", "spread": None, "depth": DepthConfig(), "wmid": None, "momentum": MomentumConfig()})
    combos.append({"name": "wmid+mom", "spread": None, "depth": None, "wmid": WeightedMidConfig(), "momentum": MomentumConfig()})
    combos.append({"name": "all_l2", "spread": None, "depth": DepthConfig(), "wmid": WeightedMidConfig(), "momentum": MomentumConfig()})

    # Spread + L2 combos
    combos.append({"name": "spread+depth", "spread": SpreadConfig(z_threshold=2.0), "depth": DepthConfig(), "wmid": None, "momentum": None})
    combos.append({"name": "spread+wmid", "spread": SpreadConfig(z_threshold=2.0), "depth": None, "wmid": WeightedMidConfig(), "momentum": None})
    combos.append({"name": "spread+all_l2", "spread": SpreadConfig(z_threshold=2.0), "depth": DepthConfig(), "wmid": WeightedMidConfig(), "momentum": MomentumConfig()})

    print(f"\n{'=' * 90}")
    print(f"VWAP L2 Filter Sweep (entry=1.5, stop=2.0, exp=30m)")
    print(f"Data: {start} to {end} (enriched 1s bars + L2 MBP-10)")
    print(f"Filter combos: {len(combos)}")
    print(f"{'=' * 90}\n")

    results = []
    for i, combo in enumerate(combos):
        strat = make_vwap()
        sm = SpreadMonitor(config=combo["spread"], persist=False) if combo["spread"] else None
        dm = DepthMonitor(config=combo["depth"], persist=False) if combo["depth"] else None
        wm = WeightedMidMonitor(config=combo["wmid"], persist=False) if combo["wmid"] else None
        mm = MidMomentumMonitor(config=combo["momentum"], persist=False) if combo["momentum"] else None

        config = BacktestConfig(
            strategies=[strat], start_date=start, end_date=end,
            prebuilt_bars=bars,
            prebuilt_l2=l2_df,
            spread_monitor=sm,
            depth_monitor=dm,
            weighted_mid_monitor=wm,
            mid_momentum_monitor=mm,
        )
        r = engine.run(config)
        m = r.metrics

        result = {
            "filter": combo["name"],
            "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
            "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
            "pf": m.profit_factor, "dd": m.max_drawdown_pct,
            "avg_win": m.avg_win, "avg_loss": m.avg_loss,
        }
        results.append(result)
        s = "+" if m.sharpe_ratio > 0 else "-"
        print(f"  [{i+1:3d}/{len(combos)}] {s} {combo['name']:20s}  "
              f"trades={m.total_trades:3d} WR={m.win_rate:.1%} Sharpe={m.sharpe_ratio:>7.3f} PnL=${m.net_pnl:>9,.2f}")
        sys.stdout.flush()

    # Sort and display
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"\n{'=' * 90}")
    print(f"RESULTS (by Sharpe)")
    print(f"{'=' * 90}")
    for i, r in enumerate(results):
        print(f"  {i+1:2d}. {r['filter']:20s}  trades={r['trades']} WR={r['wr']:.1%} "
              f"Sharpe={r['sharpe']:.3f} PnL=${r['pnl']:,.2f} DD={r['dd']:.1f}%")

    profitable = [r for r in results if r["pnl"] > 0]
    print(f"\n  {len(profitable)}/{len(results)} profitable")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
