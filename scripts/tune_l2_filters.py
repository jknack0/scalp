#!/usr/bin/env python3
"""Tier 3: Tune L2 filter combinations on 3-month L2 data.

Locks strategy params + L1 filters from Tier 2 output, sweeps L2 filter
on/off combinations and key thresholds.

Prerequisites:
    python scripts/convert_l2_parquet.py  # convert DBN to Parquet first

Usage:
    python scripts/tune_l2_filters.py --strategy orb --tier2 results/orb/tier2_l1_filters_2026-03-08.json
"""

import argparse
import json
import os
import sys
from datetime import date, datetime
from itertools import product
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.tune_log import setup_log

import logging
logging.disable(logging.CRITICAL)
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

import polars as pl

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.features.feature_hub import FeatureHub
from src.filters.depth_monitor import DepthConfig, DepthMonitor
from src.filters.mid_momentum import MidMomentumMonitor, MomentumConfig
from src.filters.spread_monitor import SpreadConfig, SpreadMonitor
from src.filters.vpin_monitor import VPINConfig, VPINMonitor
from src.filters.weighted_mid import WeightedMidConfig, WeightedMidMonitor
from src.strategies.orb_strategy import ORBConfig, ORBStrategy
from src.strategies.vwap_strategy import VWAPConfig, VWAPStrategy

_engine = BacktestEngine()
_cached_bars = None
_cached_l2 = None


def load_bars_once(start, end, bar_seconds=60):
    """Load and aggregate L1 data once, cache for reuse."""
    global _cached_bars
    if _cached_bars is not None:
        return _cached_bars

    print(f"  Loading L1 data and aggregating to {bar_seconds}s bars...")
    config = BacktestConfig(
        strategies=[], start_date=start, end_date=end,
        l1_parquet_dir="data/l1", l1_bar_seconds=bar_seconds,
    )
    _cached_bars = _engine._load_l1_bars(config)
    print(f"  Loaded {len(_cached_bars)} bars\n")
    return _cached_bars


def load_l2_once(l2_dir="data/l2_parquet"):
    """Load L2 Parquet data once, cache for reuse."""
    global _cached_l2
    if _cached_l2 is not None:
        return _cached_l2

    print(f"  Loading L2 snapshots from {l2_dir}...")
    frames = []
    if os.path.isdir(l2_dir):
        for f in sorted(os.listdir(l2_dir)):
            if f.endswith(".parquet"):
                frames.append(pl.read_parquet(os.path.join(l2_dir, f)))

    if not frames:
        print(f"  WARNING: No L2 parquet files found in {l2_dir}")
        print(f"  Run: python scripts/convert_l2_parquet.py")
        return None

    _cached_l2 = pl.concat(frames).sort("timestamp")
    print(f"  Loaded {len(_cached_l2)} L2 snapshots\n")
    return _cached_l2


def make_strategy(strategy_name, params):
    """Create a strategy instance with locked params."""
    hub = FeatureHub()
    if strategy_name == "orb":
        cfg = ORBConfig(
            require_hmm_states=[], min_confidence=0.0,
            target_multiplier=params["tgt"],
            volume_multiplier=params["vol"],
            max_signal_time=params["max_t"],
            expiry_minutes=params["exp"],
            require_vwap_alignment=params["vwap"],
            max_signals_per_day=1,
        )
        return ORBStrategy(cfg, hub)
    elif strategy_name == "vwap":
        cfg = VWAPConfig(
            reversion_hmm_states=[], pullback_hmm_states=[], min_confidence=0.3,
            entry_sd_reversion=params["entry_sd"],
            stop_sd=params["stop_sd"],
            pullback_entry_sd=0.5,
            mode_cooldown_bars=5,
            max_signals_per_day=4,
            expiry_minutes=params["expiry"],
        )
        return VWAPStrategy(cfg, hub)
    raise ValueError(f"Unknown strategy: {strategy_name}")


def make_l1_filters(best_l1):
    """Recreate L1 filters from Tier 2 best config."""
    vm, sm = None, None
    filter_name = best_l1.get("filter", "none")

    if "vpin" in filter_name or "both" in filter_name:
        vpin_kwargs = {}
        if "vpin_trending" in best_l1:
            vpin_kwargs["trending_threshold"] = best_l1["vpin_trending"]
        if "vpin_mean_rev" in best_l1:
            vpin_kwargs["mean_reversion_threshold"] = best_l1["vpin_mean_rev"]
        vm = VPINMonitor(config=VPINConfig(**vpin_kwargs), persist=False)

    if "spread" in filter_name or "both" in filter_name:
        spread_kwargs = {}
        if "spread_z" in best_l1:
            spread_kwargs["z_threshold"] = best_l1["spread_z"]
        sm = SpreadMonitor(config=SpreadConfig(**spread_kwargs), persist=False)

    return vm, sm


def build_l2_filter_sweep():
    """Build L2 filter on/off combinations with threshold variations.

    Filters available for L2 gating:
    - depth_monitor: blocks when liquidity thinning on trade side
    - weighted_mid: blocks when lean contradicts signal direction
    - mid_momentum: blocks when momentum contradicts signal direction

    We sweep on/off combos and key thresholds.
    """
    combos = []

    # No L2 filters (baseline)
    combos.append({
        "name": "no_l2",
        "depth": None, "wmid": None, "momentum": None,
    })

    # Individual filters with threshold sweep
    for thin in [0.4, 0.6, 0.8]:
        combos.append({
            "name": f"depth_t{thin}",
            "depth": DepthConfig(thin_threshold=thin),
            "wmid": None, "momentum": None,
        })

    for lean_t in [0.5, 1.0, 1.5]:
        combos.append({
            "name": f"wmid_l{lean_t}",
            "depth": None,
            "wmid": WeightedMidConfig(lean_threshold=lean_t),
            "momentum": None,
        })

    for neut in [0.2, 0.3, 0.5]:
        combos.append({
            "name": f"mom_n{neut}",
            "depth": None, "wmid": None,
            "momentum": MomentumConfig(neutral_threshold=neut),
        })

    # Pairs
    combos.append({
        "name": "depth+wmid",
        "depth": DepthConfig(), "wmid": WeightedMidConfig(), "momentum": None,
    })
    combos.append({
        "name": "depth+mom",
        "depth": DepthConfig(), "wmid": None, "momentum": MomentumConfig(),
    })
    combos.append({
        "name": "wmid+mom",
        "depth": None, "wmid": WeightedMidConfig(), "momentum": MomentumConfig(),
    })

    # All three with defaults
    combos.append({
        "name": "all_l2",
        "depth": DepthConfig(), "wmid": WeightedMidConfig(), "momentum": MomentumConfig(),
    })

    return combos


def main():
    parser = argparse.ArgumentParser(description="Tier 3: L2 filter tuning")
    parser.add_argument("--strategy", required=True, choices=["orb", "vwap"])
    parser.add_argument("--tier2", required=True, help="Path to Tier 2 JSON output")
    parser.add_argument("--l2-dir", default="data/l2_parquet", help="L2 Parquet directory")
    args = parser.parse_args()

    _log_path = setup_log(args.strategy)

    # Load Tier 2 results
    tier2 = json.loads(Path(args.tier2).read_text())
    strategy_params = tier2["locked_strategy_params"]
    best_l1 = tier2["best_config"]

    print(f"\n  Strategy params (from Tier 1): {strategy_params}")
    print(f"  L1 filter (from Tier 2): {best_l1['filter']}")

    # Determine date range from L2 data availability
    start = date(2025, 9, 1)
    end = date(2025, 11, 30)

    l2_sweep = build_l2_filter_sweep()

    print(f"\n{'=' * 90}")
    print(f"TIER 3: {args.strategy.upper()} L2 Filter Sweep")
    print(f"Locked strategy + L1 filters from: {args.tier2}")
    print(f"Data: {start} to {end} (L1 bars + L2 snapshots)")
    print(f"L2 filter combos: {len(l2_sweep)}")
    print(f"{'=' * 90}\n")

    bars = load_bars_once(start, end)
    l2_df = load_l2_once(args.l2_dir)

    if l2_df is None:
        print("  ERROR: No L2 data available. Run convert_l2_parquet.py first.")
        return

    results = []
    for i, combo in enumerate(l2_sweep):
        strat = make_strategy(args.strategy, strategy_params)
        vm, sm = make_l1_filters(best_l1)

        # Build L2 filter instances
        depth_mon = DepthMonitor(config=combo["depth"], persist=False) if combo["depth"] else None
        wmid_mon = WeightedMidMonitor(config=combo["wmid"], persist=False) if combo["wmid"] else None
        mom_mon = MidMomentumMonitor(config=combo["momentum"], persist=False) if combo["momentum"] else None

        config = BacktestConfig(
            strategies=[strat], start_date=start, end_date=end,
            prebuilt_bars=bars,
            prebuilt_l2=l2_df,
            vpin_monitor=vm,
            spread_monitor=sm,
            depth_monitor=depth_mon,
            weighted_mid_monitor=wmid_mon,
            mid_momentum_monitor=mom_mon,
        )
        r = _engine.run(config)
        m = r.metrics

        result = {
            "l2_filter": combo["name"],
            "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
            "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
            "pf": m.profit_factor, "dd": m.max_drawdown_pct,
            "avg_win": m.avg_win, "avg_loss": m.avg_loss,
        }
        results.append(result)
        s = "+" if m.sharpe_ratio > 0 else "-"
        print(f"  [{i+1:3d}/{len(l2_sweep)}] {s} {combo['name']:20s}  "
              f"trades={m.total_trades:3d} WR={m.win_rate:.1%} Sharpe={m.sharpe_ratio:>7.3f} PnL=${m.net_pnl:>9,.2f}")
        sys.stdout.flush()

    # Sort and display
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"\n{'=' * 90}")
    print(f"L2 FILTER RESULTS (by Sharpe)")
    print(f"{'=' * 90}")
    for i, r in enumerate(results):
        print(f"  {i+1:2d}. {r['l2_filter']:20s}  trades={r['trades']} WR={r['wr']:.1%} "
              f"Sharpe={r['sharpe']:.3f} PnL=${r['pnl']:,.2f} DD={r['dd']:.1f}%")

    profitable = [r for r in results if r["pnl"] > 0]
    pos_sharpe = [r for r in results if r["sharpe"] > 0]
    print(f"\n  {len(profitable)}/{len(results)} profitable, {len(pos_sharpe)}/{len(results)} +Sharpe")
    print(f"\n{'=' * 90}\n")

    # Save
    out_dir = Path("results") / args.strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now().strftime("%Y-%m-%d")
    out_path = out_dir / f"tier3_l2_filters_{run_date}.json"
    if out_path.exists():
        i = 2
        while (out_dir / f"tier3_l2_filters_{run_date}_{i}.json").exists():
            i += 1
        out_path = out_dir / f"tier3_l2_filters_{run_date}_{i}.json"

    output = {
        "tier": 3,
        "strategy": args.strategy,
        "date": run_date,
        "data_range": {"start": str(start), "end": str(end)},
        "locked_strategy_params": strategy_params,
        "locked_l1_filter": best_l1["filter"],
        "tier2_source": args.tier2,
        "total_configs": len(results),
        "profitable_configs": len(profitable),
        "positive_sharpe_configs": len(pos_sharpe),
        "best_config": results[0] if results else None,
        "all_results": results,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
