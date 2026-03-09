#!/usr/bin/env python3
"""Tier 2: Tune L1 filter combinations on 1-year L1 data.

Locks strategy params from Tier 1 output, sweeps spread/VPIN filter combos
and thresholds.

Usage:
    python scripts/tune_l1_filters.py --strategy orb --tier1 results/orb/tier1_strategy_2026-03-08.json
    python scripts/tune_l1_filters.py --strategy vwap --tier1 results/vwap/tier1_strategy_2026-03-08.json
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

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.features.feature_hub import FeatureHub
from src.filters.spread_monitor import SpreadConfig, SpreadMonitor
from src.filters.vpin_monitor import VPINConfig, VPINMonitor
from src.strategies.orb_strategy import ORBConfig, ORBStrategy
from src.strategies.vwap_strategy import VWAPConfig, VWAPStrategy

_engine = BacktestEngine()
_cached_bars = None


def load_bars_once(start, end, bar_seconds=60):
    """Load enriched 1s bars (with L1 features pre-joined), cache for reuse."""
    global _cached_bars
    if _cached_bars is not None:
        return _cached_bars

    print(f"  Loading enriched 1s bars from data/parquet_1s_enriched...")
    config = BacktestConfig(
        strategies=[], start_date=start, end_date=end,
        parquet_dir="data/parquet_1s_enriched",
    )
    _cached_bars = _engine._load_bars(config)
    print(f"  Loaded {len(_cached_bars)} bars\n")
    return _cached_bars


def make_strategy(strategy_name, params):
    """Create a strategy instance with locked params from Tier 1."""
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


# Filter combos: (name, use_vpin, use_spread, vpin_config_overrides, spread_config_overrides)
def build_filter_sweep():
    """Build all filter parameter combinations to sweep."""
    combos = []

    # No filters
    combos.append(("none", None, None))

    # VPIN only — sweep thresholds
    for trending in [0.45, 0.50, 0.55, 0.60]:
        for mean_rev in [0.30, 0.35, 0.38, 0.42]:
            combos.append((
                f"vpin_t{trending}_m{mean_rev}",
                VPINConfig(trending_threshold=trending, mean_reversion_threshold=mean_rev),
                None,
            ))

    # Spread only — sweep z-threshold
    for z in [1.5, 2.0, 2.5, 3.0]:
        combos.append((
            f"spread_z{z}",
            None,
            SpreadConfig(z_threshold=z),
        ))

    # Both — best VPIN defaults + spread sweep
    for z in [1.5, 2.0, 2.5, 3.0]:
        combos.append((
            f"both_z{z}",
            VPINConfig(),  # default thresholds
            SpreadConfig(z_threshold=z),
        ))

    return combos


def main():
    parser = argparse.ArgumentParser(description="Tier 2: L1 filter tuning")
    parser.add_argument("--strategy", required=True, choices=["orb", "vwap"])
    parser.add_argument("--tier1", required=True, help="Path to Tier 1 JSON output")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    _log_path = setup_log(args.strategy)

    # Load Tier 1 best params
    tier1 = json.loads(Path(args.tier1).read_text())
    best_params = tier1["best_config"]
    print(f"\n  Tier 1 best config: {best_params}")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    filter_sweep = build_filter_sweep()

    print(f"\n{'=' * 90}")
    print(f"TIER 2: {args.strategy.upper()} L1 Filter Sweep")
    print(f"Locked params from: {args.tier1}")
    print(f"Data: {start} to {end} (enriched 1s bars)")
    print(f"Filter combos: {len(filter_sweep)}")
    print(f"{'=' * 90}\n")

    load_bars_once(start, end)
    bars = _cached_bars

    results = []
    for i, (filter_name, vpin_cfg, spread_cfg) in enumerate(filter_sweep):
        strat = make_strategy(args.strategy, best_params)
        vm = VPINMonitor(config=vpin_cfg, persist=False) if vpin_cfg else None
        sm = SpreadMonitor(config=spread_cfg, persist=False) if spread_cfg else None

        config = BacktestConfig(
            strategies=[strat], start_date=start, end_date=end,
            prebuilt_bars=bars,
            vpin_monitor=vm,
            spread_monitor=sm,
        )
        r = _engine.run(config)
        m = r.metrics

        result = {
            "filter": filter_name,
            "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
            "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
            "pf": m.profit_factor, "dd": m.max_drawdown_pct,
            "avg_win": m.avg_win, "avg_loss": m.avg_loss,
        }
        # Store filter config details
        if vpin_cfg:
            result["vpin_trending"] = vpin_cfg.trending_threshold
            result["vpin_mean_rev"] = vpin_cfg.mean_reversion_threshold
        if spread_cfg:
            result["spread_z"] = spread_cfg.z_threshold

        results.append(result)
        s = "+" if m.sharpe_ratio > 0 else "-"
        print(f"  [{i+1:3d}/{len(filter_sweep)}] {s} {filter_name:25s}  "
              f"trades={m.total_trades:3d} WR={m.win_rate:.1%} Sharpe={m.sharpe_ratio:>7.3f} PnL=${m.net_pnl:>9,.2f}")
        sys.stdout.flush()

    # Sort and display
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"\n{'=' * 90}")
    print(f"TOP 10 FILTER CONFIGS (by Sharpe)")
    print(f"{'=' * 90}")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:2d}. {r['filter']:25s}  trades={r['trades']} WR={r['wr']:.1%} Sharpe={r['sharpe']:.3f} PnL=${r['pnl']:,.2f} DD={r['dd']:.1f}%")

    profitable = [r for r in results if r["pnl"] > 0]
    pos_sharpe = [r for r in results if r["sharpe"] > 0]
    print(f"\n  {len(profitable)}/{len(results)} profitable, {len(pos_sharpe)}/{len(results)} +Sharpe")
    print(f"\n{'=' * 90}\n")

    # Save
    out_dir = Path("results") / args.strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now().strftime("%Y-%m-%d")
    out_path = out_dir / f"tier2_l1_filters_{run_date}.json"
    if out_path.exists():
        i = 2
        while (out_dir / f"tier2_l1_filters_{run_date}_{i}.json").exists():
            i += 1
        out_path = out_dir / f"tier2_l1_filters_{run_date}_{i}.json"

    output = {
        "tier": 2,
        "strategy": args.strategy,
        "date": run_date,
        "data_range": {"start": str(start), "end": str(end)},
        "bar_seconds": 60,
        "locked_strategy_params": best_params,
        "tier1_source": args.tier1,
        "total_configs": len(results),
        "profitable_configs": len(profitable),
        "positive_sharpe_configs": len(pos_sharpe),
        "best_config": results[0] if results else None,
        "top_10": results[:10],
        "all_results": results,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Results saved to {out_path}")
    print(f"  Next: python scripts/tune_l2_filters.py --strategy {args.strategy} --tier2 {out_path}")


if __name__ == "__main__":
    main()
