#!/usr/bin/env python3
"""Tier 2: Tune filter thresholds on 1-year enriched data.

Locks strategy params from Tier 1 output, sweeps spread z-score thresholds
via FilterEngine.

Usage:
    python scripts/tune/filters.py --strategy orb --tier1 results/orb/tier1_strategy_2026-03-08.json
    python scripts/tune/filters.py --strategy vwap --tier1 results/vwap/tier1_strategy_2026-03-08.json
"""

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.tune.log import setup_log

import logging
logging.disable(logging.CRITICAL)
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.filters.filter_engine import FilterEngine
from src.signals.signal_bundle import SignalEngine
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy

_engine = BacktestEngine()
_cached_bars = None


def load_bars_once(start, end):
    """Load enriched 1s bars, cache for reuse."""
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
    if strategy_name == "orb":
        return ORBStrategy(config={
            "strategy": {"strategy_id": "orb", "max_signals_per_day": 1},
            "orb": {
                "target_multiplier": params["tgt"],
                "volume_multiplier": params["vol"],
                "max_signal_time": params["max_t"],
                "expiry_minutes": params["exp"],
                "require_vwap_alignment": params["vwap"],
            },
            "exit": {"time_stop_minutes": params["exp"]},
        })
    elif strategy_name == "vwap":
        expiry = params["expiry"]
        return VWAPReversionStrategy(config={
            "strategy": {"strategy_id": "vwap_reversion", "max_signals_per_day": 4},
            "vwap": {
                "entry_sd_reversion": params["entry_sd"],
                "stop_sd": params["stop_sd"],
                "pullback_entry_sd": 0.5,
                "mode_cooldown_bars": 5,
                "expiry_minutes": expiry,
            },
            "exit": {"target": "vwap", "stop_ticks": 8, "time_stop_minutes": expiry},
        })
    raise ValueError(f"Unknown strategy: {strategy_name}")


def build_filter_sweep():
    """Build filter parameter combinations to sweep.

    Sweeps spread z-score and relative_volume thresholds on 5m bars.
    Also tests multi-TF confirmation (5m + 15m).
    """
    combos = []

    # No filters (baseline)
    combos.append(("none", []))

    # Single-TF spread z-score thresholds (5m)
    for z in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        combos.append((f"spread_z{z}_5m", [
            {"signal": "spread", "expr": f"< {z}", "seq": 1, "bar": "5m"},
        ]))

    # Multi-TF spread confirmation (5m + 15m)
    for z in [2.0, 3.0]:
        combos.append((f"spread_z{z}_5m+15m", [
            {"signal": "spread", "expr": f"< {z}", "seq": 1, "bar": "5m"},
            {"signal": "spread", "expr": f"< {z}", "seq": 2, "bar": "15m"},
        ]))

    # Relative volume thresholds (5m)
    for rv in [0.5, 1.0, 1.5]:
        combos.append((f"rvol_{rv}_5m", [
            {"signal": "relative_volume", "expr": f"> {rv}", "seq": 1, "bar": "5m"},
        ]))

    # Combined spread + rvol (5m)
    combos.append(("spread_z2+rvol1.5_5m", [
        {"signal": "spread", "expr": "< 2.0", "seq": 1, "bar": "5m"},
        {"signal": "relative_volume", "expr": "> 1.5", "seq": 1, "bar": "5m"},
    ]))

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
    print(f"TIER 2: {args.strategy.upper()} L1 Filter Sweep (spread z-score via FilterEngine)")
    print(f"Locked params from: {args.tier1}")
    print(f"Data: {start} to {end} (enriched 1s bars)")
    print(f"Filter combos: {len(filter_sweep)}")
    print(f"{'=' * 90}\n")

    load_bars_once(start, end)
    bars = _cached_bars

    # Shared signal engine
    sig_engine = SignalEngine(["atr", "vwap_session", "relative_volume", "spread"])

    results = []
    for i, (filter_name, filter_dict) in enumerate(filter_sweep):
        strat = make_strategy(args.strategy, best_params)
        fe = FilterEngine.from_list(filter_dict) if filter_dict else None

        config = BacktestConfig(
            strategies=[strat], start_date=start, end_date=end,
            prebuilt_bars=bars,
            signal_engine=sig_engine,
            filter_engine=fe,
        )
        r = _engine.run(config)
        m = r.metrics

        result = {
            "filter": filter_name,
            "filter_rules": filter_dict,
            "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
            "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
            "pf": m.profit_factor, "dd": m.max_drawdown_pct,
            "avg_win": m.avg_win, "avg_loss": m.avg_loss,
        }
        results.append(result)
        s = "+" if m.sharpe_ratio > 0 else "-"
        print(f"  [{i+1:3d}/{len(filter_sweep)}] {s} {filter_name:25s}  "
              f"trades={m.total_trades:3d} WR={m.win_rate:.1%} Sharpe={m.sharpe_ratio:>7.3f} PnL=${m.net_pnl:>9,.2f}")
        sys.stdout.flush()

    # Sort and display
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"\n{'=' * 90}")
    print(f"FILTER RESULTS (by Sharpe)")
    print(f"{'=' * 90}")
    for i, r in enumerate(results):
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


if __name__ == "__main__":
    main()
