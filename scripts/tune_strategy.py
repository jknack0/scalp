#!/usr/bin/env python3
"""Tier 1: Tune strategy parameters on 15-year bar data (no filters).

Finds the best strategy params using the largest available dataset.
Output is consumed by tune_l1_filters.py (Tier 2).

Uses ProcessPoolExecutor for true parallelism (bypasses GIL).
Each worker loads pre-built 1m bars once via initializer.

Usage:
    python scripts/tune_strategy.py --strategy orb
    python scripts/tune_strategy.py --strategy vwap
    python scripts/tune_strategy.py --strategy orb --start 2020-01-01 --end 2025-12-31
    python scripts/tune_strategy.py --strategy orb --workers 14
"""

import argparse
import json
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from itertools import product
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structlog

from scripts.tune_log import setup_log

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.features.feature_hub import FeatureHub
from src.strategies.orb_strategy import ORBConfig, ORBStrategy
from src.strategies.vwap_strategy import VWAPConfig, VWAPStrategy

# Per-process shared bars (set by _init_worker)
_worker_bars = None


def _init_worker(start, end, parquet_dir):
    """Initializer for each worker process — loads bars once."""
    global _worker_bars
    logging.disable(logging.CRITICAL)
    engine = BacktestEngine()
    config = BacktestConfig(
        strategies=[], start_date=start, end_date=end,
        parquet_dir=parquet_dir,
    )
    _worker_bars = engine._load_bars(config)


def _run_orb(tgt, vol, max_t, exp, vwap, start, end):
    """Run a single ORB config using worker-local bars."""
    hub = FeatureHub()
    cfg = ORBConfig(
        require_hmm_states=[], min_confidence=0.0,
        target_multiplier=tgt, volume_multiplier=vol,
        max_signal_time=max_t, expiry_minutes=exp,
        require_vwap_alignment=vwap, max_signals_per_day=1,
    )
    strat = ORBStrategy(cfg, hub)
    config = BacktestConfig(
        strategies=[strat], start_date=start, end_date=end,
        prebuilt_bars=_worker_bars,
    )
    engine = BacktestEngine()
    r = engine.run(config)
    m = r.metrics
    return {
        "tgt": tgt, "vol": vol, "max_t": max_t, "exp": exp, "vwap": vwap,
        "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
        "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
        "pf": m.profit_factor, "dd": m.max_drawdown_pct,
        "avg_win": m.avg_win, "avg_loss": m.avg_loss,
    }


def _run_vwap(entry_sd, stop_sd, pb_sd, cooldown, max_sig, expiry, start, end):
    """Run a single VWAP config using worker-local bars."""
    hub = FeatureHub()
    cfg = VWAPConfig(
        reversion_hmm_states=[], pullback_hmm_states=[], min_confidence=0.3,
        entry_sd_reversion=entry_sd, stop_sd=stop_sd,
        pullback_entry_sd=pb_sd, mode_cooldown_bars=cooldown,
        max_signals_per_day=max_sig, expiry_minutes=expiry,
    )
    strat = VWAPStrategy(cfg, hub)
    config = BacktestConfig(
        strategies=[strat], start_date=start, end_date=end,
        prebuilt_bars=_worker_bars,
    )
    engine = BacktestEngine()
    r = engine.run(config)
    m = r.metrics
    return {
        "entry_sd": entry_sd, "stop_sd": stop_sd, "expiry": expiry,
        "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
        "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
        "pf": m.profit_factor, "dd": m.max_drawdown_pct,
        "avg_win": m.avg_win, "avg_loss": m.avg_loss,
    }


def sweep(strategy, start, end, workers, parquet_dir):
    """Run parameter sweep using ProcessPoolExecutor for true parallelism."""
    if strategy == "orb":
        combos = list(product(
            [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],  # target_multiplier
            [1.5],                         # volume_multiplier (locked)
            ["10:30"],                     # max_signal_time (locked)
            [15, 30, 45, 60, 75, 90, 105, 120],  # expiry_minutes
            [True],                        # require_vwap_alignment (locked)
        ))
    else:
        combos = list(product(
            [1.0, 1.5, 2.0, 2.5],     # entry_sd
            [1.5, 2.0, 2.5, 3.0, 3.5],  # stop_sd
            [0.5],                      # pullback_sd
            [5],                        # mode_cooldown
            [4],                        # max_signals
            [30, 60],                   # expiry_minutes
        ))

    total = len(combos)
    print(f"  {total} {strategy.upper()} configs to sweep ({workers} processes)\n")

    results = []
    done = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(start, end, parquet_dir),
    ) as pool:
        if strategy == "orb":
            futures = {
                pool.submit(_run_orb, *combo, start, end): combo
                for combo in combos
            }
        else:
            futures = {
                pool.submit(_run_vwap, *combo, start, end): combo
                for combo in combos
            }

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"\n  WORKER ERROR: {e}", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)
                raise
            results.append(result)
            done += 1
            s = "+" if result["sharpe"] > 0 else "-"

            if strategy == "orb":
                print(f"  [{done:3d}/{total}] {s} tgt={result['tgt']:.2f} vol={result['vol']:.1f} "
                      f"max={result['max_t']} exp={result['exp']:2d}m vwap={str(result['vwap']):5s}  "
                      f"trades={result['trades']:3d} WR={result['wr']:.1%} "
                      f"Sharpe={result['sharpe']:>7.3f} PnL=${result['pnl']:>9,.2f}")
            else:
                print(f"  [{done:2d}/{total}] {s} entry={result['entry_sd']:.1f} "
                      f"stop={result['stop_sd']:.1f} exp={result['expiry']:2d}m  "
                      f"trades={result['trades']:3d} WR={result['wr']:.1%} "
                      f"Sharpe={result['sharpe']:>7.3f} PnL=${result['pnl']:>9,.2f}")
            sys.stdout.flush()

    return results


def main():
    parser = argparse.ArgumentParser(description="Tier 1: Strategy parameter tuning")
    parser.add_argument("--strategy", required=True, choices=["orb", "vwap"])
    parser.add_argument("--start", default="2011-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=5, help="Parallel processes (default 5)")
    parser.add_argument("--bar-freq", default="1m", help="Pre-built bar frequency (e.g. 5s, 1m)")
    parser.add_argument("--rth", action="store_true", help="Use RTH-only pre-built bars (faster, smaller)")
    args = parser.parse_args()

    _log_path = setup_log(args.strategy)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    label = args.bar_freq + ("_rth" if args.rth else "")
    parquet_dir = f"data/parquet_{label}"

    rth_tag = " RTH" if args.rth else ""
    print(f"\n{'=' * 90}")
    print(f"TIER 1: {args.strategy.upper()} Strategy Parameter Sweep (no filters)")
    print(f"Data: {start} to {end} (pre-built {args.bar_freq}{rth_tag} bars)")
    print(f"Workers: {args.workers}")
    print(f"{'=' * 90}\n")

    # Verify pre-built bars exist
    if not os.path.isdir(parquet_dir) or not any(
        d.startswith("year=") for d in os.listdir(parquet_dir)
    ):
        print(f"  FATAL: Pre-built bars not found at {parquet_dir}/")
        rth_flag = " --rth" if args.rth else ""
        print(f"  Run: python -m scripts.build_bars --freq {args.bar_freq}{rth_flag}")
        sys.exit(1)

    print(f"  Each worker will load {args.bar_freq}{rth_tag} bars on startup")
    print(f"  Spawning {args.workers} processes...\n", flush=True)

    # Suppress noisy loggers in main process
    logging.disable(logging.CRITICAL)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

    results = sweep(args.strategy, start, end, args.workers, parquet_dir)

    # Composite scoring: sharpe * sqrt(trades)
    # Rewards configs that are both good AND frequent — gives filters raw material
    MIN_TRADES = 50
    metric_keys = ("trades", "wr", "pnl", "sharpe", "sortino", "pf", "dd", "avg_win", "avg_loss", "score")

    for r in results:
        if r["trades"] >= MIN_TRADES and r["sharpe"] > 0:
            r["score"] = r["sharpe"] * math.sqrt(r["trades"])
        else:
            r["score"] = -999.0  # disqualified

    results.sort(key=lambda x: x["score"], reverse=True)

    qualified = [r for r in results if r["score"] > -999]
    disqualified = len(results) - len(qualified)

    print(f"\n{'=' * 90}")
    print(f"TOP 15 {args.strategy.upper()} CONFIGS (by composite score = Sharpe x sqrt(trades))")
    print(f"Min trades: {MIN_TRADES} | Disqualified: {disqualified}/{len(results)}")
    print(f"{'=' * 90}")
    for i, r in enumerate(results[:15]):
        params = " ".join(f"{k}={v}" for k, v in r.items() if k not in metric_keys)
        print(f"  {i+1:2d}. {params}  trades={r['trades']} WR={r['wr']:.1%} "
              f"Sharpe={r['sharpe']:.3f} score={r['score']:>7.1f} PnL=${r['pnl']:,.2f}")

    profitable = [r for r in results if r["pnl"] > 0]
    pos_sharpe = [r for r in results if r["sharpe"] > 0]
    print(f"\n  {len(profitable)}/{len(results)} profitable, {len(pos_sharpe)}/{len(results)} +Sharpe")
    print(f"  {len(qualified)}/{len(results)} qualified (>={MIN_TRADES} trades + positive Sharpe)")
    print(f"\n{'=' * 90}\n")

    # Save
    out_dir = Path("results") / args.strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now().strftime("%Y-%m-%d")
    out_path = out_dir / f"tier1_strategy_{run_date}.json"
    if out_path.exists():
        i = 2
        while (out_dir / f"tier1_strategy_{run_date}_{i}.json").exists():
            i += 1
        out_path = out_dir / f"tier1_strategy_{run_date}_{i}.json"

    output = {
        "tier": 1,
        "strategy": args.strategy,
        "date": run_date,
        "data_range": {"start": str(start), "end": str(end)},
        "bar_freq": args.bar_freq,
        "ranking_method": "composite_score = sharpe * sqrt(trades)",
        "min_trades": MIN_TRADES,
        "workers": args.workers,
        "total_configs": len(results),
        "qualified_configs": len(qualified),
        "profitable_configs": len(profitable),
        "positive_sharpe_configs": len(pos_sharpe),
        "best_config": results[0] if results else None,
        "top_15": results[:15],
        "all_results": results,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Results saved to {out_path}")
    print(f"  Next: python scripts/tune_l1_filters.py --strategy {args.strategy} --tier1 {out_path}")


if __name__ == "__main__":
    main()
