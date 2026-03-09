#!/usr/bin/env python3
"""ORB + filters combined sweep on L1-enriched 1s RTH bars.

Sweeps strategy params AND filter combos together (no tiered separation).
Uses ProcessPoolExecutor for true parallelism.

Usage:
    python scripts/tune_orb_combined.py --workers 10
    python scripts/tune_orb_combined.py --workers 10 --start 2025-01-01 --end 2025-12-31
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
from src.filters.spread_monitor import SpreadConfig, SpreadMonitor
from src.filters.vpin_monitor import VPINConfig, VPINMonitor
from src.strategies.orb_strategy import ORBConfig, ORBStrategy

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


def _run_combo(tgt, exp, filter_mode, spread_z, vpin_trend, vpin_mr, start, end):
    """Run a single ORB + filter config using worker-local bars."""
    hub = FeatureHub()
    cfg = ORBConfig(
        require_hmm_states=[], min_confidence=0.0,
        target_multiplier=tgt, volume_multiplier=1.5,
        max_signal_time="10:30", expiry_minutes=exp,
        require_vwap_alignment=True, max_signals_per_day=1,
    )
    strat = ORBStrategy(cfg, hub)

    # Build filter objects based on mode
    spread_mon = None
    vpin_mon = None
    if filter_mode in ("spread", "both"):
        spread_mon = SpreadMonitor(
            config=SpreadConfig(z_threshold=spread_z),
            persist=False,
        )
    if filter_mode in ("vpin", "both"):
        vpin_mon = VPINMonitor(
            config=VPINConfig(
                trending_threshold=vpin_trend,
                mean_reversion_threshold=vpin_mr,
            ),
            persist=False,
        )

    config = BacktestConfig(
        strategies=[strat], start_date=start, end_date=end,
        prebuilt_bars=_worker_bars,
        spread_monitor=spread_mon,
        vpin_monitor=vpin_mon,
    )
    engine = BacktestEngine()
    r = engine.run(config)
    m = r.metrics
    return {
        "tgt": tgt, "exp": exp,
        "filter": filter_mode, "spread_z": spread_z,
        "vpin_trend": vpin_trend, "vpin_mr": vpin_mr,
        "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
        "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
        "pf": m.profit_factor, "dd": m.max_drawdown_pct,
        "avg_win": m.avg_win, "avg_loss": m.avg_loss,
    }


def build_combos():
    """Build all parameter combinations."""
    # Strategy params
    targets = [1.0, 2.0, 3.0]
    expiries = [30, 60]

    # Filter combos: (mode, spread_z, vpin_trending, vpin_mr)
    # vpin_mr set to 0.0 (disabled) — ORB is trending only, MR gate irrelevant
    filter_combos = [
        ("spread", 1.5, 0.0, 0.0),                   # spread only (tight)
        ("spread", 2.0, 0.0, 0.0),                   # spread only (default)
        ("spread", 2.5, 0.0, 0.0),                   # spread only (loose)
        ("vpin", 0.0, 0.45, 0.0),                    # VPIN only (tight)
        ("vpin", 0.0, 0.55, 0.0),                    # VPIN only (default)
        ("vpin", 0.0, 0.65, 0.0),                    # VPIN only (loose)
        ("both", 2.0, 0.55, 0.0),                    # both default
        ("both", 1.5, 0.45, 0.0),                    # both tight
        ("both", 2.5, 0.65, 0.0),                    # both loose
    ]

    combos = []
    for tgt, exp in product(targets, expiries):
        for fmode, sz, vt, vmr in filter_combos:
            combos.append((tgt, exp, fmode, sz, vt, vmr))

    return combos


def main():
    parser = argparse.ArgumentParser(description="ORB + filters combined sweep")
    parser.add_argument("--start", default="2025-01-01", help="Start date")
    parser.add_argument("--end", default="2025-12-31", help="End date")
    parser.add_argument("--workers", type=int, default=5, help="Parallel processes")
    parser.add_argument("--parquet-dir", default="data/parquet_1s_enriched",
                        help="Pre-built L1 enriched bars directory")
    args = parser.parse_args()

    _log_path = setup_log("orb_combined")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    combos = build_combos()
    total = len(combos)

    print(f"\n{'=' * 90}")
    print(f"ORB + FILTERS COMBINED SWEEP")
    print(f"Data: {start} to {end} (L1-enriched 1s RTH bars)")
    print(f"Strategy: 4 targets x 3 expiries = 12 strategy configs")
    print(f"Filters: 9 filter combos (spread/vpin/both x thresholds)")
    print(f"Total: {total} configs | Workers: {args.workers}")
    print(f"{'=' * 90}\n")

    # Verify bars exist
    if not os.path.isdir(args.parquet_dir) or not any(
        d.startswith("year=") for d in os.listdir(args.parquet_dir)
    ):
        print(f"  FATAL: Bars not found at {args.parquet_dir}/")
        sys.exit(1)

    print(f"  Spawning {args.workers} processes...\n", flush=True)

    logging.disable(logging.CRITICAL)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

    results = []
    done = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(start, end, args.parquet_dir),
    ) as pool:
        futures = {
            pool.submit(_run_combo, *combo, start, end): combo
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

            print(f"  [{done:3d}/{total}] {s} tgt={result['tgt']:.2f} exp={result['exp']:3d}m "
                  f"filt={result['filter']:6s}  "
                  f"trades={result['trades']:3d} WR={result['wr']:.1%} "
                  f"Sharpe={result['sharpe']:>7.3f} PnL=${result['pnl']:>9,.2f}")
            sys.stdout.flush()

    # Score and rank
    MIN_TRADES = 20  # lower threshold for 1yr data
    metric_keys = ("trades", "wr", "pnl", "sharpe", "sortino", "pf", "dd",
                   "avg_win", "avg_loss", "score")

    for r in results:
        if r["trades"] >= MIN_TRADES and r["sharpe"] > 0:
            r["score"] = r["sharpe"] * math.sqrt(r["trades"])
        else:
            r["score"] = -999.0

    results.sort(key=lambda x: x["score"], reverse=True)

    qualified = [r for r in results if r["score"] > -999]
    disqualified = len(results) - len(qualified)

    print(f"\n{'=' * 90}")
    print(f"TOP 20 ORB+FILTER CONFIGS (score = Sharpe x sqrt(trades))")
    print(f"Min trades: {MIN_TRADES} | Qualified: {len(qualified)}/{len(results)}")
    print(f"{'=' * 90}")
    for i, r in enumerate(results[:20]):
        filt_desc = r["filter"]
        if r["filter"] == "spread":
            filt_desc = f"spread(z={r['spread_z']})"
        elif r["filter"] == "vpin":
            filt_desc = f"vpin(t={r['vpin_trend']},mr={r['vpin_mr']})"
        elif r["filter"] == "both":
            filt_desc = f"both(z={r['spread_z']},t={r['vpin_trend']},mr={r['vpin_mr']})"

        print(f"  {i+1:2d}. tgt={r['tgt']:.2f} exp={r['exp']:3d}m {filt_desc:35s} "
              f"trades={r['trades']:3d} WR={r['wr']:.1%} "
              f"Sharpe={r['sharpe']:>7.3f} score={r['score']:>7.1f} PnL=${r['pnl']:>9,.2f}")

    profitable = [r for r in results if r["pnl"] > 0]
    pos_sharpe = [r for r in results if r["sharpe"] > 0]
    print(f"\n  {len(profitable)}/{len(results)} profitable, {len(pos_sharpe)}/{len(results)} +Sharpe")
    print(f"  {len(qualified)}/{len(results)} qualified (>={MIN_TRADES} trades + positive Sharpe)")

    # Filter breakdown
    print(f"\n  --- By filter mode ---")
    for mode in ["none", "spread", "vpin", "both"]:
        mode_results = [r for r in results if r["filter"] == mode]
        mode_pos = [r for r in mode_results if r["sharpe"] > 0]
        mode_profit = [r for r in mode_results if r["pnl"] > 0]
        best = max(mode_results, key=lambda x: x["sharpe"]) if mode_results else None
        print(f"  {mode:6s}: {len(mode_profit)}/{len(mode_results)} profitable, "
              f"{len(mode_pos)}/{len(mode_results)} +Sharpe, "
              f"best Sharpe={best['sharpe']:.3f}" if best else "")

    print(f"\n{'=' * 90}\n")

    # Save
    out_dir = Path("results") / "orb"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now().strftime("%Y-%m-%d")
    out_path = out_dir / f"combined_sweep_{run_date}.json"
    if out_path.exists():
        i = 2
        while (out_dir / f"combined_sweep_{run_date}_{i}.json").exists():
            i += 1
        out_path = out_dir / f"combined_sweep_{run_date}_{i}.json"

    output = {
        "type": "combined_sweep",
        "strategy": "orb",
        "date": run_date,
        "data_range": {"start": str(start), "end": str(end)},
        "parquet_dir": args.parquet_dir,
        "locked_params": {
            "volume_multiplier": 1.5,
            "max_signal_time": "10:30",
            "require_vwap_alignment": True,
        },
        "ranking_method": "composite_score = sharpe * sqrt(trades)",
        "min_trades": MIN_TRADES,
        "total_configs": len(results),
        "qualified_configs": len(qualified),
        "profitable_configs": len(profitable),
        "best_config": results[0] if results else None,
        "top_20": results[:20],
        "all_results": results,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
