#!/usr/bin/env python3
"""Donchian Breakout Trend: optimize on 2019-2024, backtest on 2025.

Tier 1 sweep over entry + exit params on in-sample period, then runs
top configs out-of-sample. Parallelized via multiprocessing.
Results saved to results/donchian_breakout_trend/.

Usage:
    python -u scripts/backtest/donchian_breakout_optimize.py
    python -u scripts/backtest/donchian_breakout_optimize.py --no-regime
    python -u scripts/backtest/donchian_breakout_optimize.py --top 10
    python -u scripts/backtest/donchian_breakout_optimize.py --workers 8
"""

import argparse
import json
import logging
import math
import os
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, time
from itertools import product
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.disable(logging.CRITICAL)
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

import polars as pl

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.data.bar_cache import BarCache
from src.exits.exit_engine import ExitEngine
from src.signals.vectorized import enrich_bars
from src.strategies.donchian_breakout_trend import DonchianBreakoutTrendStrategy


BASE_SIGNALS = ["donchian_channel", "adx", "relative_volume", "atr", "session_time"]
RESULTS_DIR = Path("results/donchian_breakout_trend")


def load_enriched_bars(use_regime: bool = True) -> tuple[pl.DataFrame, list[str]]:
    """Load 5m bars enriched with vectorized signals. Caches to disk."""
    signal_names = BASE_SIGNALS + (["regime_v2"] if use_regime else [])
    cache_name = BarCache.enriched_name("5m", signal_names)

    cached = BarCache.load(cache_name)
    if cached is not None:
        print(f"  [Cache HIT] {cache_name} ({len(cached):,} rows)")
        return cached, signal_names

    print("  Building enriched bars (first run -- will be cached)...")
    t0 = _time.perf_counter()

    engine = BacktestEngine()
    config = BacktestConfig(
        strategies=[], start_date=date(2019, 1, 1), end_date=date(2026, 1, 1),
        parquet_dir="data/parquet_5m", bar_type="5m",
    )
    bars_df = engine._load_bars(config)
    if bars_df.is_empty():
        print("  FATAL: No bars loaded")
        sys.exit(1)

    bars_df = bars_df.with_columns(
        pl.col("timestamp")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("US/Eastern")
            .alias("_et_ts"),
    )
    bars_df = bars_df.filter(
        (pl.col("_et_ts").dt.time() >= time(9, 30))
        & (pl.col("_et_ts").dt.time() < time(16, 0))
    )
    bars_df = bars_df.with_columns(
        pl.col("_et_ts").dt.date().alias("_bar_date"),
    )

    enriched = enrich_bars(bars_df, signal_names)
    save_df = enriched.drop(["_et_ts", "_bar_date"]).with_columns(
        pl.col("timestamp").dt.replace_time_zone(None).alias("timestamp"),
    )
    BarCache.save(cache_name, save_df)

    print(f"  Enriched {len(enriched):,} bars in {_time.perf_counter() - t0:.1f}s")
    return save_df, signal_names


def run_single(bars: pl.DataFrame, signal_names: list[str],
               start: date, end: date, use_regime: bool = True, **overrides) -> dict:
    """Run a single backtest config using pre-computed signals."""
    session_start_min = overrides.get("session_start", 630)   # 10:30 AM
    session_end_min = overrides.get("session_end", 900)       # 3:00 PM

    filters = [
        {"signal": "session_time", "expr": f">= {session_start_min}"},
        {"signal": "session_time", "expr": f"<= {session_end_min}"},
        {"signal": "donchian_channel", "expr": "passes"},
        {"signal": "adx", "expr": f">= {overrides.get('adx_min', 22.0)}"},
        {"signal": "relative_volume", "expr": f">= {overrides.get('rvol_min', 1.5)}"},
    ]

    if use_regime:
        filters.append({"signal": "regime_v2", "expr": "passes"})
        filters.append({"signal": "regime_v2", "expr": "== 0"})  # TRENDING

    cfg = {
        "strategy": {
            "strategy_id": "donchian_breakout_trend",
            "max_signals_per_day": overrides.get("max_signals", 3),
        },
        "exit": {"time_stop_minutes": overrides.get("time_stop_min", 45)},
        "donchian": {
            "target_atr": overrides.get("target_atr", 3.5),
            "stop_atr": overrides.get("stop_atr", 1.75),
            "width_min": overrides.get("width_min", 2.0),
            "width_max": overrides.get("width_max", 30.0),
        },
        "filters": filters,
    }

    strat = DonchianBreakoutTrendStrategy(config=cfg)

    filtered_bars = bars.filter(
        (pl.col("timestamp") >= pl.lit(datetime(start.year, start.month, start.day)))
        & (pl.col("timestamp") < pl.lit(datetime(end.year, end.month, end.day)))
    )

    time_stop_bars = overrides.get("time_stop_bars", 9)
    adx_kill = overrides.get("adx_kill", 18.0)

    exits = [
        {"type": "bracket_target", "enabled": True},
        {"type": "bracket_stop", "enabled": True},
        {"type": "time_stop", "enabled": True, "max_bars": time_stop_bars},
        {"type": "price_vs_signal_exit", "enabled": True,
         "signal": "donchian_channel", "long_field": "exit_lower", "short_field": "exit_upper"},
        {"type": "signal_bound_exit", "enabled": True,
         "signal": "adx", "lower_bound": adx_kill},
    ]
    exit_engine = ExitEngine.from_list(exits)

    bt_config = BacktestConfig(
        strategies=[strat],
        start_date=start,
        end_date=end,
        prebuilt_bars=filtered_bars,
        enriched_signal_names=signal_names,
        use_rth_bars=True,
        exit_engine=exit_engine,
    )
    engine = BacktestEngine()
    r = engine.run(bt_config)
    m = r.metrics
    return {
        **overrides,
        "trades": m.total_trades,
        "wr": m.win_rate,
        "pnl": m.net_pnl,
        "sharpe": m.sharpe_ratio,
        "sortino": m.sortino_ratio,
        "pf": m.profit_factor,
        "dd": m.max_drawdown_pct,
        "avg_win": m.avg_win,
        "avg_loss": m.avg_loss,
    }


# ── Worker function for multiprocessing ──────────────────────────────────

def _worker_init():
    """Suppress logging in worker processes."""
    logging.disable(logging.CRITICAL)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))


def _run_batch(batch_args: tuple) -> list[dict]:
    """Run a batch of configs in a worker process."""
    bars_path, signal_names, start, end, use_regime, param_list, combo_keys = batch_args
    _worker_init()
    bars = pl.read_parquet(bars_path)
    results = []
    for combo in param_list:
        params = dict(zip(combo_keys, combo))
        r = run_single(bars, signal_names, start, end, use_regime=use_regime, **params)
        results.append(r)
    return results


def format_row(r: dict) -> str:
    return (f"{r.get('target_atr',3.5):>4.1f} {r.get('stop_atr',1.75):>4.2f} "
            f"{r.get('adx_min',22):>3.0f} {r.get('rvol_min',1.5):>3.1f} "
            f"{r.get('time_stop_bars',9):>3} {r.get('adx_kill',18):>3.0f} "
            f"{r.get('session_start',630):>3} {r.get('session_end',900):>3} | "
            f"{r['trades']:>6} {r['wr']:>5.1%} ${r['pnl']:>7,.0f} "
            f"{r['sharpe']:>7.2f} {r['pf']:>5.2f} {r['dd']:>4.1f}% | "
            f"{r['score']:>6.1f}")


HEADER = (f"{'tgt':>4} {'stp':>4} {'adx':>3} {'rvl':>3} {'tBr':>3} {'aKl':>3} "
          f"{'sS':>3} {'sE':>3} | "
          f"{'trades':>6} {'WR':>6} {'PnL':>8} {'Sharpe':>7} {'PF':>5} {'DD':>5} | {'score':>6}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-regime", action="store_true", help="Disable regime_v2 gating")
    parser.add_argument("--top", type=int, default=10, help="Number of top configs to backtest OOS")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0=auto, 1=serial)")
    args = parser.parse_args()

    use_regime = not args.no_regime
    regime_label = " + regime_v2 TRENDING" if use_regime else " (no regime gate)"

    n_workers = args.workers if args.workers > 0 else max(1, os.cpu_count() - 1)

    IS_START = date(2019, 1, 1)
    IS_END = date(2025, 1, 1)
    OOS_START = date(2025, 1, 1)
    OOS_END = date(2026, 1, 1)

    print(f"\n{'='*78}")
    print(f"  DONCHIAN BREAKOUT TREND -- OPTIMIZE + BACKTEST")
    print(f"  In-Sample:      {IS_START} to {IS_END} (6 years)")
    print(f"  Out-of-Sample:  {OOS_START} to {OOS_END} (1 year)")
    print(f"  Regime gate:    {regime_label}")
    print(f"  Workers:        {n_workers}")
    print(f"{'='*78}\n")

    # ── Load enriched bars ──────────────────────────────────────────────
    bars, signal_names = load_enriched_bars(use_regime)
    bars_path = BarCache.path(BarCache.enriched_name("5m", signal_names))
    print()

    # ── Phase 1: In-sample sweep (2019-2024) ────────────────────────────
    print(f"{'='*78}")
    print(f"  PHASE 1: IN-SAMPLE PARAMETER SWEEP (2019-2024)")
    print(f"{'='*78}\n")

    # Entry params (from guide Tier 1 ranges)
    target_atrs = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    stop_atrs = [1.25, 1.5, 1.75, 2.0, 2.5]
    adx_mins = [18.0, 22.0, 26.0, 30.0]
    rvol_mins = [0.8, 1.2, 1.5, 2.0]

    # Exit params
    time_stop_bars_list = [6, 12, 18, 30]       # 30/60/90/150 min in 5m bars
    adx_kills = [12.0, 15.0, 18.0]

    # Session (minutes since midnight)
    session_starts = [600, 630]                  # 10:00, 10:30
    session_ends = [840, 900]                    # 2:00, 3:00

    combo_keys = ["target_atr", "stop_atr", "adx_min", "rvol_min",
                  "time_stop_bars", "adx_kill",
                  "session_start", "session_end"]
    combos = list(product(target_atrs, stop_atrs, adx_mins, rvol_mins,
                          time_stop_bars_list, adx_kills,
                          session_starts, session_ends))

    print(f"  {len(combos):,} configurations to evaluate\n")

    t0 = _time.perf_counter()

    if n_workers <= 1:
        # Serial execution
        results = []
        for i, combo in enumerate(combos):
            params = dict(zip(combo_keys, combo))
            r = run_single(bars, signal_names, IS_START, IS_END, use_regime=use_regime, **params)
            results.append(r)
            if (i + 1) % 500 == 0:
                elapsed = _time.perf_counter() - t0
                eta = elapsed / (i + 1) * (len(combos) - i - 1)
                print(f"  ... {i+1:,}/{len(combos):,} done ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    else:
        # Parallel execution: split combos into batches
        batch_size = max(50, len(combos) // (n_workers * 4))
        batches = []
        for i in range(0, len(combos), batch_size):
            chunk = combos[i:i + batch_size]
            batches.append((str(bars_path), signal_names, IS_START, IS_END,
                           use_regime, chunk, combo_keys))

        print(f"  Running {len(batches)} batches across {n_workers} workers "
              f"(batch_size={batch_size})...\n")

        results = []
        completed = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_batch, batch): i
                       for i, batch in enumerate(batches)}
            for future in as_completed(futures):
                batch_results = future.result()
                results.extend(batch_results)
                completed += len(batch_results)
                if completed % 1000 < batch_size or completed == len(combos):
                    elapsed = _time.perf_counter() - t0
                    eta = elapsed / completed * (len(combos) - completed) if completed else 0
                    print(f"  ... {completed:,}/{len(combos):,} done "
                          f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    total_time = _time.perf_counter() - t0

    # Score and rank
    for r in results:
        r["score"] = r["sharpe"] * math.sqrt(max(r["trades"], 1)) if r["sharpe"] > 0 else 0

    results.sort(key=lambda x: x["score"], reverse=True)

    # Filter viable configs
    viable = [r for r in results if r["sharpe"] >= 0.5 and r["pf"] >= 1.1 and r["trades"] >= 20]
    profitable = [r for r in results if r["pnl"] > 0]

    print(f"\n  Sweep completed in {total_time:.0f}s ({total_time/len(combos)*1000:.1f}ms/config)")
    print(f"  Total configs:      {len(results):,}")
    print(f"  Profitable:         {len(profitable):,} ({len(profitable)/len(results):.1%})")
    print(f"  Viable (Sh>=0.5):   {len(viable):,} ({len(viable)/len(results):.1%})")

    print(f"\n  TOP 20 IN-SAMPLE CONFIGS (ranked by Sharpe * sqrt(trades)):")
    print(f"  {HEADER}")
    print(f"  {'-'*len(HEADER)}")
    for r in results[:20]:
        print(f"  {format_row(r)}")

    if not viable and not profitable:
        print("\n  No viable configs found (Sharpe >= 0.5, PF >= 1.1, trades >= 20)")
        print("  Showing top 5 by raw PnL for diagnostics:")
        by_pnl = sorted(results, key=lambda x: x["pnl"], reverse=True)
        for r in by_pnl[:5]:
            r["score"] = r["sharpe"] * math.sqrt(max(r["trades"], 1)) if r["sharpe"] > 0 else 0
            print(f"  {format_row(r)}")

        # Save results anyway for analysis
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        is_file = RESULTS_DIR / f"tier1_is_{ts}.json"
        is_save = []
        for r in results[:50]:
            is_save.append({k: v for k, v in r.items()
                           if not isinstance(v, float) or not math.isnan(v)})
        with open(is_file, "w") as f:
            json.dump(is_save, f, indent=2, default=str)
        print(f"\n  Saved top 50 IS results -> {is_file}")
        return

    # Use viable if available, else profitable
    candidates = viable if viable else sorted(profitable, key=lambda x: x.get("score", 0), reverse=True)

    # ── Phase 2: Out-of-sample backtest (2025) ──────────────────────────
    n_oos = min(args.top, len(candidates))
    print(f"\n{'='*78}")
    print(f"  PHASE 2: OUT-OF-SAMPLE BACKTEST (2025) -- TOP {n_oos} CONFIGS")
    print(f"{'='*78}\n")

    oos_results = []
    for rank, is_result in enumerate(candidates[:n_oos], 1):
        params = {k: is_result[k] for k in combo_keys if k in is_result}
        oos = run_single(bars, signal_names, OOS_START, OOS_END, use_regime=use_regime, **params)
        oos["score"] = oos["sharpe"] * math.sqrt(max(oos["trades"], 1)) if oos["sharpe"] > 0 else 0
        oos["is_sharpe"] = is_result["sharpe"]
        oos["is_pnl"] = is_result["pnl"]
        oos["is_trades"] = is_result["trades"]
        oos["is_score"] = is_result["score"]
        oos_results.append(oos)

        print(f"  Config #{rank}:")
        print(f"    Params: tgt={params.get('target_atr')}, stp={params.get('stop_atr')}, "
              f"adx>={params.get('adx_min')}, rvol>={params.get('rvol_min')}, "
              f"tBars={params.get('time_stop_bars')}, aKill={params.get('adx_kill')}, "
              f"session={params.get('session_start')}-{params.get('session_end')}")
        print(f"    IN-SAMPLE  (2019-2024):  {is_result['trades']} trades, "
              f"Sharpe={is_result['sharpe']:.2f}, PnL=${is_result['pnl']:,.0f}, "
              f"WR={is_result['wr']:.1%}, PF={is_result['pf']:.2f}, DD={is_result['dd']:.1f}%")
        print(f"    OUT-SAMPLE (2025):       {oos['trades']} trades, "
              f"Sharpe={oos['sharpe']:.2f}, PnL=${oos['pnl']:,.0f}, "
              f"WR={oos['wr']:.1%}, PF={oos['pf']:.2f}, DD={oos['dd']:.1f}%")

        if is_result["sharpe"] > 0:
            degradation = 1.0 - (oos["sharpe"] / is_result["sharpe"]) if is_result["sharpe"] > 0 else 1.0
            flag = " << OVERFIT" if degradation > 0.3 else (" << ROBUST" if degradation < 0.15 else "")
            print(f"    Sharpe degradation:      {degradation:.1%}{flag}")
        print()

    # ── Save results ────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    is_file = RESULTS_DIR / f"tier1_is_{ts}.json"
    is_save = []
    for r in results[:50]:
        is_save.append({k: v for k, v in r.items()
                       if not isinstance(v, float) or not math.isnan(v)})
    with open(is_file, "w") as f:
        json.dump(is_save, f, indent=2, default=str)
    print(f"  Saved top 50 IS results -> {is_file}")

    oos_file = RESULTS_DIR / f"oos_2025_{ts}.json"
    oos_save = []
    for r in oos_results:
        oos_save.append({k: v for k, v in r.items()
                        if not isinstance(v, float) or not math.isnan(v)})
    with open(oos_file, "w") as f:
        json.dump(oos_save, f, indent=2, default=str)
    print(f"  Saved OOS results -> {oos_file}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")

    oos_profitable = [r for r in oos_results if r["pnl"] > 0]
    oos_robust = [r for r in oos_results
                  if r["sharpe"] > 0 and r.get("is_sharpe", 0) > 0
                  and (1.0 - r["sharpe"] / r["is_sharpe"]) < 0.3]

    print(f"  IS configs evaluated:    {len(results):,}")
    print(f"  IS viable:               {len(viable):,}")
    print(f"  OOS tested:              {n_oos}")
    print(f"  OOS profitable:          {len(oos_profitable)}")
    print(f"  OOS robust (<30% deg):   {len(oos_robust)}")
    print()


if __name__ == "__main__":
    main()
