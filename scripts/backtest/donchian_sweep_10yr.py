#!/usr/bin/env python3
"""10-year coarse sweep for Donchian Breakout: train 2014-2024, test 2025+.

Sweeps entry + exit params on 10yr data, then validates top configs on
2025 OOS data. Outputs results to file for remote execution.

Usage:
    python -u scripts/backtest/donchian_sweep_10yr.py 2>&1 | tee logs/donchian_sweep_10yr.log
"""

import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time as _time
from datetime import date, datetime, time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.disable(logging.CRITICAL)
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

import polars as pl

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.data.bar_cache import BarCache
from src.exits.exit_engine import ExitEngine
from src.signals.vectorized import enrich_bars
from src.strategies.donchian_breakout import DonchianBreakoutStrategy


SIGNAL_NAMES = ["donchian_channel", "adx", "relative_volume", "atr", "session_time"]

# Train / test split
TRAIN_START = date(2014, 1, 1)
TRAIN_END   = date(2025, 1, 1)
TEST_START  = date(2025, 1, 1)
TEST_END    = date(2026, 1, 1)


def load_enriched_bars(start: date, end: date, label: str) -> pl.DataFrame:
    """Load 5m bars enriched with vectorized signals."""
    cache_name = BarCache.enriched_name("5m", SIGNAL_NAMES)
    # Different cache for different date ranges
    if start.year < 2020:
        cache_name = f"enriched_5m_donch_{start.year}_{end.year}"

    cached = BarCache.load(cache_name)
    if cached is not None:
        # Filter to requested range
        filtered = cached.filter(
            (pl.col("timestamp") >= pl.lit(datetime(start.year, start.month, start.day)))
            & (pl.col("timestamp") < pl.lit(datetime(end.year, end.month, end.day)))
        )
        if len(filtered) > 0:
            print(f"  [{label} Cache HIT] {cache_name} ({len(filtered):,} rows)")
            return filtered

    print(f"  [{label}] Building enriched bars {start} to {end} (will be cached)...")
    t0 = _time.perf_counter()

    engine = BacktestEngine()
    config = BacktestConfig(
        strategies=[], start_date=start, end_date=end,
        parquet_dir="data/parquet_5m", bar_type="5m",
    )
    bars_df = engine._load_bars(config)
    if bars_df.is_empty():
        print(f"  FATAL: No bars loaded for {start} to {end}")
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

    enriched = enrich_bars(bars_df, SIGNAL_NAMES)
    save_df = enriched.drop(["_et_ts", "_bar_date"]).with_columns(
        pl.col("timestamp").dt.replace_time_zone(None).alias("timestamp"),
    )
    BarCache.save(cache_name, save_df)

    dt = _time.perf_counter() - t0
    print(f"  [{label}] Enriched {len(save_df):,} bars in {dt:.1f}s")
    return save_df


def run_single(bars: pl.DataFrame, start: date, end: date, **params) -> dict:
    """Run a single donchian backtest."""
    filters = [
        {"signal": "session_time", "expr": f">= {params.get('session_start', 600)}"},
        {"signal": "session_time", "expr": f"<= {params.get('session_end', 840)}"},
        {"signal": "donchian_channel", "expr": "passes"},
        {"signal": "adx", "expr": f"> {params.get('adx_min', 25.0)}"},
        {"signal": "relative_volume", "expr": f">= {params.get('rvol_min', 0.5)}"},
    ]

    cfg = {
        "strategy": {
            "strategy_id": "donchian_breakout",
            "max_signals_per_day": params.get("max_signals", 5),
        },
        "exit": {"time_stop_minutes": params.get("time_stop_bars", 12) * 5},
        "donchian": {
            "target_atr": params.get("target_atr", 2.0),
            "stop_atr": params.get("stop_atr", 1.5),
            "width_min": params.get("width_min", 2.0),
            "width_max": params.get("width_max", 30.0),
        },
        "filters": filters,
    }

    strat = DonchianBreakoutStrategy(config=cfg)

    filtered_bars = bars.filter(
        (pl.col("timestamp") >= pl.lit(datetime(start.year, start.month, start.day)))
        & (pl.col("timestamp") < pl.lit(datetime(end.year, end.month, end.day)))
    )

    exits = [
        {"type": "bracket_target", "enabled": True},
        {"type": "bracket_stop", "enabled": True},
        {"type": "time_stop", "enabled": True, "max_bars": params.get("time_stop_bars", 12)},
        {"type": "price_vs_signal_exit", "enabled": True,
         "signal": "donchian_channel", "long_field": "exit_lower", "short_field": "exit_upper"},
        {"type": "signal_bound_exit", "enabled": True,
         "signal": "adx", "lower_bound": params.get("adx_kill", 15.0)},
    ]
    exit_engine = ExitEngine.from_list(exits)

    bt_config = BacktestConfig(
        strategies=[strat],
        start_date=start,
        end_date=end,
        prebuilt_bars=filtered_bars,
        enriched_signal_names=SIGNAL_NAMES,
        use_rth_bars=True,
        exit_engine=exit_engine,
    )
    engine = BacktestEngine()
    r = engine.run(bt_config)
    m = r.metrics
    return {
        **params,
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


N_WORKERS = 10


def _worker_init(bars_path: str, start: date, end: date):
    """Initialize worker process — load bars once per worker."""
    global _worker_bars, _worker_start, _worker_end
    # Suppress logging in workers
    logging.disable(logging.CRITICAL)
    import structlog
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

    _worker_bars = pl.read_parquet(bars_path)
    _worker_start = start
    _worker_end = end


def _worker_run(params: dict) -> dict:
    """Run a single config in worker process."""
    return run_single(_worker_bars, _worker_start, _worker_end, **params)


def main():
    print(f"\n{'='*80}")
    print(f"  DONCHIAN BREAKOUT 10-YEAR SWEEP ({N_WORKERS} workers)")
    print(f"  Train: {TRAIN_START} to {TRAIN_END}  |  Test: {TEST_START} to {TEST_END}")
    print(f"{'='*80}\n")

    # ── Load data ────────────────────────────────────────
    train_bars = load_enriched_bars(TRAIN_START, TRAIN_END, "TRAIN")
    test_bars = load_enriched_bars(TEST_START, TEST_END, "TEST")
    print()

    # Save train bars to a temp parquet for worker init
    train_path = "data/_sweep_train_tmp.parquet"
    train_bars.write_parquet(train_path)

    # ── Sweep grid ───────────────────────────────────────
    # ~2,304 configs
    #
    # Entry params (4 x 3 x 3 x 2 = 72)
    target_atrs    = [1.5, 2.0, 3.0, 4.0]
    stop_atrs      = [1.0, 1.5, 2.5]
    adx_mins       = [20.0, 25.0, 30.0]
    rvol_mins      = [0.3, 0.8]

    # Exit params (4 x 2 = 8)
    time_stop_bars = [6, 12, 18, 24]        # 30/60/90/120 min
    adx_kills      = [12.0, 18.0]

    # Session window (2 x 2 = 4)
    session_starts = [570, 600]             # 9:30, 10:00
    session_ends   = [840, 900]             # 2:00, 3:00

    # Width filter (fixed — less impactful)
    width_mins     = [2.0]
    width_maxs     = [30.0]

    combo_keys = [
        "target_atr", "stop_atr", "adx_min", "rvol_min",
        "time_stop_bars", "adx_kill",
        "session_start", "session_end",
        "width_min", "width_max",
    ]
    combos = list(product(
        target_atrs, stop_atrs, adx_mins, rvol_mins,
        time_stop_bars, adx_kills,
        session_starts, session_ends,
        width_mins, width_maxs,
    ))

    # Build param dicts for workers
    all_params = [dict(zip(combo_keys, combo)) for combo in combos]

    print(f"  Sweep grid: {len(combos):,} combinations x {N_WORKERS} workers")
    print(f"  Params: tgt_atr={target_atrs}, stp_atr={stop_atrs}")
    print(f"          adx_min={adx_mins}, rvol_min={rvol_mins}")
    print(f"          time_bars={time_stop_bars}, adx_kill={adx_kills}")
    print(f"          session={session_starts}x{session_ends}")
    print(f"          width={width_mins}x{width_maxs}")
    print()
    sys.stdout.flush()

    # ── Run sweep on training data (multiprocessing) ─────
    print(f"{'='*80}")
    print(f"  PHASE 1: TRAINING SWEEP ({TRAIN_START} to {TRAIN_END})")
    print(f"{'='*80}\n")

    t0 = _time.perf_counter()
    results = []

    with mp.Pool(
        processes=N_WORKERS,
        initializer=_worker_init,
        initargs=(train_path, TRAIN_START, TRAIN_END),
    ) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker_run, all_params, chunksize=16)):
            results.append(r)
            done = i + 1
            if done % 200 == 0 or done == len(all_params):
                elapsed = _time.perf_counter() - t0
                rate = elapsed / done
                eta = rate * (len(all_params) - done)
                print(f"  ... {done:,}/{len(all_params):,} done "
                      f"({elapsed:.0f}s elapsed, {rate:.2f}s/config, ~{eta:.0f}s remaining)")
                sys.stdout.flush()

    total_time = _time.perf_counter() - t0
    print(f"\n  Sweep completed in {total_time:.0f}s ({total_time/len(combos):.2f}s/config effective)")

    # Cleanup temp file
    try:
        os.remove(train_path)
    except OSError:
        pass

    # Score and rank
    for r in results:
        trades = max(r["trades"], 1)
        # Penalize configs with too few trades (want >100 over 10yr)
        trade_penalty = min(trades / 100, 1.0)
        r["score"] = r["sharpe"] * math.sqrt(trades) * trade_penalty if r["sharpe"] > 0 else 0

    results.sort(key=lambda x: x["score"], reverse=True)

    # Print top 30
    hdr = (f"{'tgt':>4} {'stp':>4} {'adx':>3} {'rvl':>3} {'tBr':>3} {'aKl':>3} "
           f"{'sS':>3} {'sE':>3} {'wMn':>3} {'wMx':>3} | "
           f"{'trades':>6} {'WR':>6} {'PnL':>8} {'Sharpe':>7} {'PF':>5} {'DD':>5} | {'score':>6}")

    print(f"\n  TOP 30 CONFIGS (ranked by Sharpe * sqrt(trades) * trade_penalty):")
    print(f"  {hdr}")
    print(f"  {'-'*len(hdr)}")

    for r in results[:30]:
        print(f"  {r.get('target_atr',2):>4.1f} {r.get('stop_atr',1.5):>4.1f} "
              f"{r.get('adx_min',25):>3.0f} {r.get('rvol_min',0.5):>3.1f} "
              f"{r.get('time_stop_bars',12):>3} {r.get('adx_kill',15):>3.0f} "
              f"{r.get('session_start',600):>3} {r.get('session_end',840):>3} "
              f"{r.get('width_min',2):>3.0f} {r.get('width_max',20):>3.0f} | "
              f"{r['trades']:>6} {r['wr']:>5.1%} ${r['pnl']:>7,.0f} "
              f"{r['sharpe']:>7.2f} {r['pf']:>5.2f} {r['dd']:>4.1f}% | "
              f"{r['score']:>6.1f}")
    sys.stdout.flush()

    # Count profitable configs
    profitable = sum(1 for r in results if r["pnl"] > 0 and r["trades"] >= 20)
    print(f"\n  Profitable configs (>=20 trades): {profitable}/{len(results)} ({100*profitable/len(results):.1f}%)")

    # ── Phase 2: OOS test top configs ────────────────────
    print(f"\n{'='*80}")
    print(f"  PHASE 2: OUT-OF-SAMPLE TEST ({TEST_START} to {TEST_END})")
    print(f"  Testing top 20 training configs on unseen 2025 data")
    print(f"{'='*80}\n")

    top_n = min(20, len([r for r in results if r["score"] > 0]))
    oos_results = []

    print(f"  {'#':>2} | {'tgt':>4} {'stp':>4} {'adx':>3} {'rvl':>3} {'tBr':>3} {'aKl':>3} "
          f"{'sS':>3} {'sE':>3} {'wMn':>3} {'wMx':>3} | "
          f"{'TRAIN':>30} | {'TEST (OOS)':>30}")
    print(f"  {'-'*110}")

    for rank, train_r in enumerate(results[:top_n], 1):
        # Extract params from training result
        params = {k: train_r[k] for k in combo_keys}
        test_r = run_single(test_bars, TEST_START, TEST_END, **params)
        oos_results.append({"rank": rank, "train": train_r, "test": test_r, "params": params})

        t_str = f"{train_r['trades']:>4}t {train_r['wr']:>4.0%} ${train_r['pnl']:>6,.0f} Sh{train_r['sharpe']:>5.2f}"
        o_str = f"{test_r['trades']:>4}t {test_r['wr']:>4.0%} ${test_r['pnl']:>6,.0f} Sh{test_r['sharpe']:>5.2f}"

        flag = " OOS+" if test_r["pnl"] > 0 and test_r["trades"] >= 3 else ""
        print(f"  {rank:>2} | {params['target_atr']:>4.1f} {params['stop_atr']:>4.1f} "
              f"{params['adx_min']:>3.0f} {params['rvol_min']:>3.1f} "
              f"{params['time_stop_bars']:>3} {params['adx_kill']:>3.0f} "
              f"{params['session_start']:>3} {params['session_end']:>3} "
              f"{params['width_min']:>3.0f} {params['width_max']:>3.0f} | "
              f"{t_str} | {o_str}{flag}")
        sys.stdout.flush()

    # ── Summary ──────────────────────────────────────────
    oos_profitable = sum(1 for r in oos_results if r["test"]["pnl"] > 0 and r["test"]["trades"] >= 3)
    print(f"\n  OOS profitable (>=3 trades): {oos_profitable}/{top_n}")

    # Find best OOS result
    oos_with_trades = [r for r in oos_results if r["test"]["trades"] >= 3]
    if oos_with_trades:
        best_oos = max(oos_with_trades, key=lambda x: x["test"]["sharpe"])
        print(f"\n  BEST OOS CONFIG (rank #{best_oos['rank']} in training):")
        print(f"    Params: {best_oos['params']}")
        print(f"    Train:  {best_oos['train']['trades']} trades, {best_oos['train']['wr']:.1%} WR, "
              f"Sharpe {best_oos['train']['sharpe']:.2f}, ${best_oos['train']['pnl']:,.0f} PnL")
        print(f"    Test:   {best_oos['test']['trades']} trades, {best_oos['test']['wr']:.1%} WR, "
              f"Sharpe {best_oos['test']['sharpe']:.2f}, ${best_oos['test']['pnl']:,.0f} PnL")
    else:
        print(f"\n  No OOS configs with >=3 trades — strategy may not be viable")

    # Save results
    os.makedirs("results/donchian", exist_ok=True)
    output = {
        "train_period": f"{TRAIN_START} to {TRAIN_END}",
        "test_period": f"{TEST_START} to {TEST_END}",
        "total_configs": len(combos),
        "top_30_train": results[:30],
        "oos_results": [
            {"rank": r["rank"], "params": r["params"],
             "train": {k: r["train"][k] for k in ["trades","wr","pnl","sharpe","pf","dd"]},
             "test": {k: r["test"][k] for k in ["trades","wr","pnl","sharpe","pf","dd"]}}
            for r in oos_results
        ],
    }
    out_path = "results/donchian/sweep_10yr.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print()


if __name__ == "__main__":
    main()
