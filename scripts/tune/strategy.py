#!/usr/bin/env python3
"""Tier 1: Tune strategy parameters on 15-year bar data (no filters).

Reads the strategy's YAML config to determine:
  - Bar frequency (e.g. 5s for ORB, 1m fallback for dollar-bar strategies)
  - Session times (session_start / session_end)
  - Required signals (signals: [atr, vwap_session, ...])

Signal columns are pre-computed once (vectorized Polars) into a temporary
enriched parquet.  Workers load it and reconstruct SignalBundles from
columns — no per-bar Python signal computation needed.

Output is consumed by tune/filters.py (Tier 2).

Usage:
    python scripts/tune/strategy.py --strategy orb
    python scripts/tune/strategy.py --strategy vwap
    python scripts/tune/strategy.py --strategy orb --start 2020-01-01 --end 2025-12-31
    python scripts/tune/strategy.py --strategy orb --workers 3
"""

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, time
from itertools import product
from pathlib import Path

import polars as pl
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import structlog

from scripts.tune.log import setup_log

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.filters.filter_engine import FilterEngine
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy
from src.strategies.intraday_momentum import NoiseBreakoutStrategy


# Filter combos to sweep alongside strategy params (list format with bar/seq)
FILTER_COMBOS = [
    ("none", []),
    ("spread_z1.0", [{"signal": "spread", "expr": "< 1.0", "seq": 1, "bar": "5m"}]),
    ("spread_z1.5", [{"signal": "spread", "expr": "< 1.5", "seq": 1, "bar": "5m"}]),
    ("spread_z2.0", [{"signal": "spread", "expr": "< 2.0", "seq": 1, "bar": "5m"}]),
    ("spread_z3.0", [{"signal": "spread", "expr": "< 3.0", "seq": 1, "bar": "5m"}]),
    ("rvol_1.5", [{"signal": "relative_volume", "expr": ">= 1.5", "seq": 1, "bar": "5m"}]),
    ("rvol_2.0", [{"signal": "relative_volume", "expr": ">= 2.0", "seq": 1, "bar": "5m"}]),
    ("spread_z2+rvol1.5", [
        {"signal": "spread", "expr": "< 2.0", "seq": 1, "bar": "5m"},
        {"signal": "relative_volume", "expr": ">= 1.5", "seq": 1, "bar": "5m"},
    ]),
    ("spread_z2+rvol2.0", [
        {"signal": "spread", "expr": "< 2.0", "seq": 1, "bar": "5m"},
        {"signal": "relative_volume", "expr": ">= 2.0", "seq": 1, "bar": "5m"},
    ]),
]

# Strategy YAML locations
YAML_DIR = Path("config/strategies")
YAML_MAP = {
    "orb": YAML_DIR / "orb.yaml",
    "vwap": YAML_DIR / "vwap_reversion.yaml",
    "intraday_momentum": YAML_DIR / "intraday_momentum.yaml",
}

# Per-process shared state (set by _init_worker)
_worker_bars = None
_worker_bundles = None
_worker_signal_names = None


def load_strategy_yaml(strategy: str) -> dict:
    """Load and return the strategy's YAML config."""
    path = YAML_MAP[strategy]
    if not path.exists():
        print(f"  FATAL: Strategy YAML not found at {path}")
        sys.exit(1)
    return yaml.safe_load(path.read_text())


def resolve_bar_config(yaml_cfg: dict) -> tuple[str, bool]:
    """Derive bar frequency and RTH flag from YAML bar section.

    Returns:
        (bar_freq, use_rth) — e.g. ("5s", True) for ORB, ("1m", True) for VWAP.
    """
    bar_cfg = yaml_cfg.get("bar", {})
    bar_type = bar_cfg.get("type", "time")

    if bar_type == "time":
        interval_s = bar_cfg.get("interval_seconds", 60)
        if interval_s < 60:
            freq = f"{interval_s}s"
        else:
            freq = f"{interval_s // 60}m"
    else:
        # Dollar bars / other non-time types: can't pre-build, fall back to 1m
        freq = "1m"

    # Always use RTH — strategies only trade 9:30-16:00
    return freq, True


def resolve_parquet_dir(bar_freq: str, use_rth: bool) -> tuple[str, bool]:
    """Build parquet directory path from bar freq and RTH flag.

    Falls back to all-hours bars if RTH version doesn't exist
    (engine will filter to session hours at runtime).

    Returns:
        (parquet_dir, actual_rth) — actual_rth may be False if fallback used.
    """
    if use_rth:
        rth_dir = f"data/parquet_{bar_freq}_rth"
        if os.path.isdir(rth_dir):
            return rth_dir, True
        # Fall back to all-hours
        print(f"  NOTE: {rth_dir}/ not found, falling back to all-hours bars")

    all_dir = f"data/parquet_{bar_freq}"
    return all_dir, False


def build_enriched_parquet(
    start: date,
    end: date,
    parquet_dir: str,
    signal_names: list[str],
    session_start: time = time(9, 30),
    session_end: time = time(16, 0),
    bar_freq: str = "1s",
) -> tuple[str, pl.DataFrame]:
    """Load bars, compute signals vectorially, write enriched parquet.

    Uses persistent bar cache — if enriched bars already exist on disk,
    skips the entire build. Returns the cached path directly.

    Returns:
        (parquet_path, bars_df) — path to the enriched parquet file,
        bars_df is the raw bars (without sig_ columns) for reference.
    """
    from src.data.bar_cache import BarCache
    from src.signals.vectorized import enrich_bars

    cache_name = BarCache.enriched_name(bar_freq, signal_names)

    # Check persistent cache first
    cached = BarCache.load(cache_name)
    if cached is not None:
        cache_path = str(BarCache.path(cache_name))
        print(f"  [BarCache] HIT: {cache_name} ({len(cached):,} rows)")
        # Return the cached file path — workers will load it directly
        # For bars_df, strip sig_ columns
        raw_cols = [c for c in cached.columns if not c.startswith("sig_")]
        return cache_path, cached.select(raw_cols)

    t0 = _time.perf_counter()

    # Load bars the same way BacktestEngine does
    engine = BacktestEngine()
    config = BacktestConfig(
        strategies=[], start_date=start, end_date=end,
        parquet_dir=parquet_dir,
    )
    bars_df = engine._load_bars(config)
    if bars_df.is_empty():
        print("  FATAL: No bars loaded")
        sys.exit(1)

    t_load = _time.perf_counter()

    # Apply same preprocessing as BacktestEngine.run() so row count matches
    bars_df = bars_df.with_columns(
        pl.col("timestamp")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("US/Eastern")
            .alias("_et_ts"),
    )
    # Filter to RTH session
    bars_df = bars_df.filter(
        (pl.col("_et_ts").dt.time() >= session_start)
        & (pl.col("_et_ts").dt.time() < session_end)
    )
    bars_df = bars_df.with_columns(
        pl.col("_et_ts").dt.date().alias("_bar_date"),
    )

    t_prep = _time.perf_counter()

    # Vectorized signal enrichment
    enriched_df = enrich_bars(bars_df, signal_names)

    t_enrich = _time.perf_counter()

    # Save to persistent cache (drop helper columns — engine will re-add them)
    save_df = enriched_df.drop(["_et_ts", "_bar_date"])
    cache_path = str(BarCache.save(cache_name, save_df))

    t_write = _time.perf_counter()

    print(f"  Enriched parquet: {len(enriched_df):,} bars")
    print(f"  Timing: load={t_load - t0:.1f}s prep={t_prep - t_load:.1f}s "
          f"enrich={t_enrich - t_prep:.1f}s write={t_write - t_enrich:.1f}s "
          f"total={t_write - t0:.1f}s")

    return cache_path, bars_df


def _init_worker(enriched_parquet_path, signal_names, start, end):
    """Initializer for each worker process — loads enriched parquet.

    No bundle construction — the engine builds SignalBundle inline per-row
    from sig_* columns during its existing iteration.
    """
    global _worker_bars, _worker_bundles, _worker_signal_names
    import os as _os
    pid = _os.getpid()
    logging.disable(logging.CRITICAL)
    try:
        _worker_bars = pl.read_parquet(enriched_parquet_path)
        _worker_bundles = None
        _worker_signal_names = signal_names
        print(f"  [Worker {pid}] Ready: {len(_worker_bars):,} rows, "
              f"{_worker_bars.estimated_size('mb'):.1f} MB", flush=True)
    except Exception as e:
        print(f"  [Worker {pid}] INIT FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def _run_orb(tgt, vol, max_t, exp, vwap, filter_name, filter_dict, start, end):
    """Run a single ORB config using worker-local bars + signals."""
    import os as _os
    pid = _os.getpid()
    if _worker_bars is None:
        raise RuntimeError(f"Worker {pid}: _worker_bars is None — init failed silently")
    strat = ORBStrategy(config={
        "strategy": {"strategy_id": "orb", "max_signals_per_day": 1},
        "orb": {
            "target_multiplier": tgt, "volume_multiplier": vol,
            "max_signal_time": max_t, "expiry_minutes": exp,
            "require_vwap_alignment": vwap,
        },
        "filter": {"min_score": 0, "require_direction_agreement": False, "signals": {}},
        "exit": {"time_stop_minutes": exp},
    })
    fe = FilterEngine.from_list(filter_dict) if filter_dict else None
    config = BacktestConfig(
        strategies=[strat], start_date=start, end_date=end,
        prebuilt_bars=_worker_bars,
        enriched_signal_names=_worker_signal_names,
        filter_engine=fe,
    )
    engine = BacktestEngine()
    r = engine.run(config)
    m = r.metrics
    return {
        "tgt": tgt, "vol": vol, "max_t": max_t, "exp": exp, "vwap": vwap,
        "filter": filter_name,
        "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
        "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
        "pf": m.profit_factor, "dd": m.max_drawdown_pct,
        "avg_win": m.avg_win, "avg_loss": m.avg_loss,
    }


def _run_vwap(entry_sd, stop_sd, pb_sd, cooldown, max_sig, expiry, filter_name, filter_dict, start, end):
    """Run a single VWAP config using worker-local bars + signals."""
    strat = VWAPReversionStrategy(config={
        "strategy": {"strategy_id": "vwap_reversion", "max_signals_per_day": max_sig},
        "vwap": {
            "entry_sd_reversion": entry_sd, "stop_sd": stop_sd,
            "pullback_entry_sd": pb_sd, "mode_cooldown_bars": cooldown,
            "expiry_minutes": expiry,
        },
        "filter": {"min_score": 0, "require_direction_agreement": False, "signals": {}},
        "exit": {"target": "vwap", "stop_ticks": 8, "time_stop_minutes": expiry},
    })
    fe = FilterEngine.from_list(filter_dict) if filter_dict else None
    config = BacktestConfig(
        strategies=[strat], start_date=start, end_date=end,
        prebuilt_bars=_worker_bars,
        enriched_signal_names=_worker_signal_names,
        filter_engine=fe,
    )
    engine = BacktestEngine()
    r = engine.run(config)
    m = r.metrics
    return {
        "entry_sd": entry_sd, "stop_sd": stop_sd, "expiry": expiry,
        "filter": filter_name,
        "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
        "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
        "pf": m.profit_factor, "dd": m.max_drawdown_pct,
        "avg_win": m.avg_win, "avg_loss": m.avg_loss,
    }


def _run_noise_breakout(noise_mult, stop_atr, tgt_atr, expiry, max_sig, filter_name, filter_dict, start, end):
    """Run a single NoiseBreakout config using worker-local bars + signals."""
    strat = NoiseBreakoutStrategy(config={
        "strategy": {"strategy_id": "noise_breakout", "max_signals_per_day": max_sig},
        "noise_breakout": {
            "noise_lookback_days": 90,
            "noise_multiplier": noise_mult,
            "stop_atr_multiplier": stop_atr,
            "target_atr_multiplier": tgt_atr,
            "expiry_minutes": expiry,
            "require_vwap_alignment": True,
            "use_vwap_trail": True,
            "slippage_ticks": 1,
        },
        "filter": {"min_score": 0, "require_direction_agreement": False, "signals": {}},
        "exit": {"hard_close": "15:50"},
    })
    fe = FilterEngine.from_list(filter_dict) if filter_dict else None
    config = BacktestConfig(
        strategies=[strat], start_date=start, end_date=end,
        prebuilt_bars=_worker_bars,
        enriched_signal_names=_worker_signal_names,
        filter_engine=fe,
    )
    engine = BacktestEngine()
    r = engine.run(config)
    m = r.metrics
    return {
        "noise_mult": noise_mult, "stop_atr": stop_atr, "tgt_atr": tgt_atr,
        "expiry": expiry, "max_sig": max_sig, "filter": filter_name,
        "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
        "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
        "pf": m.profit_factor, "dd": m.max_drawdown_pct,
        "avg_win": m.avg_win, "avg_loss": m.avg_loss,
    }


def sweep(strategy, start, end, workers, enriched_parquet_path, signal_names):
    """Run parameter sweep using ProcessPoolExecutor for true parallelism.

    Cross-products strategy params × filter combos for a combined Tier 1+2 sweep.
    """
    if strategy == "orb":
        strat_combos = list(product(
            [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],  # target_multiplier
            [1.5],                         # volume_multiplier (locked)
            ["10:30"],                     # max_signal_time (locked)
            [15, 30, 45, 60, 75, 90, 105, 120],  # expiry_minutes
            [True],                        # require_vwap_alignment (locked)
        ))
    elif strategy == "intraday_momentum":
        strat_combos = list(product(
            [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],  # noise_multiplier
            [1.5, 2.0, 2.5, 3.0],              # stop_atr_multiplier
            [2.0, 3.0, 4.0, 5.0],              # target_atr_multiplier
            [30, 45, 60, 90],                   # expiry_minutes
            [3],                                # max_signals_per_day (locked)
        ))
    else:
        strat_combos = list(product(
            [1.0, 1.5, 2.0, 2.5],     # entry_sd
            [1.5, 2.0, 2.5, 3.0, 3.5],  # stop_sd
            [0.5],                      # pullback_sd
            [5],                        # mode_cooldown
            [4],                        # max_signals
            [30, 60],                   # expiry_minutes
        ))

    # Cross-product: strategy params × filter combos
    combos = [(sc, fn, fd) for sc in strat_combos for fn, fd in FILTER_COMBOS]

    total = len(combos)
    n_strat = len(strat_combos)
    n_filt = len(FILTER_COMBOS)
    print(f"  {n_strat} strategy × {n_filt} filter = {total} configs ({workers} processes)\n")

    results = []
    done = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(enriched_parquet_path, signal_names, start, end),
    ) as pool:
        futures = {}
        for strat_params, filter_name, filter_dict in combos:
            if strategy == "orb":
                f = pool.submit(_run_orb, *strat_params, filter_name, filter_dict, start, end)
            elif strategy == "intraday_momentum":
                f = pool.submit(_run_noise_breakout, *strat_params, filter_name, filter_dict, start, end)
            else:
                f = pool.submit(_run_vwap, *strat_params, filter_name, filter_dict, start, end)
            futures[f] = (strat_params, filter_name)

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
            filt = result.get("filter", "none")
            filt_tag = f" [{filt}]" if filt != "none" else ""

            if strategy == "orb":
                print(f"  [{done:3d}/{total}] {s} tgt={result['tgt']:.2f} exp={result['exp']:2d}m{filt_tag}  "
                      f"trades={result['trades']:3d} WR={result['wr']:.1%} "
                      f"Sharpe={result['sharpe']:>7.3f} PnL=${result['pnl']:>9,.2f}")
            elif strategy == "intraday_momentum":
                print(f"  [{done:3d}/{total}] {s} nm={result['noise_mult']:.1f} "
                      f"stop={result['stop_atr']:.1f} tgt={result['tgt_atr']:.1f} "
                      f"exp={result['expiry']:2d}m{filt_tag}  "
                      f"trades={result['trades']:3d} WR={result['wr']:.1%} "
                      f"Sharpe={result['sharpe']:>7.3f} PnL=${result['pnl']:>9,.2f}")
            else:
                print(f"  [{done:3d}/{total}] {s} entry={result['entry_sd']:.1f} "
                      f"stop={result['stop_sd']:.1f} exp={result['expiry']:2d}m{filt_tag}  "
                      f"trades={result['trades']:3d} WR={result['wr']:.1%} "
                      f"Sharpe={result['sharpe']:>7.3f} PnL=${result['pnl']:>9,.2f}")
            sys.stdout.flush()

    return results


def main():
    parser = argparse.ArgumentParser(description="Tier 1: Strategy parameter tuning")
    parser.add_argument("--strategy", required=True, choices=["orb", "vwap", "intraday_momentum"])
    parser.add_argument("--start", default="2011-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=10, help="Parallel processes (default 10)")
    args = parser.parse_args()

    _log_path = setup_log(args.strategy)

    # ── Read strategy YAML for bar config + signals ──
    yaml_cfg = load_strategy_yaml(args.strategy)
    bar_freq, want_rth = resolve_bar_config(yaml_cfg)
    parquet_dir, use_rth = resolve_parquet_dir(bar_freq, want_rth)
    signal_names = yaml_cfg.get("signals", [])
    session_start_str = yaml_cfg.get("strategy", {}).get("session_start", "09:30")
    session_end_str = yaml_cfg.get("strategy", {}).get("session_end", "16:00")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    rth_tag = " RTH" if use_rth else ""
    print(f"\n{'=' * 90}")
    print(f"COMBINED: {args.strategy.upper()} Strategy + Filter Sweep")
    print(f"Data:     {start} to {end} ({bar_freq}{rth_tag} bars)")
    print(f"Session:  {session_start_str} - {session_end_str} ET")
    print(f"Signals:  {signal_names or '(none)'}")
    print(f"Workers:  {args.workers}")
    print(f"Source:   {parquet_dir}/")
    print(f"{'=' * 90}\n")

    # Verify pre-built bars exist
    if not os.path.isdir(parquet_dir) or not any(
        d.startswith("year=") for d in os.listdir(parquet_dir)
    ):
        print(f"  FATAL: Pre-built bars not found at {parquet_dir}/")
        print(f"  Run: python -m scripts.build_bars --freq {bar_freq} --rth")
        sys.exit(1)

    # Suppress noisy loggers in main process
    logging.disable(logging.CRITICAL)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))
    # Also suppress the noise_breakout strategy logger (floods with signal_generated)
    logging.getLogger("noise_breakout").setLevel(logging.CRITICAL + 1)

    # ── Build enriched parquet (vectorized signals, computed once) ──
    print("  Building enriched parquet (vectorized signal pre-computation)...")
    session_start = time(*map(int, session_start_str.split(":")))
    session_end = time(*map(int, session_end_str.split(":")))
    enriched_path, _ = build_enriched_parquet(
        start, end, parquet_dir, signal_names,
        session_start=session_start, session_end=session_end,
        bar_freq=bar_freq,
    )

    print(f"\n  Spawning {args.workers} processes...\n", flush=True)

    results = sweep(args.strategy, start, end, args.workers, enriched_path, signal_names)

    # Composite scoring: sharpe * sqrt(trades)
    # Rewards configs that are both good AND frequent — gives filters raw material
    MIN_TRADES = 50
    metric_keys = ("trades", "wr", "pnl", "sharpe", "sortino", "pf", "dd", "avg_win", "avg_loss", "score", "filter")

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
        filt = r.get("filter", "none")
        print(f"  {i+1:2d}. {params} [{filt}]  trades={r['trades']} WR={r['wr']:.1%} "
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
        "bar_freq": bar_freq,
        "rth": use_rth,
        "signals": signal_names,
        "session": {"start": session_start_str, "end": session_end_str},
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
    print(f"  Next: python scripts/tune/filters.py --strategy {args.strategy} --tier1 {out_path}")


if __name__ == "__main__":
    main()
