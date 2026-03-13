#!/usr/bin/env python3
"""Quick backtest for Donchian Channel Breakout strategy.

Runs on 5m bars with vectorized signals for fast parameter sweeps.
Supports regime_v2 gating and full ExitEngine param sweep.

Usage:
    python -u scripts/backtest/donchian_quick.py
    python -u scripts/backtest/donchian_quick.py --sweep
    python -u scripts/backtest/donchian_quick.py --sweep --no-regime
    python -u scripts/backtest/donchian_quick.py --sweep-full
    python -u scripts/backtest/donchian_quick.py --start 2024-01-01 --end 2025-01-01 --sweep-full
"""

import argparse
import logging
import math
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


BASE_SIGNALS = ["donchian_channel", "adx", "relative_volume", "atr", "session_time"]


def load_enriched_bars(start: date, end: date, use_regime: bool = True) -> tuple[pl.DataFrame, list[str]]:
    """Load 5m bars enriched with vectorized signals. Caches to disk."""
    signal_names = BASE_SIGNALS + (["regime_v2"] if use_regime else [])
    cache_name = BarCache.enriched_name("5m", signal_names)

    cached = BarCache.load(cache_name)
    if cached is not None:
        print(f"  [Cache HIT] {cache_name} ({len(cached):,} rows)")
        return cached, signal_names

    print("  Building enriched bars (first run — will be cached)...")
    t0 = _time.perf_counter()

    engine = BacktestEngine()
    config = BacktestConfig(
        strategies=[], start_date=start, end_date=end,
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
    """Run a single donchian backtest config using pre-computed signals."""
    # Session time window (minutes since midnight)
    session_start_min = overrides.get("session_start", 585)  # 9:45 AM
    session_end_min = overrides.get("session_end", 900)      # 3:00 PM

    filters = [
        {"signal": "session_time", "expr": f">= {session_start_min}"},
        {"signal": "session_time", "expr": f"<= {session_end_min}"},
        {"signal": "donchian_channel", "expr": "passes"},
        {"signal": "adx", "expr": f"> {overrides.get('adx_min', 18.0)}"},
        {"signal": "relative_volume", "expr": f">= {overrides.get('rvol_min', 1.0)}"},
    ]

    if use_regime:
        filters.append({"signal": "regime_v2", "expr": "passes"})
        filters.append({"signal": "regime_v2", "expr": "== 0"})
        conf_min = overrides.get("conf_min", 0.0)
        if conf_min > 0:
            filters.append({"signal": "regime_v2", "field": "confidence", "expr": f">= {conf_min}"})

    cfg = {
        "strategy": {
            "strategy_id": "donchian_breakout",
            "max_signals_per_day": overrides.get("max_signals", 3),
        },
        "exit": {"time_stop_minutes": overrides.get("time_stop_min", 45)},
        "donchian": {
            "target_atr": overrides.get("target_atr", 3.0),
            "stop_atr": overrides.get("stop_atr", 1.5),
            "width_min": overrides.get("width_min", 3.0),
            "width_max": overrides.get("width_max", 20.0),
        },
        "filters": filters,
    }

    strat = DonchianBreakoutStrategy(config=cfg)

    # Filter prebuilt bars to date range
    filtered_bars = bars.filter(
        (pl.col("timestamp") >= pl.lit(datetime(start.year, start.month, start.day)))
        & (pl.col("timestamp") < pl.lit(datetime(end.year, end.month, end.day)))
    )

    # Declarative exits (sweepable params)
    time_stop_bars = overrides.get("time_stop_bars", 9)
    adx_kill = overrides.get("adx_kill", 15.0)

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


def print_results(r: dict) -> None:
    print(f"  Trades:        {r['trades']}")
    print(f"  Win Rate:      {r['wr']:.1%}")
    print(f"  Net PnL:       ${r['pnl']:,.2f}")
    print(f"  Sharpe:        {r['sharpe']:.3f}")
    print(f"  Sortino:       {r['sortino']:.3f}")
    print(f"  Profit Factor: {r['pf']:.2f}")
    print(f"  Max DD:        {r['dd']:.1f}%")
    print(f"  Avg Win/Loss:  ${r['avg_win']:.2f} / ${r['avg_loss']:.2f}")


def run_sweep(bars, signal_names, start, end, combos, combo_keys, use_regime, header_fmt, row_fmt):
    """Generic sweep runner."""
    print(f"  {len(combos)} combinations\n")

    t0 = _time.perf_counter()
    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(combo_keys, combo))
        r = run_single(bars, signal_names, start, end, use_regime=use_regime, **params)
        results.append(r)
        if (i + 1) % 50 == 0:
            elapsed = _time.perf_counter() - t0
            eta = elapsed / (i + 1) * (len(combos) - i - 1)
            print(f"  ... {i+1}/{len(combos)} done ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    total_time = _time.perf_counter() - t0

    for r in results:
        r["score"] = r["sharpe"] * math.sqrt(max(r["trades"], 1)) if r["sharpe"] > 0 else 0

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n  Sweep completed in {total_time:.0f}s ({total_time/len(combos):.1f}s/config)")
    print(f"\n  TOP 20 CONFIGS (ranked by Sharpe * sqrt(trades)):")
    print(f"  {header_fmt}")
    print(f"  {'-'*len(header_fmt)}")

    for r in results[:20]:
        print(f"  {row_fmt(r)}")

    print(f"\n  WORST 5:")
    for r in results[-5:]:
        print(f"  {row_fmt(r)}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--sweep", action="store_true", help="Coarse parameter sweep (entry params only)")
    parser.add_argument("--sweep-full", action="store_true", help="Full sweep (entry + exit + session params)")
    parser.add_argument("--no-regime", action="store_true", help="Disable regime_v2 gating")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    use_regime = not args.no_regime

    regime_label = " + regime_v2 TRENDING gate" if use_regime else " (no regime gate)"
    print(f"\n{'='*70}")
    print(f"  DONCHIAN BREAKOUT BACKTEST: {start} to {end}{regime_label}")
    print(f"{'='*70}\n")

    bars, signal_names = load_enriched_bars(start, end, use_regime)
    print()

    # ── Baseline ──
    print(f"  [Baseline] tgt=3.0, stp=1.5, adx>18, rvol>=1.0{regime_label}...")
    t0 = _time.perf_counter()
    baseline = run_single(bars, signal_names, start, end, use_regime=use_regime)
    dt = _time.perf_counter() - t0
    print_results(baseline)
    print(f"  Time:          {dt:.1f}s")
    print()

    if not args.sweep and not args.sweep_full:
        print("  Run with --sweep or --sweep-full to explore parameter space")
        return

    if args.sweep_full:
        # ── Full sweep: ALL params in one stage ──
        print(f"{'='*70}")
        print(f"  FULL PARAMETER SWEEP (all params){regime_label}")
        print(f"{'='*70}\n")

        # Entry params
        target_atrs = [2.0, 3.0]
        stop_atrs = [1.0, 1.5, 2.5]
        adx_mins = [25.0, 30.0]
        rvol_mins = [0.3, 0.8]

        # Exit params
        time_stop_bars_list = [6, 12, 18]            # 30/60/90 min in 5m bars
        adx_kills = [12.0, 15.0]                     # ADX collapse threshold

        # Width filter
        width_mins = [2.0, 5.0]
        width_maxs = [15.0, 30.0]

        # Session time (minutes since midnight)
        session_starts = [570, 600]                   # 9:30, 10:00
        session_ends = [840, 930]                     # 2:00, 3:30

        # Max signals per day
        max_signals_list = [2, 5]

        combo_keys = ["target_atr", "stop_atr", "adx_min", "rvol_min",
                      "time_stop_bars", "adx_kill",
                      "width_min", "width_max",
                      "session_start", "session_end",
                      "max_signals"]
        combos = list(product(target_atrs, stop_atrs, adx_mins, rvol_mins,
                              time_stop_bars_list, adx_kills,
                              width_mins, width_maxs,
                              session_starts, session_ends,
                              max_signals_list))

        hdr = (f"{'tgt':>4} {'stp':>4} {'adx':>3} {'rvl':>3} {'tBr':>3} {'aKl':>3} "
               f"{'wMn':>3} {'wMx':>3} {'sS':>3} {'sE':>3} {'ms':>2} | "
               f"{'trades':>6} {'WR':>6} {'PnL':>8} {'Sharpe':>7} {'PF':>5} {'DD':>5} | {'score':>6}")

        def row_full(r):
            return (f"{r.get('target_atr',3.0):>4.1f} {r.get('stop_atr',1.5):>4.1f} "
                    f"{r.get('adx_min',25):>3.0f} {r.get('rvol_min',0.5):>3.1f} "
                    f"{r.get('time_stop_bars',9):>3} {r.get('adx_kill',15):>3.0f} "
                    f"{r.get('width_min',3):>3.0f} {r.get('width_max',20):>3.0f} "
                    f"{r.get('session_start',585):>3} {r.get('session_end',900):>3} "
                    f"{r.get('max_signals',3):>2} | "
                    f"{r['trades']:>6} {r['wr']:>5.1%} ${r['pnl']:>7,.0f} "
                    f"{r['sharpe']:>7.2f} {r['pf']:>5.2f} {r['dd']:>4.1f}% | "
                    f"{r['score']:>6.1f}")

        run_sweep(bars, signal_names, start, end, combos, combo_keys,
                  use_regime, hdr, row_full)

    elif args.sweep:
        # ── Coarse sweep (same as before) ──
        print(f"{'='*70}")
        print(f"  COARSE PARAMETER SWEEP{regime_label}")
        print(f"{'='*70}\n")

        combo_keys = ["target_atr", "stop_atr", "adx_min", "rvol_min"]
        combos = list(product(
            [2.0, 2.5, 3.0, 4.0, 5.0],
            [1.0, 1.5, 2.0, 2.5],
            [15.0, 22.0, 30.0],
            [0.3, 0.5, 0.8, 1.0],
        ))

        def header_c():
            return f"{'tgt':>4} {'stp':>4} {'adx':>4} {'rvol':>4} | {'trades':>6} {'WR':>6} {'PnL':>8} {'Sharpe':>7} {'PF':>5} {'DD':>5} | {'score':>6}"

        def row_c(r):
            return (f"{r.get('target_atr',3.0):>4.1f} {r.get('stop_atr',1.5):>4.1f} "
                    f"{r.get('adx_min',18.0):>4.0f} {r.get('rvol_min',1.0):>4.1f} | "
                    f"{r['trades']:>6} {r['wr']:>5.1%} ${r['pnl']:>7,.0f} "
                    f"{r['sharpe']:>7.2f} {r['pf']:>5.2f} {r['dd']:>4.1f}% | "
                    f"{r['score']:>6.1f}")

        run_sweep(bars, signal_names, start, end, combos, combo_keys,
                  use_regime, header_c(), row_c)

    print()


if __name__ == "__main__":
    main()
