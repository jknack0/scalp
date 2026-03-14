#!/usr/bin/env python3
"""Multi-year validation for Donchian Channel Breakout strategy.

Runs the YAML-tuned params across 5+ years to check:
  1. Full-period backtest (2020-2025)
  2. Year-by-year breakdown (any year deeply negative?)
  3. Rolling 2-year window stability

Usage:
    python -u scripts/backtest/donchian_validate.py
    python -u scripts/backtest/donchian_validate.py --start 2020-01-01 --end 2025-03-01
"""

import argparse
import logging
import os
import sys
import time as _time
from datetime import date, datetime, time

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


# Current YAML params (tuned on 2024 sweep)
YAML_PARAMS = {
    "target_atr": 2.0,
    "stop_atr": 1.5,
    "adx_min": 30.0,
    "rvol_min": 0.8,
    "time_stop_bars": 18,
    "adx_kill": 12.0,
    "width_min": 2.0,
    "width_max": 30.0,
    "session_start": 600,  # 10:00 AM
    "session_end": 840,    # 2:00 PM
    "max_signals": 5,
}

SIGNAL_NAMES = ["donchian_channel", "adx", "relative_volume", "atr", "session_time"]


def load_enriched_bars(start: date, end: date) -> pl.DataFrame:
    """Load 5m bars enriched with vectorized signals."""
    cache_name = BarCache.enriched_name("5m", SIGNAL_NAMES)

    cached = BarCache.load(cache_name)
    if cached is not None:
        print(f"  [Cache HIT] {cache_name} ({len(cached):,} rows)")
        return cached

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

    enriched = enrich_bars(bars_df, SIGNAL_NAMES)
    save_df = enriched.drop(["_et_ts", "_bar_date"]).with_columns(
        pl.col("timestamp").dt.replace_time_zone(None).alias("timestamp"),
    )
    BarCache.save(cache_name, save_df)

    print(f"  Enriched {len(enriched):,} bars in {_time.perf_counter() - t0:.1f}s")
    return save_df


def run_backtest(bars: pl.DataFrame, start: date, end: date, params: dict | None = None) -> dict:
    """Run a single donchian backtest with given params."""
    p = params or YAML_PARAMS

    filters = [
        {"signal": "session_time", "expr": f">= {p['session_start']}"},
        {"signal": "session_time", "expr": f"<= {p['session_end']}"},
        {"signal": "donchian_channel", "expr": "passes"},
        {"signal": "adx", "expr": f"> {p['adx_min']}"},
        {"signal": "relative_volume", "expr": f">= {p['rvol_min']}"},
    ]

    cfg = {
        "strategy": {
            "strategy_id": "donchian_breakout",
            "max_signals_per_day": p["max_signals"],
        },
        "exit": {"time_stop_minutes": p["time_stop_bars"] * 5},
        "donchian": {
            "target_atr": p["target_atr"],
            "stop_atr": p["stop_atr"],
            "width_min": p["width_min"],
            "width_max": p["width_max"],
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
        {"type": "time_stop", "enabled": True, "max_bars": p["time_stop_bars"]},
        {"type": "price_vs_signal_exit", "enabled": True,
         "signal": "donchian_channel", "long_field": "exit_lower", "short_field": "exit_upper"},
        {"type": "signal_bound_exit", "enabled": True,
         "signal": "adx", "lower_bound": p["adx_kill"]},
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2025-03-01")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"\n{'='*70}")
    print(f"  DONCHIAN BREAKOUT VALIDATION: {start} to {end}")
    print(f"  Params tuned on 2024 sweep — testing out-of-sample stability")
    print(f"{'='*70}\n")

    print(f"  YAML params: {YAML_PARAMS}\n")

    bars = load_enriched_bars(start, end)
    print()

    # ── 1. Full-period backtest ──────────────────────────
    print(f"{'='*70}")
    print(f"  1. FULL PERIOD: {start} to {end}")
    print(f"{'='*70}")

    t0 = _time.perf_counter()
    full = run_backtest(bars, start, end)
    dt = _time.perf_counter() - t0

    print(f"  Trades:        {full['trades']}")
    print(f"  Win Rate:      {full['wr']:.1%}")
    print(f"  Net PnL:       ${full['pnl']:,.2f}")
    print(f"  Sharpe:        {full['sharpe']:.3f}")
    print(f"  Sortino:       {full['sortino']:.3f}")
    print(f"  Profit Factor: {full['pf']:.2f}")
    print(f"  Max DD:        {full['dd']:.1f}%")
    print(f"  Avg Win/Loss:  ${full['avg_win']:.2f} / ${full['avg_loss']:.2f}")
    print(f"  Time:          {dt:.1f}s")
    sys.stdout.flush()

    # ── 2. Year-by-year breakdown ────────────────────────
    print(f"\n{'='*70}")
    print(f"  2. YEAR-BY-YEAR BREAKDOWN")
    print(f"{'='*70}\n")

    hdr = f"  {'Year':>6} | {'Trades':>6} {'WR':>6} {'PnL':>8} {'Sharpe':>7} {'PF':>5} {'DD':>5} {'AvgW':>6} {'AvgL':>7}"
    print(hdr)
    print(f"  {'-'*72}")

    year_results = []
    for year in range(start.year, end.year + 1):
        y_start = date(year, 1, 1)
        y_end = date(year + 1, 1, 1)
        if y_start >= end:
            break
        y_end = min(y_end, end)

        r = run_backtest(bars, y_start, y_end)
        year_results.append((year, r))

        pnl_marker = "  " if r["pnl"] >= 0 else "!!"
        print(f"  {year:>6} | {r['trades']:>6} {r['wr']:>5.1%} ${r['pnl']:>7,.0f} "
              f"{r['sharpe']:>7.2f} {r['pf']:>5.2f} {r['dd']:>4.1f}% "
              f"${r['avg_win']:>5.0f} ${r['avg_loss']:>6.0f} {pnl_marker}")
        sys.stdout.flush()

    profitable_years = sum(1 for _, r in year_results if r["pnl"] > 0)
    total_years = len(year_results)
    print(f"\n  Profitable years: {profitable_years}/{total_years}")

    # ── 3. Rolling 2-year windows ────────────────────────
    print(f"\n{'='*70}")
    print(f"  3. ROLLING 2-YEAR WINDOWS")
    print(f"{'='*70}\n")

    print(f"  {'Window':>15} | {'Trades':>6} {'WR':>6} {'PnL':>8} {'Sharpe':>7} {'PF':>5} {'DD':>5}")
    print(f"  {'-'*65}")

    window_results = []
    for y in range(start.year, end.year - 1):
        w_start = date(y, 1, 1)
        w_end = date(y + 2, 1, 1)
        if w_end > end:
            w_end = end

        r = run_backtest(bars, w_start, w_end)
        window_results.append((f"{y}-{y+1}", r))

        pnl_marker = "  " if r["pnl"] >= 0 else "!!"
        print(f"  {y}-{y+1:>10} | {r['trades']:>6} {r['wr']:>5.1%} ${r['pnl']:>7,.0f} "
              f"{r['sharpe']:>7.2f} {r['pf']:>5.2f} {r['dd']:>4.1f}% {pnl_marker}")
        sys.stdout.flush()

    profitable_windows = sum(1 for _, r in window_results if r["pnl"] > 0)
    total_windows = len(window_results)
    print(f"\n  Profitable 2yr windows: {profitable_windows}/{total_windows}")

    # ── Summary ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Full period ({start} to {end}):")
    print(f"    {full['trades']} trades, {full['wr']:.1%} WR, Sharpe {full['sharpe']:.2f}, "
          f"${full['pnl']:,.0f} PnL, {full['dd']:.1f}% DD")
    print(f"  Year stability: {profitable_years}/{total_years} years profitable")
    print(f"  Window stability: {profitable_windows}/{total_windows} 2yr windows profitable")

    # Verdict
    if full["sharpe"] > 1.0 and profitable_years >= total_years * 0.6 and full["trades"] >= 50:
        print(f"\n  VERDICT: PASS — strategy appears robust across multiple periods")
    elif full["sharpe"] > 0.5 and profitable_years >= total_years * 0.5:
        print(f"\n  VERDICT: MARGINAL — may be overfit to training period")
    else:
        print(f"\n  VERDICT: FAIL — likely overfit or not robust")

    print()


if __name__ == "__main__":
    main()
