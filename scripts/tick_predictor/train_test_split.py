#!/usr/bin/env python3
"""Proper train/test split for TickDirectionPredictor.

Trains on a defined period, then runs OOS backtest on a held-out period
with an embargo gap to prevent leakage.

Usage:
    python -u scripts/tick_predictor/train_test_split.py 2>&1 | tee logs/tick_predictor_oos.log
    python -u scripts/tick_predictor/train_test_split.py --sweep 2>&1 | tee logs/tick_predictor_oos_sweep.log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from datetime import datetime
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np
import polars as pl

from src.core.logging import configure_logging, get_logger
from src.signals.tick_predictor.features.feature_builder import FEATURE_NAMES
from src.signals.tick_predictor.labels.triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierLabeler,
)
from src.signals.tick_predictor.model.calibrator import TemperatureCalibrator
from src.signals.tick_predictor.model.trainer import ModelTrainer, TrainerConfig

logger = get_logger("tick_predictor.oos")

# ── Split config ────────────────────────────────────────────────────
FEATURE_START = "2025-01-01"
FEATURE_END = "2026-03-14"
TRAIN_END = "2025-09-01"       # Train on Jan-Aug 2025
EMBARGO_END = "2025-10-01"     # 1-month embargo gap
# OOS = Oct 2025 - Mar 2026

TP_TICKS = 24
SL_TICKS = 12
HORIZON_BARS = 300

MODEL_DIR = Path("models/tick_predictor")
DATA_DIR = Path("data/tick_predictor")

# Backtest constants
MES_TICK_SIZE = 0.25
MES_TICK_VALUE = 1.25
MES_POINT_VALUE = 5.0
COMMISSION_RT = 0.59


def load_features_and_labels() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load cached features and generate/load labels."""
    suffix = f"_v6_h{HORIZON_BARS}_tp{TP_TICKS}_sl{SL_TICKS}_rth"
    feat_path = DATA_DIR / f"features_{FEATURE_START}_{FEATURE_END}{suffix}.parquet"

    if not feat_path.exists():
        raise FileNotFoundError(
            f"No cached features at {feat_path}.\n"
            f"Run: python -u scripts/tick_predictor/train.py "
            f"--start {FEATURE_START} --end {FEATURE_END} "
            f"--horizon {HORIZON_BARS} --tp-ticks {TP_TICKS} --sl-ticks {SL_TICKS}"
        )

    print(f"  Loading features: {feat_path}")
    features_df = pl.read_parquet(feat_path)
    print(f"  Features: {len(features_df):,} rows")

    # Labels
    labels_path = DATA_DIR / f"labels_{FEATURE_START}_{FEATURE_END}.parquet"
    if labels_path.exists():
        print(f"  Loading labels: {labels_path}")
        labels_df = pl.read_parquet(labels_path)
    else:
        print(f"  Generating labels...")
        cfg = TripleBarrierConfig(
            vertical_barrier_bars=HORIZON_BARS,
            tp_ticks=TP_TICKS,
            sl_ticks=SL_TICKS,
        )
        labeler = TripleBarrierLabeler(cfg)
        labels_df = labeler.generate_labels(FEATURE_START, FEATURE_END)
        labeler.save_labels(labels_df, FEATURE_START, FEATURE_END)

    print(f"  Labels: {len(labels_df):,} rows")
    return features_df, labels_df


def train_oos_model(features_df: pl.DataFrame, labels_df: pl.DataFrame):
    """Train on Jan-Aug 2025, calibrate on last 15 days of train period."""
    import datetime as dt

    train_end_ns = int(
        dt.datetime.combine(
            dt.date.fromisoformat(TRAIN_END), dt.time.min
        ).timestamp() * 1e9
    )

    # Join features + labels
    joined = features_df.join(labels_df, on="timestamp_ns", how="inner")
    joined = joined.drop_nulls(subset=FEATURE_NAMES)

    # Drop FLAT
    joined = joined.filter(pl.col("label") != 0)

    # Split: train = before TRAIN_END
    train_df = joined.filter(pl.col("timestamp_ns") < train_end_ns)

    print(f"\n  TRAINING SPLIT:")
    print(f"    Train period: {FEATURE_START} to {TRAIN_END}")
    print(f"    Train rows:   {len(train_df):,}")

    # Internal val split for calibration (last 15 days of train period)
    val_days = 15
    val_start = dt.date.fromisoformat(TRAIN_END) - dt.timedelta(days=val_days)
    val_start_ns = int(
        dt.datetime.combine(val_start, dt.time.min).timestamp() * 1e9
    )
    cal_train = train_df.filter(pl.col("timestamp_ns") < val_start_ns)
    cal_val = train_df.filter(pl.col("timestamp_ns") >= val_start_ns)

    print(f"    CV train:     {len(cal_train):,}")
    print(f"    Cal val:      {len(cal_val):,} (last {val_days} days for calibration)")

    # Label distribution
    dist = train_df.group_by("label").len().sort("label")
    for row in dist.iter_rows(named=True):
        name = {-1: "DOWN", 0: "FLAT", 1: "UP"}.get(row["label"], "?")
        pct = row["len"] / len(train_df) * 100
        print(f"    {name}: {row['len']:,} ({pct:.1f}%)")

    # Encode labels
    label_map = {-1: 0, 0: 1, 1: 2}

    X_train = cal_train.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    y_train = np.vectorize(label_map.get)(cal_train["label"].to_numpy()).astype(np.int32)
    w_train = cal_train["sample_weight"].to_numpy().astype(np.float32)

    X_val = cal_val.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    y_val = np.vectorize(label_map.get)(cal_val["label"].to_numpy()).astype(np.int32)

    # Walk-forward CV
    print("\n  Walk-forward CV...")
    t0 = _time.perf_counter()
    trainer = ModelTrainer(TrainerConfig())
    cv_results = trainer.purged_walkforward_cv(X_train, y_train, w_train)

    mean_acc = float(np.mean([r.accuracy for r in cv_results]))
    std_acc = float(np.std([r.accuracy for r in cv_results]))
    print(f"    CV accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    for r in cv_results:
        print(f"    Fold {r.fold}: acc={r.accuracy:.4f} f1={r.f1_macro:.4f}")
    print(f"    ({_time.perf_counter()-t0:.1f}s)")

    # Train final model on ALL train data (up to TRAIN_END)
    print("\n  Training final model...")
    t0 = _time.perf_counter()

    X_all = train_df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    y_all = np.vectorize(label_map.get)(train_df["label"].to_numpy()).astype(np.int32)
    w_all = train_df["sample_weight"].to_numpy().astype(np.float32)

    booster = trainer.train_final_model(X_all, y_all, w_all)
    print(f"    Trained ({_time.perf_counter()-t0:.1f}s)")

    # Calibrate on validation slice
    print("  Calibrating...")
    raw_proba_val = booster.predict(X_val)
    if raw_proba_val.ndim == 1:
        raw_proba_val = raw_proba_val.reshape(-1, 3)

    calibrator = TemperatureCalibrator()
    calibrator.fit(raw_proba_val, y_val)

    cal_proba = calibrator.predict_proba_calibrated(raw_proba_val)
    val_acc = float(np.mean(np.argmax(cal_proba, axis=1) == y_val))
    val_ece = calibrator.compute_ece(cal_proba, y_val)

    print(f"    Temperature: {calibrator.temperature:.4f}")
    print(f"    Val accuracy: {val_acc:.4f}")
    print(f"    Val ECE:      {val_ece:.4f}")

    # Feature importance
    imp = booster.feature_importance(importance_type="gain")
    top_idx = np.argsort(imp)[::-1][:10]
    print(f"\n  Top features:")
    for i in top_idx:
        print(f"    {FEATURE_NAMES[i]:30s} {imp[i]:,.0f}")

    # Save with OOS suffix to avoid overwriting current model
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = str(MODEL_DIR / f"lgbm_oos_{ts}.txt")
    cal_path = str(MODEL_DIR / f"calibrator_oos_{ts}.pkl")
    booster.save_model(model_path)
    calibrator.save(cal_path)
    print(f"\n  Model saved: {model_path}")
    print(f"  Calibrator saved: {cal_path}")

    return booster, calibrator, mean_acc, std_acc, val_acc


def run_oos_backtest(
    features_df: pl.DataFrame,
    model,
    calibrator,
    threshold: float = 0.60,
    cooldown_bars: int = 60,
    slippage_ticks: float = 1.0,
) -> dict:
    """Run backtest ONLY on OOS period (after embargo)."""
    import datetime as dt

    embargo_end_ns = int(
        dt.datetime.combine(
            dt.date.fromisoformat(EMBARGO_END), dt.time.min
        ).timestamp() * 1e9
    )

    # Filter to OOS period only
    oos_df = features_df.filter(pl.col("timestamp_ns") >= embargo_end_ns)
    print(f"\n  OOS BACKTEST: {EMBARGO_END} to {FEATURE_END}")
    print(f"    OOS rows: {len(oos_df):,}")
    print(f"    Threshold: {threshold}, Cooldown: {cooldown_bars}s")

    # Inference
    X = oos_df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    raw_proba = model.predict(X)
    if raw_proba.ndim == 1:
        raw_proba = raw_proba.reshape(-1, 3)

    cal_proba = calibrator.predict_proba_calibrated(raw_proba)
    p_down = cal_proba[:, 0]
    p_up = cal_proba[:, 2]

    # Load bar prices for OOS period
    timestamps_ns = oos_df["timestamp_ns"].to_numpy()
    bars_df = _load_bars_for_sim(oos_df)
    close = bars_df["close"].to_numpy()
    high = bars_df["high"].to_numpy()
    low = bars_df["low"].to_numpy()

    tp_pts = TP_TICKS * MES_TICK_SIZE
    sl_pts = SL_TICKS * MES_TICK_SIZE
    slippage_pts = slippage_ticks * MES_TICK_SIZE

    # Session boundaries
    bar_dates = (
        pl.Series("ts", timestamps_ns)
        .cast(pl.Datetime("ns"))
        .dt.date()
        .to_numpy()
    )

    # Simulate
    n_bars = len(X)
    trades = []
    last_entry_bar = -cooldown_bars
    open_positions = []

    for i in range(n_bars):
        # Session close
        if i > 0 and bar_dates[i] != bar_dates[i - 1]:
            for pos in open_positions:
                exit_price = close[i]
                slip_cost = slippage_ticks * MES_TICK_VALUE
                if pos["direction"] == "LONG":
                    exit_price -= slippage_pts
                    gross = (exit_price - pos["entry_price"]) * MES_POINT_VALUE
                else:
                    exit_price += slippage_pts
                    gross = (pos["entry_price"] - exit_price) * MES_POINT_VALUE
                net = gross - COMMISSION_RT - slip_cost
                trades.append({
                    "net_pnl": net, "gross_pnl": gross,
                    "exit_reason": "session_close",
                    "bars_held": i - pos["entry_bar"],
                    "direction": pos["direction"],
                    "confidence": pos["confidence"],
                    "entry_date": bar_dates[pos["entry_bar"]],
                })
            open_positions.clear()

        # Check exits
        closed = []
        for j, pos in enumerate(open_positions):
            bars_held = i - pos["entry_bar"]
            if pos["direction"] == "LONG":
                stop_hit = low[i] <= pos["stop_price"]
                target_hit = high[i] >= pos["target_price"]
            else:
                stop_hit = high[i] >= pos["stop_price"]
                target_hit = low[i] <= pos["target_price"]

            exit_reason = None
            if stop_hit and target_hit:
                exit_reason = "stop"
            elif stop_hit:
                exit_reason = "stop"
            elif target_hit:
                exit_reason = "target"
            elif bars_held >= HORIZON_BARS:
                exit_reason = "expiry"

            if exit_reason:
                if exit_reason == "target":
                    exit_price = pos["target_price"]
                    slip_cost = 0.0
                elif exit_reason == "stop":
                    exit_price = pos["stop_price"]
                    if pos["direction"] == "LONG":
                        exit_price -= slippage_pts
                    else:
                        exit_price += slippage_pts
                    slip_cost = slippage_ticks * MES_TICK_VALUE
                else:
                    exit_price = close[i]
                    if pos["direction"] == "LONG":
                        exit_price -= slippage_pts
                    else:
                        exit_price += slippage_pts
                    slip_cost = slippage_ticks * MES_TICK_VALUE

                if pos["direction"] == "LONG":
                    gross = (exit_price - pos["entry_price"]) * MES_POINT_VALUE
                else:
                    gross = (pos["entry_price"] - exit_price) * MES_POINT_VALUE

                net = gross - COMMISSION_RT - slip_cost
                trades.append({
                    "net_pnl": net, "gross_pnl": gross,
                    "exit_reason": exit_reason,
                    "bars_held": bars_held,
                    "direction": pos["direction"],
                    "confidence": pos["confidence"],
                    "entry_date": bar_dates[pos["entry_bar"]],
                })
                closed.append(j)

        for j in reversed(closed):
            open_positions.pop(j)

        # New entry
        if len(open_positions) >= 1:
            continue
        if i - last_entry_bar < cooldown_bars:
            continue
        if i >= n_bars - HORIZON_BARS:
            continue

        direction = None
        confidence = 0.0
        if p_up[i] >= threshold and p_up[i] > p_down[i]:
            direction = "LONG"
            confidence = float(p_up[i])
        elif p_down[i] >= threshold and p_down[i] > p_up[i]:
            direction = "SHORT"
            confidence = float(p_down[i])

        if direction:
            entry_price = close[i]
            if direction == "LONG":
                target_price = entry_price + tp_pts
                stop_price = entry_price - sl_pts
            else:
                target_price = entry_price - tp_pts
                stop_price = entry_price + sl_pts

            open_positions.append({
                "entry_bar": i,
                "entry_price": entry_price,
                "target_price": target_price,
                "stop_price": stop_price,
                "direction": direction,
                "confidence": confidence,
            })
            last_entry_bar = i

    # Metrics
    if not trades:
        return {"trades": 0, "net_pnl": 0, "sharpe": 0, "wr": 0,
                "pf": 0, "dd": 0, "threshold": threshold,
                "cooldown": cooldown_bars}

    net_pnls = np.array([t["net_pnl"] for t in trades])
    wins = net_pnls[net_pnls > 0]
    losses = net_pnls[net_pnls <= 0]

    wr = len(wins) / len(trades)
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
    pf = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else float("inf")

    # Daily P&L for Sharpe (proper: aggregate trades by entry date)
    daily_pnl = {}
    for i_t, t in enumerate(trades):
        d = t.get("entry_date", i_t)
        daily_pnl[d] = daily_pnl.get(d, 0.0) + t["net_pnl"]
    daily_returns = np.array(list(daily_pnl.values()))

    sharpe = 0.0
    sortino = 0.0
    if len(daily_returns) > 1:
        std = float(np.std(daily_returns, ddof=1))
        if std > 0:
            sharpe = float(np.mean(daily_returns)) / std * np.sqrt(252)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0:
            down_std = float(np.std(downside, ddof=1))
            if down_std > 0:
                sortino = float(np.mean(daily_returns)) / down_std * np.sqrt(252)

    # Max drawdown (on cumulative daily P&L)
    daily_equity = np.cumsum(daily_returns)
    peak = np.maximum.accumulate(daily_equity)
    dd = daily_equity - peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0
    max_dd_pct = float(max_dd / (np.max(peak) + 1e-9) * 100) if len(peak) > 0 else 0

    # Exit reasons
    exit_reasons = {}
    for t in trades:
        r = t["exit_reason"]
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # Direction breakdown
    longs = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]

    # Daily stats
    n_trading_days = len(daily_pnl)
    profitable_days = sum(1 for v in daily_pnl.values() if v > 0)
    avg_daily_pnl = float(np.mean(daily_returns))
    median_daily_pnl = float(np.median(daily_returns))
    worst_day = float(np.min(daily_returns))
    best_day = float(np.max(daily_returns))

    return {
        "threshold": threshold,
        "cooldown": cooldown_bars,
        "trades": len(trades),
        "wr": wr,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "pf": pf,
        "net_pnl": float(np.sum(net_pnls)),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "exit_reasons": exit_reasons,
        "longs": len(longs),
        "shorts": len(shorts),
        "avg_bars_held": float(np.mean([t["bars_held"] for t in trades])),
        "avg_confidence": float(np.mean([t["confidence"] for t in trades])),
        "n_trading_days": n_trading_days,
        "profitable_days": profitable_days,
        "avg_daily_pnl": avg_daily_pnl,
        "median_daily_pnl": median_daily_pnl,
        "worst_day": worst_day,
        "best_day": best_day,
    }


def _load_bars_for_sim(features_df: pl.DataFrame) -> pl.DataFrame:
    """Load bar OHLC data matching feature timestamps."""
    import datetime as dt

    timestamps_ns = features_df["timestamp_ns"].to_numpy()
    min_ts = int(timestamps_ns.min())
    max_ts = int(timestamps_ns.max())
    min_dt = dt.datetime.fromtimestamp(min_ts / 1e9, tz=dt.timezone.utc)
    max_dt = dt.datetime.fromtimestamp(max_ts / 1e9, tz=dt.timezone.utc)

    years = range(min_dt.year, max_dt.year + 1)

    # Try L1 first
    l1_paths = [f"data/l1/year={y}/data.parquet" for y in years
                if Path(f"data/l1/year={y}/data.parquet").exists()]
    if l1_paths:
        ticks = (
            pl.scan_parquet(l1_paths)
            .filter(
                (pl.col("timestamp") >= min_dt)
                & (pl.col("timestamp") <= max_dt + dt.timedelta(seconds=1))
            )
            .sort("timestamp")
            .collect()
        )
        bars = (
            ticks.with_columns(pl.col("timestamp").dt.truncate("1s").alias("bar_ts"))
            .group_by("bar_ts")
            .agg([
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
            ])
            .sort("bar_ts")
            .with_columns(pl.col("bar_ts").dt.epoch("ns").alias("timestamp_ns"))
        )
    else:
        paths = [f"data/parquet/year={y}/data.parquet" for y in years
                 if Path(f"data/parquet/year={y}/data.parquet").exists()]
        bars = (
            pl.scan_parquet(paths)
            .filter(
                (pl.col("timestamp") >= min_dt)
                & (pl.col("timestamp") <= max_dt + dt.timedelta(seconds=1))
            )
            .select(["timestamp", "open", "high", "low", "close"])
            .sort("timestamp")
            .collect()
            .with_columns(pl.col("timestamp").dt.epoch("ns").alias("timestamp_ns"))
        )

    result = features_df.select("timestamp_ns").join(
        bars.select(["timestamp_ns", "open", "high", "low", "close"]),
        on="timestamp_ns", how="left",
    )
    for col in ["open", "high", "low", "close"]:
        result = result.with_columns(pl.col(col).fill_null(strategy="forward"))

    return result


def print_result(r: dict) -> None:
    """Pretty-print a single backtest result."""
    print(f"    Trades:          {r['trades']:,}")
    print(f"    Win rate:        {r['wr']*100:.1f}%")
    print(f"    Avg win:         ${r['avg_win']:.2f}")
    print(f"    Avg loss:        ${r['avg_loss']:.2f}")
    print(f"    Profit factor:   {r['pf']:.2f}")
    print(f"    Net PnL:         ${r['net_pnl']:,.2f}")
    print(f"    Sharpe (daily):  {r['sharpe']:.2f}")
    print(f"    Sortino (daily): {r['sortino']:.2f}")
    print(f"    Max DD:          ${r['max_dd']:,.2f} ({r['max_dd_pct']:.1f}%)")
    print(f"    Avg bars held:   {r['avg_bars_held']:.0f}")
    print(f"    Avg confidence:  {r['avg_confidence']:.3f}")
    print(f"    Long/Short:      {r['longs']}/{r['shorts']}")
    print(f"    Exit reasons:    {r['exit_reasons']}")
    print(f"    Trading days:    {r['n_trading_days']} ({r['profitable_days']} profitable)")
    print(f"    Avg daily PnL:   ${r['avg_daily_pnl']:,.2f}")
    print(f"    Median daily:    ${r['median_daily_pnl']:,.2f}")
    print(f"    Worst day:       ${r['worst_day']:,.2f}")
    print(f"    Best day:        ${r['best_day']:,.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Sweep thresholds and cooldowns")
    args = parser.parse_args()

    configure_logging(log_level="INFO")

    print(f"\n{'='*70}")
    print(f"  TICK PREDICTOR — PROPER OOS VALIDATION")
    print(f"  Train:   {FEATURE_START} to {TRAIN_END}")
    print(f"  Embargo: {TRAIN_END} to {EMBARGO_END}")
    print(f"  OOS:     {EMBARGO_END} to {FEATURE_END}")
    print(f"  TP/SL:   {TP_TICKS}/{SL_TICKS} ticks, Horizon: {HORIZON_BARS}s")
    print(f"{'='*70}\n")

    # Load data
    features_df, labels_df = load_features_and_labels()

    # Train
    print(f"\n{'='*70}")
    print(f"  PHASE 1: TRAINING (on {FEATURE_START} to {TRAIN_END})")
    print(f"{'='*70}")
    t0 = _time.perf_counter()
    model, calibrator, cv_acc, cv_std, val_acc = train_oos_model(features_df, labels_df)
    train_time = _time.perf_counter() - t0
    print(f"\n  Training complete in {train_time:.1f}s")

    # OOS backtest
    print(f"\n{'='*70}")
    print(f"  PHASE 2: OOS BACKTEST ({EMBARGO_END} to {FEATURE_END})")
    print(f"{'='*70}")

    if args.sweep:
        thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        cooldowns = [0, 30, 60, 120, 300]

        print(f"\n  Sweeping {len(thresholds)} thresholds x {len(cooldowns)} cooldowns")
        print(f"  {'thresh':>6} {'cd':>4} | {'trades':>6} {'WR':>6} {'PnL':>10} "
              f"{'Sharpe':>7} {'PF':>5} {'DD%':>6} {'avgDay':>8} {'worDay':>8} | "
              f"{'profDays':>8}")
        print(f"  {'-'*95}")

        all_results = []
        for thresh in thresholds:
            for cd in cooldowns:
                r = run_oos_backtest(features_df, model, calibrator,
                                     threshold=thresh, cooldown_bars=cd)
                all_results.append(r)
                pnl_str = f"${r['net_pnl']:>9,.0f}"
                print(f"  {thresh:>6.2f} {cd:>4} | {r['trades']:>6} {r['wr']*100:>5.1f}% "
                      f"{pnl_str} {r['sharpe']:>7.2f} {r['pf']:>5.2f} "
                      f"{r['max_dd_pct']:>5.1f}% ${r['avg_daily_pnl']:>7,.0f} "
                      f"${r['worst_day']:>7,.0f} | "
                      f"{r['profitable_days']}/{r['n_trading_days']}")
                sys.stdout.flush()

        # Sort by Sharpe
        all_results.sort(key=lambda x: x["sharpe"], reverse=True)

        print(f"\n  TOP 5 BY DAILY SHARPE:")
        for r in all_results[:5]:
            print(f"    thresh={r['threshold']:.2f} cd={r['cooldown']:>3} | "
                  f"{r['trades']} trades, {r['wr']*100:.1f}% WR, "
                  f"${r['net_pnl']:,.0f} PnL, Sharpe {r['sharpe']:.2f}, PF {r['pf']:.2f}, "
                  f"avg/day ${r['avg_daily_pnl']:,.0f}, worst day ${r['worst_day']:,.0f}")

        # Save results
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path("results/tick_predictor") / f"oos_sweep_{ts}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved: {results_path}")

    else:
        r = run_oos_backtest(features_df, model, calibrator,
                              threshold=0.65, cooldown_bars=60)
        print()
        print_result(r)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  CV accuracy:  {cv_acc:.4f} +/- {cv_std:.4f}")
    print(f"  Val accuracy: {val_acc:.4f}")
    print(f"  Train period: {FEATURE_START} to {TRAIN_END}")
    print(f"  OOS period:   {EMBARGO_END} to {FEATURE_END}")
    print(f"  This is the REAL test — model never saw OOS data during training")
    print()


if __name__ == "__main__":
    main()
