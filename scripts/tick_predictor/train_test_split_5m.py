#!/usr/bin/env python3
"""Proper OOS validation for multi-timeframe TickDirectionPredictor.

Trains on a defined period, then runs OOS backtest on held-out period
with embargo gap to prevent leakage.

Usage:
    python -u scripts/tick_predictor/train_test_split_5m.py --sweep 2>&1 | tee logs/tick_5m_oos.log
    python -u scripts/tick_predictor/train_test_split_5m.py --freq 15m --tp-ticks 20 --sl-ticks 12 --horizon 8 --sweep
    python -u scripts/tick_predictor/train_test_split_5m.py --freq 30m --tp-ticks 32 --sl-ticks 20 --horizon 6 --sweep
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

import lightgbm as lgb
import numpy as np
import polars as pl

from src.core.logging import configure_logging, get_logger
from src.signals.tick_predictor.features.feature_builder_5m import FEATURE_NAMES_5M
from src.signals.tick_predictor.labels.triple_barrier_5m import (
    TripleBarrierConfig5M,
    TripleBarrierLabeler5M,
)
from src.signals.tick_predictor.model.calibrator import TemperatureCalibrator

logger = get_logger("tick_predictor.oos_5m")

# ── Split config (defaults, overridden by CLI) ─────────────────────
FEATURE_START = "2025-01-01"
FEATURE_END = "2026-03-14"
TRAIN_END = "2025-09-01"       # Train on Jan-Aug 2025
EMBARGO_END = "2025-10-01"     # 1-month embargo gap
# OOS = Oct 2025 - Mar 2026

DATA_DIR = Path("data/tick_predictor")

# Backtest constants
MES_TICK_SIZE = 0.25
MES_TICK_VALUE = 1.25
MES_POINT_VALUE = 5.0
COMMISSION_RT = 0.59


def freq_minutes(freq: str) -> int:
    return int(freq.replace("m", ""))


def load_data(freq: str, tp_ticks: int, sl_ticks: int, horizon_bars: int):
    """Load cached features and labels for a given timeframe."""
    cfg = TripleBarrierConfig5M(
        vertical_barrier_bars=horizon_bars,
        tp_ticks=tp_ticks,
        sl_ticks=sl_ticks,
    )
    suffix = f"_{freq}_v1_h{cfg.vertical_barrier_bars}_tp{cfg.tp_ticks}_sl{cfg.sl_ticks}_rth"

    feat_path = DATA_DIR / f"features{suffix}_{FEATURE_START}_{FEATURE_END}.parquet"
    labels_path = DATA_DIR / f"labels{suffix}_{FEATURE_START}_{FEATURE_END}.parquet"

    if not feat_path.exists():
        raise FileNotFoundError(
            f"No cached features at {feat_path}.\n"
            f"Run: python -u scripts/tick_predictor/train_5m.py "
            f"--freq {freq} --tp-ticks {tp_ticks} --sl-ticks {sl_ticks} "
            f"--horizon {horizon_bars} --start {FEATURE_START} --end {FEATURE_END}"
        )

    print(f"  Loading features: {feat_path}")
    features_df = pl.read_parquet(feat_path)
    print(f"  Features: {len(features_df):,} rows")

    if not labels_path.exists():
        raise FileNotFoundError(f"No cached labels at {labels_path}")

    print(f"  Loading labels: {labels_path}")
    labels_df = pl.read_parquet(labels_path)
    print(f"  Labels: {len(labels_df):,} rows")

    return features_df, labels_df


def train_oos_model(features_df, labels_df):
    """Train on Jan-Aug 2025, calibrate on last 15 days."""
    import datetime as dt

    train_end_ns = int(
        dt.datetime.combine(
            dt.date.fromisoformat(TRAIN_END), dt.time.min
        ).timestamp() * 1e9
    )

    # Join features + labels
    joined = features_df.join(labels_df, on="timestamp_ns", how="inner")
    joined = joined.drop_nulls(subset=FEATURE_NAMES_5M)
    joined = joined.filter(pl.col("label") != 0)

    # Split
    train_df = joined.filter(pl.col("timestamp_ns") < train_end_ns)
    print(f"\n  TRAINING SPLIT:")
    print(f"    Train period: {FEATURE_START} to {TRAIN_END}")
    print(f"    Train rows:   {len(train_df):,}")

    # Label distribution
    dist = train_df.group_by("label").len().sort("label")
    for row in dist.iter_rows(named=True):
        name = {-1: "DOWN", 1: "UP"}.get(row["label"], "?")
        pct = row["len"] / len(train_df) * 100
        print(f"    {name}: {row['len']:,} ({pct:.1f}%)")

    # Internal val split
    val_days = 15
    timestamps = train_df["timestamp_ns"].to_numpy()
    ts_dates = (
        pl.Series("ts", timestamps).cast(pl.Datetime("ns")).dt.date().to_numpy()
    )
    unique_dates = np.unique(ts_dates)
    if len(unique_dates) > val_days:
        val_start_date = unique_dates[-val_days]
        val_mask = ts_dates >= val_start_date
    else:
        val_mask = np.zeros(len(train_df), dtype=bool)
        val_mask[int(len(train_df) * 0.9):] = True

    train_mask = ~val_mask

    X_all = train_df.select(FEATURE_NAMES_5M).to_numpy().astype(np.float32)
    y_raw = train_df["label"].to_numpy()
    y_all = np.where(y_raw == 1, 1, 0).astype(np.int32)
    w_all = train_df["sample_weight"].to_numpy().astype(np.float32)

    X_train, X_val = X_all[train_mask], X_all[val_mask]
    y_train, y_val = y_all[train_mask], y_all[val_mask]
    w_train = w_all[train_mask]

    print(f"    CV train: {len(X_train):,}, Cal val: {len(X_val):,}")

    # Walk-forward CV
    print("\n  Walk-forward CV...")
    t0 = _time.perf_counter()
    n_splits = 5
    embargo = 6
    n = len(X_train)
    min_train = int(n * 0.5)
    fold_size = (n - min_train) // n_splits

    cv_accs = []
    for fold in range(n_splits):
        test_start = min_train + fold * fold_size
        test_end = min_train + (fold + 1) * fold_size if fold < n_splits - 1 else n
        train_end_idx = max(0, test_start - embargo)

        if test_end <= test_start or train_end_idx < 100:
            continue

        Xtr = X_train[:train_end_idx]
        ytr = y_train[:train_end_idx]
        wtr = w_train[:train_end_idx]
        Xte = X_train[test_start:test_end]
        yte = y_train[test_start:test_end]

        vs = max(1, int(len(Xtr) * 0.1))
        params = {
            "objective": "binary",
            "num_leaves": 31,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
            "is_unbalance": True,
            "device": "gpu",
            "verbose": -1,
            "seed": 42,
        }
        train_set = lgb.Dataset(Xtr[:-vs], ytr[:-vs], weight=wtr[:-vs],
                                feature_name=FEATURE_NAMES_5M)
        val_set = lgb.Dataset(Xtr[-vs:], ytr[-vs:], reference=train_set)
        callbacks = [
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        booster = lgb.train(params, train_set, num_boost_round=300,
                            valid_sets=[val_set], callbacks=callbacks)

        pred = booster.predict(Xte)
        acc = float(np.mean((pred > 0.5).astype(int) == yte))
        cv_accs.append(acc)
        print(f"    Fold {fold}: acc={acc:.4f} (trees={booster.num_trees()})")

    mean_acc = float(np.mean(cv_accs))
    std_acc = float(np.std(cv_accs))
    print(f"    Mean accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"    ({_time.perf_counter() - t0:.1f}s)")

    # Train final model on ALL train data
    print("\n  Training final model...")
    t0 = _time.perf_counter()
    vs = max(1, int(len(X_train) * 0.1))
    params = {
        "objective": "binary",
        "num_leaves": 31,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "is_unbalance": True,
        "device": "gpu",
        "verbose": -1,
        "seed": 42,
    }
    train_set = lgb.Dataset(X_train[:-vs], y_train[:-vs], weight=w_train[:-vs],
                            feature_name=FEATURE_NAMES_5M)
    val_set = lgb.Dataset(X_train[-vs:], y_train[-vs:], reference=train_set)
    callbacks = [
        lgb.early_stopping(30, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    booster = lgb.train(params, train_set, num_boost_round=300,
                        valid_sets=[val_set], callbacks=callbacks)
    print(f"    Trained ({_time.perf_counter() - t0:.1f}s, {booster.num_trees()} trees)")

    # Calibrate
    print("  Calibrating...")
    raw_val = booster.predict(X_val)
    raw_3class = np.column_stack([1 - raw_val, np.zeros(len(raw_val)), raw_val])
    y_val_3class = np.where(y_val == 1, 2, 0).astype(np.int32)

    calibrator = TemperatureCalibrator()
    calibrator.fit(raw_3class, y_val_3class)
    cal_proba = calibrator.predict_proba_calibrated(raw_3class)
    val_acc = float(np.mean(np.argmax(cal_proba, axis=1) == y_val_3class))
    val_ece = calibrator.compute_ece(cal_proba, y_val_3class)
    print(f"    Temperature: {calibrator.temperature:.4f}")
    print(f"    Val accuracy: {val_acc:.4f}")
    print(f"    Val ECE: {val_ece:.4f}")

    # Feature importance
    imp = booster.feature_importance(importance_type="gain")
    top_idx = np.argsort(imp)[::-1][:10]
    print(f"\n  Top features:")
    for i in top_idx:
        print(f"    {FEATURE_NAMES_5M[i]:25s} {imp[i]:,.0f}")

    return booster, calibrator, mean_acc, std_acc, val_acc


def run_oos_backtest(
    features_df,
    labels_df,
    model,
    calibrator,
    tp_ticks: int,
    sl_ticks: int,
    horizon_bars: int,
    freq: str = "5m",
    threshold: float = 0.60,
    cooldown_bars: int = 2,
    slippage_ticks: float = 1.0,
) -> dict:
    """Run backtest on OOS period."""
    import datetime as dt

    embargo_end_ns = int(
        dt.datetime.combine(
            dt.date.fromisoformat(EMBARGO_END), dt.time.min
        ).timestamp() * 1e9
    )

    # Join features + labels for OOS
    joined = features_df.join(labels_df, on="timestamp_ns", how="inner")
    joined = joined.drop_nulls(subset=FEATURE_NAMES_5M)
    oos_df = joined.filter(pl.col("timestamp_ns") >= embargo_end_ns)

    print(f"\n  OOS BACKTEST: {EMBARGO_END} to {FEATURE_END}")
    print(f"    OOS rows: {len(oos_df):,}")
    print(f"    Threshold: {threshold}, Cooldown: {cooldown_bars} bars")

    if len(oos_df) == 0:
        return _empty_result(threshold, cooldown_bars)

    # Inference
    X = oos_df.select(FEATURE_NAMES_5M).to_numpy().astype(np.float32)
    raw_proba = model.predict(X)  # binary P(UP)
    p_up = raw_proba
    p_down = 1.0 - raw_proba

    # Bar data for simulation
    timestamps_ns = oos_df["timestamp_ns"].to_numpy()
    bars_ohlc = _load_bars_for_sim(timestamps_ns, freq)
    close = bars_ohlc["close"]
    high = bars_ohlc["high"]
    low = bars_ohlc["low"]

    tp_pts = tp_ticks * MES_TICK_SIZE
    sl_pts = sl_ticks * MES_TICK_SIZE
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
                exit_reason = "stop"  # conservative: assume stop hit first
            elif stop_hit:
                exit_reason = "stop"
            elif target_hit:
                exit_reason = "target"
            elif bars_held >= horizon_bars:
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
        if i >= n_bars - horizon_bars:
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
        return _empty_result(threshold, cooldown_bars)

    net_pnls = np.array([t["net_pnl"] for t in trades])
    wins = net_pnls[net_pnls > 0]
    losses = net_pnls[net_pnls <= 0]

    wr = len(wins) / len(trades)
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
    pf = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else float("inf")

    # Daily P&L for Sharpe
    daily_pnl = {}
    for t in trades:
        d = t["entry_date"]
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

    # Max drawdown
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

    longs = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]

    n_trading_days = len(daily_pnl)
    profitable_days = sum(1 for v in daily_pnl.values() if v > 0)
    avg_daily_pnl = float(np.mean(daily_returns))
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
        "worst_day": worst_day,
        "best_day": best_day,
    }


def _empty_result(threshold, cooldown):
    return {
        "threshold": threshold, "cooldown": cooldown, "trades": 0,
        "wr": 0, "avg_win": 0, "avg_loss": 0, "pf": 0,
        "net_pnl": 0, "sharpe": 0, "sortino": 0,
        "max_dd": 0, "max_dd_pct": 0, "exit_reasons": {},
        "longs": 0, "shorts": 0, "avg_bars_held": 0, "avg_confidence": 0,
        "n_trading_days": 0, "profitable_days": 0,
        "avg_daily_pnl": 0, "worst_day": 0, "best_day": 0,
    }


def _load_bars_for_sim(timestamps_ns: np.ndarray, freq: str = "5m") -> dict:
    """Load bar OHLC data matching timestamps."""
    import datetime as dt
    from src.data.bars import resample_bars

    bar_mins = freq_minutes(freq)
    min_ts = int(timestamps_ns.min())
    max_ts = int(timestamps_ns.max())
    min_dt = dt.datetime.fromtimestamp(min_ts / 1e9, tz=dt.timezone.utc).replace(tzinfo=None)
    max_dt = dt.datetime.fromtimestamp(max_ts / 1e9, tz=dt.timezone.utc).replace(tzinfo=None)

    years = range(min_dt.year, max_dt.year + 1)
    paths = [f"data/parquet/year={y}/data.parquet" for y in years
             if Path(f"data/parquet/year={y}/data.parquet").exists()]

    bars_1s = (
        pl.scan_parquet(paths)
        .filter(
            (pl.col("timestamp") >= min_dt - dt.timedelta(minutes=bar_mins))
            & (pl.col("timestamp") <= max_dt + dt.timedelta(minutes=bar_mins))
        )
        .sort("timestamp")
        .collect()
    )

    bars = resample_bars(bars_1s, freq)
    if "timestamp_ns" not in bars.columns:
        bars = bars.with_columns(
            pl.col("timestamp").dt.epoch("ns").alias("timestamp_ns")
        )

    # Build lookup by timestamp_ns
    ts_to_idx = {}
    bar_ts = bars["timestamp_ns"].to_numpy()
    for idx, ts in enumerate(bar_ts):
        ts_to_idx[int(ts)] = idx

    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()

    # Map OOS timestamps to bar indices
    out_close = np.empty(len(timestamps_ns), dtype=np.float64)
    out_high = np.empty(len(timestamps_ns), dtype=np.float64)
    out_low = np.empty(len(timestamps_ns), dtype=np.float64)

    last_valid = 0
    for i, ts in enumerate(timestamps_ns):
        idx = ts_to_idx.get(int(ts))
        if idx is not None:
            out_close[i] = close[idx]
            out_high[i] = high[idx]
            out_low[i] = low[idx]
            last_valid = i
        else:
            # Forward fill from last known
            if i > 0:
                out_close[i] = out_close[i - 1]
                out_high[i] = out_high[i - 1]
                out_low[i] = out_low[i - 1]
            else:
                out_close[i] = close[0]
                out_high[i] = high[0]
                out_low[i] = low[0]

    return {"close": out_close, "high": out_high, "low": out_low}


def print_result(r: dict) -> None:
    print(f"    Trades:          {r['trades']:,}")
    print(f"    Win rate:        {r['wr']*100:.1f}%")
    print(f"    Avg win:         ${r['avg_win']:.2f}")
    print(f"    Avg loss:        ${r['avg_loss']:.2f}")
    print(f"    Profit factor:   {r['pf']:.2f}")
    print(f"    Net PnL:         ${r['net_pnl']:,.2f}")
    print(f"    Sharpe (daily):  {r['sharpe']:.2f}")
    print(f"    Sortino (daily): {r['sortino']:.2f}")
    print(f"    Max DD:          ${r['max_dd']:,.2f} ({r['max_dd_pct']:.1f}%)")
    print(f"    Avg bars held:   {r['avg_bars_held']:.1f}")
    print(f"    Avg confidence:  {r['avg_confidence']:.3f}")
    print(f"    Long/Short:      {r['longs']}/{r['shorts']}")
    print(f"    Exit reasons:    {r['exit_reasons']}")
    print(f"    Trading days:    {r['n_trading_days']} ({r['profitable_days']} profitable)")
    print(f"    Avg daily PnL:   ${r['avg_daily_pnl']:,.2f}")
    print(f"    Worst day:       ${r['worst_day']:,.2f}")
    print(f"    Best day:        ${r['best_day']:,.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--freq", default="5m", help="Bar frequency: 5m, 15m, 30m")
    parser.add_argument("--tp-ticks", type=int, default=12)
    parser.add_argument("--sl-ticks", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()

    configure_logging(log_level="INFO")

    bar_mins = freq_minutes(args.freq)

    print(f"\n{'='*70}")
    print(f"  {args.freq.upper()} TICK PREDICTOR — PROPER OOS VALIDATION")
    print(f"  Train:   {FEATURE_START} to {TRAIN_END}")
    print(f"  Embargo: {TRAIN_END} to {EMBARGO_END}")
    print(f"  OOS:     {EMBARGO_END} to {FEATURE_END}")
    print(f"  TP/SL:   {args.tp_ticks}/{args.sl_ticks} ticks, "
          f"Horizon: {args.horizon} bars ({args.horizon * bar_mins}m)")
    print(f"{'='*70}\n")

    features_df, labels_df = load_data(args.freq, args.tp_ticks, args.sl_ticks, args.horizon)

    # Train
    print(f"\n{'='*70}")
    print(f"  PHASE 1: TRAINING (on {FEATURE_START} to {TRAIN_END})")
    print(f"{'='*70}")
    t0 = _time.perf_counter()
    model, calibrator, cv_acc, cv_std, val_acc = train_oos_model(features_df, labels_df)
    print(f"\n  Training complete in {_time.perf_counter() - t0:.1f}s")

    # OOS backtest
    print(f"\n{'='*70}")
    print(f"  PHASE 2: OOS BACKTEST ({EMBARGO_END} to {FEATURE_END})")
    print(f"{'='*70}")

    if args.sweep:
        thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]
        cooldowns = [1, 2, 3, 6]

        print(f"\n  Sweeping {len(thresholds)} thresholds x {len(cooldowns)} cooldowns")
        print(f"  {'thresh':>6} {'cd':>3} | {'trades':>6} {'WR':>6} {'PnL':>10} "
              f"{'Sharpe':>7} {'PF':>5} {'DD%':>6} {'avgDay':>8} {'worDay':>8} | "
              f"{'profDays':>8}")
        print(f"  {'-'*95}")

        all_results = []
        for thresh in thresholds:
            for cd in cooldowns:
                r = run_oos_backtest(
                    features_df, labels_df, model, calibrator,
                    tp_ticks=args.tp_ticks, sl_ticks=args.sl_ticks,
                    horizon_bars=args.horizon, freq=args.freq,
                    threshold=thresh, cooldown_bars=cd,
                )
                all_results.append(r)
                pnl_str = f"${r['net_pnl']:>9,.0f}"
                print(f"  {thresh:>6.2f} {cd:>3} | {r['trades']:>6} {r['wr']*100:>5.1f}% "
                      f"{pnl_str} {r['sharpe']:>7.2f} {r['pf']:>5.2f} "
                      f"{r['max_dd_pct']:>5.1f}% ${r['avg_daily_pnl']:>7,.0f} "
                      f"${r['worst_day']:>7,.0f} | "
                      f"{r['profitable_days']}/{r['n_trading_days']}")
                sys.stdout.flush()

        all_results.sort(key=lambda x: x["sharpe"], reverse=True)

        print(f"\n  TOP 5 BY DAILY SHARPE:")
        for r in all_results[:5]:
            print(f"    thresh={r['threshold']:.2f} cd={r['cooldown']:>1} | "
                  f"{r['trades']} trades, {r['wr']*100:.1f}% WR, "
                  f"${r['net_pnl']:,.0f} PnL, Sharpe {r['sharpe']:.2f}, PF {r['pf']:.2f}, "
                  f"avg/day ${r['avg_daily_pnl']:,.0f}, worst day ${r['worst_day']:,.0f}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"results/tick_predictor_{args.freq}")
        results_path = results_dir / f"oos_sweep_{ts}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved: {results_path}")

    else:
        r = run_oos_backtest(
            features_df, labels_df, model, calibrator,
            tp_ticks=args.tp_ticks, sl_ticks=args.sl_ticks,
            horizon_bars=args.horizon, freq=args.freq,
            threshold=0.55, cooldown_bars=2,
        )
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
    print()


if __name__ == "__main__":
    main()
