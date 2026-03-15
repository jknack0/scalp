#!/usr/bin/env python3
"""Training pipeline v3 — lagged bar features.

Uses the last N 1m bars (OHLCV) flattened as features to predict
whether the next bar goes UP or DOWN.

Features per lag (5 per bar × N bars):
  - log_return:     log(close[t-i] / close[t-i-1])
  - hl_range:       (high - low) / close, normalized
  - close_position: (close - low) / (high - low)
  - volume_zscore:  volume z-scored over 50-bar rolling window
  - body_ratio:     abs(close - open) / (high - low)

Usage:
    python -u scripts/tick_predictor/train_v3_lagged.py --lag-bars 5 --sweep
    python -u scripts/tick_predictor/train_v3_lagged.py --lag-bars 10 --tp-ticks 4 --sl-ticks 4 --sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from datetime import datetime, time as dt_time
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import lightgbm as lgb
import numpy as np
import polars as pl

from src.core.logging import configure_logging, get_logger
from src.data.bars import resample_bars
from src.signals.tick_predictor.labels.triple_barrier_5m import (
    TripleBarrierConfig5M,
    TripleBarrierLabeler5M,
)
from src.signals.tick_predictor.model.calibrator import TemperatureCalibrator

logger = get_logger("tick_predictor.train_v3")

MODEL_DIR = Path("models/tick_predictor_v3")
DATA_DIR = Path("data/tick_predictor")

MES_TICK_SIZE = 0.25
MES_TICK_VALUE = 1.25
MES_POINT_VALUE = 5.0
COMMISSION_RT = 0.59

FEATURES_PER_LAG = ["log_return", "hl_range", "close_pos", "vol_zscore", "body_ratio"]


def freq_minutes(freq: str) -> int:
    return int(freq.replace("m", ""))


def build_lagged_features(bars: pl.DataFrame, n_lags: int) -> tuple[pl.DataFrame, list[str]]:
    """Build lagged bar features from OHLCV bars.

    For each of the last n_lags bars, compute 5 normalized features.
    Returns (features_df with timestamp_ns + feature columns, feature_names).
    """
    # Compute raw per-bar features
    close = bars["close"]
    open_ = bars["open"]
    high = bars["high"]
    low = bars["low"]
    volume = bars["volume"].cast(pl.Float64)

    # Log return: log(close[t] / close[t-1])
    log_ret = (close / close.shift(1)).log().fill_null(0.0).alias("_log_return")

    # HL range normalized by close
    hl_range = ((high - low) / (close + 1e-9)).alias("_hl_range")

    # Close position within bar: (close - low) / (high - low)
    close_pos = ((close - low) / (high - low + 1e-9)).alias("_close_pos")

    # Volume z-score (50-bar rolling)
    vol_mean = volume.rolling_mean(50).fill_null(volume.mean())
    vol_std = volume.rolling_std(50).fill_null(1.0)
    vol_zscore = ((volume - vol_mean) / (vol_std + 1e-9)).alias("_vol_zscore")

    # Body ratio: abs(close - open) / (high - low)
    body = ((close - open_).abs() / (high - low + 1e-9)).alias("_body_ratio")

    df = bars.select("timestamp_ns").with_columns([
        log_ret, hl_range, close_pos, vol_zscore, body,
    ])

    # Create lagged versions
    feature_names = []
    lag_cols = []
    for lag in range(1, n_lags + 1):
        for base_feat in ["_log_return", "_hl_range", "_close_pos", "_vol_zscore", "_body_ratio"]:
            feat_name = f"{base_feat[1:]}_lag{lag}"
            feature_names.append(feat_name)
            lag_cols.append(
                pl.col(base_feat).shift(lag).alias(feat_name)
            )

    df = df.with_columns(lag_cols)

    # Drop the raw columns, keep only lagged + timestamp_ns
    df = df.select(["timestamp_ns"] + feature_names)

    return df, feature_names


# ── Data loading (same as v2) ────────────────────────────────────────

def load_bars(start_date: str, end_date: str, freq: str,
              use_l1: bool = False) -> pl.DataFrame:
    import datetime as dt

    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    years = range(start.year, end.year + 1)
    start_dt = dt.datetime.combine(start, dt.time.min)
    end_dt = dt.datetime.combine(end, dt.time.min)

    paths_1s = [
        f"data/parquet/year={y}/data.parquet"
        for y in years
        if Path(f"data/parquet/year={y}/data.parquet").exists()
    ]
    if not paths_1s:
        raise FileNotFoundError(f"No parquet files for {start_date} to {end_date}")

    print(f"    Loading 1s bars...")
    bars_1s = (
        pl.scan_parquet(paths_1s)
        .filter(
            (pl.col("timestamp") >= start_dt)
            & (pl.col("timestamp") < end_dt)
        )
        .sort("timestamp")
        .collect()
    )
    print(f"    {len(bars_1s):,} 1s bars loaded")

    bars_1s = bars_1s.filter(
        (pl.col("timestamp").dt.time() >= dt_time(9, 30))
        & (pl.col("timestamp").dt.time() < dt_time(16, 0))
    )
    print(f"    After RTH filter: {len(bars_1s):,} bars")

    bars = resample_bars(bars_1s, freq)
    if "timestamp_ns" not in bars.columns:
        bars = bars.with_columns(
            pl.col("timestamp").dt.epoch("ns").alias("timestamp_ns")
        )
    print(f"    Resampled to {len(bars):,} {freq} bars")

    return bars


# ── Training ─────────────────────────────────────────────────────────

def train_model(
    features_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    feature_names: list[str],
    val_days: int = 15,
) -> tuple:
    joined = features_df.join(labels_df, on="timestamp_ns", how="inner")
    joined = joined.drop_nulls(subset=feature_names)
    joined = joined.filter(pl.col("label") != 0)

    print(f"\n  Training data: {len(joined):,} rows")
    dist = joined.group_by("label").len().sort("label")
    for row in dist.iter_rows(named=True):
        name = {-1: "DOWN", 0: "FLAT", 1: "UP"}.get(row["label"], "?")
        pct = row["len"] / len(joined) * 100
        print(f"    {name}: {row['len']:,} ({pct:.1f}%)")

    X = joined.select(feature_names).to_numpy().astype(np.float32)
    y_raw = joined["label"].to_numpy()
    y = np.where(y_raw == 1, 1, 0).astype(np.int32)
    w = joined["sample_weight"].to_numpy().astype(np.float32)

    timestamps = joined["timestamp_ns"].to_numpy()
    ts_dates = pl.Series("ts", timestamps).cast(pl.Datetime("ns")).dt.date().to_numpy()
    unique_dates = np.unique(ts_dates)

    if len(unique_dates) > val_days:
        val_start_date = unique_dates[-val_days]
        val_mask = ts_dates >= val_start_date
        train_mask = ~val_mask
    else:
        split = int(len(X) * 0.9)
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[:split] = True
        val_mask = ~train_mask

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    w_train = w[train_mask]

    print(f"    Train: {len(X_train):,}, Val: {len(X_val):,}")

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
        train_end = max(0, test_start - embargo)

        if test_end <= test_start or train_end < 100:
            continue

        Xtr = X_train[:train_end]
        ytr = y_train[:train_end]
        wtr = w_train[:train_end]
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
                                feature_name=feature_names)
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
    print(f"    Mean CV accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"    ({_time.perf_counter() - t0:.1f}s)")

    # Train final model
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
                            feature_name=feature_names)
    val_set = lgb.Dataset(X_train[-vs:], y_train[-vs:], reference=train_set)
    callbacks = [
        lgb.early_stopping(30, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    booster = lgb.train(params, train_set, num_boost_round=300,
                        valid_sets=[val_set], callbacks=callbacks)
    print(f"    Trained ({_time.perf_counter() - t0:.1f}s, {booster.num_trees()} trees)")

    # Feature importance
    imp = booster.feature_importance(importance_type="gain")
    top_idx = np.argsort(imp)[::-1][:10]
    print(f"\n  Top features:")
    for i in top_idx:
        print(f"    {feature_names[i]:25s} {imp[i]:,.0f}")

    # Calibrate
    print("\n  Calibrating...")
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

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"lgbm_{ts}.txt"
    cal_path = MODEL_DIR / f"calibrator_{ts}.pkl"
    booster.save_model(str(model_path))
    calibrator.save(str(cal_path))
    print(f"    Model saved: {model_path}")
    print(f"    Calibrator saved: {cal_path}")

    return booster, calibrator, mean_acc, std_acc, val_acc


# ── OOS Backtest ─────────────────────────────────────────────────────

def run_oos_backtest(
    features_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    feature_names: list[str],
    model,
    calibrator,
    tp_ticks: int,
    sl_ticks: int,
    horizon_bars: int,
    freq: str = "1m",
    threshold: float = 0.15,
    cooldown_bars: int = 2,
    slippage_ticks: float = 1.0,
    cost_ticks: float = 0.72,
) -> dict:
    joined = features_df.join(labels_df, on="timestamp_ns", how="inner")
    joined = joined.drop_nulls(subset=feature_names)
    joined = joined.filter(pl.col("label") != 0)

    if len(joined) == 0:
        return _empty_result(threshold, cooldown_bars)

    print(f"    OOS rows: {len(joined):,}")
    print(f"    EV threshold: {threshold}, Cooldown: {cooldown_bars} bars, Cost: {cost_ticks} ticks")

    X = joined.select(feature_names).to_numpy().astype(np.float32)
    raw_proba = model.predict(X)
    p_up = raw_proba
    p_down = 1.0 - raw_proba

    ev_long = p_up * tp_ticks - p_down * sl_ticks - cost_ticks
    ev_short = p_down * tp_ticks - p_up * sl_ticks - cost_ticks
    ev_long_norm = ev_long / tp_ticks
    ev_short_norm = ev_short / tp_ticks

    # Print p_up distribution for diagnostics
    print(f"    p_up: min={p_up.min():.4f}, max={p_up.max():.4f}, "
          f"mean={p_up.mean():.4f}, std={p_up.std():.4f}")

    timestamps_ns = joined["timestamp_ns"].to_numpy()
    bars_ohlc = _load_bars_for_sim(timestamps_ns, freq)
    close = bars_ohlc["close"]
    high = bars_ohlc["high"]
    low = bars_ohlc["low"]

    tp_pts = tp_ticks * MES_TICK_SIZE
    sl_pts = sl_ticks * MES_TICK_SIZE
    slippage_pts = slippage_ticks * MES_TICK_SIZE

    bar_dates = (
        pl.Series("ts", timestamps_ns)
        .cast(pl.Datetime("ns"))
        .dt.date()
        .to_numpy()
    )

    n_bars = len(X)
    trades = []
    last_entry_bar = -cooldown_bars
    open_positions = []

    for i in range(n_bars):
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

        if len(open_positions) >= 1:
            continue
        if i - last_entry_bar < cooldown_bars:
            continue
        if i >= n_bars - horizon_bars:
            continue

        direction = None
        confidence = 0.0
        if ev_long_norm[i] > threshold:
            direction = "LONG"
            confidence = float(ev_long_norm[i])
        elif ev_short_norm[i] > threshold:
            direction = "SHORT"
            confidence = float(ev_short_norm[i])

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

    if not trades:
        return _empty_result(threshold, cooldown_bars)

    net_pnls = np.array([t["net_pnl"] for t in trades])
    wins = net_pnls[net_pnls > 0]
    losses = net_pnls[net_pnls <= 0]

    wr = len(wins) / len(trades)
    pf = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else float("inf")

    daily_pnl = {}
    for t in trades:
        d = t["entry_date"]
        daily_pnl[d] = daily_pnl.get(d, 0.0) + t["net_pnl"]
    daily_returns = np.array(list(daily_pnl.values()))

    sharpe = 0.0
    if len(daily_returns) > 1:
        std = float(np.std(daily_returns, ddof=1))
        if std > 0:
            sharpe = float(np.mean(daily_returns)) / std * np.sqrt(252)

    daily_equity = np.cumsum(daily_returns)
    peak = np.maximum.accumulate(daily_equity)
    dd = daily_equity - peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0
    max_dd_pct = float(max_dd / (np.max(peak) + 1e-9) * 100) if len(peak) > 0 else 0

    n_trading_days = len(daily_pnl)
    profitable_days = sum(1 for v in daily_pnl.values() if v > 0)

    return {
        "threshold": threshold,
        "cooldown": cooldown_bars,
        "trades": len(trades),
        "wr": wr,
        "pf": pf,
        "net_pnl": float(np.sum(net_pnls)),
        "sharpe": sharpe,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "n_trading_days": n_trading_days,
        "profitable_days": profitable_days,
        "avg_daily_pnl": float(np.mean(daily_returns)),
        "worst_day": float(np.min(daily_returns)),
    }


def _empty_result(threshold, cooldown):
    return {
        "threshold": threshold, "cooldown": cooldown, "trades": 0,
        "wr": 0, "pf": 0, "net_pnl": 0, "sharpe": 0,
        "max_dd": 0, "max_dd_pct": 0,
        "n_trading_days": 0, "profitable_days": 0,
        "avg_daily_pnl": 0, "worst_day": 0,
    }


def _load_bars_for_sim(timestamps_ns: np.ndarray, freq: str = "1m") -> dict:
    import datetime as dt

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

    ts_to_idx = {}
    bar_ts = bars["timestamp_ns"].to_numpy()
    for idx, ts in enumerate(bar_ts):
        ts_to_idx[int(ts)] = idx

    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()

    out_close = np.empty(len(timestamps_ns), dtype=np.float64)
    out_high = np.empty(len(timestamps_ns), dtype=np.float64)
    out_low = np.empty(len(timestamps_ns), dtype=np.float64)

    for i, ts in enumerate(timestamps_ns):
        idx = ts_to_idx.get(int(ts))
        if idx is not None:
            out_close[i] = close[idx]
            out_high[i] = high[idx]
            out_low[i] = low[idx]
        elif i > 0:
            out_close[i] = out_close[i - 1]
            out_high[i] = out_high[i - 1]
            out_low[i] = out_low[i - 1]
        else:
            out_close[i] = close[0]
            out_high[i] = high[0]
            out_low[i] = low[0]

    return {"close": out_close, "high": out_high, "low": out_low}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", default="1m")
    parser.add_argument("--lag-bars", type=int, default=5)
    parser.add_argument("--tp-ticks", type=int, default=4)
    parser.add_argument("--sl-ticks", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--val-days", type=int, default=15)
    parser.add_argument("--cost-ticks", type=float, default=0.72)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--train-start", default="2011-01-01")
    parser.add_argument("--train-end", default="2025-01-01")
    parser.add_argument("--oos-start", default="2025-01-01")
    parser.add_argument("--oos-end", default="2026-03-14")
    args = parser.parse_args()

    configure_logging(log_level="INFO")

    n_features = args.lag_bars * len(FEATURES_PER_LAG)
    bar_mins = freq_minutes(args.freq)
    cfg = TripleBarrierConfig5M(
        vertical_barrier_bars=args.horizon,
        tp_ticks=args.tp_ticks,
        sl_ticks=args.sl_ticks,
    )
    suffix = f"_v3_lag{args.lag_bars}_{args.freq}_h{cfg.vertical_barrier_bars}_tp{cfg.tp_ticks}_sl{cfg.sl_ticks}_rth"

    print(f"\n{'='*70}")
    print(f"  TICK PREDICTOR v3 — LAGGED BAR FEATURES")
    print(f"  Train:     {args.train_start} to {args.train_end} (OHLCV only)")
    print(f"  OOS:       {args.oos_start} to {args.oos_end} (real L1)")
    print(f"  Bar freq:  {args.freq}")
    print(f"  Lag bars:  {args.lag_bars} (= {args.lag_bars * bar_mins}m context)")
    print(f"  Horizon:   {cfg.vertical_barrier_bars} bars ({cfg.vertical_barrier_bars * bar_mins}m)")
    print(f"  TP/SL:     {cfg.tp_ticks}/{cfg.sl_ticks} ticks "
          f"({cfg.tp_ticks * 0.25:.2f}/{cfg.sl_ticks * 0.25:.2f} pts)")
    print(f"  Cost:      {args.cost_ticks} ticks (commission + slippage)")
    print(f"  Features:  {n_features} ({args.lag_bars} lags × {len(FEATURES_PER_LAG)} per bar)")
    print(f"{'='*70}")

    # ── PHASE 1: Build train features ────────────────────────────────
    train_feat_path = DATA_DIR / f"features{suffix}_{args.train_start}_{args.train_end}.parquet"
    train_labels_path = DATA_DIR / f"labels{suffix}_{args.train_start}_{args.train_end}.parquet"

    if train_feat_path.exists() and train_labels_path.exists():
        print(f"\n  Loading cached TRAIN features: {train_feat_path}")
        train_features = pl.read_parquet(train_feat_path)
        feature_names = [c for c in train_features.columns if c != "timestamp_ns"]
        print(f"  Loading cached TRAIN labels: {train_labels_path}")
        train_labels = pl.read_parquet(train_labels_path)
    else:
        print(f"\n  STEP 1: Loading TRAIN {args.freq} bars (OHLCV only)...")
        t0 = _time.perf_counter()
        train_bars = load_bars(args.train_start, args.train_end, args.freq, use_l1=False)
        print(f"    ({_time.perf_counter() - t0:.1f}s)")

        print(f"\n  STEP 2: TRAIN feature generation (lagged bars)...")
        t0 = _time.perf_counter()
        train_features, feature_names = build_lagged_features(train_bars, args.lag_bars)
        print(f"    Features: {len(train_features):,} rows, {len(feature_names)} cols ({_time.perf_counter() - t0:.1f}s)")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_features.write_parquet(train_feat_path)
        print(f"    Saved: {train_feat_path}")

        print(f"\n  STEP 3: TRAIN label generation...")
        t0 = _time.perf_counter()
        labeler = TripleBarrierLabeler5M(cfg)
        train_labels = labeler.generate_labels_from_bars(train_bars)
        labeler.save_labels(train_labels, args.train_start, args.train_end, suffix=suffix)
        train_labels.write_parquet(train_labels_path)
        print(f"    Labels: {len(train_labels):,} rows ({_time.perf_counter() - t0:.1f}s)")

    # ── PHASE 2: Train ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PHASE 2: TRAINING ({args.train_start} to {args.train_end})")
    print(f"{'='*70}")
    t0 = _time.perf_counter()
    model, calibrator, cv_acc, cv_std, val_acc = train_model(
        train_features, train_labels, feature_names, val_days=args.val_days
    )
    print(f"\n  Training complete in {_time.perf_counter() - t0:.1f}s")

    # ── PHASE 3: Build OOS features ─────────────────────────────────
    oos_feat_path = DATA_DIR / f"features{suffix}_{args.oos_start}_{args.oos_end}.parquet"
    oos_labels_path = DATA_DIR / f"labels{suffix}_{args.oos_start}_{args.oos_end}.parquet"

    if oos_feat_path.exists() and oos_labels_path.exists():
        print(f"\n  Loading cached OOS features: {oos_feat_path}")
        oos_features = pl.read_parquet(oos_feat_path)
        print(f"  Loading cached OOS labels: {oos_labels_path}")
        oos_labels = pl.read_parquet(oos_labels_path)
    else:
        print(f"\n  STEP 4: Loading OOS {args.freq} bars...")
        t0 = _time.perf_counter()
        oos_bars = load_bars(args.oos_start, args.oos_end, args.freq, use_l1=False)
        print(f"    ({_time.perf_counter() - t0:.1f}s)")

        print(f"\n  STEP 5: OOS feature generation (lagged bars)...")
        t0 = _time.perf_counter()
        oos_features, _ = build_lagged_features(oos_bars, args.lag_bars)
        print(f"    Features: {len(oos_features):,} rows ({_time.perf_counter() - t0:.1f}s)")
        oos_features.write_parquet(oos_feat_path)
        print(f"    Saved: {oos_feat_path}")

        print(f"\n  STEP 6: OOS label generation...")
        t0 = _time.perf_counter()
        labeler = TripleBarrierLabeler5M(cfg)
        oos_labels = labeler.generate_labels_from_bars(oos_bars)
        labeler.save_labels(oos_labels, args.oos_start, args.oos_end, suffix=suffix)
        oos_labels.write_parquet(oos_labels_path)
        print(f"    Labels: {len(oos_labels):,} rows ({_time.perf_counter() - t0:.1f}s)")

    # ── PHASE 4: OOS Backtest ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PHASE 3: OOS BACKTEST ({args.oos_start} to {args.oos_end})")
    print(f"{'='*70}")

    if args.sweep:
        thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        cooldowns = [1, 2, 3, 6]

        print(f"\n  Sweeping {len(thresholds)} EV thresholds x {len(cooldowns)} cooldowns")
        print(f"  Cost: {args.cost_ticks} ticks, TP: {args.tp_ticks}, SL: {args.sl_ticks}")
        print(f"  {'ev_thr':>6} {'cd':>3} | {'trades':>6} {'WR':>6} {'PnL':>10} "
              f"{'Sharpe':>7} {'PF':>5} {'DD%':>6} {'avgDay':>8} {'worDay':>8} | "
              f"{'profDays':>8}")
        print(f"  {'-'*95}")

        all_results = []
        for thresh in thresholds:
            for cd in cooldowns:
                r = run_oos_backtest(
                    oos_features, oos_labels, feature_names, model, calibrator,
                    tp_ticks=args.tp_ticks, sl_ticks=args.sl_ticks,
                    horizon_bars=args.horizon, freq=args.freq,
                    threshold=thresh, cooldown_bars=cd,
                    cost_ticks=args.cost_ticks,
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
        results_dir = Path("results/tick_predictor_v3")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"oos_sweep_{ts}.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved: {results_path}")

    else:
        r = run_oos_backtest(
            oos_features, oos_labels, feature_names, model, calibrator,
            tp_ticks=args.tp_ticks, sl_ticks=args.sl_ticks,
            horizon_bars=args.horizon, freq=args.freq,
            threshold=0.15, cooldown_bars=2,
            cost_ticks=args.cost_ticks,
        )
        print(f"    Trades: {r['trades']}, WR: {r['wr']*100:.1f}%, "
              f"PnL: ${r['net_pnl']:,.0f}, Sharpe: {r['sharpe']:.2f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  CV accuracy:    {cv_acc:.4f} +/- {cv_std:.4f}")
    print(f"  Val accuracy:   {val_acc:.4f}")
    print(f"  Train:          {args.train_start} to {args.train_end}")
    print(f"  OOS:            {args.oos_start} to {args.oos_end}")
    print(f"  Features:       {n_features} (lag={args.lag_bars} × {len(FEATURES_PER_LAG)}/bar)")
    print()


if __name__ == "__main__":
    main()
