#!/usr/bin/env python3
"""1-minute tick predictor: build features, train, and OOS evaluation.

Order-flow-first feature set on 1m bars with strict causality.
Binary classification: DOWN vs UP (no FLAT class).

Usage:
    # Full pipeline: build features + train + OOS sweep
    python -u scripts/tick_predictor/train_1m.py 2>&1 | tee logs/tick_predictor_1m.log

    # Sweep TP/SL configs
    python -u scripts/tick_predictor/train_1m.py --sweep 2>&1 | tee logs/tick_predictor_1m_sweep.log

    # Custom dates
    python -u scripts/tick_predictor/train_1m.py --start 2024-01-01 --end 2026-03-14
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

import numpy as np
import polars as pl

from src.core.logging import configure_logging, get_logger
from src.signals.tick_predictor.features.feature_builder_1m import (
    FEATURE_NAMES_1M,
    NUM_FEATURES_1M,
    build_features_batch,
)
from src.signals.tick_predictor.labels.triple_barrier_1m import (
    TripleBarrierConfig1M,
    TripleBarrierLabeler1M,
)
from src.signals.tick_predictor.model.calibrator import TemperatureCalibrator
from src.signals.tick_predictor.model.trainer import ModelTrainer, TrainerConfig

logger = get_logger("tick_predictor.train_1m")

# ── Config ──────────────────────────────────────────────────────────
FEATURE_START = "2025-01-01"
FEATURE_END = "2026-03-14"
TRAIN_END = "2025-09-01"
EMBARGO_END = "2025-10-01"

DEFAULT_TP_TICKS = 6
DEFAULT_SL_TICKS = 4
DEFAULT_HORIZON = 15  # 15 minutes

MODEL_DIR = Path("models/tick_predictor")
DATA_DIR = Path("data/tick_predictor")
RESULTS_DIR = Path("results/tick_predictor")

MES_TICK_SIZE = 0.25
MES_TICK_VALUE = 1.25
MES_POINT_VALUE = 5.0
COMMISSION_RT = 0.59


# ── Data loading ────────────────────────────────────────────────────

def load_1m_bars(start_date: str, end_date: str, rth_only: bool = True) -> pl.DataFrame:
    """Load 1m bars, optionally with L1 enrichment."""
    import datetime as dt

    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)

    # Try L1-enriched 1m bars first
    l1_paths = [
        f"data/l1/year={y}/data.parquet"
        for y in range(start.year, end.year + 1)
        if Path(f"data/l1/year={y}/data.parquet").exists()
    ]

    if l1_paths:
        print("  Loading L1 tick data and aggregating to 1m bars...")
        df = aggregate_l1_to_1m(l1_paths, start, end)
        print(f"  L1 → 1m bars: {len(df):,} rows (real order flow)")
    else:
        # Fall back to pre-built 1m bars
        parquet_paths = [
            f"data/parquet_1m/year={y}/data.parquet"
            for y in range(start.year, end.year + 1)
            if Path(f"data/parquet_1m/year={y}/data.parquet").exists()
        ]

        if not parquet_paths:
            # Build from 1s bars
            print("  No 1m bars found. Resampling from 1s bars...")
            df = resample_1s_to_1m(start, end)
        else:
            start_dt = dt.datetime.combine(start, dt.time.min)
            end_dt = dt.datetime.combine(end, dt.time.min)
            df = (
                pl.scan_parquet(parquet_paths)
                .filter(
                    (pl.col("timestamp") >= start_dt)
                    & (pl.col("timestamp") < end_dt)
                )
                .collect()
            )
            print(f"  Loaded 1m bars: {len(df):,} rows (OHLCV approximation)")

    # Add timestamp_ns if missing
    if "timestamp_ns" not in df.columns:
        df = df.with_columns(
            pl.col("timestamp").dt.epoch("ns").alias("timestamp_ns")
        )

    # RTH filter
    if rth_only:
        ts_dtype = df.schema["timestamp"]
        ts_col = pl.col("timestamp")
        if hasattr(ts_dtype, "time_zone") and ts_dtype.time_zone:
            et_col = ts_col.dt.convert_time_zone("US/Eastern")
        else:
            et_col = ts_col.dt.replace_time_zone("UTC").dt.convert_time_zone("US/Eastern")
        df = df.with_columns(et_col.alias("_et_ts"))
        df = df.filter(
            (pl.col("_et_ts").dt.time() >= dt_time(9, 30))
            & (pl.col("_et_ts").dt.time() < dt_time(16, 0))
        ).drop("_et_ts")
        print(f"  After RTH filter: {len(df):,} bars")

    return df


def aggregate_l1_to_1m(
    l1_paths: list[str],
    start: "dt.date",
    end: "dt.date",
) -> pl.DataFrame:
    """Aggregate L1 tick data into 1m bars with real order flow columns."""
    import datetime as dt

    start_dt = dt.datetime.combine(start, dt.time.min)
    end_dt = dt.datetime.combine(end, dt.time.min)

    df = (
        pl.scan_parquet(l1_paths)
        .filter(
            (pl.col("timestamp") >= start_dt)
            & (pl.col("timestamp") < end_dt)
        )
        .collect()
    )

    # Determine available columns for aggregation
    agg_exprs = [
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("size").sum().alias("volume"),
    ]

    # Add L1 columns if available
    if "bid_size" in df.columns:
        agg_exprs.append(pl.col("bid_size").mean().alias("avg_bid_size"))
    if "ask_size" in df.columns:
        agg_exprs.append(pl.col("ask_size").mean().alias("avg_ask_size"))
    if "side" in df.columns:
        agg_exprs.extend([
            pl.col("size").filter(pl.col("side") == "B").sum().alias("aggressive_buy_vol"),
            pl.col("size").filter(pl.col("side") == "S").sum().alias("aggressive_sell_vol"),
        ])

    bars = (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every="1m")
        .agg(agg_exprs)
        .sort("timestamp")
    )

    # Fill nulls for L1 columns
    for col in ["avg_bid_size", "avg_ask_size", "aggressive_buy_vol", "aggressive_sell_vol"]:
        if col in bars.columns:
            bars = bars.with_columns(pl.col(col).fill_null(0.0))

    return bars


def resample_1s_to_1m(start: "dt.date", end: "dt.date") -> pl.DataFrame:
    """Resample 1s bars to 1m."""
    import datetime as dt

    paths = [
        f"data/parquet/year={y}/data.parquet"
        for y in range(start.year, end.year + 1)
        if Path(f"data/parquet/year={y}/data.parquet").exists()
    ]

    if not paths:
        raise FileNotFoundError("No 1s parquet data found")

    start_dt = dt.datetime.combine(start, dt.time.min)
    end_dt = dt.datetime.combine(end, dt.time.min)

    df = (
        pl.scan_parquet(paths)
        .filter(
            (pl.col("timestamp") >= start_dt)
            & (pl.col("timestamp") < end_dt)
        )
        .collect()
    )

    bars = (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every="1m")
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ])
        .sort("timestamp")
    )

    return bars


def load_regime_features(start_date: str, end_date: str) -> pl.DataFrame | None:
    """Load pre-computed HMM regime probabilities if available."""
    regime_path = Path("data/regime_v2_features.parquet")
    if not regime_path.exists():
        # Try building from model
        model_path = Path("models/regime_v2/hmm_regime_v2.pkl")
        if not model_path.exists():
            print("  No regime model found, skipping regime features")
            return None
        print("  TODO: build regime features from model")
        return None

    df = pl.read_parquet(regime_path)
    return df


# ── Training pipeline ───────────────────────────────────────────────

def build_and_cache_features(
    start_date: str, end_date: str,
    tp_ticks: int, sl_ticks: int, horizon: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build features and labels, caching to disk."""
    suffix = f"_1m_h{horizon}_tp{tp_ticks}_sl{sl_ticks}_rth"
    feat_path = DATA_DIR / f"features{suffix}_{start_date}_{end_date}.parquet"
    label_path = DATA_DIR / f"labels{suffix}_{start_date}_{end_date}.parquet"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Features
    if feat_path.exists():
        print(f"  [Cache HIT] {feat_path}")
        features_df = pl.read_parquet(feat_path)
    else:
        print("  Building features...")
        t0 = _time.perf_counter()
        bars_1m = load_1m_bars(start_date, end_date, rth_only=True)
        regime_df = load_regime_features(start_date, end_date)
        features_df = build_features_batch(bars_1m, regime_df)
        features_df.write_parquet(feat_path)
        print(f"  Features: {len(features_df):,} rows ({_time.perf_counter()-t0:.1f}s)")
        print(f"  Cached -> {feat_path}")

    # Labels
    if label_path.exists():
        print(f"  [Cache HIT] {label_path}")
        labels_df = pl.read_parquet(label_path)
    else:
        print("  Building labels...")
        t0 = _time.perf_counter()
        bars_1m = load_1m_bars(start_date, end_date, rth_only=True)
        cfg = TripleBarrierConfig1M(
            vertical_barrier_bars=horizon,
            tp_ticks=tp_ticks,
            sl_ticks=sl_ticks,
        )
        labeler = TripleBarrierLabeler1M(cfg)
        labels_df = labeler.generate_labels_from_bars(bars_1m)
        labels_df.write_parquet(label_path)
        print(f"  Labels: {len(labels_df):,} rows ({_time.perf_counter()-t0:.1f}s)")

    return features_df, labels_df


def train_model(
    features_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    train_end: str,
    embargo_end: str,
    suffix: str = "",
) -> dict:
    """Train on data before train_end, validate on last 15 days of train period."""
    import datetime as dt

    train_end_ns = int(
        dt.datetime.combine(dt.date.fromisoformat(train_end), dt.time.min).timestamp() * 1e9
    )
    embargo_end_ns = int(
        dt.datetime.combine(dt.date.fromisoformat(embargo_end), dt.time.min).timestamp() * 1e9
    )

    # Join features + labels
    joined = features_df.join(labels_df, on="timestamp_ns", how="inner")

    # Drop nulls in feature columns
    avail_feats = [f for f in FEATURE_NAMES_1M if f in joined.columns]
    joined = joined.drop_nulls(subset=avail_feats)

    # Drop FLAT labels (label == 0)
    n_before = len(joined)
    joined = joined.filter(pl.col("label") != 0)
    n_flat = n_before - len(joined)
    print(f"  Dropped FLAT: {n_flat:,} ({n_flat/n_before*100:.1f}%)")

    # Binary: remap -1 -> 0 (DOWN), +1 -> 1 (UP)
    joined = joined.with_columns(
        pl.when(pl.col("label") == -1).then(0).otherwise(1).alias("label_bin")
    )

    # Split
    train_df = joined.filter(pl.col("timestamp_ns") < train_end_ns)
    oos_df = joined.filter(pl.col("timestamp_ns") >= embargo_end_ns)

    # Validation: last 15 days of training data for calibration
    val_cutoff = train_end_ns - int(15 * 86400 * 1e9)
    cal_df = train_df.filter(pl.col("timestamp_ns") >= val_cutoff)
    cv_df = train_df.filter(pl.col("timestamp_ns") < val_cutoff)

    print(f"  Train (CV):     {len(cv_df):,} rows")
    print(f"  Train (cal):    {len(cal_df):,} rows")
    print(f"  OOS:            {len(oos_df):,} rows")

    dist = joined.group_by("label_bin").len().sort("label_bin")
    for row in dist.iter_rows(named=True):
        name = "DOWN" if row["label_bin"] == 0 else "UP"
        print(f"  {name}: {row['len']:,} ({row['len']/len(joined)*100:.1f}%)")

    if len(cv_df) < 1000 or len(oos_df) < 100:
        print("  ERROR: Insufficient data")
        return {"success": False}

    X_train = cv_df.select(avail_feats).to_numpy().astype(np.float32)
    y_train = cv_df["label_bin"].to_numpy().astype(np.int32)
    w_train = cv_df["sample_weight"].to_numpy().astype(np.float32)

    X_cal = cal_df.select(avail_feats).to_numpy().astype(np.float32)
    y_cal = cal_df["label_bin"].to_numpy().astype(np.int32)

    X_oos = oos_df.select(avail_feats).to_numpy().astype(np.float32)
    y_oos = oos_df["label_bin"].to_numpy().astype(np.int32)

    # ── Walk-forward CV ─────────────────────────────────────
    print("\n  Walk-forward CV (binary)...")
    t0 = _time.perf_counter()

    # Use binary LightGBM config
    import lightgbm as lgb

    n = len(X_train)
    n_folds = 5
    embargo = 60  # 60-minute embargo between train/test
    min_train = int(n * 0.5)
    fold_size = (n - min_train) // n_folds

    cv_accs = []
    for fold in range(n_folds):
        test_start = min_train + fold * fold_size
        test_end = min_train + (fold + 1) * fold_size if fold < n_folds - 1 else n
        train_end_idx = max(0, test_start - embargo)

        if train_end_idx < 100 or test_end <= test_start:
            continue

        X_tr = X_train[:train_end_idx]
        y_tr = y_train[:train_end_idx]
        w_tr = w_train[:train_end_idx]
        X_te = X_train[test_start:test_end]
        y_te = y_train[test_start:test_end]

        # Early stopping split
        val_n = max(1, int(len(X_tr) * 0.1))
        train_set = lgb.Dataset(
            X_tr[:-val_n], y_tr[:-val_n], weight=w_tr[:-val_n],
            feature_name=avail_feats, free_raw_data=False,
        )
        val_set = lgb.Dataset(
            X_tr[-val_n:], y_tr[-val_n:], reference=train_set, free_raw_data=False,
        )

        params = {
            "objective": "binary",
            "num_leaves": 63,
            "max_depth": 7,
            "learning_rate": 0.03,
            "n_estimators": 300,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
            "is_unbalance": True,
            "device": "gpu",
            "verbose": -1,
            "seed": 42,
        }

        booster = lgb.train(
            params, train_set, num_boost_round=300,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        proba = booster.predict(X_te)
        y_pred = (proba > 0.5).astype(int)
        acc = float(np.mean(y_pred == y_te))
        cv_accs.append(acc)
        print(f"    Fold {fold}: acc={acc:.4f} (train={len(X_tr):,}, test={len(X_te):,})")

    mean_acc = float(np.mean(cv_accs))
    std_acc = float(np.std(cv_accs))
    print(f"    CV accuracy: {mean_acc:.4f} +/- {std_acc:.4f} ({_time.perf_counter()-t0:.1f}s)")

    # ── Train final model ───────────────────────────────────
    print("\n  Training final model...")
    t0 = _time.perf_counter()

    # Use all CV data for final model
    val_n = max(1, int(len(X_train) * 0.1))
    train_set = lgb.Dataset(
        X_train[:-val_n], y_train[:-val_n], weight=w_train[:-val_n],
        feature_name=avail_feats, free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_train[-val_n:], y_train[-val_n:], reference=train_set, free_raw_data=False,
    )

    params = {
        "objective": "binary",
        "num_leaves": 63,
        "max_depth": 7,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "is_unbalance": True,
        "device": "gpu",
        "verbose": -1,
        "seed": 42,
    }

    booster = lgb.train(
        params, train_set, num_boost_round=300,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"lgbm_1m_{ts}.txt"
    booster.save_model(str(model_path))

    # Feature importances
    imp = booster.feature_importance(importance_type="gain")
    top_features = {
        avail_feats[i]: float(imp[i])
        for i in np.argsort(imp)[::-1][:10]
    }
    print(f"  Model saved: {model_path} ({_time.perf_counter()-t0:.1f}s)")
    print(f"\n  Top features:")
    for name, val in top_features.items():
        print(f"    {name:<30s} {val:>12,.0f}")

    # ── Calibration ─────────────────────────────────────────
    print("\n  Calibrating on last 15 days of train period...")
    raw_cal = booster.predict(X_cal)
    cal_acc = float(np.mean((raw_cal > 0.5).astype(int) == y_cal))
    print(f"  Calibration accuracy: {cal_acc:.4f}")

    return {
        "success": True,
        "cv_accuracy_mean": mean_acc,
        "cv_accuracy_std": std_acc,
        "cal_accuracy": cal_acc,
        "top_features": top_features,
        "model_path": str(model_path),
        "booster": booster,
        "X_oos": X_oos,
        "y_oos": y_oos,
        "oos_df": oos_df,
        "avail_feats": avail_feats,
    }


# ── OOS Backtest ────────────────────────────────────────────────────

def oos_backtest(
    booster,
    X_oos: np.ndarray,
    y_oos: np.ndarray,
    oos_df: pl.DataFrame,
    threshold: float = 0.55,
    cooldown_bars: int = 5,
    tp_ticks: int = DEFAULT_TP_TICKS,
    sl_ticks: int = DEFAULT_SL_TICKS,
    horizon_bars: int = DEFAULT_HORIZON,
) -> dict:
    """Simulate OOS trading with threshold + cooldown."""
    proba = booster.predict(X_oos)
    closes = oos_df["close"].to_numpy()
    highs = oos_df["high"].to_numpy()
    lows = oos_df["low"].to_numpy()
    timestamps = oos_df["timestamp_ns"].to_numpy()
    n = len(proba)

    tp_pts = tp_ticks * MES_TICK_SIZE
    sl_pts = sl_ticks * MES_TICK_SIZE

    trades = []
    last_trade_bar = -cooldown_bars - 1

    for i in range(n - horizon_bars):
        if i - last_trade_bar < cooldown_bars:
            continue

        p_up = float(proba[i])
        p_down = 1.0 - p_up

        # Signal: confident UP or DOWN
        if p_up >= threshold:
            direction = 1  # LONG
        elif p_down >= threshold:
            direction = -1  # SHORT
        else:
            continue

        entry = closes[i]
        last_trade_bar = i

        # Simulate with high/low for barrier touches
        pnl = 0.0
        exit_reason = "vertical"
        exit_bar = min(i + horizon_bars, n - 1)

        if direction == 1:  # LONG
            target = entry + tp_pts
            stop = entry - sl_pts
            for j in range(i + 1, exit_bar + 1):
                hit_tp = highs[j] >= target
                hit_sl = lows[j] <= stop
                if hit_tp and hit_sl:
                    pnl = (closes[j] - entry) * MES_POINT_VALUE
                    exit_reason = "both"
                    exit_bar = j
                    break
                elif hit_tp:
                    pnl = tp_pts * MES_POINT_VALUE
                    exit_reason = "tp"
                    exit_bar = j
                    break
                elif hit_sl:
                    pnl = -sl_pts * MES_POINT_VALUE
                    exit_reason = "sl"
                    exit_bar = j
                    break
            else:
                pnl = (closes[exit_bar] - entry) * MES_POINT_VALUE
        else:  # SHORT
            target = entry - tp_pts
            stop = entry + sl_pts
            for j in range(i + 1, exit_bar + 1):
                hit_tp = lows[j] <= target
                hit_sl = highs[j] >= stop
                if hit_tp and hit_sl:
                    pnl = (entry - closes[j]) * MES_POINT_VALUE
                    exit_reason = "both"
                    exit_bar = j
                    break
                elif hit_tp:
                    pnl = tp_pts * MES_POINT_VALUE
                    exit_reason = "tp"
                    exit_bar = j
                    break
                elif hit_sl:
                    pnl = -sl_pts * MES_POINT_VALUE
                    exit_reason = "sl"
                    exit_bar = j
                    break
            else:
                pnl = (entry - closes[exit_bar]) * MES_POINT_VALUE

        net_pnl = pnl - COMMISSION_RT
        trades.append({
            "bar_idx": i,
            "timestamp_ns": int(timestamps[i]),
            "direction": direction,
            "entry": entry,
            "pnl": net_pnl,
            "exit_reason": exit_reason,
            "p_up": p_up,
        })

    if not trades:
        return {
            "trades": 0, "win_rate": 0, "total_pnl": 0, "sharpe": 0,
            "profit_factor": 0, "max_dd_pct": 0, "avg_daily_pnl": 0,
            "worst_day_pnl": 0, "profitable_days": "0/0",
        }

    pnls = np.array([t["pnl"] for t in trades])
    wins = np.sum(pnls > 0)
    total_pnl = float(np.sum(pnls))
    win_rate = float(wins / len(pnls)) * 100

    # Sharpe (annualized assuming ~252 trading days)
    if np.std(pnls) > 0:
        sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0]))
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0])))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    max_dd_pct = max_dd / max(abs(total_pnl), 1.0) * 100

    # Daily P&L
    import pandas as pd
    trade_dates = pd.to_datetime(
        [t["timestamp_ns"] for t in trades], unit="ns"
    ).date
    daily_pnl = {}
    for date, pnl_val in zip(trade_dates, pnls):
        daily_pnl[date] = daily_pnl.get(date, 0.0) + pnl_val

    daily_vals = list(daily_pnl.values())
    avg_daily = float(np.mean(daily_vals)) if daily_vals else 0.0
    worst_day = float(np.min(daily_vals)) if daily_vals else 0.0
    prof_days = sum(1 for v in daily_vals if v > 0)
    total_days = len(daily_vals)

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "profit_factor": pf,
        "max_dd_pct": max_dd_pct,
        "max_dd_dollars": max_dd,
        "avg_daily_pnl": avg_daily,
        "worst_day_pnl": worst_day,
        "profitable_days": f"{prof_days}/{total_days}",
        "avg_pnl_per_trade": float(np.mean(pnls)),
        "median_pnl_per_trade": float(np.median(pnls)),
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=FEATURE_START)
    parser.add_argument("--end", default=FEATURE_END)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--embargo-end", default=EMBARGO_END)
    parser.add_argument("--tp-ticks", type=int, default=DEFAULT_TP_TICKS)
    parser.add_argument("--sl-ticks", type=int, default=DEFAULT_SL_TICKS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--sweep", action="store_true", help="Sweep thresholds and cooldowns")
    args = parser.parse_args()

    configure_logging()

    print(f"\n{'='*70}")
    print(f"  1-MINUTE TICK PREDICTOR — ORDER FLOW FIRST")
    print(f"  Train:   {args.start} to {args.train_end}")
    print(f"  Embargo: {args.train_end} to {args.embargo_end}")
    print(f"  OOS:     {args.embargo_end} to {args.end}")
    print(f"  TP/SL:   {args.tp_ticks}/{args.sl_ticks} ticks, Horizon: {args.horizon}m")
    print(f"{'='*70}\n")

    # Build features and labels
    features_df, labels_df = build_and_cache_features(
        args.start, args.end, args.tp_ticks, args.sl_ticks, args.horizon,
    )

    # Train
    result = train_model(
        features_df, labels_df, args.train_end, args.embargo_end,
    )

    if not result["success"]:
        print("\n  TRAINING FAILED")
        return

    booster = result["booster"]
    X_oos = result["X_oos"]
    y_oos = result["y_oos"]
    oos_df = result["oos_df"]

    # OOS evaluation
    if args.sweep:
        thresholds = [0.52, 0.55, 0.58, 0.60, 0.65, 0.70]
        cooldowns = [0, 3, 5, 10, 15]

        print(f"\n  Sweeping {len(thresholds)} thresholds x {len(cooldowns)} cooldowns")
        print(f"  {'thresh':>6} {'cd':>4} | {'trades':>6} {'WR':>6} {'PnL':>10} "
              f"{'Sharpe':>7} {'PF':>6} {'DD%':>6} {'avgDay':>8} {'worDay':>8} | profDays")
        print(f"  {'-'*95}")

        best_sharpe = -999
        best_config = None
        all_results = []

        for thresh in thresholds:
            for cd in cooldowns:
                r = oos_backtest(
                    booster, X_oos, y_oos, oos_df,
                    threshold=thresh, cooldown_bars=cd,
                    tp_ticks=args.tp_ticks, sl_ticks=args.sl_ticks,
                    horizon_bars=args.horizon,
                )
                print(
                    f"  {thresh:>6.2f} {cd:>4} | {r['trades']:>6} {r['win_rate']:>5.1f}% "
                    f"${r['total_pnl']:>9,.0f} {r['sharpe']:>7.2f} {r['profit_factor']:>6.2f} "
                    f"{r['max_dd_pct']:>5.1f}% ${r['avg_daily_pnl']:>7,.0f} "
                    f"${r['worst_day_pnl']:>7,.0f} | {r['profitable_days']}"
                )
                r["threshold"] = thresh
                r["cooldown"] = cd
                all_results.append(r)

                if r["sharpe"] > best_sharpe and r["trades"] >= 20:
                    best_sharpe = r["sharpe"]
                    best_config = r

        if best_config:
            print(f"\n  BEST CONFIG:")
            print(f"    Threshold: {best_config['threshold']}, Cooldown: {best_config['cooldown']}")
            print(f"    Trades: {best_config['trades']}, WR: {best_config['win_rate']:.1f}%")
            print(f"    Sharpe: {best_config['sharpe']:.2f}, PF: {best_config['profit_factor']:.2f}")
            print(f"    Total PnL: ${best_config['total_pnl']:,.0f}")

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = RESULTS_DIR / f"oos_1m_sweep_{ts}.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved -> {results_path}")

    else:
        # Single evaluation
        print(f"\n{'='*70}")
        print(f"  OOS BACKTEST (threshold=0.55, cooldown=5)")
        print(f"{'='*70}")

        r = oos_backtest(
            booster, X_oos, y_oos, oos_df,
            threshold=0.55, cooldown_bars=5,
            tp_ticks=args.tp_ticks, sl_ticks=args.sl_ticks,
            horizon_bars=args.horizon,
        )

        print(f"  Trades:      {r['trades']}")
        print(f"  Win rate:    {r['win_rate']:.1f}%")
        print(f"  Total PnL:   ${r['total_pnl']:,.2f}")
        print(f"  Sharpe:      {r['sharpe']:.2f}")
        print(f"  PF:          {r['profit_factor']:.2f}")
        print(f"  Max DD:      ${r.get('max_dd_dollars', 0):,.0f} ({r['max_dd_pct']:.1f}%)")
        print(f"  Avg daily:   ${r['avg_daily_pnl']:,.2f}")
        print(f"  Worst day:   ${r['worst_day_pnl']:,.2f}")
        print(f"  Prof days:   {r['profitable_days']}")
        print(f"  Avg/trade:   ${r.get('avg_pnl_per_trade', 0):,.2f}")

    print(f"\n  CV accuracy:  {result['cv_accuracy_mean']:.4f} +/- {result['cv_accuracy_std']:.4f}")
    print(f"  Model:        {result['model_path']}")
    print()


if __name__ == "__main__":
    main()
