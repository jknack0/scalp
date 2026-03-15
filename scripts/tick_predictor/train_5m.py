#!/usr/bin/env python3
"""Training pipeline for multi-timeframe TickDirectionPredictor.

Builds features on Nm bars with strict causality (all features shifted by 1 bar),
generates triple-barrier labels, trains LightGBM with walk-forward CV.

Usage:
    python -u scripts/tick_predictor/train_5m.py --start 2025-01-01 --end 2026-03-14
    python -u scripts/tick_predictor/train_5m.py --freq 15m --tp-ticks 20 --sl-ticks 12 --horizon 8
    python -u scripts/tick_predictor/train_5m.py --freq 30m --tp-ticks 32 --sl-ticks 20 --horizon 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np
import polars as pl

from src.core.logging import configure_logging, get_logger
from src.data.bars import resample_bars
from src.signals.tick_predictor.features.feature_builder_5m import (
    FEATURE_NAMES_5M,
    build_features_batch,
)
from src.signals.tick_predictor.labels.triple_barrier_5m import (
    TripleBarrierConfig5M,
    TripleBarrierLabeler5M,
)
from src.signals.tick_predictor.model.calibrator import TemperatureCalibrator

logger = get_logger("tick_predictor.train_5m")

MODEL_DIR = Path("models/tick_predictor_5m")
DATA_DIR = Path("data/tick_predictor")

# Label encoding: binary (no FLAT)
LABEL_MAP = {-1: 0, 1: 1}  # DOWN=0, UP=1


@dataclass
class TrainingResult:
    cv_accuracy_mean: float
    cv_accuracy_std: float
    val_ece: float
    val_accuracy: float
    temperature: float
    n_train: int
    n_val: int
    top_features: dict


def cache_suffix(cfg: TripleBarrierConfig5M, freq: str) -> str:
    return f"_{freq}_v1_h{cfg.vertical_barrier_bars}_tp{cfg.tp_ticks}_sl{cfg.sl_ticks}_rth"


def freq_minutes(freq: str) -> int:
    """Parse freq string like '5m', '15m', '30m' to minutes."""
    return int(freq.replace("m", ""))


def load_bars(start_date: str, end_date: str, freq: str = "5m",
              use_l1: bool = True) -> pl.DataFrame:
    """Load 1s bars and resample to target frequency. Optionally enrich with L1 data."""
    import datetime as dt

    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    years = range(start.year, end.year + 1)

    start_dt = dt.datetime.combine(start, dt.time.min)
    end_dt = dt.datetime.combine(end, dt.time.min)

    # Load 1s bars
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

    # RTH filter
    bars_1s = bars_1s.filter(
        (pl.col("timestamp").dt.time() >= dt_time(9, 30))
        & (pl.col("timestamp").dt.time() < dt_time(16, 0))
    )
    print(f"    After RTH filter: {len(bars_1s):,} bars")

    # Resample to target frequency
    bars = resample_bars(bars_1s, freq)
    if "timestamp_ns" not in bars.columns:
        bars = bars.with_columns(
            pl.col("timestamp").dt.epoch("ns").alias("timestamp_ns")
        )
    print(f"    Resampled to {len(bars):,} {freq} bars")

    # Enrich with L1 data if available
    if use_l1:
        l1_paths = [
            f"data/l1/year={y}/data.parquet"
            for y in years
            if Path(f"data/l1/year={y}/data.parquet").exists()
        ]
        if l1_paths:
            print(f"    Loading L1 ticks for order flow enrichment...")
            # L1 data may have UTC-aware timestamps — cast to naive for comparison
            ticks = (
                pl.scan_parquet(l1_paths)
                .with_columns(
                    pl.col("timestamp").dt.replace_time_zone(None).alias("timestamp")
                )
                .filter(
                    (pl.col("timestamp") >= start_dt)
                    & (pl.col("timestamp") < end_dt)
                    & (pl.col("timestamp").dt.time() >= dt_time(9, 30))
                    & (pl.col("timestamp").dt.time() < dt_time(16, 0))
                )
                .sort("timestamp")
                .collect()
            )
            print(f"    {len(ticks):,} L1 ticks loaded")

            if len(ticks) > 0:
                bars = _aggregate_l1_to_bars(ticks, bars, freq)

    return bars


def _aggregate_l1_to_bars(ticks: pl.DataFrame, bars: pl.DataFrame,
                           freq: str) -> pl.DataFrame:
    """Aggregate L1 tick data into bar-level order flow columns."""
    # Truncate tick timestamps to target frequency
    ticks = ticks.with_columns(
        pl.col("timestamp").dt.truncate(freq).alias("bar_ts")
    )

    # Classify ticks as buy/sell using tick rule
    ticks = ticks.with_columns(
        pl.col("price").diff().sign().fill_null(0).alias("tick_dir")
    )

    agg = ticks.group_by("bar_ts").agg([
        pl.col("bid_size").mean().alias("avg_bid_size"),
        pl.col("ask_size").mean().alias("avg_ask_size"),
        # Buy volume: ticks where price went up
        pl.when(pl.col("tick_dir") > 0)
        .then(pl.col("size"))
        .otherwise(0)
        .sum()
        .alias("aggressive_buy_vol"),
        # Sell volume: ticks where price went down
        pl.when(pl.col("tick_dir") < 0)
        .then(pl.col("size"))
        .otherwise(0)
        .sum()
        .alias("aggressive_sell_vol"),
    ]).sort("bar_ts")

    # Join to bars
    freq_ns = freq_minutes(freq) * 60 * 1_000_000_000
    agg = agg.with_columns(
        pl.col("bar_ts").dt.epoch("ns").alias("_l1_ts_ns")
    )
    bars = bars.sort("timestamp_ns").join_asof(
        agg.select(["_l1_ts_ns", "avg_bid_size", "avg_ask_size",
                     "aggressive_buy_vol", "aggressive_sell_vol"]).sort("_l1_ts_ns"),
        left_on="timestamp_ns",
        right_on="_l1_ts_ns",
        strategy="nearest",
        tolerance=freq_ns,
    )
    print(f"    L1 order flow joined to {bars.select('avg_bid_size').drop_nulls().height:,} / {len(bars):,} bars")

    return bars


def build_regime_df(bars: pl.DataFrame, freq: str = "5m") -> pl.DataFrame | None:
    """Build regime predictions from bars using causal forward-only HMM."""
    from src.models.regime_detector_v2 import RegimeDetectorV2

    regime_model_path = Path("models/regime_v2")
    if not regime_model_path.exists():
        print("    [SKIP] Regime model not found — filling with zeros")
        return None

    detector = RegimeDetectorV2.load(str(regime_model_path))

    # Build regime features from bars
    from src.models.regime_detector_v2 import build_features_v2
    features, ts = build_features_v2(bars, detector.config)
    probas = detector.predict_proba_sequence(features, causal=True)

    shift_ns = freq_minutes(freq) * 60 * 1_000_000_000
    regime_records = []
    for i, proba in enumerate(probas):
        regime_records.append({
            # Shift timestamp by bar freq so join_asof only sees PREVIOUS completed regime
            "timestamp_ns": int(ts[i]) + shift_ns,
            "regime_p_trending": float(proba.probabilities[0]),
            "regime_p_ranging": float(proba.probabilities[1]),
            "regime_p_highvol": float(proba.probabilities[2]),
        })

    regime_df = pl.DataFrame(regime_records).sort("timestamp_ns")
    print(f"    Regime predictions: {len(regime_df):,} rows (shifted +{freq})")
    return regime_df


def train_model(
    features_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    val_days: int = 15,
) -> tuple:
    """Train LightGBM on features + labels."""
    import lightgbm as lgb

    # Join
    joined = features_df.join(labels_df, on="timestamp_ns", how="inner")
    joined = joined.drop_nulls(subset=FEATURE_NAMES_5M)

    # Drop FLAT
    joined = joined.filter(pl.col("label") != 0)

    print(f"\n  Training data: {len(joined):,} rows")

    # Label distribution
    dist = joined.group_by("label").len().sort("label")
    for row in dist.iter_rows(named=True):
        name = {-1: "DOWN", 0: "FLAT", 1: "UP"}.get(row["label"], "?")
        pct = row["len"] / len(joined) * 100
        print(f"    {name}: {row['len']:,} ({pct:.1f}%)")

    # Encode labels (binary: DOWN=0, UP=1)
    X = joined.select(FEATURE_NAMES_5M).to_numpy().astype(np.float32)
    y_raw = joined["label"].to_numpy()
    y = np.where(y_raw == 1, 1, 0).astype(np.int32)
    w = joined["sample_weight"].to_numpy().astype(np.float32)

    # Train/val split (last val_days days for calibration)
    timestamps = joined["timestamp_ns"].to_numpy()
    ts_series = pl.Series("ts", timestamps).cast(pl.Datetime("ns"))
    dates = ts_series.dt.date().to_numpy()
    unique_dates = np.unique(dates)

    if len(unique_dates) > val_days:
        val_start_date = unique_dates[-val_days]
        val_mask = dates >= val_start_date
        train_mask = ~val_mask
    else:
        # Not enough days, use last 10% as val
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
    embargo = 6  # 6 bars = 30 min
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

        # Internal val for early stopping
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
    print(f"    Mean CV accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"    ({_time.perf_counter() - t0:.1f}s)")

    # Train final model on all training data
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

    # Feature importance
    imp = booster.feature_importance(importance_type="gain")
    top_idx = np.argsort(imp)[::-1][:10]
    print(f"\n  Top features:")
    for i in top_idx:
        print(f"    {FEATURE_NAMES_5M[i]:25s} {imp[i]:,.0f}")

    # Calibrate on validation data
    print("\n  Calibrating...")
    raw_val = booster.predict(X_val)
    # Convert binary proba to 3-class format for calibrator
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

    # Latest symlink
    import shutil
    shutil.copy2(str(model_path), str(MODEL_DIR / "lgbm_latest.txt"))
    shutil.copy2(str(cal_path), str(MODEL_DIR / "calibrator_latest.pkl"))

    print(f"    Model saved: {model_path}")
    print(f"    Calibrator saved: {cal_path}")

    top_features = {FEATURE_NAMES_5M[i]: float(imp[i]) for i in top_idx}

    result = TrainingResult(
        cv_accuracy_mean=mean_acc,
        cv_accuracy_std=std_acc,
        val_ece=val_ece,
        val_accuracy=val_acc,
        temperature=calibrator.temperature,
        n_train=len(X_train),
        n_val=len(X_val),
        top_features=top_features,
    )

    # Save result JSON
    result_path = MODEL_DIR / f"training_result_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(vars(result), f, indent=2)

    return booster, calibrator, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-03-14")
    parser.add_argument("--freq", default="5m", help="Bar frequency: 5m, 15m, 30m")
    parser.add_argument("--val-days", type=int, default=15)
    parser.add_argument("--no-l1", action="store_true", help="Skip L1 enrichment")
    parser.add_argument("--tp-ticks", type=int, default=12)
    parser.add_argument("--sl-ticks", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()

    configure_logging(log_level="INFO")

    bar_mins = freq_minutes(args.freq)

    cfg = TripleBarrierConfig5M(
        vertical_barrier_bars=args.horizon,
        tp_ticks=args.tp_ticks,
        sl_ticks=args.sl_ticks,
    )
    suffix = cache_suffix(cfg, args.freq)

    print(f"\n{'='*70}")
    print(f"  {args.freq.upper()} TICK DIRECTION PREDICTOR — TRAINING PIPELINE")
    print(f"  Data range:  {args.start} to {args.end}")
    print(f"  Bar freq:    {args.freq}")
    print(f"  Val days:    {args.val_days}")
    print(f"  Horizon:     {cfg.vertical_barrier_bars} bars ({cfg.vertical_barrier_bars * bar_mins}m)")
    print(f"  TP/SL:       {cfg.tp_ticks}/{cfg.sl_ticks} ticks "
          f"({cfg.tp_ticks * 0.25:.2f}/{cfg.sl_ticks * 0.25:.2f} pts)")
    print(f"  L1 enrich:   {'Yes' if not args.no_l1 else 'No'}")
    print(f"{'='*70}\n")

    # Check for cached features
    feat_path = DATA_DIR / f"features{suffix}_{args.start}_{args.end}.parquet"
    labels_path = DATA_DIR / f"labels{suffix}_{args.start}_{args.end}.parquet"

    if feat_path.exists() and labels_path.exists():
        print(f"  Loading cached features: {feat_path}")
        features_df = pl.read_parquet(feat_path)
        print(f"  Loading cached labels: {labels_path}")
        labels_df = pl.read_parquet(labels_path)
    else:
        # Step 1: Load and resample bars
        print(f"  STEP 1: Loading {args.freq} bars...")
        t0 = _time.perf_counter()
        bars = load_bars(args.start, args.end, freq=args.freq, use_l1=not args.no_l1)
        print(f"    ({_time.perf_counter() - t0:.1f}s)")

        # Step 2: Build regime predictions
        print(f"\n  STEP 2: Regime predictions...")
        t0 = _time.perf_counter()
        regime_df = build_regime_df(bars, freq=args.freq)
        print(f"    ({_time.perf_counter() - t0:.1f}s)")

        # Step 3: Build features
        print(f"\n  STEP 3: Feature generation...")
        t0 = _time.perf_counter()
        features_df = build_features_batch(bars, regime_df=regime_df)
        print(f"    Features: {len(features_df):,} rows ({_time.perf_counter() - t0:.1f}s)")

        # Cache features
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        features_df.write_parquet(feat_path)
        print(f"    Saved: {feat_path}")

        # Step 4: Generate labels
        print(f"\n  STEP 4: Label generation...")
        t0 = _time.perf_counter()
        labeler = TripleBarrierLabeler5M(cfg)
        labels_df = labeler.generate_labels_from_bars(bars)
        labeler.save_labels(labels_df, args.start, args.end, suffix=suffix)
        print(f"    Labels: {len(labels_df):,} rows ({_time.perf_counter() - t0:.1f}s)")

        # Cache labels
        labels_df.write_parquet(labels_path)

    # Step 5: Train
    print(f"\n{'='*70}")
    print(f"  STEP 5: Training")
    print(f"{'='*70}")
    booster, calibrator, result = train_model(
        features_df, labels_df, val_days=args.val_days
    )

    # Summary
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  CV accuracy:    {result.cv_accuracy_mean:.4f} +/- {result.cv_accuracy_std:.4f}")
    print(f"  Val accuracy:   {result.val_accuracy:.4f}")
    print(f"  Val ECE:        {result.val_ece:.4f}")
    print(f"  Temperature:    {result.temperature:.4f}")
    print(f"  Train samples:  {result.n_train:,}")
    print(f"  Val samples:    {result.n_val:,}")
    print()


if __name__ == "__main__":
    main()
