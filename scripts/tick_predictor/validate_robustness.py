"""Robustness validation for TickDirectionPredictor.

Two tests:
  Test A — Reverse walk-forward: Train on RECENT data, test on OLDER data.
           If accuracy holds, the signal captures structural microstructure
           rather than regime-specific artifacts.

  Test B — Flipped labels: Evaluate the existing model but with UP↔DOWN swapped.
           A good model should score well BELOW 50% — proving its predictions
           are anti-correlated with flipped truth (i.e. genuinely correlated
           with real truth).

Usage:
    python -u scripts/tick_predictor/validate_robustness.py \
        --start 2025-01-01 --end 2026-03-14 \
        --tp-ticks 24 --sl-ticks 12
"""

from __future__ import annotations

import argparse
import os
import sys
import time as _time
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np
import polars as pl

from src.core.logging import configure_logging, get_logger
from src.signals.tick_predictor.features.feature_builder import FEATURE_NAMES
from src.signals.tick_predictor.model.trainer import ModelTrainer, TrainerConfig

logger = get_logger("tick_predictor.validate")

DATA_DIR = Path("data/tick_predictor")


def load_joined_data(start_date: str, end_date: str, tp_ticks: int, sl_ticks: int,
                     drop_flat: bool = True) -> pl.DataFrame:
    """Load and join cached features + labels."""
    suffix = f"_v4_h300_tp{tp_ticks}_sl{sl_ticks}_rth"
    feat_path = DATA_DIR / f"features_{start_date}_{end_date}{suffix}.parquet"
    # Labels don't include the full suffix in the filename
    label_path = DATA_DIR / f"labels_{start_date}_{end_date}.parquet"

    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found: {feat_path}\nRun train.py first.")
    if not label_path.exists():
        raise FileNotFoundError(f"Labels not found: {label_path}\nRun train.py first.")

    print(f"  Loading features: {feat_path}")
    features_df = pl.read_parquet(feat_path)
    print(f"  Loading labels:   {label_path}")
    labels_df = pl.read_parquet(label_path)

    joined = features_df.join(labels_df, on="timestamp_ns", how="inner")
    joined = joined.drop_nulls(subset=FEATURE_NAMES)
    print(f"  Joined: {len(joined):,} rows")

    if drop_flat:
        n_before = len(joined)
        joined = joined.filter(pl.col("label") != 0)
        print(f"  Dropped FLAT: {n_before - len(joined):,} rows -> {len(joined):,}")

    return joined


def test_reverse_walkforward(joined: pl.DataFrame, train_frac: float = 0.5) -> None:
    """Test A: Train on recent half, test on older half."""
    print(f"\n{'='*70}")
    print("  TEST A: REVERSE WALK-FORWARD")
    print(f"  Train on RECENT data, test on OLDER data")
    print(f"{'='*70}\n")

    n = len(joined)
    midpoint = n // 2

    # Chronological order: older is first, recent is last
    # REVERSE: train on second half, test on first half
    test_df = joined.slice(0, midpoint)          # older
    train_df = joined.slice(midpoint, n - midpoint)  # recent

    ts_test_min = test_df["timestamp_ns"].min()
    ts_test_max = test_df["timestamp_ns"].max()
    ts_train_min = train_df["timestamp_ns"].min()
    ts_train_max = train_df["timestamp_ns"].max()

    # Convert ns timestamps to readable dates
    from datetime import datetime, timezone
    def ns_to_date(ns):
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")

    print(f"  Train (RECENT): {len(train_df):,} rows  ({ns_to_date(ts_train_min)} -> {ns_to_date(ts_train_max)})")
    print(f"  Test  (OLDER):  {len(test_df):,} rows  ({ns_to_date(ts_test_min)} -> {ns_to_date(ts_test_max)})")

    X_train = train_df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    y_train_raw = train_df["label"].to_numpy()
    X_test = test_df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    y_test_raw = test_df["label"].to_numpy()

    # Encode: -1->0, 0->1, 1->2
    label_map = {-1: 0, 0: 1, 1: 2}
    y_train = np.vectorize(label_map.get)(y_train_raw).astype(np.int32)
    y_test = np.vectorize(label_map.get)(y_test_raw).astype(np.int32)

    # Use sample weights = 1.0 (uniform)
    w_train = np.ones(len(y_train), dtype=np.float32)

    # --- Walk-forward CV on training data (recent half) ---
    print(f"\n  Running walk-forward CV on training set...")
    t0 = _time.perf_counter()
    trainer = ModelTrainer(TrainerConfig())
    cv_results = trainer.purged_walkforward_cv(X_train, y_train, w_train)

    mean_acc = float(np.mean([r.accuracy for r in cv_results]))
    std_acc = float(np.std([r.accuracy for r in cv_results]))
    print(f"  CV accuracy (recent data): {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  ({_time.perf_counter()-t0:.1f}s)")

    # --- Train final model on recent, evaluate on older ---
    print(f"\n  Training final model on recent half...")
    t0 = _time.perf_counter()
    booster = trainer.train_final_model(X_train, y_train, w_train)
    print(f"  Trained in {_time.perf_counter()-t0:.1f}s")

    # Predict on older test data
    raw_proba = booster.predict(X_test)
    if raw_proba.ndim == 1:
        raw_proba = raw_proba.reshape(-1, 3)

    y_pred = np.argmax(raw_proba, axis=1)
    accuracy = float(np.mean(y_pred == y_test))

    # Per-class accuracy
    for cls_id, cls_name in [(0, "DOWN"), (2, "UP")]:
        mask = y_test == cls_id
        if mask.sum() > 0:
            cls_acc = float(np.mean(y_pred[mask] == y_test[mask]))
            print(f"  {cls_name} accuracy: {cls_acc:.4f} ({mask.sum():,} samples)")

    print(f"\n  >>> REVERSE TEST ACCURACY: {accuracy:.4f} <<<")
    print(f"  (Compare to normal val accuracy ~0.67)")

    # --- Also run NORMAL direction for comparison ---
    print(f"\n  [Comparison] Training on OLDER half, testing on RECENT half...")
    t0 = _time.perf_counter()
    trainer2 = ModelTrainer(TrainerConfig())
    booster2 = trainer2.train_final_model(
        test_df.select(FEATURE_NAMES).to_numpy().astype(np.float32),  # older = train
        y_test,  # older labels
        np.ones(len(y_test), dtype=np.float32),
    )
    raw_proba2 = booster2.predict(X_train)
    if raw_proba2.ndim == 1:
        raw_proba2 = raw_proba2.reshape(-1, 3)
    y_pred2 = np.argmax(raw_proba2, axis=1)
    normal_acc = float(np.mean(y_pred2 == y_train))
    print(f"  Normal direction accuracy: {normal_acc:.4f}")
    print(f"  ({_time.perf_counter()-t0:.1f}s)")

    print(f"\n  SUMMARY:")
    print(f"    Train RECENT -> Test OLDER:  {accuracy:.4f}")
    print(f"    Train OLDER  -> Test RECENT: {normal_acc:.4f}")
    diff = abs(accuracy - normal_acc)
    if diff < 0.03:
        print(f"    Gap: {diff:.4f} — EXCELLENT: signal is time-symmetric")
    elif diff < 0.06:
        print(f"    Gap: {diff:.4f} — GOOD: minor temporal drift but signal holds")
    else:
        print(f"    Gap: {diff:.4f} — WARNING: significant regime dependency")


def test_flipped_labels(joined: pl.DataFrame) -> None:
    """Test B: Evaluate existing model with UP↔DOWN swapped."""
    print(f"\n{'='*70}")
    print("  TEST B: FLIPPED LABELS")
    print(f"  Evaluate model against swapped UP<->DOWN labels")
    print(f"  A good model should score WELL BELOW 50%")
    print(f"{'='*70}\n")

    import lightgbm as lgb
    from src.signals.tick_predictor.model.calibrator import TemperatureCalibrator

    model_path = "models/tick_predictor/lgbm_latest.txt"
    cal_path = "models/tick_predictor/calibrator_latest.pkl"

    if not Path(model_path).exists():
        print(f"  Model not found: {model_path}")
        return

    model = lgb.Booster(model_file=model_path)
    if model.num_feature() != len(FEATURE_NAMES):
        print(f"  Model feature mismatch: {model.num_feature()} vs {len(FEATURE_NAMES)}")
        return

    calibrator = TemperatureCalibrator()
    calibrator.load(cal_path)

    X = joined.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    y_raw = joined["label"].to_numpy()

    # Encode: -1->0, 0->1, 1->2
    label_map = {-1: 0, 0: 1, 1: 2}
    y_true = np.vectorize(label_map.get)(y_raw).astype(np.int32)

    # Flip: DOWN(0) <-> UP(2), FLAT(1) stays
    flip_map = {0: 2, 1: 1, 2: 0}
    y_flipped = np.vectorize(flip_map.get)(y_true).astype(np.int32)

    print(f"  Running inference on {len(X):,} bars...")
    raw_proba = model.predict(X)
    if raw_proba.ndim == 1:
        raw_proba = raw_proba.reshape(-1, 3)

    cal_proba = calibrator.predict_proba_calibrated(raw_proba)
    y_pred = np.argmax(cal_proba, axis=1)

    # Normal accuracy
    normal_acc = float(np.mean(y_pred == y_true))
    # Flipped accuracy
    flipped_acc = float(np.mean(y_pred == y_flipped))
    # Random baseline (marginal distribution)
    from collections import Counter
    dist = Counter(y_true.tolist())
    random_acc = sum((v/len(y_true))**2 for v in dist.values())

    print(f"\n  Results:")
    print(f"    Normal accuracy:     {normal_acc:.4f}")
    print(f"    Flipped accuracy:    {flipped_acc:.4f}")
    print(f"    Random baseline:     {random_acc:.4f}")
    print(f"    Normal - Flipped:    {normal_acc - flipped_acc:.4f}")

    # Confidence analysis on flipped
    max_conf = np.max(cal_proba, axis=1)
    high_conf_mask = max_conf > 0.65

    if high_conf_mask.sum() > 0:
        hc_normal = float(np.mean(y_pred[high_conf_mask] == y_true[high_conf_mask]))
        hc_flipped = float(np.mean(y_pred[high_conf_mask] == y_flipped[high_conf_mask]))
        print(f"\n  High-confidence trades (>{0.65:.0%}):")
        print(f"    Count:               {high_conf_mask.sum():,}")
        print(f"    Normal accuracy:     {hc_normal:.4f}")
        print(f"    Flipped accuracy:    {hc_flipped:.4f}")

    if flipped_acc < 0.40:
        print(f"\n  >>> PASS: Flipped accuracy {flipped_acc:.4f} << 50%")
        print(f"      Model predictions are genuinely directional")
    elif flipped_acc < 0.48:
        print(f"\n  >>> WEAK PASS: Flipped accuracy {flipped_acc:.4f} is below 50%")
        print(f"      Some directional signal but not strongly anti-correlated")
    else:
        print(f"\n  >>> FAIL: Flipped accuracy {flipped_acc:.4f} ~ 50%")
        print(f"      Model may be fitting noise, not direction")


def main():
    parser = argparse.ArgumentParser(description="Robustness validation for tick predictor")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2026-03-14")
    parser.add_argument("--tp-ticks", type=int, default=24)
    parser.add_argument("--sl-ticks", type=int, default=12)
    parser.add_argument("--test", type=str, default="both",
                        choices=["reverse", "flipped", "both"],
                        help="Which test to run (default: both)")
    args = parser.parse_args()

    configure_logging(log_level="WARNING")

    joined = load_joined_data(args.start, args.end, args.tp_ticks, args.sl_ticks)

    if args.test in ("reverse", "both"):
        test_reverse_walkforward(joined)

    if args.test in ("flipped", "both"):
        test_flipped_labels(joined)

    print(f"\n{'='*70}")
    print("  VALIDATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
