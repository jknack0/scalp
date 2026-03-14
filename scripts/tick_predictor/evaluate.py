"""Evaluation CLI for TickDirectionPredictor.

Usage:
    python -u scripts/tick_predictor/evaluate.py
    python -u scripts/tick_predictor/evaluate.py --test-data data/tick_predictor/features_2025-01-01_2025-03-01.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np
import polars as pl

from src.core.logging import configure_logging, get_logger
from src.signals.tick_predictor.evaluation.metrics import PredictorEvaluator
from src.signals.tick_predictor.features.feature_builder import FEATURE_NAMES
from src.signals.tick_predictor.model.calibrator import TemperatureCalibrator

logger = get_logger("tick_predictor.evaluate")

MODEL_DIR = Path("models/tick_predictor")
DATA_DIR = Path("data/tick_predictor")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TickDirectionPredictor")
    parser.add_argument("--model", default=str(MODEL_DIR / "lgbm_latest.txt"))
    parser.add_argument("--calibrator", default=str(MODEL_DIR / "calibrator_latest.pkl"))
    parser.add_argument("--test-data", default=None,
                        help="Path to test features Parquet (auto-detect if omitted)")
    parser.add_argument("--test-labels", default=None,
                        help="Path to test labels Parquet (auto-detect if omitted)")
    args = parser.parse_args()

    configure_logging(log_level="INFO")

    # Load model
    import lightgbm as lgb
    if not Path(args.model).exists():
        print(f"  Model not found: {args.model}")
        sys.exit(1)
    model = lgb.Booster(model_file=args.model)
    print(f"  Model loaded: {args.model}")

    # Load calibrator
    calibrator = TemperatureCalibrator()
    if Path(args.calibrator).exists():
        calibrator.load(args.calibrator)
        print(f"  Calibrator loaded: {args.calibrator} (T={calibrator.temperature:.4f})")
    else:
        print("  No calibrator found — using T=1.0")

    # Find test data
    if args.test_data:
        feat_path = args.test_data
    else:
        # Find latest features file
        feat_files = sorted(DATA_DIR.glob("features_*.parquet"))
        if not feat_files:
            print("  No feature files found in data/tick_predictor/")
            sys.exit(1)
        feat_path = str(feat_files[-1])

    if args.test_labels:
        label_path = args.test_labels
    else:
        label_files = sorted(DATA_DIR.glob("labels_*.parquet"))
        if not label_files:
            print("  No label files found in data/tick_predictor/")
            sys.exit(1)
        label_path = str(label_files[-1])

    print(f"  Features: {feat_path}")
    print(f"  Labels:   {label_path}")

    # Load and join
    features_df = pl.read_parquet(feat_path)
    labels_df = pl.read_parquet(label_path)
    df = features_df.join(labels_df, on="timestamp_ns", how="inner")
    df = df.drop_nulls(subset=FEATURE_NAMES)

    X = df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    y_raw = df["label"].to_numpy()
    label_map = {-1: 0, 0: 1, 1: 2}
    y = np.vectorize(label_map.get)(y_raw).astype(np.int32)

    print(f"  Test samples: {len(X):,}")

    # Run evaluation
    evaluator = PredictorEvaluator()
    report = evaluator.generate_full_report(
        model=model,
        calibrator=calibrator,
        X_test=X,
        y_test=y,
        save_dir=str(MODEL_DIR),
    )

    print(f"\n{'='*50}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"  Accuracy:  {report['accuracy']:.4f}")
    print(f"  ECE:       {report['ece']:.4f}")
    print(f"  Brier:     {report['brier_score']:.4f}")
    print(f"  Report:    {MODEL_DIR}")


if __name__ == "__main__":
    main()
