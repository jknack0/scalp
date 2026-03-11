#!/usr/bin/env python3
"""Train the HMM intraday regime classifier on year-partitioned Parquet data."""

import argparse
from datetime import timedelta
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import polars as pl

from src.data.bars import resample_bars
from src.models.hmm_regime import (
    HMMRegimeClassifier,
    HMMRegimeConfig,
    RegimeState,
    build_feature_matrix,
    compute_persistence_accuracy,
    validate_model,
)


def load_parquet_range(parquet_dir: str, start_year: int, end_year: int) -> pl.DataFrame:
    """Load and concatenate year-partitioned Parquet files."""
    frames = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(parquet_dir, f"year={year}", "data.parquet")
        if os.path.exists(path):
            df = pl.read_parquet(path)
            frames.append(df)
            print(f"  Loaded {path} ({len(df):,} rows)")
        else:
            print(f"  Skipped {path} (not found)")

    if not frames:
        print("Error: No Parquet files found.")
        sys.exit(1)

    return pl.concat(frames).sort("timestamp")


def main():
    parser = argparse.ArgumentParser(description="Train HMM Intraday Regime Classifier")
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default="data/parquet",
        help="Directory with year-partitioned Parquet files (default: data/parquet)",
    )
    parser.add_argument(
        "--start-year", type=int, default=2024, help="Start year (default: 2024)"
    )
    parser.add_argument(
        "--end-year", type=int, default=2024, help="End year (default: 2024)"
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=2,
        help="Months held out from end for validation (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/hmm/v1",
        help="Output directory for trained model (default: models/hmm/v1)",
    )
    parser.add_argument(
        "--n-states", type=int, default=2, help="Number of HMM states (default: 2)"
    )
    parser.add_argument(
        "--bar-freq",
        type=str,
        default="1m",
        help="Resample 1s bars to this frequency before training (default: 1m)",
    )

    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOADING 1-SECOND BARS")
    print("=" * 60)
    t0 = time.time()
    df_all = load_parquet_range(args.parquet_dir, args.start_year, args.end_year)
    print(f"  Total: {len(df_all):,} rows in {time.time() - t0:.1f}s")

    # ── Train/test split by time ─────────────────────────────────────
    ts_col = df_all["timestamp"]
    ts_max = ts_col.max()
    ts_min = ts_col.min()

    # Approximate split: test_months from end
    split_ts = ts_max - timedelta(days=args.test_months * 30)

    df_train = df_all.filter(pl.col("timestamp") <= split_ts)
    df_test = df_all.filter(pl.col("timestamp") > split_ts)
    print(f"\n  Train: {len(df_train):,} bars | Test: {len(df_test):,} bars (1s)")

    # ── Resample to target bar frequency ─────────────────────────────
    if args.bar_freq != "1s":
        print(f"\n  Resampling to {args.bar_freq}...")
        t0 = time.time()
        df_train = resample_bars(df_train, args.bar_freq)
        df_test = resample_bars(df_test, args.bar_freq)
        print(f"  Train: {len(df_train):,} | Test: {len(df_test):,} ({args.bar_freq}) in {time.time() - t0:.1f}s")

    # ── Build feature matrices ───────────────────────────────────────
    config = HMMRegimeConfig(n_states=args.n_states)

    print("\n" + "=" * 60)
    print("BUILDING FEATURE MATRICES")
    print("=" * 60)

    t0 = time.time()
    print("  Building train matrix...")
    train_features, train_ts = build_feature_matrix(df_train, config)
    print(f"    Shape: {train_features.shape} in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("  Building test matrix...")
    test_features, test_ts = build_feature_matrix(df_test, config)
    print(f"    Shape: {test_features.shape} in {time.time() - t0:.1f}s")

    # ── Fit HMM ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FITTING HMM")
    print("=" * 60)
    t0 = time.time()
    clf = HMMRegimeClassifier(config)
    clf.fit(train_features)
    print(f"  Fitted in {time.time() - t0:.1f}s")

    # ── Validate ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION (test set)")
    print("=" * 60)

    report = validate_model(clf, test_features)

    print(f"\n  Samples:            {report.n_samples:,}")
    print(f"  Log-likelihood:     {report.log_likelihood:.2f}")
    print(f"  Persistence (5-bar): {report.persistence_5bar:.1%}")

    print("\n  State distribution:")
    for name, frac in sorted(report.state_distribution.items()):
        bar = "#" * int(frac * 40)
        print(f"    {name:<20s} {frac:6.1%}  {bar}")

    print("\n  Transition matrix:")
    labels = [s.name for s in RegimeState]
    header = "  " + " " * 20 + "  ".join(f"{l[:8]:>8s}" for l in labels)
    print(header)
    for i, row in enumerate(report.transition_matrix):
        vals = "  ".join(f"{v:8.3f}" for v in row)
        print(f"    {labels[i]:<20s}{vals}")

    # Also check train persistence
    train_states = clf.predict_sequence(train_features)
    train_pers = compute_persistence_accuracy(train_states, horizon=5)
    print(f"\n  Train persistence (5-bar): {train_pers:.1%}")

    # ── Pass/fail ────────────────────────────────────────────────────
    threshold = 0.65
    passed = report.persistence_5bar >= threshold
    print("\n" + "=" * 60)
    if passed:
        print(f"  PASS — persistence {report.persistence_5bar:.1%} >= {threshold:.0%}")
    else:
        print(f"  FAIL — persistence {report.persistence_5bar:.1%} < {threshold:.0%}")
    print("=" * 60)

    # ── Save model ───────────────────────────────────────────────────
    clf.save(args.output)
    print(f"\n  Model saved to {args.output}/")


if __name__ == "__main__":
    main()
