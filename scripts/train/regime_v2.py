"""Train Phase 7 Regime Detector V2 on 5-minute bars.

Usage:
    # Train on all 5m parquet data
    python scripts/train/regime_v2.py --parquet-dir data/parquet_1m --output models/regime_v2

    # Train on date range
    python scripts/train/regime_v2.py --parquet-dir data/parquet_1m --output models/regime_v2 \
        --start 2020-01-01 --end 2024-01-01

    # Walk-forward validation
    python scripts/train/regime_v2.py --parquet-dir data/parquet_1m --output models/regime_v2 \
        --walk-forward --wf-train-months 6 --wf-test-months 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.bars import resample_bars
from src.data.csv_to_parquet import read_parquet_range
from src.models.regime_detector_v2 import (
    RegimeDetectorV2,
    RegimeDetectorV2Config,
    RegimeLabel,
    build_features_v2,
    compute_regime_stats,
)


def load_5m_bars(parquet_dir: str, start: str | None, end: str | None) -> pl.DataFrame:
    """Load 1m parquet data and resample to 5m bars."""
    # Determine year range from directory or args
    from pathlib import Path
    import re
    parquet_path = Path(parquet_dir)
    years = sorted(
        int(d.name.split("=")[1])
        for d in parquet_path.iterdir()
        if d.is_dir() and re.match(r"year=\d{4}", d.name)
    )
    if not years:
        raise FileNotFoundError(f"No year= partitions in {parquet_dir}")

    start_year = int(start[:4]) if start else years[0]
    end_year = int(end[:4]) if end else years[-1]

    df = read_parquet_range(parquet_dir, start_year, end_year)
    print(f"Loaded {len(df):,} raw 1m bars from {parquet_dir} ({start_year}-{end_year})")

    # Date filter
    if start:
        df = df.filter(pl.col("timestamp") >= pl.lit(start).str.to_datetime())
    if end:
        df = df.filter(pl.col("timestamp") < pl.lit(end).str.to_datetime())

    print(f"After date filter: {len(df):,} bars")

    # Resample to 5m
    df_5m = resample_bars(df, freq="5m")
    print(f"Resampled to 5m: {len(df_5m):,} bars")
    return df_5m


def train_single(
    df: pl.DataFrame,
    config: RegimeDetectorV2Config,
    output_dir: str,
) -> None:
    """Train a single V2 detector and save."""
    features, timestamps = build_features_v2(df, config)
    print(f"Feature matrix: {features.shape}")

    detector = RegimeDetectorV2(config)
    detector.fit(features)

    # Batch predict (forward-only) for stats
    probas = detector.predict_proba_sequence(features)
    stats = compute_regime_stats(probas)

    print("\n=== Regime Stats ===")
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv:.3f}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Save
    detector.save(output_dir)
    print(f"\nModel saved to {output_dir}/")


def walk_forward_validate(
    df: pl.DataFrame,
    config: RegimeDetectorV2Config,
    train_months: int = 6,
    test_months: int = 1,
) -> None:
    """Walk-forward validation: train on [T-W, T], predict on [T, T+M]."""
    # Get date range
    ts_col = df["timestamp"]
    min_date = ts_col.min()
    max_date = ts_col.max()
    print(f"Date range: {min_date} to {max_date}")

    # Generate splits
    from datetime import timedelta

    train_delta = timedelta(days=train_months * 30)
    test_delta = timedelta(days=test_months * 30)

    splits = []
    current_train_start = min_date
    while current_train_start + train_delta + test_delta <= max_date:
        train_end = current_train_start + train_delta
        test_end = train_end + test_delta
        splits.append((current_train_start, train_end, test_end))
        current_train_start = train_end  # slide by test_months

    print(f"\n{len(splits)} walk-forward splits")

    all_oos_stats = []

    for i, (train_start, train_end, test_end) in enumerate(splits):
        print(f"\n--- Split {i+1}/{len(splits)} ---")
        print(f"  Train: {train_start.date()} -> {train_end.date()}")
        print(f"  Test:  {train_end.date()} -> {test_end.date()}")

        # Split data
        train_df = df.filter(
            (pl.col("timestamp") >= train_start) & (pl.col("timestamp") < train_end)
        )
        test_df = df.filter(
            (pl.col("timestamp") >= train_end) & (pl.col("timestamp") < test_end)
        )

        if len(train_df) < 1000 or len(test_df) < 100:
            print(f"  Skipping: train={len(train_df)}, test={len(test_df)} bars")
            continue

        # Build features
        train_features, _ = build_features_v2(train_df, config)
        test_features, _ = build_features_v2(test_df, config)

        if len(train_features) < 500 or len(test_features) < 50:
            print(f"  Skipping: train_feat={len(train_features)}, test_feat={len(test_features)}")
            continue

        # Fit
        detector = RegimeDetectorV2(config)
        detector.fit(train_features)

        # OOS predict (forward-only)
        oos_probas = detector.predict_proba_sequence(test_features)
        oos_stats = compute_regime_stats(oos_probas)

        print(f"  OOS bars: {oos_stats['n_bars']}")
        print(f"  OOS dist: {oos_stats['state_distribution']}")
        print(f"  OOS avg conf: {oos_stats['avg_confidence']:.3f}")
        print(f"  OOS transitions: {oos_stats['transitions']}")
        print(f"  OOS halt%: {oos_stats['halt_fraction']:.3f}")
        print(f"  OOS avg stint: {oos_stats['avg_stint_length']:.1f} bars")
        print(f"  OOS size dist: {oos_stats['position_size_distribution']}")

        all_oos_stats.append(oos_stats)

    # Summary
    if all_oos_stats:
        print("\n=== Walk-Forward Summary ===")
        avg_conf = np.mean([s["avg_confidence"] for s in all_oos_stats])
        avg_halt = np.mean([s["halt_fraction"] for s in all_oos_stats])
        avg_stint = np.mean([s["avg_stint_length"] for s in all_oos_stats])
        avg_transitions = np.mean([s["transitions"] for s in all_oos_stats])

        # Aggregate state distribution
        agg_dist = {label.name: 0.0 for label in RegimeLabel}
        for s in all_oos_stats:
            for k, v in s["state_distribution"].items():
                agg_dist[k] += v
        agg_dist = {k: v / len(all_oos_stats) for k, v in agg_dist.items()}

        # Aggregate position size distribution
        agg_size = {"full": 0.0, "half": 0.0, "flat": 0.0}
        for s in all_oos_stats:
            for k, v in s["position_size_distribution"].items():
                agg_size[k] += v
        agg_size = {k: v / len(all_oos_stats) for k, v in agg_size.items()}

        print(f"  Splits: {len(all_oos_stats)}")
        print(f"  Avg confidence: {avg_conf:.3f}")
        print(f"  Avg halt fraction: {avg_halt:.3f}")
        print(f"  Avg stint length: {avg_stint:.1f} bars")
        print(f"  Avg transitions/split: {avg_transitions:.1f}")
        print(f"  Avg state dist: {agg_dist}")
        print(f"  Avg size dist: {agg_size}")

        # Red flags
        if avg_conf > 0.90:
            print("  WARNING: Avg confidence > 0.90 — possible overfit")
        if avg_halt > 0.30:
            print("  WARNING: >30% bars halted — anti-whipsaw too aggressive or model unstable")
        if avg_stint < 3:
            print("  WARNING: Avg stint < 3 bars — model is flipping too fast")


def main():
    parser = argparse.ArgumentParser(description="Train Phase 7 Regime Detector V2")
    parser.add_argument("--parquet-dir", default="data/parquet_1m",
                        help="Directory with year-partitioned 1m parquet files")
    parser.add_argument("--output", default="models/regime_v2",
                        help="Output directory for saved model")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")

    # Walk-forward
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation instead of single train")
    parser.add_argument("--wf-train-months", type=int, default=6,
                        help="Training window in months (default: 6)")
    parser.add_argument("--wf-test-months", type=int, default=1,
                        help="Test window in months (default: 1)")

    # Config overrides
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--gk-window", type=int, default=20)
    parser.add_argument("--zscore-window", type=int, default=500)
    parser.add_argument("--dof", type=int, default=5,
                        help="Student-t degrees of freedom (default: 5)")
    parser.add_argument("--hurst-window", type=int, default=250)
    parser.add_argument("--autocorr-window", type=int, default=100)
    parser.add_argument("--emission", default="gaussian",
                        choices=["gaussian", "studentt"],
                        help="Emission distribution type (default: gaussian)")

    args = parser.parse_args()

    config = RegimeDetectorV2Config(
        n_states=args.n_states,
        gk_vol_window=args.gk_window,
        zscore_window=args.zscore_window,
        studentt_dof=args.dof,
        hurst_window=args.hurst_window,
        autocorr_window=args.autocorr_window,
        emission_type=args.emission,
    )

    print(f"Config: {config}")

    df_5m = load_5m_bars(args.parquet_dir, args.start, args.end)

    if args.walk_forward:
        walk_forward_validate(df_5m, config, args.wf_train_months, args.wf_test_months)
    else:
        train_single(df_5m, config, args.output)


if __name__ == "__main__":
    main()
