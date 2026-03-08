#!/usr/bin/env python3
"""Analyze mean reversion / momentum transition across timescales on MES 1s bars."""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl

from src.analysis.regime_characterization import (
    print_regime_report,
    run_full_analysis,
    segment_by_time_of_day,
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
    parser = argparse.ArgumentParser(
        description="MES Mean Reversion / Momentum Transition Analysis"
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default="data/parquet",
        help="Directory with year-partitioned Parquet files (default: data/parquet)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2024,
        help="Start year (default: 2024)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="End year (default: 2024)",
    )
    parser.add_argument(
        "--max-acf-lag",
        type=int,
        default=200,
        help="Maximum ACF lag in 1s bars (default: 200)",
    )
    parser.add_argument(
        "--by-tod",
        action="store_true",
        help="Also run analysis segmented by time of day (open/midday/close)",
    )

    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOADING 1-SECOND BARS")
    print("=" * 60)
    t0 = time.time()
    df_1s = load_parquet_range(args.parquet_dir, args.start_year, args.end_year)
    print(f"  Total: {len(df_1s):,} rows in {time.time() - t0:.1f}s")

    # ── Full session analysis ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RUNNING REGIME ANALYSIS (full session)")
    print("=" * 60)
    t0 = time.time()
    report_all = run_full_analysis(df_1s, label="full-session", max_acf_lag=args.max_acf_lag)
    reports = [report_all]
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Time-of-day segmented analysis ───────────────────────────────
    if args.by_tod:
        print("\n" + "=" * 60)
        print("RUNNING TIME-OF-DAY SEGMENTED ANALYSIS")
        print("=" * 60)
        segments = segment_by_time_of_day(df_1s)
        for name, seg_df in segments.items():
            if len(seg_df) < 100:
                print(f"  Skipping {name}: only {len(seg_df)} bars")
                continue
            print(f"  Analyzing {name} ({len(seg_df):,} bars)...")
            t0 = time.time()
            seg_report = run_full_analysis(seg_df, label=name, max_acf_lag=args.max_acf_lag)
            reports.append(seg_report)
            print(f"    Done in {time.time() - t0:.1f}s")

    # ── Print all reports ────────────────────────────────────────────
    print_regime_report(reports)


if __name__ == "__main__":
    main()
