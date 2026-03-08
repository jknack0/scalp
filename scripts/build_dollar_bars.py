#!/usr/bin/env python3
"""Build dollar bars from 1s Parquet data and optionally run statistical comparison."""

import argparse
import os
import sys
import time

import numpy as np
import polars as pl

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.bars import build_dollar_bars, resample_bars
from src.analysis.bar_statistics import analyze_bars, compare_bar_types, print_comparison_report


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
        description="Build MES dollar bars from 1s Parquet data"
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default="data/parquet",
        help="Directory with year-partitioned Parquet files (default: data/parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dollar_bars",
        help="Output directory for dollar bar Parquet files (default: data/dollar_bars)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2023,
        help="Start year (default: 2023)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year (default: 2025)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=15000.0,
        help="Dollar volume threshold per bar (default: 15000.0)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run statistical comparison vs 1m and 5m time bars",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Load existing dollar bars and run stats only (skip build)",
    )

    args = parser.parse_args()

    # ── Load 1s bars ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOADING 1-SECOND BARS")
    print("=" * 60)
    t0 = time.time()
    df_1s = load_parquet_range(args.parquet_dir, args.start_year, args.end_year)
    print(f"  Total: {len(df_1s):,} rows in {time.time() - t0:.1f}s")

    # ── Build or load dollar bars ────────────────────────────────────
    output_path = os.path.join(args.output_dir, "dollar_bars.parquet")

    if args.stats_only:
        if not os.path.exists(output_path):
            print(f"Error: {output_path} not found. Run without --stats-only first.")
            sys.exit(1)
        print(f"\n  Loading existing dollar bars from {output_path}")
        df_dollar = pl.read_parquet(output_path)
    else:
        print("\n" + "=" * 60)
        print(f"BUILDING DOLLAR BARS (threshold=${args.threshold:,.0f})")
        print("=" * 60)
        t0 = time.time()
        df_dollar = build_dollar_bars(df_1s, args.threshold)
        elapsed = time.time() - t0
        print(f"  Built {len(df_dollar):,} dollar bars in {elapsed:.1f}s")
        print(f"  Avg duration: {df_dollar['bar_duration_s'].mean():.1f}s")
        print(f"  Median duration: {df_dollar['bar_duration_s'].median():.1f}s")

        # Save
        os.makedirs(args.output_dir, exist_ok=True)
        df_dollar.write_parquet(output_path)
        print(f"  Saved to {output_path}")

    # ── Statistical comparison ───────────────────────────────────────
    if args.compare or args.stats_only:
        print("\n" + "=" * 60)
        print("STATISTICAL COMPARISON")
        print("=" * 60)

        bar_sets: dict[str, np.ndarray] = {
            "dollar": df_dollar["close"].to_numpy(),
        }

        # Build time bars for comparison
        print("  Resampling 1m bars...")
        df_1m = resample_bars(df_1s, "1m")
        bar_sets["1m"] = df_1m["close"].to_numpy()
        print(f"    {len(df_1m):,} bars")

        print("  Resampling 5m bars...")
        df_5m = resample_bars(df_1s, "5m")
        bar_sets["5m"] = df_5m["close"].to_numpy()
        print(f"    {len(df_5m):,} bars")

        print("  Running statistical tests...")
        t0 = time.time()
        reports = compare_bar_types(bar_sets)
        elapsed = time.time() - t0

        print_comparison_report(reports)
        print(f"  Stats completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
