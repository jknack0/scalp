#!/usr/bin/env python3
"""Analyze intraday volatility & volume profiles on MES 1s bars."""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl

from src.analysis.intraday_profile import (
    build_daily_profiles,
    compute_realized_vol_heatmap,
    compute_spread_cost_by_slot,
    compute_u_shape_metrics,
    identify_dead_zone,
    print_volatility_summary,
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
        description="MES Intraday Volatility & Volume Profile Analysis"
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
        "--output-dir",
        type=str,
        default="data/volume_profiles",
        help="Output directory for volume profiles (default: data/volume_profiles)",
    )
    parser.add_argument(
        "--save-profiles",
        action="store_true",
        help="Save daily volume profiles as Parquet files",
    )

    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOADING 1-SECOND BARS")
    print("=" * 60)
    t0 = time.time()
    df_1s = load_parquet_range(args.parquet_dir, args.start_year, args.end_year)
    print(f"  Total: {len(df_1s):,} rows in {time.time() - t0:.1f}s")

    # ── Volatility heatmap ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPUTING 15-MIN REALIZED VOLATILITY HEATMAP")
    print("=" * 60)
    t0 = time.time()
    heatmap = compute_realized_vol_heatmap(df_1s, window_minutes=15)
    print(f"  {len(heatmap.time_slots)} time slots × {len(heatmap.months)} months")
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Dead zone ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("IDENTIFYING DEAD ZONE")
    print("=" * 60)
    dead_zone = identify_dead_zone(heatmap, threshold=1.5)

    # ── U-shape metrics ──────────────────────────────────────────────
    u_shape = compute_u_shape_metrics(heatmap)

    # ── Spread cost ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPUTING SPREAD COST BY TIME SLOT")
    print("=" * 60)
    t0 = time.time()
    spread = compute_spread_cost_by_slot(df_1s, window_minutes=15)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Print summary ────────────────────────────────────────────────
    print_volatility_summary(heatmap, dead_zone, u_shape, spread)

    # ── Volume profiles ──────────────────────────────────────────────
    if args.save_profiles:
        print("=" * 60)
        print("BUILDING DAILY VOLUME PROFILES")
        print("=" * 60)
        t0 = time.time()
        profiles = build_daily_profiles(df_1s, tick_size=0.25)
        print(f"  Built {len(profiles)} daily profiles in {time.time() - t0:.1f}s")

        os.makedirs(args.output_dir, exist_ok=True)
        for p in profiles:
            out_path = os.path.join(args.output_dir, f"{p.date}.parquet")
            df_out = pl.DataFrame({
                "price_level": p.price_levels,
                "volume": p.volumes,
            })
            df_out.write_parquet(out_path)

        print(f"  Saved to {args.output_dir}/")
        if profiles:
            sample = profiles[0]
            print(f"  Sample ({sample.date}): POC={sample.poc}, VAL={sample.val}, VAH={sample.vah}")


if __name__ == "__main__":
    main()
