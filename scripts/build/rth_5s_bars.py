"""Build RTH-only 5s bars from 1s Parquet files.

Reads each year=YYYY/data.parquet, resamples to 5s, filters to
RTH (9:30-16:00 ET), and writes to data/parquet_5s_rth/year=YYYY/data.parquet.

Usage:
    python scripts/build/rth_5s_bars.py
    python scripts/build/rth_5s_bars.py --freq 5s --years 2020 2021 2022
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import time as dt_time
from pathlib import Path

import polars as pl

from src.data.bars import resample_bars

INPUT_DIR = "data/parquet"
OUTPUT_DIR = "data/parquet_5s_rth"
RTH_START = dt_time(9, 30)
RTH_END = dt_time(16, 0)


def build_year(year: int, freq: str, input_dir: str, output_dir: str) -> int:
    """Process one year: resample + RTH filter. Returns row count."""
    src_path = os.path.join(input_dir, f"year={year}", "data.parquet")
    if not os.path.exists(src_path):
        print(f"  {year}: no source file, skipping")
        return 0

    t0 = time.time()
    df = pl.read_parquet(src_path)
    n_raw = len(df)

    # Resample 1s → 5s (or whatever freq)
    df = resample_bars(df, freq)

    # Convert to ET for RTH filtering
    df = df.with_columns(
        pl.col("timestamp")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("US/Eastern")
        .alias("_et_ts")
    )

    # Filter to RTH
    df = df.filter(
        (pl.col("_et_ts").dt.time() >= RTH_START)
        & (pl.col("_et_ts").dt.time() < RTH_END)
    )

    # Drop helper column, keep timestamps as naive UTC (matches original schema)
    df = df.drop("_et_ts")

    # Write output
    dest_dir = os.path.join(output_dir, f"year={year}")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "data.parquet")
    df.write_parquet(dest_path, compression="zstd")

    elapsed = time.time() - t0
    n_out = len(df)
    ratio = n_out / n_raw * 100 if n_raw > 0 else 0
    print(f"  {year}: {n_raw:>10,} 1s -> {n_out:>8,} {freq} RTH ({ratio:.1f}%) in {elapsed:.1f}s")
    return n_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RTH-only resampled bars")
    parser.add_argument("--freq", default="5s", help="Resample frequency (default: 5s)")
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--years", type=int, nargs="*", help="Specific years (default: all)")
    args = parser.parse_args()

    if args.years:
        years = args.years
    else:
        # Auto-detect from input dir
        years = sorted(
            int(d.replace("year=", ""))
            for d in os.listdir(args.input_dir)
            if d.startswith("year=")
        )

    if not years:
        print("No year directories found")
        sys.exit(1)

    print(f"Building {args.freq} RTH bars: {years[0]}-{years[-1]}")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print()

    total = 0
    t_start = time.time()
    for year in years:
        total += build_year(year, args.freq, args.input_dir, args.output_dir)

    elapsed = time.time() - t_start
    print(f"\nDone: {total:,} total bars in {elapsed:.1f}s")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
