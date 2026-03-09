"""Resample 1s bars to any frequency and store as Parquet (year-partitioned).

Output: data/parquet_{label}/year=YYYY/data.parquet

With --rth, filters to RTH (9:30-16:00 ET) before resampling, dropping ~72% of bars.
Output: data/parquet_{label}_rth/year=YYYY/data.parquet

Usage:
    python -m scripts.build_bars              # default: 1m bars (all hours)
    python -m scripts.build_bars --rth        # 1m RTH-only bars
    python -m scripts.build_bars --freq 5s --rth  # 5s RTH-only bars
"""

import argparse
import os
import time
from datetime import time as dt_time
from zoneinfo import ZoneInfo

import polars as pl

from src.data.bars import resample_bars

SRC_DIR = "data/parquet"
_ET = ZoneInfo("US/Eastern")
RTH_START = dt_time(9, 30)
RTH_END = dt_time(16, 0)


def freq_to_label(freq: str) -> str:
    """Convert Polars freq string to directory label: '5s' -> '5s', '1m' -> '1m'."""
    return freq.replace(" ", "")


def filter_rth(df: pl.DataFrame) -> pl.DataFrame:
    """Filter DataFrame to RTH hours (9:30-16:00 ET)."""
    return df.with_columns(
        pl.col("timestamp").dt.convert_time_zone("US/Eastern").dt.time().alias("_et_time")
    ).filter(
        (pl.col("_et_time") >= RTH_START) & (pl.col("_et_time") < RTH_END)
    ).drop("_et_time")


def main():
    parser = argparse.ArgumentParser(description="Resample 1s bars to coarser timeframe")
    parser.add_argument("--freq", default="1m", help="Target frequency (e.g. 5s, 15s, 1m)")
    parser.add_argument("--rth", action="store_true", help="Filter to RTH (9:30-16:00 ET) before resampling")
    args = parser.parse_args()

    label = freq_to_label(args.freq)
    if args.rth:
        label += "_rth"
    dst_dir = f"data/parquet_{label}"

    years = sorted(
        int(d.replace("year=", ""))
        for d in os.listdir(SRC_DIR)
        if d.startswith("year=")
    )
    mode = "RTH only (9:30-16:00 ET)" if args.rth else "all hours"
    print(f"Resampling 1s -> {args.freq} ({mode})")
    print(f"Found {len(years)} year partitions: {years[0]}-{years[-1]}")
    print(f"Output: {dst_dir}/\n")

    total_rows_in = 0
    total_rows_rth = 0
    total_rows_out = 0
    t0 = time.perf_counter()

    for year in years:
        src_path = os.path.join(SRC_DIR, f"year={year}", "data.parquet")
        year_dst_dir = os.path.join(dst_dir, f"year={year}")
        dst_path = os.path.join(year_dst_dir, "data.parquet")

        if not os.path.exists(src_path):
            print(f"  SKIP year={year} (no source file)")
            continue

        df = pl.read_parquet(src_path)
        rows_in = len(df)
        total_rows_in += rows_in

        # Make timestamps timezone-aware (UTC) if naive
        if df["timestamp"].dtype == pl.Datetime("us"):
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))

        if args.rth:
            df = filter_rth(df)
            total_rows_rth += len(df)

        # Strip timezone before resample+write (engine expects naive UTC)
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

        if args.freq == "1s":
            df_out = df  # no-op: source is already 1s
        else:
            df_out = resample_bars(df, args.freq)
        total_rows_out += len(df_out)

        os.makedirs(year_dst_dir, exist_ok=True)
        df_out.write_parquet(dst_path, compression="zstd")

        if args.rth:
            print(f"  year={year}: {rows_in:>10,} 1s -> {len(df):>8,} RTH -> {len(df_out):>8,} {args.freq} bars")
        else:
            print(f"  year={year}: {rows_in:>10,} 1s bars -> {len(df_out):>8,} {args.freq} bars")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    if args.rth:
        pct = (1 - total_rows_rth / total_rows_in) * 100 if total_rows_in else 0
        print(f"  RTH filter dropped {pct:.0f}% of bars ({total_rows_in:,} -> {total_rows_rth:,})")
    print(f"  Total: {total_rows_in:,} 1s bars -> {total_rows_out:,} {label} bars")
    print(f"  Output: {dst_dir}/")


if __name__ == "__main__":
    main()
