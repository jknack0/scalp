"""Filter L1 tick data to RTH (9:30-16:00 ET) and store as Parquet.

Input:  data/l1/year=YYYY/data.parquet
Output: data/l1_rth/year=YYYY/data.parquet

Usage:
    python -m scripts.build_l1_rth
"""

import os
import time
from datetime import time as dt_time

import polars as pl

SRC_DIR = "data/l1"
DST_DIR = "data/l1_rth"
RTH_START = dt_time(9, 30)
RTH_END = dt_time(16, 0)


def main():
    years = sorted(
        int(d.replace("year=", ""))
        for d in os.listdir(SRC_DIR)
        if d.startswith("year=")
    )
    print(f"Filtering L1 ticks to RTH (9:30-16:00 ET)")
    print(f"Found {len(years)} year partitions: {years[0]}-{years[-1]}")
    print(f"Output: {DST_DIR}/\n")

    total_in = 0
    total_out = 0
    t0 = time.perf_counter()

    for year in years:
        src_path = os.path.join(SRC_DIR, f"year={year}", "data.parquet")
        dst_path = os.path.join(DST_DIR, f"year={year}", "data.parquet")

        if not os.path.exists(src_path):
            print(f"  SKIP year={year} (no source file)")
            continue

        df = pl.read_parquet(src_path)
        rows_in = len(df)
        total_in += rows_in

        # Convert to ET, filter RTH, drop helper column
        df_rth = df.with_columns(
            pl.col("timestamp").dt.convert_time_zone("US/Eastern").dt.time().alias("_et_time")
        ).filter(
            (pl.col("_et_time") >= RTH_START) & (pl.col("_et_time") < RTH_END)
        ).drop("_et_time")

        total_out += len(df_rth)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        df_rth.write_parquet(dst_path, compression="zstd")

        pct = (1 - len(df_rth) / rows_in) * 100 if rows_in else 0
        print(f"  year={year}: {rows_in:>12,} ticks -> {len(df_rth):>10,} RTH ticks ({pct:.0f}% dropped)")

    elapsed = time.perf_counter() - t0
    pct_total = (1 - total_out / total_in) * 100 if total_in else 0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total: {total_in:,} -> {total_out:,} RTH ticks ({pct_total:.0f}% dropped)")
    print(f"  Output: {DST_DIR}/")


if __name__ == "__main__":
    main()
