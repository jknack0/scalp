#!/usr/bin/env python3
"""Convert DataBento L2 MBP-10 DBN files to Parquet for backtesting.

Samples the full 10-level book at configurable intervals (default: 1 second)
and writes to data/l2_parquet/ with columns:
  timestamp, bid_px_1..10, bid_sz_1..10, ask_px_1..10, ask_sz_1..10

Usage:
    python scripts/build/convert_l2.py
    python scripts/build/convert_l2.py --interval 5  # sample every 5 seconds
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def find_dbn_files(l2_dir: str = "data/l2") -> list[str]:
    """Find all .dbn.zst files in the L2 data directory."""
    files = []
    for root, _, filenames in os.walk(l2_dir):
        for f in sorted(filenames):
            if f.endswith(".dbn.zst"):
                files.append(os.path.join(root, f))
    return files


def convert_file(dbn_path: str, out_dir: str, interval_seconds: int = 1) -> int:
    """Convert a single DBN file to sampled Parquet.

    Returns number of snapshots written.
    """
    import databento as db
    import polars as pl

    print(f"  Reading {dbn_path}...")
    store = db.DBNStore.from_file(dbn_path)

    rows = []
    last_sample_ns = 0
    interval_ns = interval_seconds * 1_000_000_000
    count = 0

    for msg in store:
        ts_ns = msg.ts_event
        # Sample at interval
        if ts_ns - last_sample_ns < interval_ns:
            continue
        last_sample_ns = ts_ns

        row = {"timestamp": datetime.fromtimestamp(ts_ns / 1e9)}
        for i, level in enumerate(msg.levels):
            idx = i + 1
            row[f"bid_px_{idx}"] = level.bid_px / 1e9
            row[f"bid_sz_{idx}"] = level.bid_sz
            row[f"ask_px_{idx}"] = level.ask_px / 1e9
            row[f"ask_sz_{idx}"] = level.ask_sz

        rows.append(row)
        count += 1

        if count % 100_000 == 0:
            print(f"    {count} snapshots sampled...")

    if not rows:
        print(f"  No data in {dbn_path}")
        return 0

    df = pl.DataFrame(rows)

    # Extract month for partitioning
    first_ts = rows[0]["timestamp"]
    month_str = first_ts.strftime("%Y-%m")
    out_path = Path(out_dir) / f"l2_{month_str}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.write_parquet(str(out_path), compression="zstd")
    print(f"  Wrote {count} snapshots to {out_path}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert L2 DBN to Parquet")
    parser.add_argument("--l2-dir", default="data/l2", help="L2 DBN directory")
    parser.add_argument("--out-dir", default="data/l2_parquet", help="Output Parquet directory")
    parser.add_argument("--interval", type=int, default=1, help="Sampling interval in seconds")
    args = parser.parse_args()

    files = find_dbn_files(args.l2_dir)
    if not files:
        print(f"No .dbn.zst files found in {args.l2_dir}")
        return

    print(f"Found {len(files)} DBN files, sampling at {args.interval}s intervals\n")

    total = 0
    for f in files:
        total += convert_file(f, args.out_dir, args.interval)

    print(f"\nDone. Total snapshots: {total}")


if __name__ == "__main__":
    main()
