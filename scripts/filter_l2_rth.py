#!/usr/bin/env python3
"""Filter L2 MBP-10 DBN files to RTH-only Parquet.

Reads raw .dbn.zst files, keeps only RTH messages (9:30-16:00 ET),
and writes to Parquet with all fields needed by L2ReplayEngine:
  timestamp_ns, action, side, price, size, + 10 levels of bid/ask px/sz

This cuts ~70% of the data (23hr→6.5hr) and makes replay much faster
since Parquet reads are ~10x faster than streaming compressed DBN.

Usage:
    python scripts/filter_l2_rth.py
    python scripts/filter_l2_rth.py --l2-dir data/l2 --out-dir data/l2_rth
"""

import argparse
import os
import sys
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ET = ZoneInfo("US/Eastern")
_UTC = ZoneInfo("UTC")
RTH_START = dt_time(9, 30)
RTH_END = dt_time(16, 0)


def find_dbn_files(l2_dir: str) -> list[str]:
    files = []
    for root, _, filenames in os.walk(l2_dir):
        for f in sorted(filenames):
            if f.endswith(".dbn.zst"):
                files.append(os.path.join(root, f))
    return files


def filter_file(dbn_path: str, out_dir: str) -> int:
    """Filter a single DBN file to RTH-only Parquet. Returns row count."""
    import databento as db
    import polars as pl

    print(f"\n  Reading {Path(dbn_path).name}...")
    store = db.DBNStore.from_file(dbn_path)

    rows = []
    total = 0
    kept = 0

    for msg in store:
        total += 1

        if total % 5_000_000 == 0:
            print(f"    {total:>12,} read | {kept:,} kept ({kept/max(total,1)*100:.0f}%)")

        # Parse timestamp
        ts_ns = msg.ts_event
        ts_utc = datetime.fromtimestamp(ts_ns / 1e9, tz=_UTC)
        ts_et = ts_utc.astimezone(_ET)
        t = ts_et.time()

        if t < RTH_START or t >= RTH_END:
            continue

        kept += 1

        # Extract action/side as chars
        action = msg.action
        action_val = getattr(action, "value", action)
        side = msg.side
        side_val = getattr(side, "value", side)

        row = {
            "ts_event": ts_ns,
            "action": str(action_val),
            "side": str(side_val),
            "price": msg.price,  # keep as fixed-point int (divide by 1e9 at read)
            "size": msg.size,
        }

        # 10 levels
        for i, level in enumerate(msg.levels):
            idx = i + 1
            row[f"bid_px_{idx}"] = level.bid_px
            row[f"bid_sz_{idx}"] = level.bid_sz
            row[f"ask_px_{idx}"] = level.ask_px
            row[f"ask_sz_{idx}"] = level.ask_sz

        rows.append(row)

        # Flush in chunks to manage memory (~5M rows per chunk)
        if len(rows) >= 5_000_000:
            _flush_rows(rows, out_dir, dbn_path, kept)
            rows = []

    # Final flush
    if rows:
        _flush_rows(rows, out_dir, dbn_path, kept)

    print(f"    Done: {total:,} total → {kept:,} RTH ({kept/max(total,1)*100:.1f}%)")
    return kept


def _flush_rows(rows: list[dict], out_dir: str, dbn_path: str, count: int) -> None:
    """Write accumulated rows to Parquet."""
    import polars as pl

    # Name based on source file + chunk
    stem = Path(dbn_path).name.replace(".dbn.zst", "")
    out_path = Path(out_dir) / f"{stem}_rth.parquet"

    df = pl.DataFrame(rows)

    if out_path.exists():
        # Append by reading existing + concat
        existing = pl.read_parquet(str(out_path))
        df = pl.concat([existing, df])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(out_path), compression="zstd")
    print(f"    Flushed {count:,} rows to {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Filter L2 DBN to RTH-only Parquet")
    parser.add_argument("--l2-dir", default="data/l2", help="Input DBN directory")
    parser.add_argument("--out-dir", default="data/l2_rth", help="Output Parquet directory")
    args = parser.parse_args()

    files = find_dbn_files(args.l2_dir)
    if not files:
        print(f"No .dbn.zst files found in {args.l2_dir}")
        return

    print(f"Found {len(files)} DBN files")
    for f in files:
        size_gb = os.path.getsize(f) / (1024 ** 3)
        print(f"  {Path(f).name} ({size_gb:.1f} GB)")

    t0 = time.perf_counter()
    total_kept = 0
    for f in files:
        total_kept += filter_file(f, args.out_dir)

    elapsed = time.perf_counter() - t0
    print(f"\nDone. {total_kept:,} RTH messages in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Show output sizes
    out_path = Path(args.out_dir)
    if out_path.exists():
        print(f"\nOutput files:")
        for f in sorted(out_path.glob("*.parquet")):
            size_mb = f.stat().st_size / (1024 ** 2)
            print(f"  {f.name} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
