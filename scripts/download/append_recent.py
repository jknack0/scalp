#!/usr/bin/env python3
"""Fetch recent MES data from Databento and append to existing Parquet files.

Downloads both L1 (TBBO) and OHLCV-1s data, then appends to the
year-partitioned Parquet stores in data/l1/ and data/parquet/.

Usage:
    python -u scripts/download/append_recent.py --dry-run
    python -u scripts/download/append_recent.py
    python -u scripts/download/append_recent.py --start 2026-03-01 --end 2026-03-14
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import polars as pl


def _get_last_timestamp(parquet_dir: str) -> str | None:
    """Find the latest timestamp across all year partitions."""
    base = Path(parquet_dir)
    if not base.exists():
        return None
    parts = sorted(base.glob("year=*/data.parquet"))
    if not parts:
        return None
    df = pl.read_parquet(str(parts[-1]))
    if "timestamp" not in df.columns:
        return None
    last = df["timestamp"].max()
    return str(last)


def _download_l1(client, start: str, end: str, output_dir: str) -> int:
    """Download L1 TBBO and append to existing partitions."""
    print(f"\n  Downloading L1 (TBBO): {start} -> {end}")

    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["MES.c.0"],
        schema="tbbo",
        stype_in="continuous",
        start=start,
        end=end,
    )

    df = data.to_df()
    if len(df) == 0:
        print("  No L1 data returned")
        return 0

    print(f"  Downloaded {len(df):,} L1 trades")
    df_pl = pl.from_pandas(df.reset_index())

    # Normalize columns to match existing schema
    rename_map = {}
    if "ts_event" in df_pl.columns:
        rename_map["ts_event"] = "timestamp"
    for col_from, col_to in [
        ("bid_px_00", "bid_price"),
        ("ask_px_00", "ask_price"),
        ("bid_sz_00", "bid_size"),
        ("ask_sz_00", "ask_size"),
    ]:
        if col_from in df_pl.columns:
            rename_map[col_from] = col_to
    if rename_map:
        df_pl = df_pl.rename(rename_map)

    target_cols = ["timestamp", "price", "size", "side",
                   "bid_price", "ask_price", "bid_size", "ask_size"]
    available = [c for c in target_cols if c in df_pl.columns]
    df_pl = df_pl.select(available)

    _append_partitioned(df_pl, Path(output_dir))
    return len(df_pl)


def _download_ohlcv(client, start: str, end: str, output_dir: str) -> int:
    """Download OHLCV-1s and append to existing partitions."""
    print(f"\n  Downloading OHLCV-1s: {start} -> {end}")

    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["MES.c.0"],
        schema="ohlcv-1s",
        stype_in="continuous",
        start=start,
        end=end,
    )

    df = data.to_df()
    if len(df) == 0:
        print("  No OHLCV data returned")
        return 0

    print(f"  Downloaded {len(df):,} OHLCV bars")
    df_pl = pl.from_pandas(df.reset_index())

    # Normalize to match existing schema: timestamp, open, high, low, close, volume, vwap
    rename_map = {}
    if "ts_event" in df_pl.columns:
        rename_map["ts_event"] = "timestamp"

    if rename_map:
        df_pl = df_pl.rename(rename_map)

    # Compute VWAP if not present (Databento ohlcv-1s doesn't include it)
    # Use (open + high + low + close) / 4 as proxy, or just midpoint
    if "vwap" not in df_pl.columns:
        # Typical VWAP = sum(price*volume) / sum(volume), but for 1s bars
        # the close is a reasonable approximation. For consistency with existing
        # data which has real VWAP, use the close as placeholder.
        df_pl = df_pl.with_columns(
            ((pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4.0)
            .alias("vwap")
        )

    # Strip timezone from timestamp to match existing naive timestamps
    if df_pl["timestamp"].dtype == pl.Datetime("ns", "UTC") or str(df_pl["timestamp"].dtype).endswith("UTC"):
        df_pl = df_pl.with_columns(
            pl.col("timestamp").dt.replace_time_zone(None).alias("timestamp")
        )
    # Cast to microsecond precision to match existing schema
    if df_pl["timestamp"].dtype != pl.Datetime("us"):
        df_pl = df_pl.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")).alias("timestamp")
        )

    target_cols = ["timestamp", "open", "high", "low", "close", "volume", "vwap"]
    available = [c for c in target_cols if c in df_pl.columns]
    df_pl = df_pl.select(available)

    # Cast types to match existing schema
    df_pl = df_pl.with_columns([
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
        pl.col("vwap").cast(pl.Float64),
    ])

    _append_partitioned(df_pl, Path(output_dir))
    return len(df_pl)


def _append_partitioned(df: pl.DataFrame, output_dir: Path) -> None:
    """Append DataFrame to year-partitioned Parquet files, deduplicating."""
    df = df.with_columns(pl.col("timestamp").dt.year().alias("_year"))
    years = sorted(df["_year"].unique().to_list())

    for year in years:
        year_dir = output_dir / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        new_data = df.filter(pl.col("_year") == year).drop("_year")

        out_path = year_dir / "data.parquet"
        if out_path.exists():
            existing = pl.read_parquet(str(out_path))
            # Deduplicate by timestamp
            combined = pl.concat([existing, new_data])
            combined = combined.unique(subset=["timestamp"], keep="last")
            combined = combined.sort("timestamp")
            print(f"  {year}: {len(existing):,} existing + {len(new_data):,} new "
                  f"= {len(combined):,} total (after dedup)")
        else:
            combined = new_data.sort("timestamp")
            print(f"  {year}: {len(combined):,} rows (new partition)")

        combined.write_parquet(str(out_path), compression="zstd")


def main():
    parser = argparse.ArgumentParser(description="Append recent MES data from Databento")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: auto-detect from last timestamp)")
    parser.add_argument("--end", type=str, default="2026-03-14",
                        help="End date YYYY-MM-DD (default: 2026-03-14)")
    parser.add_argument("--l1-dir", type=str, default="data/l1")
    parser.add_argument("--ohlcv-dir", type=str, default="data/parquet")
    parser.add_argument("--dry-run", action="store_true", help="Cost estimate only")
    args = parser.parse_args()

    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        print("Error: DATABENTO_API_KEY not set.")
        sys.exit(1)

    try:
        import databento as db
    except ImportError:
        print("Error: databento package required: uv add databento")
        sys.exit(1)

    # Auto-detect start dates from existing data
    l1_last = _get_last_timestamp(args.l1_dir)
    ohlcv_last = _get_last_timestamp(args.ohlcv_dir)

    # Use day before last timestamp to ensure overlap for dedup
    l1_start = args.start or (l1_last[:10] if l1_last else "2026-03-01")
    ohlcv_start = args.start or (ohlcv_last[:10] if ohlcv_last else "2026-02-28")

    print(f"\n{'=' * 60}")
    print("  MES Data Append (Databento)")
    print(f"{'=' * 60}")
    print(f"  L1 (TBBO):   {l1_start} -> {args.end}  (last: {l1_last})")
    print(f"  OHLCV-1s:    {ohlcv_start} -> {args.end}  (last: {ohlcv_last})")

    client = db.Historical(key=api_key)

    if args.dry_run:
        cost_l1 = client.metadata.get_cost(
            dataset="GLBX.MDP3", symbols=["MES.c.0"], schema="tbbo",
            stype_in="continuous", start=l1_start, end=args.end,
        )
        cost_ohlcv = client.metadata.get_cost(
            dataset="GLBX.MDP3", symbols=["MES.c.0"], schema="ohlcv-1s",
            stype_in="continuous", start=ohlcv_start, end=args.end,
        )
        print(f"\n  L1 cost:    ${cost_l1:.2f}")
        print(f"  OHLCV cost: ${cost_ohlcv:.2f}")
        print(f"  Total:      ${cost_l1 + cost_ohlcv:.2f}")
        print(f"\n  Run without --dry-run to download.")
    else:
        n_l1 = _download_l1(client, l1_start, args.end, args.l1_dir)
        n_ohlcv = _download_ohlcv(client, ohlcv_start, args.end, args.ohlcv_dir)

        print(f"\n{'=' * 60}")
        print(f"  Done! Appended {n_l1:,} L1 trades + {n_ohlcv:,} OHLCV bars")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
