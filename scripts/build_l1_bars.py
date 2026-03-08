"""Aggregate L1 TBBO tick data into 1s bars enriched with order book features.

Usage:
    python scripts/build_l1_bars.py
    python scripts/build_l1_bars.py --freq 1m --output-dir data/parquet_1m_l1

Reads from data/l1/year=YYYY/data.parquet, produces enriched bars with:
  - Standard OHLCV columns (from trade prices/sizes)
  - avg_bid_size, avg_ask_size (mean BBO sizes per bar)
  - aggressive_buy_vol, aggressive_sell_vol (classified by side)

Output: data/parquet_l1/year=YYYY/data.parquet
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl


def aggregate_l1_to_bars(
    l1_dir: str = "data/l1",
    output_dir: str = "data/parquet_l1",
    freq: str = "1s",
    rth_only: bool = True,
) -> None:
    """Aggregate L1 ticks to enriched bars."""
    l1_path = Path(l1_dir)
    out_path = Path(output_dir)

    year_dirs = sorted(l1_path.glob("year=*"))
    if not year_dirs:
        print(f"No data found in {l1_dir}")
        return

    for year_dir in year_dirs:
        year = year_dir.name.split("=")[1]
        parquet_file = year_dir / "data.parquet"
        if not parquet_file.exists():
            continue

        print(f"Processing {year}...")
        df = pl.read_parquet(str(parquet_file))
        print(f"  Loaded {len(df):,} ticks")

        # RTH filter: 9:30-16:00 ET (14:30-21:00 UTC)
        if rth_only:
            df = df.filter(
                (pl.col("timestamp").dt.hour() >= 14)
                & (pl.col("timestamp").dt.hour() < 21)
            )
            # Finer filter: exclude before 14:30 and after 21:00
            df = df.filter(
                ~(
                    (pl.col("timestamp").dt.hour() == 14)
                    & (pl.col("timestamp").dt.minute() < 30)
                )
            )
            print(f"  RTH filtered: {len(df):,} ticks")

        if df.is_empty():
            print(f"  No RTH data for {year}, skipping")
            continue

        # Truncate timestamp to bar frequency
        bars = df.group_by_dynamic(
            "timestamp", every=freq, label="left"
        ).agg([
            # OHLCV from trade prices
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
            # Order book features
            pl.col("bid_size").mean().alias("avg_bid_size"),
            pl.col("ask_size").mean().alias("avg_ask_size"),
            # Aggressive volume by side
            pl.when(pl.col("side") == "B")
            .then(pl.col("size"))
            .otherwise(0)
            .sum()
            .alias("aggressive_buy_vol"),
            pl.when(pl.col("side") == "A")
            .then(pl.col("size"))
            .otherwise(0)
            .sum()
            .alias("aggressive_sell_vol"),
            # Trade count
            pl.len().alias("tick_count"),
        ])

        # Drop empty bars (no trades)
        bars = bars.filter(pl.col("volume") > 0)

        # Cast to match existing bar schema
        bars = bars.with_columns([
            pl.col("avg_bid_size").cast(pl.Float64),
            pl.col("avg_ask_size").cast(pl.Float64),
            pl.col("aggressive_buy_vol").cast(pl.Float64),
            pl.col("aggressive_sell_vol").cast(pl.Float64),
        ])

        # Strip timezone for consistency with existing parquet files
        if bars["timestamp"].dtype == pl.Datetime("ns", "UTC"):
            bars = bars.with_columns(
                pl.col("timestamp").dt.replace_time_zone(None)
            )

        # Write output
        year_out = out_path / f"year={year}"
        year_out.mkdir(parents=True, exist_ok=True)
        out_file = year_out / "data.parquet"
        bars.write_parquet(str(out_file), compression="zstd")

        print(f"  Wrote {len(bars):,} bars to {out_file}")
        print(f"  Columns: {bars.columns}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Aggregate L1 ticks to enriched bars")
    parser.add_argument("--l1-dir", default="data/l1", help="L1 tick data directory")
    parser.add_argument("--output-dir", default="data/parquet_l1", help="Output directory")
    parser.add_argument("--freq", default="1s", help="Bar frequency (1s, 5s, 1m)")
    parser.add_argument("--no-rth", action="store_true", help="Include all hours, not just RTH")
    args = parser.parse_args()

    aggregate_l1_to_bars(
        l1_dir=args.l1_dir,
        output_dir=args.output_dir,
        freq=args.freq,
        rth_only=not args.no_rth,
    )


if __name__ == "__main__":
    main()
