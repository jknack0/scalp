"""Join regular OHLCV bars with L1 tick features to create enriched bars.

Takes the ground-truth OHLCV+VWAP from data/parquet_1s_rth and merges on
L1 order book features (avg bid/ask price, aggressive buy/sell vol) from
data/l1_rth, aggregated per-second.

Output: data/parquet_1s_enriched/year=YYYY/data.parquet

Usage:
    python -m scripts.build_enriched_bars
"""

import os
import time
from datetime import time as dt_time

import polars as pl

BARS_DIR = "data/parquet_1s_rth"
L1_DIR = "data/l1_rth"
OUT_DIR = "data/parquet_1s_enriched"


def aggregate_l1_features(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate L1 ticks to 1s features for joining."""
    # Strip timezone if present (to match bar timestamps)
    ts_dtype = df["timestamp"].dtype
    if hasattr(ts_dtype, "time_zone") and ts_dtype.time_zone is not None:
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    # Truncate to 1s for grouping
    return df.group_by_dynamic(
        "timestamp", every="1s", label="left"
    ).agg([
        # Bid/ask prices (what spread monitor needs)
        pl.col("bid_price").mean().alias("avg_bid_price"),
        pl.col("ask_price").mean().alias("avg_ask_price"),
        # Bid/ask sizes
        pl.col("bid_size").mean().cast(pl.Float64).alias("avg_bid_size"),
        pl.col("ask_size").mean().cast(pl.Float64).alias("avg_ask_size"),
        # Aggressive volume by side
        pl.when(pl.col("side") == "B")
        .then(pl.col("size"))
        .otherwise(0)
        .sum()
        .cast(pl.Float64)
        .alias("aggressive_buy_vol"),
        pl.when(pl.col("side") == "A")
        .then(pl.col("size"))
        .otherwise(0)
        .sum()
        .cast(pl.Float64)
        .alias("aggressive_sell_vol"),
        # Trade count
        pl.len().alias("tick_count"),
    ])


def main():
    bar_years = sorted(
        int(d.replace("year=", ""))
        for d in os.listdir(BARS_DIR)
        if d.startswith("year=")
    )
    l1_years = set(
        int(d.replace("year=", ""))
        for d in os.listdir(L1_DIR)
        if d.startswith("year=")
    )
    # Only process years that exist in both
    years = [y for y in bar_years if y in l1_years]

    if not years:
        print(f"No overlapping years between {BARS_DIR} and {L1_DIR}")
        return

    print(f"Joining OHLCV bars with L1 features")
    print(f"Bar source: {BARS_DIR}")
    print(f"L1 source:  {L1_DIR}")
    print(f"Overlapping years: {years}")
    print(f"Output: {OUT_DIR}/\n")

    total_bars = 0
    total_matched = 0
    t0 = time.perf_counter()

    for year in years:
        bars_path = os.path.join(BARS_DIR, f"year={year}", "data.parquet")
        l1_path = os.path.join(L1_DIR, f"year={year}", "data.parquet")

        if not os.path.exists(bars_path) or not os.path.exists(l1_path):
            print(f"  SKIP year={year} (missing file)")
            continue

        # Load bars
        bars = pl.read_parquet(bars_path)
        total_bars += len(bars)

        # Load and aggregate L1
        l1 = pl.read_parquet(l1_path)
        l1_agg = aggregate_l1_features(l1)

        # Ensure matching timestamp types
        if bars["timestamp"].dtype != l1_agg["timestamp"].dtype:
            # Cast both to us precision, no timezone
            bars = bars.with_columns(
                pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None)
            )
            l1_agg = l1_agg.with_columns(
                pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None)
            )

        # Left join: keep all bars, add L1 features where available
        joined = bars.join(l1_agg, on="timestamp", how="left")

        # Fill missing L1 values with 0
        for col in ["avg_bid_price", "avg_ask_price", "avg_bid_size", "avg_ask_size",
                     "aggressive_buy_vol", "aggressive_sell_vol", "tick_count"]:
            if col in joined.columns:
                joined = joined.with_columns(pl.col(col).fill_null(0.0))

        matched = joined.filter(pl.col("avg_bid_price") > 0).height
        total_matched += matched

        # Write
        out_path = os.path.join(OUT_DIR, f"year={year}", "data.parquet")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        joined.write_parquet(out_path, compression="zstd")

        pct = matched / len(bars) * 100 if len(bars) > 0 else 0
        print(f"  year={year}: {len(bars):>8,} bars + {len(l1_agg):>8,} L1 agg -> "
              f"{len(joined):>8,} enriched ({pct:.0f}% L1 match)")

    elapsed = time.perf_counter() - t0
    pct_total = total_matched / total_bars * 100 if total_bars > 0 else 0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total: {total_bars:,} bars, {total_matched:,} with L1 data ({pct_total:.0f}%)")
    print(f"  Columns: timestamp, open, high, low, close, volume, vwap, "
          f"avg_bid_price, avg_ask_price, avg_bid_size, avg_ask_size, "
          f"aggressive_buy_vol, aggressive_sell_vol, tick_count")
    print(f"  Output: {OUT_DIR}/")


if __name__ == "__main__":
    main()
