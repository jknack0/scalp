"""Convert MES 1-second bar CSV to partitioned Parquet files."""

import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from tqdm import tqdm


@dataclass
class ConvertConfig:
    csv_path: str
    parquet_dir: str = "data/parquet"
    chunk_size: int = 5_000_000  # rows per read chunk


def convert_csv_to_parquet(config: ConvertConfig) -> dict:
    """Convert a large CSV of 1s bars to Parquet files partitioned by year.

    CSV expected columns: timestamp, open, high, low, close, volume, vwap

    Returns dict with stats: {total_rows, years, files_created}.
    """
    csv_path = Path(config.csv_path)
    parquet_dir = Path(config.parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Reading {csv_path} ...")

    # Scan CSV lazily to get row count for progress bar
    schema = {
        "timestamp": pl.Float64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
        "vwap": pl.Float64,
    }

    # Read in chunks to handle 5+ GB files
    reader = pl.read_csv_batched(
        csv_path,
        schema_overrides=schema,
        batch_size=config.chunk_size,
        try_parse_dates=False,
    )

    total_rows = 0
    year_counts: dict[int, int] = {}
    existing_years = {
        int(d.name.split("=")[1])
        for d in parquet_dir.iterdir()
        if d.is_dir() and d.name.startswith("year=")
    }

    if existing_years:
        print(f"  Skipping already-converted years: {sorted(existing_years)}")

    # Accumulate data by year
    year_buffers: dict[int, list[pl.DataFrame]] = {}

    print("Processing chunks ...")
    batch_num = 0
    while True:
        batch = reader.next_batches(1)
        if not batch:
            break
        df = batch[0]
        batch_num += 1

        # Parse timestamps (Unix epoch seconds → datetime)
        df = df.with_columns(
            pl.from_epoch(pl.col("timestamp").cast(pl.Int64), time_unit="s")
            .alias("timestamp")
        )

        # Extract year
        df = df.with_columns(pl.col("timestamp").dt.year().alias("year"))

        # Split by year and accumulate
        for year in df["year"].unique().to_list():
            if year in existing_years:
                continue
            year_df = df.filter(pl.col("year") == year).drop("year")
            if year not in year_buffers:
                year_buffers[year] = []
            year_buffers[year].append(year_df)

        rows_in_batch = len(df)
        total_rows += rows_in_batch
        print(f"  Chunk {batch_num}: {rows_in_batch:,} rows (total: {total_rows:,})")

    # Write each year to Parquet
    files_created = 0
    for year in tqdm(sorted(year_buffers.keys()), desc="Writing Parquet"):
        year_dir = parquet_dir / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        out_path = year_dir / "data.parquet"

        combined = pl.concat(year_buffers[year])
        combined = combined.sort("timestamp")
        combined.write_parquet(out_path, compression="zstd")

        row_count = len(combined)
        year_counts[year] = row_count
        files_created += 1

    # Summary
    print(f"\nConversion complete:")
    print(f"  Total rows processed: {total_rows:,}")
    print(f"  Files created: {files_created}")
    for year in sorted(year_counts):
        size_mb = (parquet_dir / f"year={year}" / "data.parquet").stat().st_size / 1e6
        print(f"    {year}: {year_counts[year]:>12,} rows ({size_mb:.1f} MB)")

    return {
        "total_rows": total_rows,
        "years": year_counts,
        "files_created": files_created,
    }


def read_parquet_range(
    parquet_dir: str, start_year: int, end_year: int
) -> pl.DataFrame:
    """Read Parquet files for a range of years into a single DataFrame."""
    frames = []
    for year in range(start_year, end_year + 1):
        path = Path(parquet_dir) / f"year={year}" / "data.parquet"
        if path.exists():
            frames.append(pl.read_parquet(path))
    if not frames:
        raise FileNotFoundError(
            f"No Parquet files found for years {start_year}-{end_year}"
        )
    return pl.concat(frames).sort("timestamp")
