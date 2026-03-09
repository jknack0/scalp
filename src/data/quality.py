"""Data quality validation for MES 1-second bar data."""

from dataclasses import dataclass, field
from datetime import date, time, timedelta

import polars as pl


RTH_START = time(9, 30)
RTH_END = time(16, 0)
GAP_THRESHOLD_SECONDS = 30


@dataclass
class ValidationReport:
    total_rows: int = 0
    date_range: tuple = ()
    trading_days: int = 0
    missing_dates: list = field(default_factory=list)
    gap_count: int = 0
    gaps: list = field(default_factory=list)  # list of (date, start_ts, end_ts, gap_seconds)
    duplicate_count: int = 0
    outlier_count: int = 0
    outliers: list = field(default_factory=list)  # list of (timestamp, price, mean, std)

    def print_summary(self):
        print("\n" + "=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        print(f"  Total rows:     {self.total_rows:,}")
        if self.date_range:
            print(f"  Date range:     {self.date_range[0]} to {self.date_range[1]}")
        print(f"  Trading days:   {self.trading_days}")
        print(f"  Missing dates:  {len(self.missing_dates)}")
        print(f"  Gaps (>{GAP_THRESHOLD_SECONDS}s RTH): {self.gap_count}")
        print(f"  Duplicates:     {self.duplicate_count}")
        print(f"  Price outliers: {self.outlier_count}")

        if self.missing_dates:
            print(f"\n  Missing dates (first 10):")
            for d in self.missing_dates[:10]:
                print(f"    {d}")
            if len(self.missing_dates) > 10:
                print(f"    ... and {len(self.missing_dates) - 10} more")

        if self.gaps:
            print(f"\n  Gaps (first 10):")
            for d, start, end, secs in self.gaps[:10]:
                print(f"    {d}: {start} → {end} ({secs:.0f}s)")
            if len(self.gaps) > 10:
                print(f"    ... and {len(self.gaps) - 10} more")

        if self.outliers:
            print(f"\n  Outliers (first 10):")
            for ts, price, mean, std in self.outliers[:10]:
                deviation = abs(price - mean) / std if std > 0 else 0
                print(f"    {ts}: close={price:.2f} (mean={mean:.2f}, {deviation:.1f}σ)")

        status = "PASS" if self.is_clean() else "ISSUES FOUND"
        print(f"\n  Status: {status}")
        print("=" * 60)

    def is_clean(self) -> bool:
        return (
            self.gap_count == 0
            and self.duplicate_count == 0
            and self.outlier_count == 0
            and len(self.missing_dates) == 0
        )


def validate(df: pl.DataFrame) -> ValidationReport:
    """Run quality checks on a DataFrame of 1s bars.

    Expects columns: timestamp, open, high, low, close, volume, vwap
    """
    report = ValidationReport()
    report.total_rows = len(df)

    if len(df) == 0:
        return report

    df = df.sort("timestamp")
    report.date_range = (
        df["timestamp"].min().date(),
        df["timestamp"].max().date(),
    )

    # --- Trading days and missing dates ---
    dates = df.with_columns(pl.col("timestamp").dt.date().alias("date"))
    unique_dates = sorted(dates["date"].unique().to_list())
    report.trading_days = len(unique_dates)

    if len(unique_dates) >= 2:
        all_weekdays = set()
        current = unique_dates[0]
        end = unique_dates[-1]
        while current <= end:
            if current.weekday() < 5:  # Mon-Fri
                all_weekdays.add(current)
            current += timedelta(days=1)
        report.missing_dates = sorted(all_weekdays - set(unique_dates))

    # --- Gaps during RTH ---
    rth = df.filter(
        (pl.col("timestamp").dt.time() >= RTH_START)
        & (pl.col("timestamp").dt.time() < RTH_END)
    )
    if len(rth) > 1:
        rth = rth.with_columns(
            pl.col("timestamp").diff().dt.total_seconds().alias("gap_seconds"),
            pl.col("timestamp").dt.date().alias("date"),
        )
        gaps = rth.filter(pl.col("gap_seconds") > GAP_THRESHOLD_SECONDS)
        report.gap_count = len(gaps)
        for row in gaps.iter_rows(named=True):
            report.gaps.append((
                row["date"],
                row["timestamp"] - timedelta(seconds=row["gap_seconds"]),
                row["timestamp"],
                row["gap_seconds"],
            ))

    # --- Duplicate timestamps ---
    dup_count = len(df) - df["timestamp"].n_unique()
    report.duplicate_count = dup_count

    # --- Price outliers (close > 5 std from rolling mean) ---
    with_stats = df.with_columns([
        pl.col("close").rolling_mean(window_size=300).alias("rolling_mean"),
        pl.col("close").rolling_std(window_size=300).alias("rolling_std"),
    ])
    with_stats = with_stats.filter(pl.col("rolling_std").is_not_null())
    outliers = with_stats.filter(
        ((pl.col("close") - pl.col("rolling_mean")).abs())
        > (5.0 * pl.col("rolling_std"))
    )
    report.outlier_count = len(outliers)
    for row in outliers.head(20).iter_rows(named=True):
        report.outliers.append((
            row["timestamp"],
            row["close"],
            row["rolling_mean"],
            row["rolling_std"],
        ))

    return report


def validate_from_parquet(parquet_dir: str, start_year: int, end_year: int) -> ValidationReport:
    """Validate data from Parquet files for a year range."""
    from src.data.csv_to_parquet import read_parquet_range

    df = read_parquet_range(parquet_dir, start_year, end_year)
    return validate(df)
