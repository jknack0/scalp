"""DataBento L1 data downloader for MES futures.

Downloads TBBO (trades with best bid/offer) data from CME Globex via DataBento.
Outputs year-partitioned Parquet files for use by CVD and other feature calculators.

Schema: tbbo — each row is a trade with the BBO at time of execution.
Dataset: GLBX.MDP3 (CME Globex)
Symbol: MES.c.0 (continuous front month, calendar roll)

Requires: pip install databento (or uv add databento)
"""

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import polars as pl


@dataclass
class DatabentoConfig:
    """Configuration for DataBento L1 download."""

    api_key: str = field(
        default_factory=lambda: os.environ.get("DATABENTO_API_KEY", "")
    )
    dataset: str = "GLBX.MDP3"
    schema: str = "tbbo"
    symbols: list[str] = field(default_factory=lambda: ["MES.c.0"])
    stype_in: str = "continuous"
    start_date: str = "2025-01-01"
    end_date: str = "2026-03-03"
    output_dir: str = "data/l1"


class DatabentoDownloader:
    """Downloads L1 (TBBO) tick data from DataBento.

    Each row contains: timestamp, price, size, side classification,
    and BBO (bid_price, ask_price, bid_size, ask_size) at time of trade.
    """

    def __init__(self, config: DatabentoConfig) -> None:
        self.config = config

    def estimate_cost(self) -> dict:
        """Get cost estimate from DataBento without downloading.

        Returns:
            Dict with 'size_bytes' and 'cost_usd' estimates.
        """
        try:
            import databento as db
        except ImportError:
            raise ImportError(
                "databento package required: uv add databento"
            )

        if not self.config.api_key:
            raise ValueError("DATABENTO_API_KEY not set in environment")

        client = db.Historical(key=self.config.api_key)
        cost = client.metadata.get_cost(
            dataset=self.config.dataset,
            symbols=self.config.symbols,
            schema=self.config.schema,
            stype_in=self.config.stype_in,
            start=self.config.start_date,
            end=self.config.end_date,
        )
        return {"cost_usd": cost}

    def download(self) -> Path:
        """Download L1 TBBO data and save as year-partitioned Parquet files.

        Returns:
            Path to the output directory containing partitioned Parquet files.
        """
        try:
            import databento as db
        except ImportError:
            raise ImportError(
                "databento package required: uv add databento"
            )

        if not self.config.api_key:
            raise ValueError("DATABENTO_API_KEY not set in environment")

        client = db.Historical(key=self.config.api_key)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Requesting TBBO data: {self.config.start_date} -> {self.config.end_date}")
        print(f"Dataset: {self.config.dataset}, Symbols: {self.config.symbols}")

        # Download via streaming request
        data = client.timeseries.get_range(
            dataset=self.config.dataset,
            symbols=self.config.symbols,
            schema=self.config.schema,
            stype_in=self.config.stype_in,
            start=self.config.start_date,
            end=self.config.end_date,
        )

        # Convert to DataFrame
        df = data.to_df()
        print(f"Downloaded {len(df):,} trades")

        # Convert to Polars for consistent processing
        df_pl = pl.from_pandas(df.reset_index())

        # Normalize columns to our standard schema
        df_pl = self._normalize_columns(df_pl)

        # Partition by year and save
        self._save_partitioned(df_pl, output_dir)

        return output_dir

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize DataBento TBBO columns to our standard schema.

        Output columns: timestamp, price, size, side, bid_price, ask_price,
        bid_size, ask_size.
        """
        # DataBento TBBO schema has: ts_event, price, size, side,
        # bid_px_00, ask_px_00, bid_sz_00, ask_sz_00 (among others)
        rename_map = {}

        # Find the timestamp column
        if "ts_event" in df.columns:
            rename_map["ts_event"] = "timestamp"

        # Price columns (DataBento uses fixed-point, may need scaling)
        if "price" in df.columns:
            rename_map["price"] = "price"

        # BBO columns
        for col_from, col_to in [
            ("bid_px_00", "bid_price"),
            ("ask_px_00", "ask_price"),
            ("bid_sz_00", "bid_size"),
            ("ask_sz_00", "ask_size"),
        ]:
            if col_from in df.columns:
                rename_map[col_from] = col_to

        if rename_map:
            df = df.rename(rename_map)

        # Select only the columns we need (keep whatever exists)
        target_cols = [
            "timestamp", "price", "size", "side",
            "bid_price", "ask_price", "bid_size", "ask_size",
        ]
        available = [c for c in target_cols if c in df.columns]
        return df.select(available)

    def _save_partitioned(self, df: pl.DataFrame, output_dir: Path) -> None:
        """Save DataFrame as year-partitioned Parquet files."""
        if "timestamp" not in df.columns:
            print("Warning: no timestamp column, saving as single file")
            df.write_parquet(output_dir / "data.parquet")
            return

        # Extract year
        df = df.with_columns(
            pl.col("timestamp").dt.year().alias("year")
        )

        years = sorted(df["year"].unique().to_list())
        for year in years:
            year_dir = output_dir / f"year={year}"
            year_dir.mkdir(parents=True, exist_ok=True)
            year_df = df.filter(pl.col("year") == year).drop("year")

            out_path = year_dir / "data.parquet"
            year_df.write_parquet(out_path, compression="zstd")
            print(f"  Saved {len(year_df):,} trades -> {out_path}")
