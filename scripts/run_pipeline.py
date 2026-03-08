#!/usr/bin/env python3
"""MES Data Pipeline CLI — Convert CSV → Parquet and validate."""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.csv_to_parquet import ConvertConfig, convert_csv_to_parquet
from src.data.quality import validate_from_parquet


def main():
    parser = argparse.ArgumentParser(
        description="MES 1s Bar Data Pipeline: CSV -> Parquet -> Validate"
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert CSV to Parquet files (partitioned by year)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run data quality validation on Parquet files",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps: convert -> validate",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to the 1s bar CSV file (required for --convert)",
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default="data/parquet",
        help="Directory for Parquet files (default: data/parquet)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2023,
        help="Start year for validation (default: 2023)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="End year for validation (default: 2026)",
    )

    args = parser.parse_args()

    if not any([args.convert, args.validate, args.all]):
        parser.print_help()
        sys.exit(1)

    run_convert = args.convert or args.all
    run_validate = args.validate or args.all

    # -- Convert CSV -> Parquet -------------------------------------------
    if run_convert:
        if not args.csv:
            print("Error: --csv is required for conversion step.")
            print("Usage: python scripts/run_pipeline.py --convert --csv path/to/mes_1s.csv")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("STEP 1: CSV -> Parquet")
        print("=" * 60)
        t0 = time.time()
        config = ConvertConfig(csv_path=args.csv, parquet_dir=args.parquet_dir)
        result = convert_csv_to_parquet(config)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

    # -- Validate ---------------------------------------------------------
    if run_validate:
        print("\n" + "=" * 60)
        print("STEP 2: Data Quality Validation")
        print("=" * 60)
        t0 = time.time()
        report = validate_from_parquet(
            args.parquet_dir, args.start_year, args.end_year
        )
        report.print_summary()
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
