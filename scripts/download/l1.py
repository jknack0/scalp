#!/usr/bin/env python3
"""Download MES L1 (TBBO) data from DataBento.

Usage:
    python scripts/download_l1.py --dry-run                    # cost estimate only
    python scripts/download_l1.py                              # download default range
    python scripts/download_l1.py --start-date 2025-06-01      # custom range
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from src.data.databento_downloader import DatabentoConfig, DatabentoDownloader


def main():
    parser = argparse.ArgumentParser(
        description="Download MES L1 (TBBO) trades+BBO data from DataBento"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-01-01",
        help="Start date YYYY-MM-DD (default: 2025-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-03-03",
        help="End date YYYY-MM-DD (default: 2026-03-03)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/l1",
        help="Output directory (default: data/l1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="DataBento API key (default: from DATABENTO_API_KEY env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show cost estimate without downloading",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        print("Error: DATABENTO_API_KEY not set.")
        print("Set it in .env or pass via --api-key")
        sys.exit(1)

    config = DatabentoConfig(
        api_key=api_key,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
    )

    downloader = DatabentoDownloader(config)

    print(f"\n{'=' * 60}")
    print("MES L1 (TBBO) Data Download")
    print(f"{'=' * 60}")
    print(f"  Dataset:    {config.dataset}")
    print(f"  Schema:     {config.schema}")
    print(f"  Symbols:    {config.symbols}")
    print(f"  Date range: {config.start_date} -> {config.end_date}")
    print(f"  Output:     {config.output_dir}/")

    if args.dry_run:
        print(f"\n  [DRY RUN] Fetching cost estimate...")
        try:
            estimate = downloader.estimate_cost()
            print(f"  Estimated cost: ${estimate['cost_usd']:.2f}")
        except ImportError as e:
            print(f"  Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"  Error getting estimate: {e}")
            sys.exit(1)
        print(f"\n  To download, run without --dry-run")
    else:
        print(f"\n  Downloading...")
        try:
            output_path = downloader.download()
            print(f"\n  Done! Data saved to {output_path}/")
        except ImportError as e:
            print(f"  Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"  Error: {e}")
            sys.exit(1)

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
