#!/usr/bin/env python3
"""Download MES L2 (MBP-10) raw data from DataBento.

Saves the raw .dbn.zst file directly — no parsing or transformation.
Parse it later with databento's DBNStore or to_df().

Usage:
    python scripts/download_l2.py --dry-run                     # cost estimate only
    python scripts/download_l2.py                                # download 3 months
    python scripts/download_l2.py --start-date 2025-12-01       # custom start
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Download MES L2 (MBP-10) raw data from DataBento"
    )
    parser.add_argument(
        "--start-date", type=str, default="2025-12-06",
        help="Start date YYYY-MM-DD (default: 2025-12-06, ~3 months ago)",
    )
    parser.add_argument(
        "--end-date", type=str, default="2026-03-06",
        help="End date YYYY-MM-DD (default: 2026-03-06)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/l2_raw",
        help="Output directory (default: data/l2_raw)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="DataBento API key (default: from DATABENTO_API_KEY env var)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show cost estimate without downloading",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        print("Error: DATABENTO_API_KEY not set.")
        print("Set it in .env or pass via --api-key")
        sys.exit(1)

    try:
        import databento as db
    except ImportError:
        print("Error: databento package required: uv add databento")
        sys.exit(1)

    dataset = "GLBX.MDP3"
    schema = "mbp-10"
    symbols = ["MES.c.0"]
    stype_in = "continuous"

    print(f"\n{'=' * 60}")
    print("MES L2 (MBP-10) Raw Data Download")
    print(f"{'=' * 60}")
    print(f"  Dataset:    {dataset}")
    print(f"  Schema:     {schema}")
    print(f"  Symbols:    {symbols}")
    print(f"  Date range: {args.start_date} -> {args.end_date}")
    print(f"  Output:     {args.output_dir}/")

    client = db.Historical(key=api_key)

    if args.dry_run:
        print(f"\n  [DRY RUN] Fetching cost estimate...")
        try:
            cost = client.metadata.get_cost(
                dataset=dataset,
                symbols=symbols,
                schema=schema,
                stype_in=stype_in,
                start=args.start_date,
                end=args.end_date,
            )
            print(f"  Estimated cost: ${cost:.2f}")
        except Exception as e:
            print(f"  Error getting estimate: {e}")
            sys.exit(1)
        print(f"\n  To download, run without --dry-run")
    else:
        from datetime import date, timedelta

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Split into monthly chunks to avoid >5 GB streaming limit
        start = date.fromisoformat(args.start_date)
        end = date.fromisoformat(args.end_date)

        chunk_start = start
        total_size = 0
        chunk_num = 0

        while chunk_start < end:
            # Next month boundary
            if chunk_start.month == 12:
                chunk_end = date(chunk_start.year + 1, 1, 1)
            else:
                chunk_end = date(chunk_start.year, chunk_start.month + 1, 1)
            chunk_end = min(chunk_end, end)

            label = f"{chunk_start.isoformat()}_{chunk_end.isoformat()}"
            out_file = output_dir / f"mes_mbp10_{label}.dbn.zst"

            if out_file.exists():
                size_mb = out_file.stat().st_size / (1024 * 1024)
                print(f"\n  [{label}] Already exists ({size_mb:.1f} MB), skipping")
                total_size += out_file.stat().st_size
                chunk_start = chunk_end
                chunk_num += 1
                continue

            print(f"\n  [{label}] Submitting batch job...")
            try:
                import time as _time

                job = client.batch.submit_job(
                    dataset=dataset,
                    symbols=symbols,
                    schema=schema,
                    stype_in=stype_in,
                    start=chunk_start.isoformat(),
                    end=chunk_end.isoformat(),
                    encoding="dbn",
                    compression="zstd",
                    split_duration="month",
                )
                job_id = job["id"]
                print(f"  [{label}] Batch job submitted: {job_id}")
                print(f"  [{label}] Waiting for server-side processing...")

                while True:
                    jobs_list = client.batch.list_jobs()
                    current = None
                    for j in jobs_list:
                        if j.get("id") == job_id:
                            current = j
                            break
                    if current is None:
                        _time.sleep(10)
                        continue

                    state = current.get("state", "unknown")
                    if state == "done":
                        break
                    elif state in ("expired", "failed"):
                        raise RuntimeError(f"Batch job {state}: {current}")
                    print(f"  [{label}] Status: {state}, waiting...   ", end="\r")
                    _time.sleep(15)

                print(f"\n  [{label}] Downloading result...")
                downloaded = client.batch.download(job_id, output_dir=str(output_dir))
                for p in downloaded:
                    size_mb = p.stat().st_size / (1024 * 1024)
                    total_size += p.stat().st_size
                    print(f"  [{label}] Saved {size_mb:.1f} MB -> {p}")

            except Exception as e:
                print(f"\n  [{label}] Error: {e}")
                print(f"  Stopping. Re-run to resume (completed chunks are skipped).")
                sys.exit(1)

            chunk_start = chunk_end
            chunk_num += 1

        total_mb = total_size / (1024 * 1024)
        print(f"\n  Done! {chunk_num} chunks, {total_mb:.1f} MB total in {output_dir}/")
        print(f"  Files are raw .dbn.zst — parse later with databento.")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
