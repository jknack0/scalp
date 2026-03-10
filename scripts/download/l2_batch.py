#!/usr/bin/env python3
"""Submit a batch request for MES L2 (MBP-10) data from DataBento.

This does NOT download data — it submits a batch job to DataBento's servers.
Once processing completes, download the files from https://databento.com/portal/jobs

Usage:
    python scripts/request_l2_batch.py --dry-run          # cost estimate only
    python scripts/request_l2_batch.py                     # submit batch job
    python scripts/request_l2_batch.py --status            # check job status
    python scripts/request_l2_batch.py --start 2025-09-01 --end 2025-12-01
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()


DATASET = "GLBX.MDP3"
SCHEMA = "mbp-10"
SYMBOLS = ["MES.c.0"]
STYPE_IN = "continuous"

# Sept-Nov: best mix of trending + mean reversion regimes (per roadmap)
DEFAULT_START = "2025-09-01"
DEFAULT_END = "2025-12-01"


def _get_client():
    try:
        import databento as db
    except ImportError:
        print("Error: databento package required. Run: uv add databento")
        sys.exit(1)

    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        print("Error: DATABENTO_API_KEY not set in .env or environment")
        sys.exit(1)

    return db.Historical(key=api_key)


def estimate_cost(start: str, end: str) -> None:
    client = _get_client()
    print(f"  Estimating cost for MBP-10: {start} -> {end} ...")
    cost = client.metadata.get_cost(
        dataset=DATASET,
        symbols=SYMBOLS,
        schema=SCHEMA,
        stype_in=STYPE_IN,
        start=start,
        end=end,
    )
    print(f"  Estimated cost: ${cost:.2f}")


def submit_batch(start: str, end: str) -> None:
    client = _get_client()
    print(f"  Submitting batch job...")
    result = client.batch.submit_job(
        dataset=DATASET,
        symbols=SYMBOLS,
        schema=SCHEMA,
        stype_in=STYPE_IN,
        start=start,
        end=end,
        encoding="dbn",
        compression="zstd",
        split_duration="month",
        delivery="download",
    )
    print(f"\n  Job submitted successfully!")
    print(f"  Job ID: {result.get('id', result.get('job_id', 'unknown'))}")
    print(f"  State:  {result.get('state', 'unknown')}")
    print(f"\n  Monitor progress at: https://databento.com/portal/jobs")
    print(f"  Or run: python scripts/request_l2_batch.py --status")


def check_status() -> None:
    client = _get_client()
    print(f"  Checking batch job status...\n")
    jobs = client.batch.list_jobs()
    if not jobs:
        print("  No jobs found.")
        return

    for job in jobs:
        job_id = job.get("id", job.get("job_id", "?"))
        state = job.get("state", "?")
        dataset = job.get("dataset", "?")
        schema = job.get("schema", "?")
        start = job.get("start", "?")
        end = job.get("end", "?")
        ts = job.get("ts_received", "")
        cost = job.get("actual_cost", job.get("cost", "?"))

        print(f"  Job {job_id}")
        print(f"    State:   {state}")
        print(f"    Dataset: {dataset} / {schema}")
        print(f"    Range:   {start} -> {end}")
        if ts:
            print(f"    Submitted: {ts}")
        if cost and cost != "?":
            print(f"    Cost:    ${float(cost):.2f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Submit batch request for MES L2 (MBP-10) data from DataBento"
    )
    parser.add_argument(
        "--start", type=str, default=DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end", type=str, default=DEFAULT_END,
        help=f"End date YYYY-MM-DD (default: {DEFAULT_END})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show cost estimate without submitting",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Check status of existing batch jobs",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("MES L2 (MBP-10) Batch Request")
    print(f"{'=' * 60}")

    if args.status:
        check_status()
    else:
        print(f"  Dataset:    {DATASET}")
        print(f"  Schema:     {SCHEMA}")
        print(f"  Symbols:    {SYMBOLS}")
        print(f"  Date range: {args.start} -> {args.end}")
        print(f"  Delivery:   download (from databento.com/portal/jobs)")
        print(f"  Split:      monthly files")
        print()

        if args.dry_run:
            estimate_cost(args.start, args.end)
            print(f"\n  To submit, run without --dry-run")
        else:
            estimate_cost(args.start, args.end)
            print()
            confirm = input("  Submit batch job? [y/N] ")
            if confirm.lower() == "y":
                submit_batch(args.start, args.end)
            else:
                print("  Cancelled.")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
