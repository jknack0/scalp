"""Auto-retrain Regime Detector V2 from Postgres bar data.

Reads rolling window of 1m bars from bars_1m table, resamples to 5m,
trains regime_v2 with walk-forward validation, and swaps the model
if OOS metrics pass quality gates.

Usage (cron, weekly Saturday 6 AM UTC):
    0 6 * * 6  cd /opt/mes-bot && /home/botuser/.local/bin/uv run python scripts/train/auto_retrain.py >> logs/retrain.log 2>&1

Manual:
    python scripts/train/auto_retrain.py
    python scripts/train/auto_retrain.py --train-months 12 --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import psycopg2

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.bars import resample_bars
from src.models.regime_detector_v2 import (
    RegimeDetectorV2,
    RegimeDetectorV2Config,
    RegimeLabel,
    build_features_v2,
    compute_regime_stats,
)

MODEL_DIR = "models/regime_v2"
BACKUP_DIR = "models/regime_v2_backup"


# ── Quality gates ──────────────────────────────────────────────────────────

# OOS metrics must pass ALL of these to swap the model
QUALITY_GATES = {
    "min_oos_bars": 500,           # Enough data to be meaningful
    "min_avg_confidence": 0.55,    # Model isn't guessing
    "max_avg_confidence": 0.95,    # Not overfit
    "max_halt_fraction": 0.40,     # Not halting too much
    "min_avg_stint": 3.0,          # Not flipping every bar
    "min_ranging_frac": 0.10,      # RANGING state exists in OOS
    "max_ranging_frac": 0.70,      # Not classifying everything as RANGING
}


def load_bars_from_postgres(
    dsn: str,
    train_months: int,
    symbol: str = "MESH6",
) -> pl.DataFrame:
    """Load 1m bars from Postgres for the last N months."""
    conn = psycopg2.connect(dsn)
    cutoff = datetime.now(timezone.utc) - timedelta(days=train_months * 30)

    query = """
        SELECT timestamp, open, high, low, close, volume, vwap
        FROM bars_1m
        WHERE symbol = %s AND timestamp >= %s
        ORDER BY timestamp
    """

    with conn.cursor() as cur:
        cur.execute(query, (symbol, cutoff))
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"ERROR: No bars found for {symbol} after {cutoff.date()}")
        sys.exit(1)

    df = pl.DataFrame(
        {
            "timestamp": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [r[5] for r in rows],
            "vwap": [r[6] for r in rows],
        },
    )

    # Strip timezone for consistency with rest of codebase
    df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    print(f"Loaded {len(df):,} 1m bars from Postgres ({symbol})")
    print(f"  Range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def walk_forward_validate(
    df_5m: pl.DataFrame,
    config: RegimeDetectorV2Config,
    wf_train_months: int = 6,
    wf_test_months: int = 1,
) -> tuple[list[dict], RegimeDetectorV2 | None]:
    """Walk-forward validation. Returns (oos_stats_list, last_trained_detector)."""
    features, timestamps = build_features_v2(df_5m, config)
    print(f"Feature matrix: {features.shape}")

    ts_col = df_5m["timestamp"]
    min_date = ts_col.min()
    max_date = ts_col.max()
    print(f"Date range: {min_date} → {max_date}")

    train_delta = timedelta(days=wf_train_months * 30)
    test_delta = timedelta(days=wf_test_months * 30)

    splits = []
    current = min_date
    while current + train_delta + test_delta <= max_date:
        train_end = current + train_delta
        test_end = train_end + test_delta
        splits.append((current, train_end, test_end))
        current = train_end

    print(f"{len(splits)} walk-forward splits")

    all_stats = []
    last_detector = None

    for i, (train_start, train_end, test_end) in enumerate(splits):
        print(f"\n--- Split {i+1}/{len(splits)} ---")
        print(f"  Train: {train_start.date()} → {train_end.date()}")
        print(f"  Test:  {train_end.date()} → {test_end.date()}")

        train_df = df_5m.filter(
            (pl.col("timestamp") >= train_start) & (pl.col("timestamp") < train_end)
        )
        test_df = df_5m.filter(
            (pl.col("timestamp") >= train_end) & (pl.col("timestamp") < test_end)
        )

        if len(train_df) < 1000 or len(test_df) < 100:
            print(f"  Skipping: train={len(train_df)}, test={len(test_df)}")
            continue

        train_feat, _ = build_features_v2(train_df, config)
        test_feat, _ = build_features_v2(test_df, config)

        if len(train_feat) < 500 or len(test_feat) < 50:
            print(f"  Skipping: features too small")
            continue

        detector = RegimeDetectorV2(config)
        detector.fit(train_feat)

        oos_probas = detector.predict_proba_sequence(test_feat)
        stats = compute_regime_stats(oos_probas)

        print(f"  OOS bars: {stats['n_bars']}")
        print(f"  OOS dist: {stats['state_distribution']}")
        print(f"  OOS conf: {stats['avg_confidence']:.3f}")
        print(f"  OOS halt: {stats['halt_fraction']:.3f}")
        print(f"  OOS stint: {stats['avg_stint_length']:.1f}")

        all_stats.append(stats)
        last_detector = detector

    return all_stats, last_detector


def check_quality_gates(stats_list: list[dict]) -> tuple[bool, list[str]]:
    """Check if OOS stats pass quality gates. Returns (passed, reasons)."""
    if not stats_list:
        return False, ["no_oos_splits"]

    failures = []

    # Aggregate
    total_bars = sum(s["n_bars"] for s in stats_list)
    avg_conf = np.mean([s["avg_confidence"] for s in stats_list])
    avg_halt = np.mean([s["halt_fraction"] for s in stats_list])
    avg_stint = np.mean([s["avg_stint_length"] for s in stats_list])

    # Aggregate RANGING fraction
    ranging_fracs = []
    for s in stats_list:
        dist = s["state_distribution"]
        ranging_fracs.append(dist.get("RANGING", 0.0))
    avg_ranging = np.mean(ranging_fracs)

    gates = QUALITY_GATES

    if total_bars < gates["min_oos_bars"]:
        failures.append(f"oos_bars={total_bars} < {gates['min_oos_bars']}")
    if avg_conf < gates["min_avg_confidence"]:
        failures.append(f"confidence={avg_conf:.3f} < {gates['min_avg_confidence']}")
    if avg_conf > gates["max_avg_confidence"]:
        failures.append(f"confidence={avg_conf:.3f} > {gates['max_avg_confidence']} (overfit?)")
    if avg_halt > gates["max_halt_fraction"]:
        failures.append(f"halt_frac={avg_halt:.3f} > {gates['max_halt_fraction']}")
    if avg_stint < gates["min_avg_stint"]:
        failures.append(f"avg_stint={avg_stint:.1f} < {gates['min_avg_stint']} (flipping too fast)")
    if avg_ranging < gates["min_ranging_frac"]:
        failures.append(f"ranging={avg_ranging:.3f} < {gates['min_ranging_frac']} (never RANGING)")
    if avg_ranging > gates["max_ranging_frac"]:
        failures.append(f"ranging={avg_ranging:.3f} > {gates['max_ranging_frac']} (always RANGING)")

    passed = len(failures) == 0

    print(f"\n=== Quality Gates {'PASSED' if passed else 'FAILED'} ===")
    print(f"  OOS bars: {total_bars}")
    print(f"  Avg confidence: {avg_conf:.3f}")
    print(f"  Avg halt fraction: {avg_halt:.3f}")
    print(f"  Avg stint length: {avg_stint:.1f}")
    print(f"  Avg RANGING fraction: {avg_ranging:.3f}")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")

    return passed, failures


def swap_model(detector: RegimeDetectorV2, model_dir: str, backup_dir: str) -> None:
    """Backup existing model and save new one."""
    model_path = Path(model_dir)
    backup_path = Path(backup_dir)

    # Backup existing
    if model_path.exists():
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(model_path, backup_path)
        print(f"Backed up existing model to {backup_dir}/")

    # Save new
    detector.save(model_dir)
    print(f"New model saved to {model_dir}/")


def restart_bot() -> bool:
    """Restart the mes-bot systemd service."""
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "restart", "mes-bot"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print("Bot restarted successfully")
            return True
        else:
            print(f"Bot restart failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Bot restart error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Auto-retrain Regime Detector V2")
    parser.add_argument("--train-months", type=int, default=12,
                        help="Months of data to use for training (default: 12)")
    parser.add_argument("--wf-train-months", type=int, default=6,
                        help="Walk-forward train window in months (default: 6)")
    parser.add_argument("--wf-test-months", type=int, default=1,
                        help="Walk-forward test window in months (default: 1)")
    parser.add_argument("--symbol", default="MESH6",
                        help="Symbol to train on (default: MESH6)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate only, don't swap model or restart")
    parser.add_argument("--no-restart", action="store_true",
                        help="Swap model but don't restart the bot")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--backup-dir", default=BACKUP_DIR)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Auto-Retrain Regime V2 — {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}")

    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    # 1. Load data from Postgres
    df_1m = load_bars_from_postgres(dsn, args.train_months, args.symbol)

    if len(df_1m) < 5000:
        print(f"ERROR: Only {len(df_1m)} 1m bars — need at least 5000 (~1 week RTH)")
        sys.exit(1)

    # 2. Resample to 5m
    df_5m = resample_bars(df_1m, freq="5m")
    print(f"Resampled to 5m: {len(df_5m):,} bars")

    # 3. Walk-forward validation
    config = RegimeDetectorV2Config()
    oos_stats, last_detector = walk_forward_validate(
        df_5m, config, args.wf_train_months, args.wf_test_months,
    )

    # 4. Check quality gates
    passed, failures = check_quality_gates(oos_stats)

    if not passed:
        print("\nModel did NOT pass quality gates — keeping existing model")
        sys.exit(0)

    if args.dry_run:
        print("\n[DRY RUN] Would swap model and restart bot")
        sys.exit(0)

    # 5. Train final model on full dataset
    print("\n--- Training final model on full dataset ---")
    full_features, _ = build_features_v2(df_5m, config)
    final_detector = RegimeDetectorV2(config)
    final_detector.fit(full_features)

    # Final model stats
    probas = final_detector.predict_proba_sequence(full_features)
    stats = compute_regime_stats(probas)
    print(f"Full model — bars: {stats['n_bars']}, "
          f"conf: {stats['avg_confidence']:.3f}, "
          f"dist: {stats['state_distribution']}")

    # 6. Swap model
    swap_model(final_detector, args.model_dir, args.backup_dir)

    # 7. Restart bot
    if not args.no_restart:
        restart_bot()
    else:
        print("Skipping bot restart (--no-restart)")

    print(f"\nRetrain complete — {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
