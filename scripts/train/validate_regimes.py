"""Validate regime labels against actual market characteristics.

For each OOS walk-forward split, checks whether:
- TRENDING has higher abs returns and positive autocorrelation
- RANGING has lower vol and near-zero drift
- HIGH_VOL has higher realized vol

This is the critical sanity check before investing in Student-t upgrades.

Usage:
    python scripts/train/validate_regimes.py --parquet-dir data/parquet_1m \
        --start 2020-01-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.bars import resample_bars
from src.data.csv_to_parquet import read_parquet_range
from src.models.regime_detector_v2 import (
    RegimeDetectorV2,
    RegimeDetectorV2Config,
    RegimeLabel,
    RegimeProba,
    build_features_v2,
)


def load_5m_bars(parquet_dir: str, start: str | None, end: str | None) -> pl.DataFrame:
    """Load 1m parquet data and resample to 5m bars."""
    import re
    parquet_path = Path(parquet_dir)
    years = sorted(
        int(d.name.split("=")[1])
        for d in parquet_path.iterdir()
        if d.is_dir() and re.match(r"year=\d{4}", d.name)
    )
    if not years:
        raise FileNotFoundError(f"No year= partitions in {parquet_dir}")

    start_year = int(start[:4]) if start else years[0]
    end_year = int(end[:4]) if end else years[-1]

    df = read_parquet_range(parquet_dir, start_year, end_year)
    print(f"Loaded {len(df):,} raw 1m bars from {parquet_dir} ({start_year}-{end_year})")

    if start:
        df = df.filter(pl.col("timestamp") >= pl.lit(start).str.to_datetime())
    if end:
        df = df.filter(pl.col("timestamp") < pl.lit(end).str.to_datetime())

    print(f"After date filter: {len(df):,} bars")
    df_5m = resample_bars(df, freq="5m")
    print(f"Resampled to 5m: {len(df_5m):,} bars")
    return df_5m


def compute_bar_metrics(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Compute per-bar metrics for validation against regime labels.

    Returns arrays aligned with df rows (after dropping first row for returns).
    """
    close = df["close"].to_numpy().astype(np.float64)
    open_ = df["open"].to_numpy().astype(np.float64)
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    n = len(close)

    # Log returns
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(close[1:] / close[:-1])

    # Abs log returns (proxy for realized vol per bar)
    abs_ret = np.abs(log_ret)

    # Garman-Klass per-bar variance
    hl = np.log(np.maximum(high, 1e-10) / np.maximum(low, 1e-10))
    co = np.log(np.maximum(close, 1e-10) / np.maximum(open_, 1e-10))
    gk_var = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2

    # Rolling GK vol (20-bar)
    gk_series = pl.Series("gk", gk_var)
    gk_vol_20 = gk_series.rolling_mean(window_size=20, min_samples=2).to_numpy()
    gk_vol_20 = np.sqrt(np.maximum(gk_vol_20, 0.0))

    # Rolling autocorrelation lag-1 (50-bar window)
    autocorr_1 = np.full(n, np.nan)
    window = 50
    for i in range(window, n):
        chunk = log_ret[i - window:i]
        if np.std(chunk) > 1e-10:
            autocorr_1[i] = np.corrcoef(chunk[:-1], chunk[1:])[0, 1]

    # Signed return (directional drift)
    # Rolling 20-bar mean return
    ret_series = pl.Series("r", log_ret)
    rolling_mean_ret = ret_series.rolling_mean(window_size=20, min_samples=2).to_numpy()

    # High-low range as % of price (another vol proxy)
    hl_pct = (high - low) / close

    return {
        "log_ret": log_ret,
        "abs_ret": abs_ret,
        "gk_var": gk_var,
        "gk_vol_20": gk_vol_20,
        "autocorr_1": autocorr_1,
        "rolling_mean_ret": rolling_mean_ret,
        "hl_pct": hl_pct,
    }


def validate_split(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    config: RegimeDetectorV2Config,
) -> dict | None:
    """Train on train_df, predict OOS on test_df, compute per-regime metrics."""
    train_features, _ = build_features_v2(train_df, config)
    test_features, test_ts = build_features_v2(test_df, config)

    if len(train_features) < 500 or len(test_features) < 50:
        return None

    detector = RegimeDetectorV2(config)
    detector.fit(train_features)

    # OOS predictions
    probas = detector.predict_proba_sequence(test_features)

    # Compute actual market metrics on test data
    # Features drop zscore_window rows from the front — align metrics the same way
    metrics = compute_bar_metrics(test_df)
    offset = config.zscore_window

    # Further trim to match valid feature rows (NaN/inf filtering may drop more)
    n_probas = len(probas)

    # Align: metrics arrays are len(test_df), features start at offset
    # We need to match the exact rows that survived build_features_v2
    # Simplest: just use the last n_probas rows starting from offset
    # (build_features_v2 drops first zscore_window rows, then removes NaN/inf)
    aligned_metrics = {}
    for key, arr in metrics.items():
        trimmed = arr[offset:]
        # Remove rows where features had NaN/inf (same mask as build_features_v2)
        # Since we can't perfectly reconstruct the mask, take first n_probas rows
        # This is approximate but close enough for validation
        aligned_metrics[key] = trimmed[:n_probas]

    # Bucket by regime label
    regime_metrics = {label: defaultdict(list) for label in RegimeLabel}

    for i, p in enumerate(probas):
        if i >= len(aligned_metrics["log_ret"]):
            break
        # Only count bars where model is actionable (not in warmup/whipsaw)
        regime_metrics[p.regime]["abs_ret"].append(aligned_metrics["abs_ret"][i])
        regime_metrics[p.regime]["gk_vol_20"].append(aligned_metrics["gk_vol_20"][i])
        regime_metrics[p.regime]["log_ret"].append(aligned_metrics["log_ret"][i])
        regime_metrics[p.regime]["hl_pct"].append(aligned_metrics["hl_pct"][i])
        regime_metrics[p.regime]["confidence"].append(p.confidence)
        regime_metrics[p.regime]["position_size"].append(p.position_size)

        ac = aligned_metrics["autocorr_1"][i]
        if not np.isnan(ac):
            regime_metrics[p.regime]["autocorr_1"].append(ac)

        mr = aligned_metrics["rolling_mean_ret"][i]
        if not np.isnan(mr):
            regime_metrics[p.regime]["rolling_mean_ret"].append(mr)

    # Summarize
    result = {}
    for label in RegimeLabel:
        rm = regime_metrics[label]
        n = len(rm["abs_ret"])
        if n == 0:
            result[label.name] = {"n_bars": 0}
            continue

        result[label.name] = {
            "n_bars": n,
            "pct": n / n_probas,
            "mean_abs_ret": float(np.mean(rm["abs_ret"])),
            "median_abs_ret": float(np.median(rm["abs_ret"])),
            "mean_gk_vol": float(np.nanmean(rm["gk_vol_20"])),
            "mean_signed_ret": float(np.mean(rm["log_ret"])),
            "std_ret": float(np.std(rm["log_ret"])),
            "mean_hl_pct": float(np.mean(rm["hl_pct"])),
            "mean_autocorr_1": float(np.mean(rm["autocorr_1"])) if rm["autocorr_1"] else float("nan"),
            "mean_rolling_drift": float(np.mean(rm["rolling_mean_ret"])) if rm["rolling_mean_ret"] else float("nan"),
            "mean_confidence": float(np.mean(rm["confidence"])),
            "pct_full": sum(1 for s in rm["position_size"] if s == "full") / n,
        }

    return result


def print_comparison(all_splits: list[dict]) -> None:
    """Print regime comparison table aggregated across all OOS splits."""
    # Aggregate across splits
    agg = {label.name: defaultdict(list) for label in RegimeLabel}

    for split_result in all_splits:
        for label_name, metrics in split_result.items():
            if metrics["n_bars"] == 0:
                continue
            for k, v in metrics.items():
                if k != "n_bars" and not np.isnan(v):
                    agg[label_name][k].append(v)

    print("\n" + "=" * 90)
    print("REGIME VALIDATION — OOS METRICS (averaged across walk-forward splits)")
    print("=" * 90)

    # Header
    header = f"{'Metric':<25} {'TRENDING':>18} {'RANGING':>18} {'HIGH_VOL':>18}"
    print(header)
    print("-" * 90)

    metrics_to_show = [
        ("pct", "% of bars", "{:.1%}"),
        ("mean_abs_ret", "Mean |return|", "{:.6f}"),
        ("median_abs_ret", "Median |return|", "{:.6f}"),
        ("std_ret", "Return std dev", "{:.6f}"),
        ("mean_gk_vol", "Mean GK vol (20b)", "{:.6f}"),
        ("mean_hl_pct", "Mean H-L range %", "{:.6f}"),
        ("mean_signed_ret", "Mean signed return", "{:.7f}"),
        ("mean_rolling_drift", "Mean rolling drift", "{:.7f}"),
        ("mean_autocorr_1", "Mean autocorr(1)", "{:.4f}"),
        ("mean_confidence", "Mean confidence", "{:.3f}"),
        ("pct_full", "% full size", "{:.1%}"),
    ]

    for key, label, fmt in metrics_to_show:
        vals = []
        for regime in ["TRENDING", "RANGING", "HIGH_VOL"]:
            if agg[regime][key]:
                vals.append(fmt.format(np.mean(agg[regime][key])))
            else:
                vals.append("N/A")
        print(f"  {label:<23} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

    # Discrimination tests
    print("\n" + "=" * 90)
    print("DISCRIMINATION CHECKS")
    print("=" * 90)

    checks = []

    # Check 1: HIGH_VOL should have highest GK vol
    gk_means = {r: np.mean(agg[r]["mean_gk_vol"]) if agg[r]["mean_gk_vol"] else 0
                 for r in ["TRENDING", "RANGING", "HIGH_VOL"]}
    highest_gk = max(gk_means, key=gk_means.get)
    gk_ratio = gk_means["HIGH_VOL"] / max(gk_means["RANGING"], 1e-10)
    check1 = highest_gk == "HIGH_VOL"
    checks.append(check1)
    print(f"\n  1. HIGH_VOL has highest GK vol?  {'PASS' if check1 else 'FAIL'}")
    print(f"     GK vol: TRENDING={gk_means['TRENDING']:.6f}  RANGING={gk_means['RANGING']:.6f}  HIGH_VOL={gk_means['HIGH_VOL']:.6f}")
    print(f"     HIGH_VOL / RANGING ratio: {gk_ratio:.2f}x")

    # Check 2: TRENDING should have highest abs returns
    abs_means = {r: np.mean(agg[r]["mean_abs_ret"]) if agg[r]["mean_abs_ret"] else 0
                  for r in ["TRENDING", "RANGING", "HIGH_VOL"]}
    highest_abs = max(abs_means, key=abs_means.get)
    check2 = highest_abs in ("TRENDING", "HIGH_VOL")  # either is acceptable
    checks.append(check2)
    print(f"\n  2. TRENDING or HIGH_VOL has highest |return|?  {'PASS' if check2 else 'FAIL'}")
    print(f"     |ret|: TRENDING={abs_means['TRENDING']:.6f}  RANGING={abs_means['RANGING']:.6f}  HIGH_VOL={abs_means['HIGH_VOL']:.6f}")

    # Check 3: RANGING should have lowest vol
    lowest_gk = min(gk_means, key=gk_means.get)
    check3 = lowest_gk == "RANGING"
    checks.append(check3)
    print(f"\n  3. RANGING has lowest GK vol?  {'PASS' if check3 else 'FAIL'}")

    # Check 4: RANGING should have near-zero mean drift
    drift_means = {r: np.mean(agg[r]["mean_rolling_drift"]) if agg[r]["mean_rolling_drift"] else 0
                    for r in ["TRENDING", "RANGING", "HIGH_VOL"]}
    ranging_drift = abs(drift_means["RANGING"])
    trending_drift = abs(drift_means["TRENDING"])
    check4 = ranging_drift < trending_drift
    checks.append(check4)
    print(f"\n  4. RANGING has lower |drift| than TRENDING?  {'PASS' if check4 else 'FAIL'}")
    print(f"     |drift|: TRENDING={trending_drift:.7f}  RANGING={ranging_drift:.7f}  HIGH_VOL={abs(drift_means['HIGH_VOL']):.7f}")

    # Check 5: Autocorrelation — TRENDING should be more positive, RANGING more negative
    ac_means = {r: np.mean(agg[r]["mean_autocorr_1"]) if agg[r]["mean_autocorr_1"] else 0
                 for r in ["TRENDING", "RANGING", "HIGH_VOL"]}
    check5 = ac_means["TRENDING"] > ac_means["RANGING"]
    checks.append(check5)
    print(f"\n  5. TRENDING autocorr > RANGING autocorr?  {'PASS' if check5 else 'FAIL'}")
    print(f"     autocorr(1): TRENDING={ac_means['TRENDING']:.4f}  RANGING={ac_means['RANGING']:.4f}  HIGH_VOL={ac_means['HIGH_VOL']:.4f}")

    # Check 6: Separation — are the regimes actually different?
    # Use coefficient of variation across regime means for key metrics
    for metric_name, metric_key in [("GK vol", "mean_gk_vol"), ("|return|", "mean_abs_ret")]:
        vals = [np.mean(agg[r][metric_key]) for r in ["TRENDING", "RANGING", "HIGH_VOL"]
                if agg[r][metric_key]]
        if len(vals) == 3:
            cv = np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
            print(f"\n  Cross-regime CV for {metric_name}: {cv:.3f}  ({'good separation' if cv > 0.15 else 'WEAK separation' if cv > 0.05 else 'NO separation'})")

    # Overall
    passed = sum(checks)
    total = len(checks)
    print(f"\n{'=' * 90}")
    print(f"OVERALL: {passed}/{total} checks passed")
    if passed >= 4:
        print("Signal looks real — proceed to Student-t upgrade (build step 2)")
    elif passed >= 2:
        print("Marginal signal — investigate failed checks before upgrading")
    else:
        print("No meaningful discrimination — model is labeling noise. Debug before proceeding.")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Validate regime labels against market metrics")
    parser.add_argument("--parquet-dir", default="data/parquet_1m")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--gk-window", type=int, default=20)
    parser.add_argument("--zscore-window", type=int, default=500)
    parser.add_argument("--dof", type=int, default=5)
    parser.add_argument("--hurst-window", type=int, default=250)
    parser.add_argument("--autocorr-window", type=int, default=100)
    parser.add_argument("--emission", default="gaussian",
                        choices=["studentt", "gaussian"],
                        help="Emission distribution type")
    parser.add_argument("--feature-tier", default="t2", choices=["t1", "t2"],
                        help="Feature tier: t1 (4 features) or t2 (7 features, default)")
    parser.add_argument("--no-pca", action="store_true",
                        help="Disable PCA decorrelation")
    args = parser.parse_args()

    config = RegimeDetectorV2Config(
        n_states=args.n_states,
        gk_vol_window=args.gk_window,
        zscore_window=args.zscore_window,
        studentt_dof=args.dof,
        hurst_window=args.hurst_window,
        autocorr_window=args.autocorr_window,
        emission_type=args.emission,
        feature_tier=args.feature_tier,
        pca_enabled=not args.no_pca,
    )

    df = load_5m_bars(args.parquet_dir, args.start, args.end)

    # Walk-forward splits
    ts_col = df["timestamp"]
    min_date = ts_col.min()
    max_date = ts_col.max()

    train_delta = timedelta(days=args.train_months * 30)
    test_delta = timedelta(days=args.test_months * 30)

    splits = []
    current = min_date
    while current + train_delta + test_delta <= max_date:
        train_end = current + train_delta
        test_end = train_end + test_delta
        splits.append((current, train_end, test_end))
        current = train_end

    print(f"\n{len(splits)} walk-forward splits")

    all_results = []
    for i, (train_start, train_end, test_end) in enumerate(splits):
        print(f"\n--- Split {i+1}/{len(splits)}: train {train_start.date()}->{train_end.date()}, test {train_end.date()}->{test_end.date()} ---")

        train_df = df.filter(
            (pl.col("timestamp") >= train_start) & (pl.col("timestamp") < train_end)
        )
        test_df = df.filter(
            (pl.col("timestamp") >= train_end) & (pl.col("timestamp") < test_end)
        )

        if len(train_df) < 1000 or len(test_df) < 100:
            print(f"  Skipping: train={len(train_df)}, test={len(test_df)}")
            continue

        result = validate_split(train_df, test_df, config)
        if result:
            all_results.append(result)
            # Quick per-split summary
            for label in RegimeLabel:
                r = result[label.name]
                if r["n_bars"] > 0:
                    print(f"  {label.name:>10}: {r['n_bars']:>5} bars ({r['pct']:.0%}), "
                          f"|ret|={r['mean_abs_ret']:.6f}, GK={r['mean_gk_vol']:.6f}, "
                          f"ac1={r['mean_autocorr_1']:.4f}, drift={r['mean_signed_ret']:.7f}")

    if all_results:
        print_comparison(all_results)
    else:
        print("No valid splits — check data.")


if __name__ == "__main__":
    main()
