"""Tick Predictor v4 — Pure L1 microstructure features.

Train on real L1 order flow data (2025-2026) instead of OHLCV estimates.
Extracts rich microstructure features from raw tick data: actual spread,
trade intensity, aggressive flow, order book imbalance, price impact, etc.

Usage:
    python -u scripts/tick_predictor/train_v4_l1.py 2>&1 | tee logs/tick_pred_v4_l1.log
    python -u scripts/tick_predictor/train_v4_l1.py --freq 5m --horizon 6 --tp-ticks 12 --sl-ticks 12
"""

from __future__ import annotations

import argparse
import json
import time as _time
from datetime import datetime, time as dt_time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.core.logging import configure_logging
from src.data.bars import resample_bars
from src.signals.tick_predictor.labels.triple_barrier_5m import (
    TripleBarrierConfig5M,
    TripleBarrierLabeler5M,
)
from src.signals.tick_predictor.model.calibrator import TemperatureCalibrator

MES_TICK_SIZE = 0.25
MES_TICK_VALUE = 1.25
MES_POINT_VALUE = 5.00
COMMISSION_RT = 0.59

DATA_DIR = Path("data/tick_predictor")
MODEL_DIR = Path("models/tick_predictor_v4")
RESULTS_DIR = Path("results/tick_predictor_v4")


# ── L1 feature names ─────────────────────────────────────────────────

# Per-bar microstructure features (computed from raw ticks within each bar)
L1_BAR_FEATURES = [
    # Trade activity
    "trade_count",           # number of trades in bar
    "avg_trade_size",        # mean trade size
    "trade_size_cv",         # coefficient of variation of trade sizes
    "large_trade_pct",       # fraction of volume from trades > 2x median
    # Aggressive flow
    "aggressive_buy_pct",    # aggressive buy vol / total vol
    "aggressive_imbalance",  # (buy - sell) / (buy + sell)
    # Order book
    "obi",                   # (bid_size - ask_size) / (bid_size + ask_size)
    "book_depth_ratio",      # bid_size / ask_size
    "microprice_vs_mid",     # microprice - midprice (in ticks)
    # Spread
    "actual_spread",         # mean spread in ticks
    "spread_max",            # max spread in bar (in ticks)
    # Price dynamics
    "tick_up_pct",           # fraction of upticks
    "price_impact",          # abs(return) / log(volume) — how much price moves per unit flow
    "vwap_distance",         # (close - vwap) / tick_size — signed distance to bar VWAP
]

# OHLCV basics (keep a few for context)
OHLCV_FEATURES = [
    "return_1",
    "return_5",
    "realized_vol_20",
    "hl_range",
    "volume_zscore_20",
    "close_position",
    "body_ratio",
]

# Rolling L1 features (computed across bars)
ROLLING_L1_FEATURES = [
    "aggressive_imbalance_5",    # 5-bar rolling mean
    "aggressive_imbalance_10",   # 10-bar rolling mean
    "obi_5",                     # 5-bar rolling OBI
    "obi_10",                    # 10-bar rolling OBI
    "ofi_real_10",               # 10-bar order flow imbalance
    "ofi_real_30",               # 30-bar order flow imbalance
    "cvd_real_10",               # 10-bar cumulative real CVD
    "cvd_real_30",               # 30-bar cumulative real CVD
    "cvd_real_slope_10",         # slope of 10-bar CVD
    "spread_zscore",             # z-score of actual spread (50-bar)
    "trade_intensity_zscore",    # z-score of trade count (20-bar)
    "aggressive_buy_pct_5",      # 5-bar rolling buy pressure
]

FEATURE_NAMES = L1_BAR_FEATURES + OHLCV_FEATURES + ROLLING_L1_FEATURES
NUM_FEATURES = len(FEATURE_NAMES)

# Features that skip outer z-score normalization (already z-scored)
_SKIP_NORM = {
    "volume_zscore_20", "spread_zscore", "trade_intensity_zscore",
}


def freq_minutes(freq: str) -> int:
    if freq.endswith("m"):
        return int(freq[:-1])
    if freq.endswith("s"):
        return max(1, int(freq[:-1]) // 60)
    raise ValueError(f"Unknown freq: {freq}")


# ── L1 tick aggregation per bar ───────────────────────────────────────

def aggregate_l1_to_bars(
    ticks: pl.DataFrame,
    freq: str,
) -> pl.DataFrame:
    """Aggregate raw L1 ticks into bars with rich microstructure features.

    Args:
        ticks: Raw L1 ticks with columns:
            timestamp, price, size, side, bid_price, ask_price, bid_size, ask_size
        freq: Bar frequency (e.g., '1m', '5m')

    Returns:
        DataFrame with OHLCV + L1 microstructure columns per bar
    """
    # Truncate to bar boundaries
    ticks = ticks.with_columns(
        pl.col("timestamp").dt.truncate(freq).alias("bar_ts"),
    )

    # Compute per-tick derived columns
    mid = (pl.col("bid_price") + pl.col("ask_price")) / 2.0
    spread = (pl.col("ask_price") - pl.col("bid_price")) / MES_TICK_SIZE

    ticks = ticks.with_columns([
        mid.alias("mid_price"),
        spread.alias("spread_ticks"),
        pl.col("price").diff().sign().fill_null(0).alias("tick_dir"),
        (pl.col("price") * pl.col("size").cast(pl.Float64)).alias("dollar_vol"),
    ])

    # Lee-Ready aggressive classification using midprice
    ticks = ticks.with_columns([
        pl.when(pl.col("price") > pl.col("mid_price"))
        .then(pl.col("size").cast(pl.Float64))
        .otherwise(0.0)
        .alias("agg_buy_vol"),
        pl.when(pl.col("price") < pl.col("mid_price"))
        .then(pl.col("size").cast(pl.Float64))
        .otherwise(0.0)
        .alias("agg_sell_vol"),
    ])

    # Aggregate per bar
    bars = ticks.group_by("bar_ts").agg([
        # OHLCV
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("size").cast(pl.Float64).sum().alias("volume"),
        # Trade activity
        pl.col("price").count().alias("trade_count"),
        pl.col("size").cast(pl.Float64).mean().alias("avg_trade_size"),
        pl.col("size").cast(pl.Float64).std().alias("trade_size_std"),
        pl.col("size").cast(pl.Float64).median().alias("trade_size_median"),
        # Aggressive flow
        pl.col("agg_buy_vol").sum().alias("aggressive_buy_vol"),
        pl.col("agg_sell_vol").sum().alias("aggressive_sell_vol"),
        # Order book
        pl.col("bid_size").cast(pl.Float64).mean().alias("avg_bid_size"),
        pl.col("ask_size").cast(pl.Float64).mean().alias("avg_ask_size"),
        # Spread
        pl.col("spread_ticks").mean().alias("actual_spread"),
        pl.col("spread_ticks").max().alias("spread_max"),
        # Tick direction
        pl.when(pl.col("tick_dir") > 0).then(1).otherwise(0).sum().alias("uptick_count"),
        # VWAP
        pl.col("dollar_vol").sum().alias("total_dollar_vol"),
    ]).sort("bar_ts")

    # Compute derived bar-level features
    total_vol = pl.col("volume") + 1e-9
    total_agg = pl.col("aggressive_buy_vol") + pl.col("aggressive_sell_vol") + 1e-9
    total_book = pl.col("avg_bid_size") + pl.col("avg_ask_size") + 1e-9
    hl = pl.col("high") - pl.col("low") + 1e-9

    bars = bars.with_columns([
        # Trade activity
        (pl.col("trade_size_std") / (pl.col("avg_trade_size") + 1e-9)).alias("trade_size_cv"),
        # Large trade pct: fraction of volume from trades > 2x median
        # (we approximate with std > mean — actual per-tick calc done below)
        pl.lit(0.0).alias("large_trade_pct_placeholder"),
        # Aggressive flow
        (pl.col("aggressive_buy_vol") / total_agg).alias("aggressive_buy_pct"),
        ((pl.col("aggressive_buy_vol") - pl.col("aggressive_sell_vol")) / total_agg)
            .alias("aggressive_imbalance"),
        # Order book imbalance
        ((pl.col("avg_bid_size") - pl.col("avg_ask_size")) / total_book).alias("obi"),
        (pl.col("avg_bid_size").cast(pl.Float64) / (pl.col("avg_ask_size").cast(pl.Float64) + 1e-9))
            .alias("book_depth_ratio"),
        # Microprice
        ((pl.col("avg_ask_size").cast(pl.Float64) * pl.col("open")
          + pl.col("avg_bid_size").cast(pl.Float64) * pl.col("close")) / total_book)
            .alias("microprice_raw"),
        # Tick direction
        (pl.col("uptick_count").cast(pl.Float64) / (pl.col("trade_count").cast(pl.Float64) + 1e-9))
            .alias("tick_up_pct"),
        # VWAP
        (pl.col("total_dollar_vol") / total_vol).alias("vwap"),
        # Add timestamp_ns
        pl.col("bar_ts").dt.epoch("ns").alias("timestamp_ns"),
    ])

    # Microprice vs mid
    bars = bars.with_columns([
        ((pl.col("microprice_raw") - (pl.col("open") + pl.col("close")) / 2.0) / MES_TICK_SIZE)
            .alias("microprice_vs_mid"),
        # Price impact: abs(return) / log(volume)
        ((pl.col("close") - pl.col("open")).abs()
         / (pl.col("volume").log() + 1e-9) / MES_TICK_SIZE)
            .alias("price_impact"),
        # VWAP distance
        ((pl.col("close") - pl.col("vwap")) / MES_TICK_SIZE).alias("vwap_distance"),
        # Close position
        ((pl.col("close") - pl.col("low")) / hl).alias("close_position"),
        # Body ratio
        ((pl.col("close") - pl.col("open")).abs() / hl).alias("body_ratio"),
        # HL range in ticks
        (hl / MES_TICK_SIZE).alias("hl_range"),
    ])

    # Rename bar_ts -> timestamp
    bars = bars.rename({"bar_ts": "timestamp"})

    return bars


def compute_large_trade_pct(ticks: pl.DataFrame, freq: str) -> pl.DataFrame:
    """Compute fraction of volume from large trades (> 2x median) per bar."""
    ticks = ticks.with_columns(
        pl.col("timestamp").dt.truncate(freq).alias("bar_ts"),
    )

    # Per-bar median trade size
    medians = ticks.group_by("bar_ts").agg(
        pl.col("size").cast(pl.Float64).median().alias("median_size"),
    )

    ticks = ticks.join(medians, on="bar_ts", how="left")
    ticks = ticks.with_columns(
        pl.when(pl.col("size").cast(pl.Float64) > 2.0 * pl.col("median_size"))
        .then(pl.col("size").cast(pl.Float64))
        .otherwise(0.0)
        .alias("large_vol"),
    )

    result = ticks.group_by("bar_ts").agg([
        (pl.col("large_vol").sum() / (pl.col("size").cast(pl.Float64).sum() + 1e-9))
            .alias("large_trade_pct"),
    ]).sort("bar_ts")

    return result


# ── Feature builder ───────────────────────────────────────────────────

def build_l1_features(
    bars: pl.DataFrame,
    norm_window: int = 100,
) -> pl.DataFrame:
    """Build features from L1-enriched bars.

    All features use .shift(1) for strict causality.
    """
    df = bars.clone()
    s = 1  # causality shift

    c = pl.col("close")
    o = pl.col("open")
    vol = pl.col("volume").cast(pl.Float64)

    # ── Per-bar L1 features (already computed in aggregate step) ──────
    # Just apply causality shift
    for feat in L1_BAR_FEATURES:
        if feat in df.columns:
            df = df.with_columns(pl.col(feat).shift(s).alias(feat))

    # ── OHLCV features ────────────────────────────────────────────────
    log_ret_1 = (c / c.shift(1)).log()
    df = df.with_columns(log_ret_1.alias("_log_ret_1"))

    df = df.with_columns([
        pl.col("_log_ret_1").shift(s).alias("return_1"),
        (c / c.shift(5)).log().shift(s).alias("return_5"),
        pl.col("_log_ret_1").rolling_std(20).shift(s).alias("realized_vol_20"),
    ])

    vol_mean_20 = vol.rolling_mean(20)
    vol_std_20 = vol.rolling_std(20)
    df = df.with_columns(
        pl.when(vol_std_20 > 1e-9).then(
            (vol - vol_mean_20) / vol_std_20
        ).otherwise(0.0).shift(s).alias("volume_zscore_20")
    )

    # close_position and body_ratio already computed — just shift
    if "close_position" in df.columns:
        df = df.with_columns(pl.col("close_position").shift(s).alias("close_position"))
    if "body_ratio" in df.columns:
        df = df.with_columns(pl.col("body_ratio").shift(s).alias("body_ratio"))
    if "hl_range" in df.columns:
        df = df.with_columns(pl.col("hl_range").shift(s).alias("hl_range"))

    # ── Rolling L1 features ───────────────────────────────────────────
    # Need to compute these from un-shifted bar values, then shift

    # Aggressive imbalance rolling
    agg_imb = ((pl.col("aggressive_buy_vol") - pl.col("aggressive_sell_vol"))
               / (pl.col("aggressive_buy_vol") + pl.col("aggressive_sell_vol") + 1e-9))
    df = df.with_columns(agg_imb.alias("_agg_imb_raw"))
    df = df.with_columns([
        pl.col("_agg_imb_raw").rolling_mean(5).shift(s).alias("aggressive_imbalance_5"),
        pl.col("_agg_imb_raw").rolling_mean(10).shift(s).alias("aggressive_imbalance_10"),
    ])

    # OBI rolling
    obi_raw = ((pl.col("avg_bid_size") - pl.col("avg_ask_size"))
               / (pl.col("avg_bid_size") + pl.col("avg_ask_size") + 1e-9))
    df = df.with_columns(obi_raw.alias("_obi_raw"))
    df = df.with_columns([
        pl.col("_obi_raw").rolling_mean(5).shift(s).alias("obi_5"),
        pl.col("_obi_raw").rolling_mean(10).shift(s).alias("obi_10"),
    ])

    # OFI: changes in bid/ask size (real order flow imbalance)
    ofi = (pl.col("avg_bid_size").diff().fill_null(0.0)
           - pl.col("avg_ask_size").diff().fill_null(0.0))
    df = df.with_columns(ofi.alias("_ofi_raw"))
    df = df.with_columns([
        pl.col("_ofi_raw").rolling_sum(10).shift(s).alias("ofi_real_10"),
        pl.col("_ofi_raw").rolling_sum(30).shift(s).alias("ofi_real_30"),
    ])

    # Real CVD (aggressive buy - sell cumulative)
    cvd_delta = pl.col("aggressive_buy_vol") - pl.col("aggressive_sell_vol")
    df = df.with_columns(cvd_delta.alias("_cvd_delta"))
    df = df.with_columns([
        pl.col("_cvd_delta").rolling_sum(10).shift(s).alias("cvd_real_10"),
        pl.col("_cvd_delta").rolling_sum(30).shift(s).alias("cvd_real_30"),
    ])

    # CVD slope
    cvd_cs10 = pl.col("_cvd_delta").rolling_sum(10)
    df = df.with_columns(cvd_cs10.alias("_cvd_cs10"))
    df = df.with_columns(
        ((pl.col("_cvd_cs10") - pl.col("_cvd_cs10").shift(10)) / 10.0)
        .shift(s).alias("cvd_real_slope_10")
    )

    # Spread z-score
    sp = pl.col("actual_spread")
    sp_mean = sp.rolling_mean(50)
    sp_std = sp.rolling_std(50)
    df = df.with_columns(
        pl.when(sp_std > 1e-9).then((sp - sp_mean) / sp_std)
        .otherwise(0.0).shift(s).alias("spread_zscore")
    )

    # Trade intensity z-score
    tc = pl.col("trade_count").cast(pl.Float64)
    tc_mean = tc.rolling_mean(20)
    tc_std = tc.rolling_std(20)
    df = df.with_columns(
        pl.when(tc_std > 1e-9).then((tc - tc_mean) / tc_std)
        .otherwise(0.0).shift(s).alias("trade_intensity_zscore")
    )

    # Aggressive buy pct rolling
    abp = pl.col("aggressive_buy_vol") / (
        pl.col("aggressive_buy_vol") + pl.col("aggressive_sell_vol") + 1e-9
    )
    df = df.with_columns(
        abp.rolling_mean(5).shift(s).alias("aggressive_buy_pct_5")
    )

    # ── Z-score normalize features ────────────────────────────────────
    for feat in FEATURE_NAMES:
        if feat in _SKIP_NORM:
            continue
        if feat in df.columns:
            raw = pl.col(feat)
            mean = raw.rolling_mean(norm_window)
            std = raw.rolling_std(norm_window)
            df = df.with_columns(
                pl.when(std > 1e-9).then(
                    (raw - mean) / std
                ).otherwise(0.0).alias(feat)
            )

    # ── Select output ─────────────────────────────────────────────────
    out_cols = ["timestamp_ns"] + FEATURE_NAMES
    available = [c for c in out_cols if c in df.columns]
    return df.select(available)


# ── Data loading ──────────────────────────────────────────────────────

def load_l1_bars(
    start_date: str,
    end_date: str,
    freq: str,
) -> pl.DataFrame:
    """Load raw L1 ticks and aggregate to bars with microstructure features."""
    import datetime as dt

    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    years = range(start.year, end.year + 1)
    start_dt = dt.datetime.combine(start, dt.time.min)
    end_dt = dt.datetime.combine(end, dt.time.min)

    l1_paths = [
        f"data/l1/year={y}/data.parquet"
        for y in years
        if Path(f"data/l1/year={y}/data.parquet").exists()
    ]
    if not l1_paths:
        raise FileNotFoundError(f"No L1 data found for {start_date} to {end_date}")

    print(f"    Loading L1 ticks...")
    t0 = _time.perf_counter()
    ticks = (
        pl.scan_parquet(l1_paths)
        .with_columns(
            pl.col("timestamp").dt.replace_time_zone(None).alias("timestamp")
        )
        .filter(
            (pl.col("timestamp") >= start_dt)
            & (pl.col("timestamp") < end_dt)
            & (pl.col("timestamp").dt.time() >= dt_time(9, 30))
            & (pl.col("timestamp").dt.time() < dt_time(16, 0))
        )
        .sort("timestamp")
        .collect()
    )
    print(f"    {len(ticks):,} L1 ticks loaded ({_time.perf_counter() - t0:.1f}s)")

    if len(ticks) == 0:
        raise ValueError("No L1 ticks found after filtering")

    # Aggregate to bars
    print(f"    Aggregating to {freq} bars...")
    t0 = _time.perf_counter()
    bars = aggregate_l1_to_bars(ticks, freq)
    print(f"    {len(bars):,} bars ({_time.perf_counter() - t0:.1f}s)")

    # Compute large trade pct (requires per-tick median)
    print(f"    Computing large trade fractions...")
    t0 = _time.perf_counter()
    large_pct = compute_large_trade_pct(ticks, freq)
    bars = bars.join(
        large_pct.rename({"bar_ts": "timestamp"}),
        on="timestamp",
        how="left",
    ).with_columns(pl.col("large_trade_pct").fill_null(0.0))
    print(f"    ({_time.perf_counter() - t0:.1f}s)")

    return bars


# ── Training ──────────────────────────────────────────────────────────

def train_model(
    features_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    val_days: int = 15,
) -> tuple:
    """Train LightGBM with walk-forward CV."""
    joined = features_df.join(labels_df, on="timestamp_ns", how="inner")
    available_feats = [f for f in FEATURE_NAMES if f in joined.columns]
    joined = joined.drop_nulls(subset=available_feats)
    joined = joined.filter(pl.col("label") != 0)

    print(f"\n  Training data: {len(joined):,} rows")
    dist = joined.group_by("label").len().sort("label")
    for row in dist.iter_rows(named=True):
        name = {-1: "DOWN", 0: "FLAT", 1: "UP"}.get(row["label"], "?")
        pct = row["len"] / len(joined) * 100
        print(f"    {name}: {row['len']:,} ({pct:.1f}%)")

    X = joined.select(available_feats).to_numpy().astype(np.float32)
    y_raw = joined["label"].to_numpy()
    y = np.where(y_raw == 1, 1, 0).astype(np.int32)
    w = joined["sample_weight"].to_numpy().astype(np.float32)

    # Train/val split
    timestamps = joined["timestamp_ns"].to_numpy()
    ts_dates = pl.Series("ts", timestamps).cast(pl.Datetime("ns")).dt.date().to_numpy()
    unique_dates = np.unique(ts_dates)

    if len(unique_dates) > val_days:
        val_start_date = unique_dates[-val_days]
        val_mask = ts_dates >= val_start_date
        train_mask = ~val_mask
    else:
        split = int(len(X) * 0.9)
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[:split] = True
        val_mask = ~train_mask

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    w_train = w[train_mask]

    print(f"    Train: {len(X_train):,}, Val: {len(X_val):,}")

    # Walk-forward CV
    print("\n  Walk-forward CV...")
    t0 = _time.perf_counter()
    n_splits = 5
    embargo = 6
    n = len(X_train)
    min_train = int(n * 0.5)
    fold_size = (n - min_train) // n_splits

    cv_accs = []
    for fold in range(n_splits):
        test_start = min_train + fold * fold_size
        test_end = min_train + (fold + 1) * fold_size if fold < n_splits - 1 else n
        train_end = max(0, test_start - embargo)

        if test_end <= test_start or train_end < 100:
            continue

        Xtr = X_train[:train_end]
        ytr = y_train[:train_end]
        wtr = w_train[:train_end]
        Xte = X_train[test_start:test_end]
        yte = y_train[test_start:test_end]

        vs = max(1, int(len(Xtr) * 0.1))
        params = {
            "objective": "binary",
            "num_leaves": 31,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
            "is_unbalance": True,
            "device": "gpu",
            "verbose": -1,
            "seed": 42,
        }
        train_set = lgb.Dataset(Xtr[:-vs], ytr[:-vs], weight=wtr[:-vs],
                                feature_name=available_feats)
        val_set = lgb.Dataset(Xtr[-vs:], ytr[-vs:], reference=train_set)
        callbacks = [
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        booster = lgb.train(params, train_set, num_boost_round=300,
                            valid_sets=[val_set], callbacks=callbacks)

        pred = booster.predict(Xte)
        acc = float(np.mean((pred > 0.5).astype(int) == yte))
        cv_accs.append(acc)
        print(f"    Fold {fold}: acc={acc:.4f} (trees={booster.num_trees()})")

    mean_acc = float(np.mean(cv_accs))
    std_acc = float(np.std(cv_accs))
    print(f"    Mean CV accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"    ({_time.perf_counter() - t0:.1f}s)")

    # Train final model
    print("\n  Training final model...")
    t0 = _time.perf_counter()
    params = {
        "objective": "binary",
        "num_leaves": 31,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "is_unbalance": True,
        "device": "gpu",
        "verbose": -1,
        "seed": 42,
    }
    vs = max(1, int(len(X_train) * 0.1))
    train_set = lgb.Dataset(X_train[:-vs], y_train[:-vs], weight=w_train[:-vs],
                            feature_name=available_feats)
    val_set = lgb.Dataset(X_train[-vs:], y_train[-vs:], reference=train_set)
    callbacks = [
        lgb.early_stopping(30, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    booster = lgb.train(params, train_set, num_boost_round=300,
                        valid_sets=[val_set], callbacks=callbacks)
    print(f"    Trained ({_time.perf_counter() - t0:.1f}s, {booster.num_trees()} trees)")

    # Feature importance
    imp = booster.feature_importance(importance_type="gain")
    top_idx = np.argsort(imp)[::-1][:15]
    print(f"\n  Top features:")
    for i in top_idx:
        print(f"    {available_feats[i]:30s} {imp[i]:,.0f}")

    # Calibrate
    print("\n  Calibrating...")
    raw_val = booster.predict(X_val)
    raw_3class = np.column_stack([1 - raw_val, np.zeros(len(raw_val)), raw_val])
    y_val_3class = np.where(y_val == 1, 2, 0).astype(np.int32)

    calibrator = TemperatureCalibrator()
    calibrator.fit(raw_3class, y_val_3class)
    cal_proba = calibrator.predict_proba_calibrated(raw_3class)
    val_acc = float(np.mean(np.argmax(cal_proba, axis=1) == y_val_3class))
    val_ece = calibrator.compute_ece(cal_proba, y_val_3class)
    print(f"    Temperature: {calibrator.temperature:.4f}")
    print(f"    Val accuracy: {val_acc:.4f}")
    print(f"    Val ECE: {val_ece:.4f}")

    # Prediction distribution diagnostics
    print(f"\n  Prediction distribution (val set):")
    print(f"    p_up mean: {raw_val.mean():.4f}")
    print(f"    p_up std:  {raw_val.std():.4f}")
    print(f"    p_up min:  {raw_val.min():.4f}")
    print(f"    p_up max:  {raw_val.max():.4f}")
    pcts = np.percentile(raw_val, [5, 25, 50, 75, 95])
    print(f"    p_up percentiles: 5%={pcts[0]:.4f}  25%={pcts[1]:.4f}  "
          f"50%={pcts[2]:.4f}  75%={pcts[3]:.4f}  95%={pcts[4]:.4f}")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"lgbm_{ts}.txt"
    cal_path = MODEL_DIR / f"calibrator_{ts}.pkl"
    booster.save_model(str(model_path))
    calibrator.save(str(cal_path))
    print(f"    Model saved: {model_path}")
    print(f"    Calibrator saved: {cal_path}")

    return booster, calibrator, mean_acc, std_acc, val_acc, available_feats


# ── Main ──────────────────────────────────────────────────────────────

def cache_suffix(cfg, freq):
    return (f"_v4_l1_{freq}_h{cfg.vertical_barrier_bars}"
            f"_tp{cfg.tp_ticks}_sl{cfg.sl_ticks}_rth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", default="1m")
    parser.add_argument("--tp-ticks", type=int, default=4)
    parser.add_argument("--sl-ticks", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--val-days", type=int, default=15)
    parser.add_argument("--cost-ticks", type=float, default=0.72)
    # L1 data range
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-03-14")
    args = parser.parse_args()

    configure_logging(log_level="INFO")

    bar_mins = freq_minutes(args.freq)
    cfg = TripleBarrierConfig5M(
        vertical_barrier_bars=args.horizon,
        tp_ticks=args.tp_ticks,
        sl_ticks=args.sl_ticks,
    )
    suffix = cache_suffix(cfg, args.freq)

    print(f"\n{'='*70}")
    print(f"  TICK PREDICTOR v4 — PURE L1 MICROSTRUCTURE")
    print(f"  Data:      {args.start} to {args.end} (real L1 ticks)")
    print(f"  Bar freq:  {args.freq}")
    print(f"  Horizon:   {cfg.vertical_barrier_bars} bars ({cfg.vertical_barrier_bars * bar_mins}m)")
    print(f"  TP/SL:     {cfg.tp_ticks}/{cfg.sl_ticks} ticks "
          f"({cfg.tp_ticks * 0.25:.2f}/{cfg.sl_ticks * 0.25:.2f} pts)")
    print(f"  Cost:      {args.cost_ticks} ticks (commission + slippage)")
    print(f"  Features:  {NUM_FEATURES} ({len(L1_BAR_FEATURES)} bar-level + "
          f"{len(OHLCV_FEATURES)} OHLCV + {len(ROLLING_L1_FEATURES)} rolling)")
    print(f"{'='*70}")

    # ── Load L1 ticks and build bars ──────────────────────────────────
    feat_path = DATA_DIR / f"features{suffix}_{args.start}_{args.end}.parquet"
    labels_path = DATA_DIR / f"labels{suffix}_{args.start}_{args.end}.parquet"

    if feat_path.exists() and labels_path.exists():
        print(f"\n  Loading cached features: {feat_path}")
        features = pl.read_parquet(feat_path)
        print(f"  Loading cached labels: {labels_path}")
        labels = pl.read_parquet(labels_path)
    else:
        print(f"\n  STEP 1: Loading L1 ticks and building {args.freq} bars...")
        t0 = _time.perf_counter()
        bars = load_l1_bars(args.start, args.end, args.freq)
        print(f"    Total: {_time.perf_counter() - t0:.1f}s")

        print(f"\n  STEP 2: Feature generation (L1 microstructure)...")
        t0 = _time.perf_counter()
        features = build_l1_features(bars)
        print(f"    Features: {len(features):,} rows ({_time.perf_counter() - t0:.1f}s)")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        features.write_parquet(feat_path)
        print(f"    Saved: {feat_path}")

        print(f"\n  STEP 3: Label generation...")
        t0 = _time.perf_counter()
        labeler = TripleBarrierLabeler5M(cfg)
        labels = labeler.generate_labels_from_bars(bars)
        labeler.save_labels(labels, args.start, args.end, suffix=suffix)
        labels.write_parquet(labels_path)
        print(f"    Labels: {len(labels):,} rows ({_time.perf_counter() - t0:.1f}s)")

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TRAINING ({args.start} to {args.end}, pure L1)")
    print(f"{'='*70}")
    t0 = _time.perf_counter()
    model, calibrator, cv_acc, cv_std, val_acc, feat_names = train_model(
        features, labels, val_days=args.val_days
    )
    print(f"\n  Training complete in {_time.perf_counter() - t0:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  CV accuracy:    {cv_acc:.4f} +/- {cv_std:.4f}")
    print(f"  Val accuracy:   {val_acc:.4f}")
    print(f"  Data:           {args.start} to {args.end} (real L1)")
    print(f"  Features:       {len(feat_names)} L1 microstructure")
    print(f"  Bar freq:       {args.freq}")
    print(f"  Horizon:        {cfg.vertical_barrier_bars} bars ({cfg.vertical_barrier_bars * bar_mins}m)")
    print(f"  TP/SL:          {cfg.tp_ticks}/{cfg.sl_ticks} ticks")

    cost_threshold = (args.sl_ticks + args.cost_ticks) / (args.tp_ticks + args.sl_ticks)
    print(f"  Cost threshold: {cost_threshold:.1%} accuracy needed for +EV")
    if cv_acc > cost_threshold:
        print(f"  >>> CV accuracy EXCEEDS cost threshold! Possible edge detected.")
    else:
        deficit = cost_threshold - cv_acc
        print(f"  >>> CV accuracy {deficit:.1%} below cost threshold. No edge.")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "cv_accuracy": cv_acc,
        "cv_std": cv_std,
        "val_accuracy": val_acc,
        "cost_threshold": cost_threshold,
        "has_edge": cv_acc > cost_threshold,
        "freq": args.freq,
        "horizon": args.horizon,
        "tp_ticks": args.tp_ticks,
        "sl_ticks": args.sl_ticks,
        "features": len(feat_names),
        "data_start": args.start,
        "data_end": args.end,
    }
    result_path = RESULTS_DIR / f"result_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {result_path}")


if __name__ == "__main__":
    main()
