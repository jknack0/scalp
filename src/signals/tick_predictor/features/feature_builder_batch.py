"""Batch (Polars-vectorized) feature builder for TickDirectionPredictor.

Mirrors the 26-feature set in feature_builder.py but computes all features
in a single vectorized pass over a DataFrame of OHLCV bars.

Feature layout (26 total):
  SET A [0-17]  Pure OHLCV (price action, volume, CVD, candle structure)
  SET B [18-25] Estimated L1 (approximated from OHLCV bar shape)

Strict causality: all features use .shift(1) so features at row t
are computed from bars [0..t-1] only.
"""

from __future__ import annotations

import numpy as np
import polars as pl

MES_TICK_SIZE = 0.25
MES_TICK_VALUE = 1.25
MES_POINT_VALUE = 5.00

SET_A_NAMES: list[str] = [
    "return_1", "return_5", "return_15", "realized_vol_20",
    "hl_range", "hl_range_zscore",
    "volume_zscore_20", "volume_zscore_50", "volume_ratio_5",
    "volume_accel", "up_vol_ratio_10",
    "cvd_delta", "cvd_cumsum_10", "cvd_cumsum_30", "cvd_slope_10",
    "close_position", "close_pos_zscore", "body_ratio",
]

SET_B_NAMES: list[str] = [
    "obi_est", "obi_est_5",
    "microprice_est", "microprice_vs_mid",
    "ofi_est_10", "ofi_est_30",
    "spread_proxy", "spread_zscore",
]

FEATURE_NAMES: list[str] = SET_A_NAMES + SET_B_NAMES
NUM_FEATURES = len(FEATURE_NAMES)

# Features that skip outer z-score normalization (already scaled)
_SKIP_NORM = {
    "volume_zscore_20", "volume_zscore_50",
    "close_pos_zscore", "hl_range_zscore",
    "obi_est_5", "spread_zscore",
}


def build_features_batch(
    bars: pl.DataFrame,
    norm_window: int = 100,
) -> pl.DataFrame:
    """Vectorized feature generation from OHLCV bars.

    Strict causality: all features use .shift(1) so features at row t
    are computed from bars [0..t-1] only.

    Args:
        bars: OHLCV bars with columns:
            timestamp, timestamp_ns, open, high, low, close, volume
            Optional L1: avg_bid_size, avg_ask_size, aggressive_buy_vol, aggressive_sell_vol
        norm_window: rolling window for z-score normalization (default 100)

    Returns:
        DataFrame with timestamp_ns + 26 feature columns
    """
    df = bars.clone()
    s = 1  # causality shift

    c = pl.col("close")
    o = pl.col("open")
    h = pl.col("high")
    l = pl.col("low")
    vol = pl.col("volume").cast(pl.Float64)

    hl = h - l
    hl_denom = hl + 1e-9

    # ══════════════════════════════════════════════════════════════════
    # SET A — Pure OHLCV (18 features)
    # ══════════════════════════════════════════════════════════════════

    # ── Price action (6) ─────────────────────────────────────────────
    log_ret_1 = (c / c.shift(1)).log()
    df = df.with_columns(log_ret_1.alias("_log_ret_1"))

    df = df.with_columns([
        pl.col("_log_ret_1").shift(s).alias("return_1"),
        (c / c.shift(5)).log().shift(s).alias("return_5"),
        (c / c.shift(15)).log().shift(s).alias("return_15"),
        pl.col("_log_ret_1").rolling_std(20).shift(s).alias("realized_vol_20"),
    ])

    # hl_range in ticks
    hl_ticks = hl / MES_TICK_SIZE
    df = df.with_columns(hl_ticks.shift(s).alias("hl_range"))

    # hl_range_zscore (50-bar rolling)
    hl_mean_50 = hl_ticks.rolling_mean(50)
    hl_std_50 = hl_ticks.rolling_std(50)
    df = df.with_columns(
        pl.when(hl_std_50 > 1e-9).then(
            (hl_ticks - hl_mean_50) / hl_std_50
        ).otherwise(0.0).shift(s).alias("hl_range_zscore")
    )

    # ── Volume (5) ──────────────────────────────────────────────────
    vol_mean_20 = vol.rolling_mean(20)
    vol_std_20 = vol.rolling_std(20)
    vz20 = pl.when(vol_std_20 > 1e-9).then(
        (vol - vol_mean_20) / vol_std_20
    ).otherwise(0.0)
    df = df.with_columns(vz20.alias("_vz20"))

    df = df.with_columns([
        pl.col("_vz20").shift(s).alias("volume_zscore_20"),
    ])

    vol_mean_50 = vol.rolling_mean(50)
    vol_std_50 = vol.rolling_std(50)
    df = df.with_columns(
        pl.when(vol_std_50 > 1e-9).then(
            (vol - vol_mean_50) / vol_std_50
        ).otherwise(0.0).shift(s).alias("volume_zscore_50")
    )

    vol_mean_5 = vol.rolling_mean(5)
    df = df.with_columns(
        pl.when(vol_mean_5 > 1e-9).then(
            vol / vol_mean_5
        ).otherwise(1.0).shift(s).alias("volume_ratio_5")
    )

    # volume_accel = vz20[t] - vz20[t-5]
    df = df.with_columns(
        (pl.col("_vz20") - pl.col("_vz20").shift(5)).shift(s).alias("volume_accel")
    )

    # up_vol_ratio_10: fraction of volume on up-bars over last 10 bars
    up_vol = pl.when(c > o).then(vol).otherwise(pl.lit(0.0))
    df = df.with_columns(up_vol.alias("_up_vol"))
    df = df.with_columns(
        pl.when(vol.rolling_sum(10) > 0).then(
            pl.col("_up_vol").rolling_sum(10) / vol.rolling_sum(10)
        ).otherwise(0.5).shift(s).alias("up_vol_ratio_10")
    )

    # ── OHLCV CVD (4) ───────────────────────────────────────────────
    cvd_d = vol * (2.0 * (c - l) / hl_denom - 1.0)
    df = df.with_columns(cvd_d.alias("_cvd_delta"))

    df = df.with_columns([
        pl.col("_cvd_delta").shift(s).alias("cvd_delta"),
        pl.col("_cvd_delta").rolling_sum(10).shift(s).alias("cvd_cumsum_10"),
        pl.col("_cvd_delta").rolling_sum(30).shift(s).alias("cvd_cumsum_30"),
    ])

    # cvd_slope_10: (cumsum_now - cumsum_10ago) / 10
    cvd_cumsum_10_col = pl.col("_cvd_delta").rolling_sum(10)
    df = df.with_columns(cvd_cumsum_10_col.alias("_cvd_cs10"))
    df = df.with_columns(
        ((pl.col("_cvd_cs10") - pl.col("_cvd_cs10").shift(10)) / 10.0)
        .shift(s).alias("cvd_slope_10")
    )

    # ── Candle structure (3) ─────────────────────────────────────────
    close_pos = (c - l) / hl_denom
    df = df.with_columns(close_pos.alias("_close_pos"))

    df = df.with_columns([
        pl.col("_close_pos").shift(s).alias("close_position"),
        (pl.col("close") - pl.col("open")).abs() / hl_denom,
    ])

    # close_pos_zscore (20-bar rolling)
    cp_mean = pl.col("_close_pos").rolling_mean(20)
    cp_std = pl.col("_close_pos").rolling_std(20)
    df = df.with_columns(
        pl.when(cp_std > 1e-9).then(
            (pl.col("_close_pos") - cp_mean) / cp_std
        ).otherwise(0.0).shift(s).alias("close_pos_zscore")
    )

    body_r = (c - o).abs() / hl_denom
    df = df.with_columns(body_r.shift(s).alias("body_ratio"))

    # ══════════════════════════════════════════════════════════════════
    # SET B — estimated from OHLCV, replace with real L1 values when available
    # ══════════════════════════════════════════════════════════════════

    # Check for real L1 data
    has_l1 = (
        "avg_bid_size" in df.columns
        and df["avg_bid_size"].sum() > 0
    )

    if has_l1:
        df = df.with_columns([
            pl.col("avg_bid_size").fill_null(0.0).alias("_bid_est"),
            pl.col("avg_ask_size").fill_null(0.0).alias("_ask_est"),
        ])
    else:
        df = df.with_columns([
            (vol * (h - c) / hl_denom).alias("_bid_est"),
            (vol * (c - l) / hl_denom).alias("_ask_est"),
        ])

    total_est = pl.col("_bid_est") + pl.col("_ask_est") + 1e-9
    obi_expr = (pl.col("_bid_est") - pl.col("_ask_est")) / total_est
    df = df.with_columns(obi_expr.alias("_obi_est"))

    df = df.with_columns([
        pl.col("_obi_est").shift(s).alias("obi_est"),
        pl.col("_obi_est").rolling_mean(5).shift(s).alias("obi_est_5"),
    ])

    # Microprice
    hl_range_pts = hl_ticks * MES_TICK_SIZE
    half_spread = hl_range_pts / 2.0
    mp = (pl.col("_bid_est") * (c + half_spread)
          + pl.col("_ask_est") * (c - half_spread)) / total_est
    df = df.with_columns(mp.alias("_microprice_est"))

    df = df.with_columns([
        pl.col("_microprice_est").shift(s).alias("microprice_est"),
        (pl.col("_microprice_est") - c).shift(s).alias("microprice_vs_mid"),
    ])

    # OFI
    ofi_raw = (pl.col("_bid_est").diff().fill_null(0.0)
               - pl.col("_ask_est").diff().fill_null(0.0))
    df = df.with_columns(ofi_raw.alias("_ofi_raw"))

    df = df.with_columns([
        pl.col("_ofi_raw").rolling_sum(10).shift(s).alias("ofi_est_10"),
        pl.col("_ofi_raw").rolling_sum(30).shift(s).alias("ofi_est_30"),
    ])

    # Spread proxy & zscore
    df = df.with_columns(hl_ticks.shift(s).alias("spread_proxy"))

    sp_mean_50 = hl_ticks.rolling_mean(50)
    sp_std_50 = hl_ticks.rolling_std(50)
    df = df.with_columns(
        pl.when(sp_std_50 > 1e-9).then(
            (hl_ticks - sp_mean_50) / sp_std_50
        ).otherwise(0.0).shift(s).alias("spread_zscore")
    )

    # ── Z-score normalize features ──────────────────────────────────
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

    # ── Select output ────────────────────────────────────────────────
    out_cols = ["timestamp_ns"] + FEATURE_NAMES
    available = [c for c in out_cols if c in df.columns]
    return df.select(available)
