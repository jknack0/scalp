"""5-minute feature builder for direction prediction.

Order-flow-first feature set with strict causality:
- Features at bar t use ONLY data from bars [0..t-1] (previous completed bars)
- The label for bar t predicts what happens AFTER bar t closes
- No same-bar leakage: features never peek at the bar being predicted

Feature layout (22 total):
  [0-8]   Order flow (5m aggregated from L1 or OHLCV approx)
  [9-14]  Price action & volatility (5m)
  [15-18] Technicals (5m, from completed bars)
  [19-21] HMM regime (5m, causal forward-only)

Supports two modes:
  - Batch: vectorized Polars for training (fast)
  - Streaming: bar-by-bar for live inference
"""

from __future__ import annotations

import numpy as np
import polars as pl

MES_TICK_SIZE = 0.25

FEATURE_NAMES_5M: list[str] = [
    # ── Order Flow (5m, aggregated from L1) ───────────────
    "ofi_1bar",             # Order flow imbalance: Δbid - Δask
    "ofi_3bar",             # 3-bar rolling sum of OFI
    "cvd_1bar",             # Cumulative volume delta (buy - sell)
    "cvd_3bar",             # 3-bar rolling sum of CVD
    "cvd_slope_5bar",       # Slope of cumulative CVD over 5 bars (25 min)
    "obi_1bar",             # Order book imbalance: (bid_sz - ask_sz) / total
    "microprice_drift",     # Microprice change (directional pressure)
    "aggressive_ratio",     # buy_vol / total_aggressive_vol
    "vpin_12bar",           # VPIN over 12 bars (1 hour)

    # ── Price Action & Volatility (5m) ────────────────────
    "return_1",             # 1-bar log return
    "return_3",             # 3-bar log return
    "return_6",             # 6-bar (30 min) log return
    "realized_vol_12",      # 12-bar (1 hour) realized vol
    "autocorr_6",           # 6-bar return autocorrelation
    "hl_range_zscore",      # High-low range z-scored vs 12-bar rolling

    # ── Technicals (5m, completed bars) ───────────────────
    "rsi_14",               # RSI-14
    "bb_pctb_20",           # Bollinger %B (20-bar, 2 std)
    "stoch_k_14",           # Stochastic K-14
    "donchian_pos_20",      # Position within 20-bar Donchian channel

    # ── HMM Regime (5m, causal forward-only) ──────────────
    "regime_p_trending",
    "regime_p_ranging",
    "regime_p_highvol",

    # ── Time of Day ──────────────────────────────────────
    "tod_sin",                  # sin(2π * minutes_since_930 / 390)
    "tod_cos",                  # cos(2π * minutes_since_930 / 390)
    "session_segment",          # 0=open(9:30-10:30), 1=mid(10:30-14:30), 2=close(14:30-16:00)
]

NUM_FEATURES_5M = len(FEATURE_NAMES_5M)


def build_features_batch(
    bars_5m: pl.DataFrame,
    regime_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Vectorized feature generation from 5m bars.

    Strict causality: all features use .shift(1) so features at row t
    are computed from bars [0..t-1] only.

    Args:
        bars_5m: 5m OHLCV bars with columns:
            timestamp, timestamp_ns, open, high, low, close, volume
            Optional L1: avg_bid_size, avg_ask_size, aggressive_buy_vol, aggressive_sell_vol
        regime_df: optional regime probabilities with timestamp_ns column
            (must already be shifted +5m so join_asof only sees previous bar)

    Returns:
        DataFrame with timestamp_ns + 22 feature columns
    """
    df = bars_5m.clone()

    # ── Detect L1 availability ───────────────────────────
    has_l1 = (
        "avg_bid_size" in df.columns
        and df["avg_bid_size"].sum() > 0
    )

    if has_l1:
        df = df.with_columns([
            pl.col("avg_bid_size").fill_null(0.0).alias("bid_size"),
            pl.col("avg_ask_size").fill_null(0.0).alias("ask_size"),
            (pl.col("aggressive_buy_vol").fill_null(0.0)).alias("buy_vol"),
            (pl.col("aggressive_sell_vol").fill_null(0.0)).alias("sell_vol"),
            (pl.col("aggressive_buy_vol").fill_null(0.0)
             - pl.col("aggressive_sell_vol").fill_null(0.0)).alias("delta"),
        ])
    else:
        # OHLCV approximation (same as 1m builder)
        hl_denom = (pl.col("high") - pl.col("low")) + 1e-9
        ratio = (pl.col("close") - pl.col("low")) / hl_denom
        vol_f = pl.col("volume").cast(pl.Float64)
        df = df.with_columns([
            (vol_f * (1.0 - ratio)).alias("bid_size"),
            (vol_f * ratio).alias("ask_size"),
            (vol_f * ratio).alias("buy_vol"),
            (vol_f * (1.0 - ratio)).alias("sell_vol"),
            (vol_f * (2.0 * ratio - 1.0)).alias("delta"),
        ])

    vol_f = pl.col("volume").cast(pl.Float64)
    total_sz = pl.col("bid_size") + pl.col("ask_size")

    # ── Microprice ───────────────────────────────────────
    half_spread = pl.lit(MES_TICK_SIZE)
    bid_px = pl.col("close") - half_spread
    ask_px = pl.col("close") + half_spread
    mp = pl.when(total_sz > 0).then(
        (pl.col("bid_size") * ask_px + pl.col("ask_size") * bid_px) / total_sz
    ).otherwise(pl.col("close"))
    df = df.with_columns(mp.alias("microprice"))

    # ── Compute raw features (on current bar) ────────────
    # OFI: change in bid/ask sizes
    df = df.with_columns([
        (pl.col("bid_size").diff().fill_null(0.0)
         - pl.col("ask_size").diff().fill_null(0.0)).alias("_ofi"),
    ])

    # OBI
    obi = pl.when(total_sz > 0).then(
        (pl.col("bid_size") - pl.col("ask_size")) / total_sz
    ).otherwise(0.0)

    # Aggressive ratio
    total_aggressive = pl.col("buy_vol") + pl.col("sell_vol")
    agg_ratio = pl.when(total_aggressive > 0).then(
        pl.col("buy_vol") / total_aggressive
    ).otherwise(0.5)

    # Log returns
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1)).log().alias("_log_ret_1"),
    ])

    # ── Build feature columns (all shifted by 1 for causality) ──
    s = 1  # causality shift

    # Order Flow features
    df = df.with_columns([
        pl.col("_ofi").shift(s).alias("ofi_1bar"),
        pl.col("_ofi").rolling_sum(3).shift(s).alias("ofi_3bar"),
        pl.col("delta").shift(s).alias("cvd_1bar"),
        pl.col("delta").rolling_sum(3).shift(s).alias("cvd_3bar"),
        obi.shift(s).alias("obi_1bar"),
        agg_ratio.shift(s).alias("aggressive_ratio"),
        pl.col("microprice").diff().shift(s).alias("microprice_drift"),
    ])

    # CVD slope (5-bar linear regression on cumulative delta)
    df = df.with_columns(pl.col("delta").cum_sum().alias("_cvd_cumsum"))
    n_w = 5
    cvd_arr = df["_cvd_cumsum"].to_numpy()
    if len(cvd_arr) >= n_w:
        from numpy.lib.stride_tricks import sliding_window_view
        x = np.arange(n_w, dtype=np.float64)
        x_mean = x.mean()
        x_var = float(np.sum((x - x_mean) ** 2))
        windows = sliding_window_view(cvd_arr, n_w)
        y_means = windows.mean(axis=1)
        slopes_valid = np.einsum(
            "j,ij->i", x - x_mean, windows - y_means[:, None]
        ) / x_var
        slopes = np.full(len(cvd_arr), np.nan, dtype=np.float64)
        slopes[n_w - 1:] = slopes_valid
    else:
        slopes = np.full(len(cvd_arr), np.nan, dtype=np.float64)
    slopes_shifted = np.empty_like(slopes)
    slopes_shifted[0] = np.nan
    slopes_shifted[1:] = slopes[:-1]
    df = df.with_columns(pl.Series("cvd_slope_5bar", slopes_shifted))

    # VPIN (12-bar = 1 hour): |buy - sell| / total_volume, rolling mean
    df = df.with_columns([
        pl.when(vol_f > 0).then(
            (pl.col("buy_vol") - pl.col("sell_vol")).abs() / vol_f
        ).otherwise(0.0).rolling_mean(12).shift(s).alias("vpin_12bar"),
    ])

    # ── Price Action features ────────────────────────────
    df = df.with_columns([
        pl.col("_log_ret_1").shift(s).alias("return_1"),
        (pl.col("close") / pl.col("close").shift(3)).log().shift(s).alias("return_3"),
        (pl.col("close") / pl.col("close").shift(6)).log().shift(s).alias("return_6"),
        pl.col("_log_ret_1").rolling_std(12).shift(s).alias("realized_vol_12"),
    ])

    # High-low range z-score
    hl_range = pl.col("high") - pl.col("low")
    hl_std = hl_range.rolling_std(12)
    hl_mean = hl_range.rolling_mean(12)
    df = df.with_columns([
        pl.when(hl_std > 1e-12).then(
            (hl_range - hl_mean) / hl_std
        ).otherwise(0.0).shift(s).alias("hl_range_zscore"),
    ])

    # Autocorrelation (6-bar)
    log_rets = df["_log_ret_1"].to_numpy()
    n_ac = 6
    if len(log_rets) >= n_ac + 1:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(log_rets, n_ac + 1)
        autocorrs = np.full(len(log_rets), np.nan, dtype=np.float64)
        for i in range(len(windows)):
            w = windows[i]
            if np.all(np.isfinite(w)):
                r1 = w[:-1]
                r2 = w[1:]
                m1, m2 = r1.mean(), r2.mean()
                s1, s2 = r1.std(), r2.std()
                if s1 > 1e-12 and s2 > 1e-12:
                    autocorrs[n_ac + i] = float(
                        np.mean((r1 - m1) * (r2 - m2)) / (s1 * s2)
                    )
    else:
        autocorrs = np.full(len(log_rets), np.nan, dtype=np.float64)
    ac_shifted = np.empty_like(autocorrs)
    ac_shifted[0] = np.nan
    ac_shifted[1:] = autocorrs[:-1]
    df = df.with_columns(pl.Series("autocorr_6", ac_shifted))

    # ── Technicals (all from completed bars via shift) ───
    c = pl.col("close")

    # RSI-14
    gain = pl.col("_log_ret_1").clip(lower_bound=0.0)
    loss = (-pl.col("_log_ret_1")).clip(lower_bound=0.0)
    avg_gain = gain.rolling_mean(14)
    avg_loss = loss.rolling_mean(14)
    rsi = pl.when(avg_loss > 1e-12).then(
        100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    ).otherwise(50.0)
    df = df.with_columns(rsi.shift(s).alias("rsi_14"))

    # Bollinger %B (20-bar)
    c20_mean = c.rolling_mean(20)
    c20_std = c.rolling_std(20)
    bb_upper = c20_mean + 2.0 * c20_std
    bb_lower = c20_mean - 2.0 * c20_std
    bb_pctb = pl.when((bb_upper - bb_lower) > 1e-12).then(
        (c - bb_lower) / (bb_upper - bb_lower)
    ).otherwise(0.5)
    df = df.with_columns(bb_pctb.shift(s).alias("bb_pctb_20"))

    # Stochastic K-14
    h14 = pl.col("high").rolling_max(14)
    l14 = pl.col("low").rolling_min(14)
    stoch_range = h14 - l14
    stoch_k = pl.when(stoch_range > 1e-12).then(
        (c - l14) / stoch_range * 100.0
    ).otherwise(50.0)
    df = df.with_columns(stoch_k.shift(s).alias("stoch_k_14"))

    # Donchian position (20-bar)
    h20 = pl.col("high").rolling_max(20)
    l20 = pl.col("low").rolling_min(20)
    don_range = h20 - l20
    don_pos = pl.when(don_range > 1e-12).then(
        (c - l20) / don_range
    ).otherwise(0.5)
    df = df.with_columns(don_pos.shift(s).alias("donchian_pos_20"))

    # ── HMM Regime features ──────────────────────────────
    if regime_df is not None and len(regime_df) > 0:
        regime_cols = ["timestamp_ns", "regime_p_trending", "regime_p_ranging",
                       "regime_p_highvol"]
        avail = [c for c in regime_cols if c in regime_df.columns]
        r = regime_df.select(avail).sort("timestamp_ns")
        df = df.sort("timestamp_ns").join_asof(
            r, on="timestamp_ns", strategy="backward"
        )
        # Shift regime features by 1 for causality
        for col in ["regime_p_trending", "regime_p_ranging", "regime_p_highvol"]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).shift(s))
            else:
                df = df.with_columns(pl.lit(0.0).alias(col))
    else:
        for col in ["regime_p_trending", "regime_p_ranging", "regime_p_highvol"]:
            df = df.with_columns(pl.lit(0.0).alias(col))

    # ── Time of Day features ─────────────────────────────
    # These are NOT z-score normalized (they're already bounded)
    if "timestamp_ns" in df.columns:
        ts_arr = df["timestamp_ns"].cast(pl.Datetime("ns"))
        hours = ts_arr.dt.hour().to_numpy()
        minutes = ts_arr.dt.minute().to_numpy()
        mins_since_open = (hours - 9) * 60 + minutes - 30
        frac = mins_since_open.astype(np.float64) / 390.0
        df = df.with_columns([
            pl.Series("tod_sin", np.sin(frac * 2.0 * np.pi)),
            pl.Series("tod_cos", np.cos(frac * 2.0 * np.pi)),
            pl.Series("session_segment", np.where(
                mins_since_open < 60, 0.0,
                np.where(mins_since_open < 300, 0.5, 1.0)
            )),
        ])

    # ── Z-score normalize features (skip time-of-day) ───
    skip_norm = {"tod_sin", "tod_cos", "session_segment"}
    norm_window = 100
    for feat in FEATURE_NAMES_5M:
        if feat in skip_norm:
            continue
        if feat in df.columns:
            raw = pl.col(feat)
            mean = raw.rolling_mean(norm_window)
            std = raw.rolling_std(norm_window)
            df = df.with_columns(
                pl.when(std > 1e-12).then(
                    (raw - mean) / std
                ).otherwise(0.0).alias(feat)
            )

    # ── Select output ────────────────────────────────────
    out_cols = ["timestamp_ns"] + FEATURE_NAMES_5M
    available = [c for c in out_cols if c in df.columns]
    return df.select(available)
