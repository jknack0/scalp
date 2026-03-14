"""1-minute feature builder for direction prediction.

Order-flow-first feature set with strict causality:
- Features at bar t use ONLY data from bars [0..t-1] (previous completed bars)
- The label for bar t predicts what happens AFTER bar t closes
- No same-bar leakage: features never peek at the bar being predicted

Feature layout (24 total):
  [0-9]   Order flow (1m aggregated from 1s L1 data)
  [10-13] Volume profile
  [14-17] Price action & volatility (1m)
  [18-21] HMM regime (5m)
  [22-23] Context (5m)

Supports two modes:
  - Batch: vectorized Polars for training (fast)
  - Streaming: bar-by-bar for live inference
"""

from __future__ import annotations

import numpy as np
import polars as pl

MES_TICK_SIZE = 0.25

FEATURE_NAMES_1M: list[str] = [
    # ── Order Flow (1m, aggregated from 1s) ──────────────────
    "ofi_1m",              # Order flow imbalance: Σ(Δbid - Δask) over 1m
    "ofi_5m",              # OFI over 5m (5-bar rolling sum)
    "cvd_1m",              # Cumulative volume delta (buy - sell) over 1m
    "cvd_5m",              # CVD over 5m
    "cvd_slope_5m",        # Slope of cumulative CVD over 5m (trend in aggression)
    "obi_1m",              # Order book imbalance: (bid_sz - ask_sz) / total
    "aggressive_ratio_1m", # aggressive_buy_vol / total_aggressive_vol
    "trade_intensity_1m",  # 1m volume / rolling 20m mean volume
    "microprice_drift_1m", # microprice change over 1m (directional pressure)
    "vpin_20m",            # Volume-sync'd probability of informed trading

    # ── Volume Profile ───────────────────────────────────────
    "volume_zscore_20m",   # 1m volume vs rolling 20m (participation spike)
    "delta_zscore_20m",    # 1m delta vs rolling 20m (aggression spike)
    "buy_sell_ratio_1m",   # buy_vol / sell_vol (directional conviction)
    "volume_trend_10m",    # slope of volume over 10m (escalation)

    # ── Price Action & Volatility (1m) ───────────────────────
    "return_1m",           # 1-bar log return
    "return_5m",           # 5-bar log return
    "realized_vol_20m",    # 20-bar realized vol
    "autocorr_5m",         # 5-bar return autocorrelation (momentum vs mean-rev)

    # ── HMM Regime (5m) ─────────────────────────────────────
    "regime_p_trending",
    "regime_p_ranging",
    "regime_p_highvol",
    "regime_bars_in",

    # ── Context (5m) ─────────────────────────────────────────
    "bb_pctb_5m",          # Bollinger %B: position within bands
    "rsi_14_5m",           # RSI-14: overbought/oversold
]

NUM_FEATURES_1M = len(FEATURE_NAMES_1M)


def build_features_batch(
    bars_1m: pl.DataFrame,
    regime_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Vectorized feature generation from 1m bars.

    Strict causality: all features use .shift(1) so features at row t
    are computed from bars [0..t-1] only.

    Args:
        bars_1m: 1m OHLCV bars with columns:
            timestamp, timestamp_ns, open, high, low, close, volume,
            avg_bid_size, avg_ask_size, aggressive_buy_vol, aggressive_sell_vol
            (L1 columns optional — falls back to OHLCV approximation)
        regime_df: optional regime probabilities with timestamp_ns column

    Returns:
        DataFrame with timestamp_ns + 24 feature columns
    """
    df = bars_1m.clone()

    # ── Detect L1 availability ───────────────────────────────
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
        # OHLCV approximation
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

    # ── Microprice ───────────────────────────────────────────
    half_spread = pl.lit(MES_TICK_SIZE)
    bid_px = pl.col("close") - half_spread
    ask_px = pl.col("close") + half_spread
    mp = pl.when(total_sz > 0).then(
        (pl.col("bid_size") * ask_px + pl.col("ask_size") * bid_px) / total_sz
    ).otherwise(pl.col("close"))
    df = df.with_columns(mp.alias("microprice"))

    # ── Compute raw features (on current bar) ────────────────
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

    # Buy/sell ratio (clamped)
    bs_ratio = pl.when(pl.col("sell_vol") > 0).then(
        (pl.col("buy_vol") / pl.col("sell_vol")).clip(0.0, 10.0)
    ).otherwise(1.0)

    # Log returns
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1)).log().alias("_log_ret_1"),
    ])

    # ── Build feature columns (all shifted by 1 for causality) ──
    # Every feature uses .shift(1) — value at row t comes from bar t-1
    s = 1  # causality shift

    # Order Flow features
    df = df.with_columns([
        # ofi_1m: single-bar OFI, shifted
        pl.col("_ofi").shift(s).alias("ofi_1m"),
        # ofi_5m: 5-bar rolling sum of OFI, shifted
        pl.col("_ofi").rolling_sum(5).shift(s).alias("ofi_5m"),
        # cvd_1m: single-bar delta, shifted
        pl.col("delta").shift(s).alias("cvd_1m"),
        # cvd_5m: 5-bar rolling sum of delta, shifted
        pl.col("delta").rolling_sum(5).shift(s).alias("cvd_5m"),
        # obi_1m: order book imbalance, shifted
        obi.shift(s).alias("obi_1m"),
        # aggressive_ratio_1m: shifted
        agg_ratio.shift(s).alias("aggressive_ratio_1m"),
        # trade_intensity_1m: volume / rolling 20m mean, shifted
        pl.when(vol_f.rolling_mean(20) > 0).then(
            vol_f / vol_f.rolling_mean(20)
        ).otherwise(1.0).shift(s).alias("trade_intensity_1m"),
        # microprice_drift_1m: change in microprice, shifted
        pl.col("microprice").diff().shift(s).alias("microprice_drift_1m"),
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
    # Shift for causality
    slopes_shifted = np.empty_like(slopes)
    slopes_shifted[0] = np.nan
    slopes_shifted[1:] = slopes[:-1]
    df = df.with_columns(pl.Series("cvd_slope_5m", slopes_shifted))

    # VPIN (20-bar): |buy - sell| / total_volume, rolling mean
    df = df.with_columns([
        pl.when(vol_f > 0).then(
            (pl.col("buy_vol") - pl.col("sell_vol")).abs() / vol_f
        ).otherwise(0.0).rolling_mean(20).shift(s).alias("vpin_20m"),
    ])

    # ── Volume Profile features ──────────────────────────────
    vol_std_20 = vol_f.rolling_std(20)
    vol_mean_20 = vol_f.rolling_mean(20)
    delta_std_20 = pl.col("delta").rolling_std(20)
    delta_mean_20 = pl.col("delta").rolling_mean(20)

    df = df.with_columns([
        # volume_zscore_20m
        pl.when(vol_std_20 > 0).then(
            (vol_f - vol_mean_20) / vol_std_20
        ).otherwise(0.0).shift(s).alias("volume_zscore_20m"),
        # delta_zscore_20m
        pl.when(delta_std_20 > 0).then(
            (pl.col("delta") - delta_mean_20) / delta_std_20
        ).otherwise(0.0).shift(s).alias("delta_zscore_20m"),
        # buy_sell_ratio_1m
        bs_ratio.shift(s).alias("buy_sell_ratio_1m"),
    ])

    # Volume trend (10-bar slope of volume)
    n_vt = 10
    vol_arr = df["volume"].to_numpy().astype(np.float64)
    if len(vol_arr) >= n_vt:
        from numpy.lib.stride_tricks import sliding_window_view
        x = np.arange(n_vt, dtype=np.float64)
        x_mean = x.mean()
        x_var = float(np.sum((x - x_mean) ** 2))
        windows = sliding_window_view(vol_arr, n_vt)
        y_means = windows.mean(axis=1)
        vol_slopes = np.einsum(
            "j,ij->i", x - x_mean, windows - y_means[:, None]
        ) / x_var
        vol_trend = np.full(len(vol_arr), np.nan, dtype=np.float64)
        vol_trend[n_vt - 1:] = vol_slopes
    else:
        vol_trend = np.full(len(vol_arr), np.nan, dtype=np.float64)
    vol_trend_shifted = np.empty_like(vol_trend)
    vol_trend_shifted[0] = np.nan
    vol_trend_shifted[1:] = vol_trend[:-1]
    df = df.with_columns(pl.Series("volume_trend_10m", vol_trend_shifted))

    # ── Price Action features ────────────────────────────────
    df = df.with_columns([
        pl.col("_log_ret_1").shift(s).alias("return_1m"),
        (pl.col("close") / pl.col("close").shift(5)).log().shift(s).alias("return_5m"),
        pl.col("_log_ret_1").rolling_std(20).shift(s).alias("realized_vol_20m"),
    ])

    # Autocorrelation (5-bar)
    log_rets = df["_log_ret_1"].to_numpy()
    n_ac = 5
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
    df = df.with_columns(pl.Series("autocorr_5m", ac_shifted))

    # ── HMM Regime features ─────────────────────────────────
    if regime_df is not None and len(regime_df) > 0:
        # Join regime probabilities via asof join on timestamp_ns
        # Regime is computed on 5m bars, so we forward-fill to 1m
        regime_cols = ["timestamp_ns", "regime_p_trending", "regime_p_ranging",
                       "regime_p_highvol", "regime_bars_in"]
        avail = [c for c in regime_cols if c in regime_df.columns]
        r = regime_df.select(avail).sort("timestamp_ns")
        df = df.sort("timestamp_ns").join_asof(
            r, on="timestamp_ns", strategy="backward"
        )
        # Shift regime features by 1 for causality
        for col in ["regime_p_trending", "regime_p_ranging",
                     "regime_p_highvol", "regime_bars_in"]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).shift(s))
            else:
                df = df.with_columns(pl.lit(0.0).alias(col))
    else:
        for col in ["regime_p_trending", "regime_p_ranging",
                     "regime_p_highvol", "regime_bars_in"]:
            df = df.with_columns(pl.lit(0.0).alias(col))

    # ── Context features (5m, computed from 1m bars) ────────
    # Bollinger %B on 20-bar window (= 20 minutes)
    c20_mean = pl.col("close").rolling_mean(20)
    c20_std = pl.col("close").rolling_std(20)
    bb_upper = c20_mean + 2.0 * c20_std
    bb_lower = c20_mean - 2.0 * c20_std
    bb_pctb = pl.when((bb_upper - bb_lower) > 1e-12).then(
        (pl.col("close") - bb_lower) / (bb_upper - bb_lower)
    ).otherwise(0.5)

    df = df.with_columns([
        bb_pctb.shift(s).alias("bb_pctb_5m"),
    ])

    # RSI-14
    gain = pl.col("_log_ret_1").clip(lower_bound=0.0)
    loss = (-pl.col("_log_ret_1")).clip(lower_bound=0.0)
    avg_gain = gain.rolling_mean(14)
    avg_loss = loss.rolling_mean(14)
    rsi = pl.when(avg_loss > 1e-12).then(
        100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    ).otherwise(50.0)
    df = df.with_columns([
        rsi.shift(s).alias("rsi_14_5m"),
    ])

    # ── Z-score normalize all features ───────────────────────
    norm_window = 100  # rolling z-score window
    for feat in FEATURE_NAMES_1M:
        if feat in df.columns:
            raw = pl.col(feat)
            mean = raw.rolling_mean(norm_window)
            std = raw.rolling_std(norm_window)
            df = df.with_columns(
                pl.when(std > 1e-12).then(
                    (raw - mean) / std
                ).otherwise(0.0).alias(feat)
            )

    # ── Select output ────────────────────────────────────────
    out_cols = ["timestamp_ns"] + FEATURE_NAMES_1M
    available = [c for c in out_cols if c in df.columns]
    return df.select(available)
