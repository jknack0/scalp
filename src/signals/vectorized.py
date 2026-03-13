"""Vectorized signal computation using Polars.

Replaces the per-bar Python loop in SignalEngine.compute() with columnar
operations.  Each function adds `sig_*` columns to the DataFrame.

Used by the tuning pipeline: compute signals once in the main process,
write enriched parquet, workers reconstruct SignalBundles from columns.
"""

from __future__ import annotations

import logging
import numpy as np
import polars as pl

_log = logging.getLogger(__name__)

MES_TICK_SIZE = 0.25


def enrich_bars(
    df: pl.DataFrame,
    signal_names: list[str],
) -> pl.DataFrame:
    """Add signal columns to a bars DataFrame.

    Args:
        df: Bars with columns: timestamp, open, high, low, close, volume.
            Must already have ``_et_ts`` (datetime US/Eastern) and
            ``_bar_date`` (date) columns from BacktestEngine preprocessing.

    Returns:
        DataFrame with original columns plus ``sig_*`` signal columns.
    """
    for name in signal_names:
        if name == "atr":
            df = _compute_atr(df)
        elif name == "vwap_session":
            df = _compute_vwap_session(df)
        elif name == "relative_volume":
            df = _compute_relative_volume(df)
        elif name == "spread":
            df = _compute_spread(df)
        elif name == "adx":
            df = _compute_adx(df)
        elif name == "donchian_channel":
            df = _compute_donchian_channel(df)
        elif name == "session_time":
            df = _compute_session_time(df)
        elif name == "regime_v2":
            df = _compute_regime_v2(df)
        else:
            raise ValueError(f"Unknown signal for vectorized computation: {name}")
    return df


# ── ATR ──────────────────────────────────────────────────────────────────


def _wilder_smooth(tr: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing matching the Python ATR signal exactly."""
    n = len(tr)
    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    for i in range(1, n):
        if i < period:
            # Build-up: simple running average
            atr[i] = (atr[i - 1] * i + tr[i]) / (i + 1)
        else:
            atr[i] = atr[i - 1] + (tr[i] - atr[i - 1]) / period
    return atr


def _compute_atr(
    df: pl.DataFrame,
    period: int = 14,
    regime_lookback: int = 100,
) -> pl.DataFrame:
    """Add ATR signal columns: sig_atr_value, sig_atr_raw, sig_atr_percentile, sig_atr_vol_regime."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    # True range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    prev_close = np.roll(close, 1)
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_close[1:]), np.abs(low[1:] - prev_close[1:])),
    )

    # Wilder smoothing
    atr = _wilder_smooth(tr, period)
    atr_ticks = atr / MES_TICK_SIZE

    # Percentile rank within rolling window
    percentile = np.full(n, 50.0, dtype=np.float64)
    for i in range(1, n):
        start = max(0, i - regime_lookback + 1)
        window = atr[start : i + 1]
        below = float(np.sum(window < atr[i]))
        percentile[i] = (below / len(window)) * 100.0

    # Vol regime
    vol_regime = np.where(
        percentile < 25.0, "LOW", np.where(percentile > 75.0, "HIGH", "NORMAL")
    )

    return df.with_columns(
        pl.Series("sig_atr_value", atr_ticks),
        pl.Series("sig_atr_raw", atr),
        pl.Series("sig_atr_percentile", percentile),
        pl.Series("sig_atr_vol_regime", vol_regime),
    )


# ── VWAP Session ─────────────────────────────────────────────────────────


def _compute_vwap_session(
    df: pl.DataFrame,
    slope_window: int = 20,
    first_kiss_lookback: int = 6,
    first_kiss_sd_threshold: float = 1.5,
    flat_slope_threshold: float = 0.003,
    trending_slope_threshold: float = 0.008,
) -> pl.DataFrame:
    """Add VWAP session signal columns."""
    # Typical price and session-grouped cumulative sums
    df = df.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_tp"),
    )

    df = df.with_columns(
        (pl.col("_tp") * pl.col("volume")).cum_sum().over("_bar_date").alias("_cum_pv"),
        pl.col("volume").cum_sum().over("_bar_date").alias("_cum_vol"),
        (pl.col("volume") * pl.col("_tp") * pl.col("_tp"))
        .cum_sum()
        .over("_bar_date")
        .alias("_cum_pv2"),
        pl.lit(1).cum_sum().over("_bar_date").alias("sig_vwap_session_age"),
    )

    # VWAP
    df = df.with_columns(
        (pl.col("_cum_pv") / pl.col("_cum_vol")).alias("sig_vwap_vwap"),
    )

    # SD
    df = df.with_columns(
        (pl.col("_cum_pv2") / pl.col("_cum_vol") - pl.col("sig_vwap_vwap") ** 2)
        .clip(lower_bound=0.0)
        .sqrt()
        .alias("sig_vwap_sd"),
    )

    # Deviation SD
    df = df.with_columns(
        pl.when(pl.col("sig_vwap_sd") > 0)
        .then((pl.col("close") - pl.col("sig_vwap_vwap")) / pl.col("sig_vwap_sd"))
        .otherwise(0.0)
        .alias("sig_vwap_value"),
    )

    # Direction
    df = df.with_columns(
        pl.when((pl.col("sig_vwap_sd") > 0) & (pl.col("sig_vwap_value").abs() >= 1.0))
        .then(
            pl.when(pl.col("sig_vwap_value") < 0)
            .then(pl.lit("long"))
            .otherwise(pl.lit("short"))
        )
        .otherwise(pl.lit("none"))
        .alias("sig_vwap_direction"),
    )

    # Slope + first kiss + mode via numpy (session-grouped)
    vwap_col = df["sig_vwap_vwap"].to_numpy()
    close_col = df["close"].to_numpy()
    sd_col = df["sig_vwap_sd"].to_numpy()
    dev_col = df["sig_vwap_value"].to_numpy()
    dates = df["_bar_date"].to_list()
    session_age = df["sig_vwap_session_age"].to_numpy()

    n = len(df)
    slopes = np.zeros(n, dtype=np.float64)
    first_kiss = np.zeros(n, dtype=bool)

    # Process per-session
    session_start = 0
    for i in range(n):
        # Detect session boundary
        if i > 0 and dates[i] != dates[i - 1]:
            session_start = i

        # Slope: linear regression of running VWAP over last slope_window bars within session
        sess_len = i - session_start + 1
        win = min(slope_window, sess_len)
        if win >= 2:
            tail = vwap_col[i - win + 1 : i + 1]
            x = np.arange(win, dtype=np.float64)
            x_mean = x.mean()
            y_mean = tail.mean()
            dx = x - x_mean
            dy = tail - y_mean
            den = float(np.sum(dx * dx))
            if den > 0:
                slopes[i] = float(np.sum(dx * dy)) / den

        # First kiss: current deviation < 0.5 SD, and any of prior lookback bars >= threshold
        if sd_col[i] > 0 and abs(dev_col[i]) < 0.5:
            lb_start = max(session_start, i - first_kiss_lookback)
            if lb_start < i:
                prior_devs = np.abs(dev_col[lb_start:i])
                if np.any(prior_devs >= first_kiss_sd_threshold):
                    first_kiss[i] = True

    # Mode
    abs_slopes = np.abs(slopes)
    mode = np.where(
        abs_slopes < flat_slope_threshold,
        "REVERSION",
        np.where(abs_slopes > trending_slope_threshold, "PULLBACK", "NEUTRAL"),
    )

    df = df.with_columns(
        pl.Series("sig_vwap_slope", slopes),
        pl.Series("sig_vwap_first_kiss", first_kiss),
        pl.Series("sig_vwap_mode", mode),
    )

    # Drop temp columns
    df = df.drop(["_tp", "_cum_pv", "_cum_vol", "_cum_pv2"])

    return df


# ── Relative Volume ──────────────────────────────────────────────────────


def _compute_relative_volume(
    df: pl.DataFrame,
    lookback: int = 20,
    high_threshold: float = 1.5,
) -> pl.DataFrame:
    """Add relative volume columns: sig_rvol_value, sig_rvol_passes."""
    vol = df["volume"].to_numpy().astype(np.float64)
    n = len(vol)

    rvol = np.ones(n, dtype=np.float64)
    passes = np.zeros(n, dtype=bool)

    for i in range(lookback, n):
        prior_mean = float(np.mean(vol[i - lookback + 1 : i]))
        if prior_mean > 1e-9:
            rvol[i] = vol[i] / prior_mean
            passes[i] = rvol[i] >= high_threshold

    return df.with_columns(
        pl.Series("sig_rvol_value", rvol),
        pl.Series("sig_rvol_passes", passes),
    )


# ── Spread ────────────────────────────────────────────────────────────────


def _compute_spread(
    df: pl.DataFrame,
    z_threshold: float = 2.0,
    min_bars: int = 30,
) -> pl.DataFrame:
    """Add spread columns: sig_spread_value, sig_spread_passes."""
    # Check if bid/ask price columns exist
    has_bid_ask = (
        "avg_bid_price" in df.columns
        and "avg_ask_price" in df.columns
    )

    if not has_bid_ask:
        # No bid/ask data — mark as unavailable so FilterEngine blocks
        _log.warning(
            "Spread signal requires avg_bid_price/avg_ask_price columns but "
            "bars lack L1 data. Spread filter will block all bars."
        )
        return df.with_columns(
            pl.lit(0.0).alias("sig_spread_value"),
            pl.lit(False).alias("sig_spread_passes"),
            pl.lit(True).alias("sig_spread_unavailable"),
        )

    # Compute spread and z-score
    df = df.with_columns(
        (pl.col("avg_ask_price") - pl.col("avg_bid_price")).alias("_raw_spread"),
    )

    # Rolling mean and std of valid spreads
    spread = df["_raw_spread"].to_numpy()
    n = len(spread)
    z_scores = np.zeros(n, dtype=np.float64)
    passes = np.ones(n, dtype=bool)

    valid_spreads: list[float] = []
    for i in range(n):
        s = spread[i]
        if s is not None and s > 0:
            valid_spreads.append(s)

        if len(valid_spreads) >= min_bars:
            arr = np.array(valid_spreads, dtype=np.float64)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1))
            if std > 0:
                z = (valid_spreads[-1] - mean) / std
                z_scores[i] = z
                passes[i] = abs(z) < z_threshold

    df = df.with_columns(
        pl.Series("sig_spread_value", z_scores),
        pl.Series("sig_spread_passes", passes),
    )
    df = df.drop(["_raw_spread"])
    return df


# ── ADX ──────────────────────────────────────────────────────────────────


def _compute_adx(
    df: pl.DataFrame,
    period: int = 14,
    threshold: float = 25.0,
) -> pl.DataFrame:
    """Add ADX columns: sig_adx_value, sig_adx_passes, sig_adx_direction, sig_adx_plus_di, sig_adx_minus_di."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    if n < 2:
        return df.with_columns(
            pl.lit(0.0).alias("sig_adx_value"),
            pl.lit(False).alias("sig_adx_passes"),
            pl.lit("none").alias("sig_adx_direction"),
            pl.lit(0.0).alias("sig_adx_plus_di"),
            pl.lit(0.0).alias("sig_adx_minus_di"),
        )

    # True Range, +DM, -DM (from bar 1 onward)
    tr = np.empty(n - 1, dtype=np.float64)
    plus_dm = np.empty(n - 1, dtype=np.float64)
    minus_dm = np.empty(n - 1, dtype=np.float64)

    for i in range(1, n):
        tr[i - 1] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i - 1] = up if (up > down and up > 0) else 0.0
        minus_dm[i - 1] = down if (down > up and down > 0) else 0.0

    # Wilder smooth
    atr_s = _wilder_smooth(tr, period)
    plus_di_s = _wilder_smooth(plus_dm, period)
    minus_di_s = _wilder_smooth(minus_dm, period)

    # +DI and -DI
    plus_di = np.where(atr_s > 0, 100.0 * plus_di_s / atr_s, 0.0)
    minus_di = np.where(atr_s > 0, 100.0 * minus_di_s / atr_s, 0.0)

    # DX -> ADX
    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)
    adx = _wilder_smooth(dx, period)

    # Pad first bar with zeros (signals start from bar 1)
    adx_full = np.zeros(n, dtype=np.float64)
    adx_full[1:] = adx
    plus_di_full = np.zeros(n, dtype=np.float64)
    plus_di_full[1:] = plus_di
    minus_di_full = np.zeros(n, dtype=np.float64)
    minus_di_full[1:] = minus_di

    passes = adx_full >= threshold
    direction = np.where(plus_di_full > minus_di_full, "long",
                         np.where(minus_di_full > plus_di_full, "short", "none"))

    return df.with_columns(
        pl.Series("sig_adx_value", adx_full),
        pl.Series("sig_adx_passes", passes),
        pl.Series("sig_adx_direction", direction),
        pl.Series("sig_adx_plus_di", plus_di_full),
        pl.Series("sig_adx_minus_di", minus_di_full),
    )


# ── Donchian Channel ────────────────────────────────────────────────────


def _compute_donchian_channel(
    df: pl.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
) -> pl.DataFrame:
    """Add Donchian channel columns: sig_dc_value (width), sig_dc_passes, sig_dc_direction, etc."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    min_bars = max(entry_period, exit_period) + 1

    entry_upper = np.full(n, np.nan, dtype=np.float64)
    entry_lower = np.full(n, np.nan, dtype=np.float64)
    exit_upper = np.full(n, np.nan, dtype=np.float64)
    exit_lower = np.full(n, np.nan, dtype=np.float64)
    width = np.zeros(n, dtype=np.float64)
    passes = np.zeros(n, dtype=bool)
    direction = np.full(n, "none", dtype=object)

    for i in range(min_bars, n):
        # Channels from N bars BEFORE current bar (exclude current)
        eu = float(np.max(high[i - entry_period:i]))
        el = float(np.min(low[i - entry_period:i]))
        xu = float(np.max(high[i - exit_period:i]))
        xl = float(np.min(low[i - exit_period:i]))

        entry_upper[i] = eu
        entry_lower[i] = el
        exit_upper[i] = xu
        exit_lower[i] = xl
        width[i] = eu - el

        c = close[i]
        if c > eu:
            passes[i] = True
            direction[i] = "long"
        elif c < el:
            passes[i] = True
            direction[i] = "short"

    return df.with_columns(
        pl.Series("sig_dc_value", width),
        pl.Series("sig_dc_passes", passes),
        pl.Series("sig_dc_direction", direction.astype(str)),
        pl.Series("sig_dc_entry_upper", entry_upper),
        pl.Series("sig_dc_entry_lower", entry_lower),
        pl.Series("sig_dc_exit_upper", exit_upper),
        pl.Series("sig_dc_exit_lower", exit_lower),
    )


# ── Session Time ────────────────────────────────────────────────────────


def _compute_session_time(df: pl.DataFrame) -> pl.DataFrame:
    """Add session_time column: minutes since midnight in US/Eastern."""
    return df.with_columns(
        (pl.col("_et_ts").dt.hour().cast(pl.Int32) * 60 + pl.col("_et_ts").dt.minute().cast(pl.Int32))
        .cast(pl.Float64)
        .alias("sig_session_time_value"),
    )


# ── Regime V2 ──────────────────────────────────────────────────────────


def _compute_regime_v2(df: pl.DataFrame) -> pl.DataFrame:
    """Add regime_v2 columns using batch HMM inference.

    Loads the trained RegimeDetectorV2 model and runs predict_proba_sequence
    over all bars.  Warm-up rows (where features can't be computed) get
    regime=1 (RANGING), confidence=0, position_size="flat".
    """
    from src.models.regime_detector_v2 import (
        RegimeDetectorV2,
        build_features_v2,
    )

    detector = RegimeDetectorV2.load("models/regime_v2")
    features, feat_ts = build_features_v2(df, detector.config)
    probas = detector.predict_proba_sequence(features)

    n_total = len(df)
    n_feat = len(features)
    warmup = n_total - n_feat  # rows dropped by feature extraction

    # Pre-fill arrays for warmup rows
    regime = np.full(n_total, 1, dtype=np.int32)  # RANGING default
    confidence = np.zeros(n_total, dtype=np.float64)
    p_trending = np.zeros(n_total, dtype=np.float64)
    p_ranging = np.zeros(n_total, dtype=np.float64)
    p_high_vol = np.zeros(n_total, dtype=np.float64)
    position_size = np.full(n_total, "flat", dtype=object)
    whipsaw_halt = np.full(n_total, True, dtype=bool)

    for i, p in enumerate(probas):
        idx = warmup + i
        regime[idx] = p.regime.value
        confidence[idx] = p.confidence
        p_trending[idx] = p.probabilities[0]
        p_ranging[idx] = p.probabilities[1]
        p_high_vol[idx] = p.probabilities[2]
        position_size[idx] = p.position_size
        whipsaw_halt[idx] = p.whipsaw_halt

    return df.with_columns(
        pl.Series("sig_regime_v2_value", regime),
        pl.Series("sig_regime_v2_confidence", confidence),
        pl.Series("sig_regime_v2_p_trending", p_trending),
        pl.Series("sig_regime_v2_p_ranging", p_ranging),
        pl.Series("sig_regime_v2_p_high_vol", p_high_vol),
        pl.Series("sig_regime_v2_position_size", position_size.astype(str)),
        pl.Series("sig_regime_v2_whipsaw_halt", whipsaw_halt),
    )
