"""Bar resampling utilities: 1s bars → any timeframe + dollar bars."""

import polars as pl


def resample_bars(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    """Resample 1-second OHLCV+VWAP bars to a larger timeframe.

    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume, vwap
        freq: Polars duration string, e.g. "1m", "5m", "15m", "1h"

    Returns:
        Resampled DataFrame with same columns.
    """
    return (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every=freq)
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            # Volume-weighted VWAP
            (pl.col("vwap") * pl.col("volume")).sum().alias("_vwap_vol"),
            pl.col("volume").sum().alias("_vol_sum"),
        ])
        .with_columns(
            (pl.col("_vwap_vol") / pl.col("_vol_sum")).alias("vwap")
        )
        .drop(["_vwap_vol", "_vol_sum"])
        .sort("timestamp")
    )


def build_dollar_bars(df: pl.DataFrame, dollar_threshold: float) -> pl.DataFrame:
    """Construct dollar bars from 1-second bar data.

    A new bar is formed each time cumulative dollar volume crosses the threshold.
    Dollar volume per 1s bar = close × volume × 5.0 (MES point value).

    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume, vwap
        dollar_threshold: Dollar volume per bar (e.g., 50_000 for MES).

    Returns:
        DataFrame of dollar bars with columns:
        timestamp, open, high, low, close, volume, vwap, dollar_volume, bar_duration_s
    """
    MES_POINT_VALUE = 5.0

    df = df.sort("timestamp")

    if df.is_empty():
        return pl.DataFrame(schema={
            "timestamp": pl.Datetime,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
            "vwap": pl.Float64,
            "dollar_volume": pl.Float64,
            "bar_duration_s": pl.Float64,
        })

    timestamps = df["timestamp"].to_list()
    opens = df["open"].to_list()
    highs = df["high"].to_list()
    lows = df["low"].to_list()
    closes = df["close"].to_list()
    volumes = df["volume"].to_list()
    vwaps = df["vwap"].to_list()

    bars = []
    bar_open = opens[0]
    bar_high = highs[0]
    bar_low = lows[0]
    bar_ts_start = timestamps[0]
    cum_dollar_vol = 0.0
    cum_volume = 0
    cum_vwap_vol = 0.0

    for i in range(len(timestamps)):
        dollar_vol = closes[i] * volumes[i] * MES_POINT_VALUE
        cum_dollar_vol += dollar_vol
        cum_volume += volumes[i]
        cum_vwap_vol += vwaps[i] * volumes[i]
        bar_high = max(bar_high, highs[i])
        bar_low = min(bar_low, lows[i])

        if cum_dollar_vol >= dollar_threshold:
            duration = (timestamps[i] - bar_ts_start).total_seconds()
            bar_vwap = cum_vwap_vol / cum_volume if cum_volume > 0 else closes[i]
            bars.append({
                "timestamp": bar_ts_start,
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "close": closes[i],
                "volume": cum_volume,
                "vwap": bar_vwap,
                "dollar_volume": cum_dollar_vol,
                "bar_duration_s": duration,
            })
            # Reset for next bar
            if i + 1 < len(timestamps):
                bar_open = opens[i + 1] if i + 1 < len(opens) else closes[i]
                bar_high = 0.0
                bar_low = float("inf")
                bar_ts_start = timestamps[i + 1] if i + 1 < len(timestamps) else timestamps[i]
            cum_dollar_vol = 0.0
            cum_volume = 0
            cum_vwap_vol = 0.0

    if not bars:
        return pl.DataFrame(schema={
            "timestamp": pl.Datetime,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
            "vwap": pl.Float64,
            "dollar_volume": pl.Float64,
            "bar_duration_s": pl.Float64,
        })

    return pl.DataFrame(bars)
