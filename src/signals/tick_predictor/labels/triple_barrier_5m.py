"""Triple-barrier labeler for 5-minute bars.

Generates direction labels (DOWN=-1, UP=+1) from 5m OHLCV bars.
Uses high/low within each future bar for barrier touches (not just close).

Strict causality: label at bar t is determined by bars [t+1 .. t+horizon].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.core.logging import get_logger

logger = get_logger("tick_predictor.labels_5m")

MES_TICK_SIZE = 0.25


@dataclass(frozen=True)
class TripleBarrierConfig5M:
    """Config for 5m triple barrier labels.

    Default: 12-tick TP (3.00 pts), 8-tick SL (2.00 pts), 6-bar (30 min) horizon.
    1.5:1 reward:risk ratio.
    Volatility scaling adjusts barriers by realized vol / median vol.
    """
    vertical_barrier_bars: int = 6        # 6 bars = 30 minutes
    tp_ticks: int = 12                    # 12 ticks = 3.00 points
    sl_ticks: int = 8                     # 8 ticks = 2.00 points
    volatility_scale: bool = True
    vol_lookback_bars: int = 48           # 48 bars = 4 hours
    min_vol_scale: float = 0.5
    max_vol_scale: float = 3.0
    tick_size: float = MES_TICK_SIZE


class TripleBarrierLabeler5M:
    """Generate triple-barrier labels from 5m bars."""

    def __init__(self, config: TripleBarrierConfig5M | None = None) -> None:
        self.config = config or TripleBarrierConfig5M()

    def generate_labels_from_bars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate labels from a pre-loaded 5m bars DataFrame.

        Uses high/low of future bars to detect barrier touches intra-bar.
        """
        cfg = self.config
        logger.info("labels_5m_bars_loaded", rows=len(df))

        closes = df["close"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        timestamps = df["timestamp_ns"].to_numpy()
        n = len(closes)

        # Realized vol for dynamic barriers
        log_rets = np.empty(n, dtype=np.float64)
        log_rets[0] = 0.0
        log_rets[1:] = np.log(closes[1:] / closes[:-1])

        vol_lb = cfg.vol_lookback_bars
        realized_vol = np.full(n, np.nan, dtype=np.float64)
        for i in range(vol_lb, n):
            realized_vol[i] = float(np.std(log_rets[i - vol_lb + 1:i + 1], ddof=1))

        # Base barriers
        tp_base = cfg.tp_ticks * cfg.tick_size
        sl_base = cfg.sl_ticks * cfg.tick_size
        tp_points = np.full(n, tp_base, dtype=np.float64)
        sl_points = np.full(n, sl_base, dtype=np.float64)

        if cfg.volatility_scale:
            valid = ~np.isnan(realized_vol)
            if np.any(valid):
                median_vol = float(np.nanmedian(realized_vol))
                if median_vol > 1e-12:
                    scale = np.clip(
                        realized_vol / median_vol,
                        cfg.min_vol_scale,
                        cfg.max_vol_scale,
                    )
                    scale = np.where(np.isnan(scale), 1.0, scale)
                    tp_points *= scale
                    sl_points *= scale

        # Triple barrier scan using high/low
        labels = np.zeros(n, dtype=np.int8)
        exit_offsets = np.zeros(n, dtype=np.int16)
        exit_reasons = np.empty(n, dtype="U10")
        exit_reasons[:] = "end"
        tp_actual = np.zeros(n, dtype=np.float32)
        sl_actual = np.zeros(n, dtype=np.float32)
        vb = cfg.vertical_barrier_bars

        for i in range(n - 1):
            entry = closes[i]
            tp_pts = tp_points[i]
            sl_pts = sl_points[i]
            tp_actual[i] = tp_pts
            sl_actual[i] = sl_pts

            upper = entry + tp_pts
            lower = entry - sl_pts
            end_idx = min(i + vb, n - 1)

            label = 0
            offset = 0
            reason = "vertical"

            for j in range(i + 1, end_idx + 1):
                hit_tp = highs[j] >= upper
                hit_sl = lows[j] <= lower

                if hit_tp and hit_sl:
                    # Ambiguous — use close to decide
                    if closes[j] >= entry:
                        label = 1
                    else:
                        label = -1
                    offset = j - i
                    reason = "both"
                    break
                elif hit_tp:
                    label = 1
                    offset = j - i
                    reason = "tp"
                    break
                elif hit_sl:
                    label = -1
                    offset = j - i
                    reason = "sl"
                    break
            else:
                # Vertical barrier
                offset = end_idx - i
                if offset > 0:
                    move = closes[end_idx] - entry
                    if move > 0:
                        label = 1
                    elif move < 0:
                        label = -1

            labels[i] = label
            exit_offsets[i] = offset
            exit_reasons[i] = reason

        # Sample uniqueness weights
        weights = self._compute_sample_weights(n, exit_offsets)

        result = pl.DataFrame({
            "timestamp_ns": timestamps,
            "label": labels,
            "sample_weight": weights.astype(np.float32),
            "tp_points_actual": tp_actual,
            "sl_points_actual": sl_actual,
            "exit_bar_offset": exit_offsets,
            "exit_reason": exit_reasons,
        })

        # Drop warmup rows
        if cfg.volatility_scale:
            result = result.slice(vol_lb)

        # Log distribution
        dist = result.group_by("label").len().sort("label")
        total = len(result)
        for row in dist.iter_rows(named=True):
            pct = row["len"] / total * 100 if total > 0 else 0
            name = {-1: "DOWN", 0: "FLAT", 1: "UP"}.get(row["label"], "?")
            logger.info("label_5m_distribution",
                        label=name, count=row["len"], pct=f"{pct:.1f}%")

        return result

    def save_labels(self, df: pl.DataFrame, start_date: str, end_date: str,
                    suffix: str = "") -> Path:
        out_dir = Path("data/tick_predictor")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"labels_5m_{start_date}_{end_date}{suffix}.parquet"
        df.write_parquet(path)
        logger.info("labels_5m_saved", path=str(path), rows=len(df))
        return path

    @staticmethod
    def _compute_sample_weights(n: int, exit_offsets: np.ndarray) -> np.ndarray:
        """Sample uniqueness weights: 1 / avg concurrent labels."""
        concurrent = np.zeros(n, dtype=np.float64)
        for i in range(n):
            end = min(i + int(exit_offsets[i]), n)
            concurrent[i:end] += 1.0

        weights = np.ones(n, dtype=np.float64)
        for i in range(n):
            end = min(i + int(exit_offsets[i]), n)
            if end > i:
                avg_c = float(np.mean(concurrent[i:end]))
                weights[i] = 1.0 / avg_c if avg_c > 0 else 1.0

        mean_w = float(np.mean(weights))
        if mean_w > 0:
            weights /= mean_w
        return weights
