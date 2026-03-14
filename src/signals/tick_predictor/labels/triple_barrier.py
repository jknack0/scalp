"""Triple-barrier labeler for TickDirectionPredictor training data.

Adapted from Lopez de Prado's *Advances in Financial Machine Learning*.
Generates direction labels (-1/0/+1) with sample uniqueness weights from
1s OHLCV bar Parquet files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.core.logging import get_logger

logger = get_logger("tick_predictor.labels")

MES_TICK_SIZE = 0.25


@dataclass(frozen=True)
class TripleBarrierConfig:
    vertical_barrier_bars: int = 15
    tp_ticks: int = 4
    sl_ticks: int = 3
    volatility_scale: bool = True
    vol_lookback_bars: int = 50
    min_vol_scale: float = 0.5
    max_vol_scale: float = 2.5
    tick_size: float = MES_TICK_SIZE


class TripleBarrierLabeler:
    """Generate triple-barrier labels from 1s bar Parquet."""

    def __init__(self, config: TripleBarrierConfig | None = None) -> None:
        self.config = config or TripleBarrierConfig()

    def generate_labels(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Generate labels for date range.

        Args:
            start_date: ISO date string e.g. "2024-01-01"
            end_date: ISO date string e.g. "2025-03-01"

        Returns:
            Polars DataFrame with columns: timestamp_ns, label, sample_weight,
            tp_points_actual, sl_points_actual, exit_bar_offset, exit_reason
        """
        cfg = self.config

        # Load 1s bars
        df = self._load_bars(start_date, end_date)
        return self.generate_labels_from_bars(df)

    def generate_labels_from_bars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate labels from a pre-loaded bars DataFrame.

        Expects columns: timestamp_ns, close (at minimum).
        """
        cfg = self.config
        logger.info("labels_bars_loaded", rows=len(df))

        closes = df["close"].to_numpy()
        timestamps = df["timestamp_ns"].to_numpy()
        n = len(closes)

        # Compute realized vol for dynamic barriers
        log_rets = np.empty(n, dtype=np.float64)
        log_rets[0] = 0.0
        log_rets[1:] = np.log(closes[1:] / closes[:-1])

        # Rolling std
        vol_lb = cfg.vol_lookback_bars
        realized_vol = np.full(n, np.nan, dtype=np.float64)
        for i in range(vol_lb, n):
            realized_vol[i] = float(np.std(log_rets[i - vol_lb + 1 : i + 1], ddof=1))

        # Compute dynamic barriers
        tp_points = np.full(n, cfg.tp_ticks * cfg.tick_size, dtype=np.float64)
        sl_points = np.full(n, cfg.sl_ticks * cfg.tick_size, dtype=np.float64)

        if cfg.volatility_scale:
            valid_mask = ~np.isnan(realized_vol)
            if np.any(valid_mask):
                median_vol = float(np.nanmedian(realized_vol))
                if median_vol > 1e-12:
                    vol_factor = np.clip(
                        realized_vol / median_vol,
                        cfg.min_vol_scale,
                        cfg.max_vol_scale,
                    )
                    vol_factor = np.where(np.isnan(vol_factor), 1.0, vol_factor)
                    tp_points = tp_points * vol_factor
                    sl_points = sl_points * vol_factor

        # Triple barrier scan
        labels = np.zeros(n, dtype=np.int8)
        exit_offsets = np.zeros(n, dtype=np.int16)
        exit_reasons = [""] * n
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
                if closes[j] >= upper:
                    label = 1
                    offset = j - i
                    reason = "tp"
                    break
                if closes[j] <= lower:
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
                    else:
                        label = 0

            labels[i] = label
            exit_offsets[i] = offset
            exit_reasons[i] = reason

        # Last bar — can't label
        exit_reasons[-1] = "end"

        # Compute sample uniqueness weights
        weights = self._compute_sample_weights(n, exit_offsets, vb)

        result = pl.DataFrame({
            "timestamp_ns": timestamps,
            "label": labels,
            "sample_weight": weights.astype(np.float32),
            "tp_points_actual": tp_actual,
            "sl_points_actual": sl_actual,
            "exit_bar_offset": exit_offsets,
            "exit_reason": exit_reasons,
        })

        # Drop rows where vol is NaN (warmup period)
        if cfg.volatility_scale:
            result = result.slice(vol_lb)

        # Log distribution
        dist = result.group_by("label").len().sort("label")
        total = len(result)
        for row in dist.iter_rows(named=True):
            pct = row["len"] / total * 100 if total > 0 else 0
            logger.info("label_distribution",
                        label=int(row["label"]),
                        count=row["len"],
                        pct=f"{pct:.1f}%")

        return result

    def save_labels(self, df: pl.DataFrame, start_date: str, end_date: str) -> Path:
        """Write labels to Parquet."""
        out_dir = Path("data/tick_predictor")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"labels_{start_date}_{end_date}.parquet"
        df.write_parquet(path)
        logger.info("labels_saved", path=str(path), rows=len(df))
        return path

    # ── internals ───────────────────────────────────────────────

    def _load_bars(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Load 1s bars from year-partitioned Parquet."""
        import datetime as dt

        start = dt.date.fromisoformat(start_date)
        end = dt.date.fromisoformat(end_date)

        # Build list of year partitions to scan
        years = range(start.year, end.year + 1)
        paths = [
            f"data/parquet/year={y}/data.parquet"
            for y in years
            if Path(f"data/parquet/year={y}/data.parquet").exists()
        ]

        if not paths:
            raise FileNotFoundError(
                f"No parquet files found for {start_date} to {end_date}"
            )

        start_dt = dt.datetime.combine(start, dt.time.min)
        end_dt = dt.datetime.combine(end, dt.time.min)

        df = (
            pl.scan_parquet(paths)
            .filter(
                (pl.col("timestamp") >= start_dt)
                & (pl.col("timestamp") < end_dt)
            )
            .select(["timestamp", "open", "high", "low", "close", "volume"])
            .collect()
        )
        # Add timestamp_ns for downstream joins
        df = df.with_columns(
            (pl.col("timestamp").dt.epoch("ns")).alias("timestamp_ns")
        )
        return df

    @staticmethod
    def _compute_sample_weights(
        n: int, exit_offsets: np.ndarray, max_horizon: int
    ) -> np.ndarray:
        """Compute sample uniqueness weights.

        weight[i] = 1 / avg_concurrent_labels at bar i.
        Normalized so mean ~ 1.0.
        """
        # Count concurrent labels at each bar
        concurrent = np.zeros(n, dtype=np.float64)
        for i in range(n):
            end = min(i + int(exit_offsets[i]), n)
            for j in range(i, end):
                concurrent[j] += 1.0

        # Weight = inverse of average concurrency during label's life
        weights = np.ones(n, dtype=np.float64)
        for i in range(n):
            end = min(i + int(exit_offsets[i]), n)
            if end > i:
                avg_c = float(np.mean(concurrent[i:end]))
                weights[i] = 1.0 / avg_c if avg_c > 0 else 1.0

        # Normalize to mean=1.0
        mean_w = float(np.mean(weights))
        if mean_w > 0:
            weights /= mean_w

        return weights
