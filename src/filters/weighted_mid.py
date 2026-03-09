"""Weighted mid-price — size-weighted mid for VWAP reversion anchoring.

Computes a weighted mid-price from top-of-book sizes, tracks divergence
from raw mid as a directional hint, and provides a drop-in VWAP anchor.
Flushes snapshots to date-partitioned Parquet files.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl


@dataclass(frozen=True)
class WeightedMidSnapshot:
    """Top-of-book quote with sizes for weighted mid calculation."""

    timestamp: datetime
    bid: float
    ask: float
    bid_size_top: int
    ask_size_top: int

    @property
    def raw_mid(self) -> float:
        """Simple arithmetic mid-price."""
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class WeightedMidSignal:
    """Computed weighted mid state after each push."""

    timestamp: datetime
    raw_mid: float
    weighted_mid: float
    divergence: float
    divergence_z_score: float
    lean: Literal["up", "down", "neutral"]


@dataclass
class WeightedMidConfig:
    """Tuning knobs for the weighted mid monitor."""

    maxlen: int = 200
    lean_threshold: float = 1.0
    persist_every_n: int = 5
    parquet_dir: str = "data/weighted_mid"


@dataclass
class WeightedMidMonitor:
    """Rolling weighted mid-price monitor with Parquet persistence.

    Args:
        config: Tuning parameters.
        persist: Whether to flush signal snapshots to Parquet files.
    """

    config: WeightedMidConfig = field(default_factory=WeightedMidConfig)
    persist: bool = False

    def __post_init__(self) -> None:
        self._divergences: deque[float] = deque(maxlen=self.config.maxlen)
        self._push_count: int = 0
        self._latest_signal: WeightedMidSignal | None = None
        self._pending: list[WeightedMidSignal] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def push(self, snapshot: WeightedMidSnapshot) -> WeightedMidSignal:
        """Ingest a snapshot, compute weighted mid and divergence.

        Args:
            snapshot: Top-of-book quote with sizes.

        Returns:
            Computed WeightedMidSignal for this tick.
        """
        self._push_count += 1

        raw_mid = snapshot.raw_mid
        weighted_mid = self._compute_weighted_mid(snapshot)
        divergence = weighted_mid - raw_mid

        self._divergences.append(divergence)

        # Rolling z-score of divergence
        arr = np.array(self._divergences)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        z_score = (divergence - mean) / std if std > 0 else 0.0

        # Directional lean based on z-score threshold
        if z_score > self.config.lean_threshold:
            lean: Literal["up", "down", "neutral"] = "up"
        elif z_score < -self.config.lean_threshold:
            lean = "down"
        else:
            lean = "neutral"

        signal = WeightedMidSignal(
            timestamp=snapshot.timestamp,
            raw_mid=raw_mid,
            weighted_mid=weighted_mid,
            divergence=divergence,
            divergence_z_score=z_score,
            lean=lean,
        )
        self._latest_signal = signal

        if self.persist:
            self._pending.append(signal)
            if self._push_count % self.config.persist_every_n == 0:
                self._flush_parquet()

        return signal

    def push_sync(self, snapshot: WeightedMidSnapshot) -> WeightedMidSignal:
        """Synchronous push for backtesting (no Parquet persistence)."""
        self._push_count += 1

        raw_mid = snapshot.raw_mid
        weighted_mid = self._compute_weighted_mid(snapshot)
        divergence = weighted_mid - raw_mid

        self._divergences.append(divergence)

        arr = np.array(self._divergences)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        z_score = (divergence - mean) / std if std > 0 else 0.0

        if z_score > self.config.lean_threshold:
            lean: Literal["up", "down", "neutral"] = "up"
        elif z_score < -self.config.lean_threshold:
            lean = "down"
        else:
            lean = "neutral"

        signal = WeightedMidSignal(
            timestamp=snapshot.timestamp,
            raw_mid=raw_mid,
            weighted_mid=weighted_mid,
            divergence=divergence,
            divergence_z_score=z_score,
            lean=lean,
        )
        self._latest_signal = signal
        return signal

    def get_vwap_anchor(self) -> float:
        """Return the best mid-price estimate for VWAP reversion anchoring.

        Uses weighted_mid when divergence z-score is significant (abs > 1.0),
        otherwise falls back to raw_mid.

        Returns:
            Weighted mid or raw mid price.

        Raises:
            RuntimeError: If no data has been pushed yet.
        """
        if self._latest_signal is None:
            raise RuntimeError("no data pushed yet")

        if abs(self._latest_signal.divergence_z_score) > self.config.lean_threshold:
            return self._latest_signal.weighted_mid
        return self._latest_signal.raw_mid

    @property
    def buffer_size(self) -> int:
        """Current number of divergence observations in buffer."""
        return len(self._divergences)

    @property
    def latest_signal(self) -> WeightedMidSignal | None:
        """Most recently computed signal, or None if no data pushed."""
        return self._latest_signal

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_weighted_mid(snapshot: WeightedMidSnapshot) -> float:
        """Compute size-weighted mid-price.

        weighted_mid = (ask_size * bid + bid_size * ask) / (bid_size + ask_size)

        Falls back to raw mid when both sizes are zero.
        """
        total_size = snapshot.bid_size_top + snapshot.ask_size_top
        if total_size == 0:
            return snapshot.raw_mid
        return (
            snapshot.ask_size_top * snapshot.bid + snapshot.bid_size_top * snapshot.ask
        ) / total_size

    def _flush_parquet(self) -> None:
        """Append pending signals to a date-partitioned Parquet file."""
        if not self._pending:
            return

        rows = [
            {
                "timestamp": s.timestamp,
                "raw_mid": s.raw_mid,
                "weighted_mid": s.weighted_mid,
                "divergence": s.divergence,
                "divergence_z_score": s.divergence_z_score,
                "lean": s.lean,
            }
            for s in self._pending
        ]
        df = pl.DataFrame(rows)

        date_str = self._pending[-1].timestamp.strftime("%Y-%m-%d")
        out_dir = Path(self.config.parquet_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"weighted_mid_{date_str}.parquet"

        if out_path.exists():
            existing = pl.read_parquet(out_path)
            df = pl.concat([existing, df])

        df.write_parquet(out_path, compression="zstd")
        self._pending.clear()
