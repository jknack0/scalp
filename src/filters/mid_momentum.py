"""Mid-price momentum signal — detects persistent quote-level drift before candle close.

Computes linear regression slope of mid-price over a rolling window, normalized
by rolling stddev to produce a dimensionless drift score. Flushes snapshots to
date-partitioned Parquet files.
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
class MidSnapshot:
    """Single point-in-time bid/ask observation."""

    timestamp: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        """Mid-price."""
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class MomentumSignal:
    """Computed momentum state after each mid push."""

    timestamp: datetime
    mid: float
    slope: float
    drift_score: float
    direction: Literal["up", "down", "neutral"]
    strength: float


@dataclass
class MomentumConfig:
    """Tuning knobs for the mid-momentum monitor."""

    regression_window: int = 20
    neutral_threshold: float = 0.3
    maxlen: int = 100
    persist_every_n: int = 5
    parquet_dir: str = "data/mid_momentum"


@dataclass
class MidMomentumMonitor:
    """Rolling mid-price momentum monitor with Parquet persistence.

    Args:
        config: Tuning parameters.
        persist: Whether to flush signal snapshots to Parquet files.
    """

    config: MomentumConfig = field(default_factory=MomentumConfig)
    persist: bool = False

    def __post_init__(self) -> None:
        self._buffer: deque[MidSnapshot] = deque(maxlen=self.config.maxlen)
        self._push_count: int = 0
        self._latest_signal: MomentumSignal | None = None
        self._pending: list[MomentumSignal] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def push(self, snapshot: MidSnapshot) -> MomentumSignal:
        """Ingest a snapshot, compute momentum signal, optionally persist.

        Args:
            snapshot: A bid/ask observation.

        Returns:
            Computed MomentumSignal for this tick.
        """
        self._buffer.append(snapshot)
        self._push_count += 1

        # Determine effective window (may be smaller than config at startup)
        window = min(len(self._buffer), self.config.regression_window)
        mids = np.array([s.mid for s in list(self._buffer)[-window:]])

        # Linear regression slope
        if window >= 2:
            x = np.arange(window, dtype=np.float64)
            slope = float(np.polyfit(x, mids, 1)[0])
        else:
            slope = 0.0

        # Normalize by rolling stddev of mid across the full buffer
        all_mids = np.array([s.mid for s in self._buffer])
        std = float(np.std(all_mids, ddof=1)) if len(all_mids) > 1 else 0.0

        drift_score = slope / std if std > 0 else 0.0

        # Direction and strength
        if abs(drift_score) < self.config.neutral_threshold:
            direction: Literal["up", "down", "neutral"] = "neutral"
        elif drift_score > 0:
            direction = "up"
        else:
            direction = "down"

        strength = min(abs(drift_score), 1.0)

        signal = MomentumSignal(
            timestamp=snapshot.timestamp,
            mid=snapshot.mid,
            slope=slope,
            drift_score=drift_score,
            direction=direction,
            strength=strength,
        )
        self._latest_signal = signal

        # Batch for periodic Parquet flush
        if self.persist:
            self._pending.append(signal)
            if self._push_count % self.config.persist_every_n == 0:
                await self._flush_parquet()

        return signal

    def push_sync(self, snapshot: MidSnapshot) -> MomentumSignal:
        """Synchronous push for backtesting (no Parquet persistence)."""
        self._buffer.append(snapshot)
        self._push_count += 1

        window = min(len(self._buffer), self.config.regression_window)
        mids = np.array([s.mid for s in list(self._buffer)[-window:]])

        if window >= 2:
            x = np.arange(window, dtype=np.float64)
            slope = float(np.polyfit(x, mids, 1)[0])
        else:
            slope = 0.0

        all_mids = np.array([s.mid for s in self._buffer])
        std = float(np.std(all_mids, ddof=1)) if len(all_mids) > 1 else 0.0
        drift_score = slope / std if std > 0 else 0.0

        if abs(drift_score) < self.config.neutral_threshold:
            direction: Literal["up", "down", "neutral"] = "neutral"
        elif drift_score > 0:
            direction = "up"
        else:
            direction = "down"

        strength = min(abs(drift_score), 1.0)

        signal = MomentumSignal(
            timestamp=snapshot.timestamp,
            mid=snapshot.mid,
            slope=slope,
            drift_score=drift_score,
            direction=direction,
            strength=strength,
        )
        self._latest_signal = signal
        return signal

    @property
    def buffer_size(self) -> int:
        """Current number of mid observations in buffer."""
        return len(self._buffer)

    @property
    def latest_signal(self) -> MomentumSignal | None:
        """Most recently computed signal, or None if no data pushed."""
        return self._latest_signal

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _flush_parquet(self) -> None:
        """Append pending signals to a date-partitioned Parquet file."""
        if not self._pending:
            return

        rows = [
            {
                "timestamp": s.timestamp,
                "mid": s.mid,
                "slope": s.slope,
                "drift_score": s.drift_score,
                "direction": s.direction,
                "strength": s.strength,
            }
            for s in self._pending
        ]
        df = pl.DataFrame(rows)

        date_str = self._pending[-1].timestamp.strftime("%Y-%m-%d")
        out_dir = Path(self.config.parquet_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"mid_momentum_{date_str}.parquet"

        if out_path.exists():
            existing = pl.read_parquet(out_path)
            df = pl.concat([existing, df])

        df.write_parquet(out_path, compression="zstd")
        self._pending.clear()
