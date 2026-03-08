"""Spread mean reversion filter — blocks entries during anomalous spread conditions.

Maintains a rolling window of bid-ask spreads and flags when the current spread
deviates beyond a configurable z-score threshold from the rolling mean.
Periodically flushes state snapshots to Parquet files.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True)
class SpreadSnapshot:
    """Single point-in-time bid/ask observation."""

    timestamp: datetime
    bid: float
    ask: float

    @property
    def spread(self) -> float:
        """Raw spread in price units."""
        return self.ask - self.bid


@dataclass(frozen=True)
class SpreadState:
    """Computed state after each spread push."""

    timestamp: datetime
    current_spread: float
    rolling_mean: float
    rolling_std: float
    z_score: float
    is_normal: bool


@dataclass
class SpreadConfig:
    """Tuning knobs for the spread monitor."""

    z_threshold: float = 2.0
    maxlen: int = 500
    persist_every_n: int = 10
    min_samples: int = 30
    parquet_dir: str = "data/spread_states"


@dataclass
class SpreadMonitor:
    """Rolling spread monitor with z-score gating and Parquet persistence.

    Args:
        config: Tuning parameters.
        persist: Whether to flush state snapshots to Parquet files.
    """

    config: SpreadConfig = field(default_factory=SpreadConfig)
    persist: bool = False

    def __post_init__(self) -> None:
        self._buffer: deque[float] = deque(maxlen=self.config.maxlen)
        self._push_count: int = 0
        self._latest_state: SpreadState | None = None
        self._pending: list[SpreadState] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def push(self, snapshot: SpreadSnapshot) -> SpreadState:
        """Ingest a snapshot, compute z-score, optionally persist.

        Args:
            snapshot: A bid/ask observation.

        Returns:
            Computed SpreadState for this tick.
        """
        spread = snapshot.spread
        self._buffer.append(spread)
        self._push_count += 1

        # Compute rolling stats via numpy (fast on small arrays)
        arr = np.array(self._buffer)
        rolling_mean = float(np.mean(arr))
        rolling_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        # Z-score
        if rolling_std > 0:
            z_score = (spread - rolling_mean) / rolling_std
        else:
            z_score = 0.0

        is_normal = self._evaluate_normal(spread, rolling_mean, rolling_std, z_score)

        state = SpreadState(
            timestamp=snapshot.timestamp,
            current_spread=spread,
            rolling_mean=rolling_mean,
            rolling_std=rolling_std,
            z_score=z_score,
            is_normal=is_normal,
        )
        self._latest_state = state

        # Batch for periodic Parquet flush
        if self.persist:
            self._pending.append(state)
            if self._push_count % self.config.persist_every_n == 0:
                await self._flush_parquet()

        return state

    def push_sync(self, snapshot: SpreadSnapshot) -> SpreadState:
        """Synchronous push for backtesting (no Parquet persistence)."""
        spread = snapshot.spread
        self._buffer.append(spread)
        self._push_count += 1

        arr = np.array(self._buffer)
        rolling_mean = float(np.mean(arr))
        rolling_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        if rolling_std > 0:
            z_score = (spread - rolling_mean) / rolling_std
        else:
            z_score = 0.0

        is_normal = self._evaluate_normal(spread, rolling_mean, rolling_std, z_score)

        state = SpreadState(
            timestamp=snapshot.timestamp,
            current_spread=spread,
            rolling_mean=rolling_mean,
            rolling_std=rolling_std,
            z_score=z_score,
            is_normal=is_normal,
        )
        self._latest_state = state
        return state

    def is_spread_normal(self) -> tuple[bool, str]:
        """Check whether current spread conditions allow trading.

        Returns:
            (is_normal, reason) — reason is empty string when normal.
        """
        if self._latest_state is None:
            return False, "no data yet"

        state = self._latest_state

        if state.current_spread <= 0:
            return False, "spread is zero or negative (data error)"

        if len(self._buffer) < self.config.min_samples:
            return True, ""  # not enough history to filter

        if not state.is_normal:
            return (
                False,
                f"spread {state.current_spread:.4f} exceeds "
                f"mean+{self.config.z_threshold}sd "
                f"(z={state.z_score:.2f})",
            )

        return True, ""

    @property
    def buffer_size(self) -> int:
        """Current number of spread observations in buffer."""
        return len(self._buffer)

    @property
    def latest_state(self) -> SpreadState | None:
        """Most recently computed state, or None if no data pushed."""
        return self._latest_state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate_normal(
        self,
        spread: float,
        rolling_mean: float,
        rolling_std: float,
        z_score: float,
    ) -> bool:
        """Determine if spread is within normal bounds."""
        if spread <= 0:
            return False
        if len(self._buffer) < self.config.min_samples:
            return True  # insufficient data — don't filter
        if rolling_std > 0 and z_score > self.config.z_threshold:
            return False
        return True

    async def _flush_parquet(self) -> None:
        """Append pending states to a date-partitioned Parquet file."""
        if not self._pending:
            return

        rows = [
            {
                "timestamp": s.timestamp,
                "current_spread": s.current_spread,
                "rolling_mean": s.rolling_mean,
                "rolling_std": s.rolling_std,
                "z_score": s.z_score,
                "is_normal": s.is_normal,
            }
            for s in self._pending
        ]
        df = pl.DataFrame(rows)

        # Partition by date
        date_str = self._pending[-1].timestamp.strftime("%Y-%m-%d")
        out_dir = Path(self.config.parquet_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"spread_states_{date_str}.parquet"

        if out_path.exists():
            existing = pl.read_parquet(out_path)
            df = pl.concat([existing, df])

        df.write_parquet(out_path, compression="zstd")
        self._pending.clear()
