"""Depth deterioration signal — detects one-sided book thinning as a breakout precursor.

Tracks total bid/ask depth across top N levels, computes depth ratios against
rolling means, and flags thinning sides. Flushes signals to date-partitioned
Parquet files.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from src.filters import L2Snapshot


@dataclass(frozen=True)
class DepthSignal:
    """Computed depth deterioration state after each push."""

    timestamp: datetime
    bid_depth: int
    ask_depth: int
    bid_depth_ratio: float
    ask_depth_ratio: float
    bid_thinning: bool
    ask_thinning: bool
    breakout_lean: Literal["up", "down", "none"]


@dataclass
class DepthConfig:
    """Tuning knobs for the depth deterioration monitor."""

    depth_levels: int = 10
    thin_threshold: float = 0.6
    min_samples: int = 60
    maxlen: int = 300
    persist_every_n: int = 5
    parquet_dir: str = "data/depth_signals"


@dataclass
class DepthMonitor:
    """Rolling depth deterioration monitor with Parquet persistence.

    Args:
        config: Tuning parameters.
        persist: Whether to flush signal snapshots to Parquet files.
    """

    config: DepthConfig = field(default_factory=DepthConfig)
    persist: bool = False

    def __post_init__(self) -> None:
        self._bid_depths: deque[int] = deque(maxlen=self.config.maxlen)
        self._ask_depths: deque[int] = deque(maxlen=self.config.maxlen)
        self._push_count: int = 0
        self._latest_signal: DepthSignal | None = None
        self._pending: list[DepthSignal] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def push(self, snapshot: L2Snapshot) -> DepthSignal:
        """Ingest an L2 snapshot, compute depth ratios and thinning flags.

        Args:
            snapshot: A top-of-book depth snapshot.

        Returns:
            Computed DepthSignal for this update.
        """
        n = self.config.depth_levels
        bid_depth = sum(size for _, size in snapshot.bids[:n])
        ask_depth = sum(size for _, size in snapshot.asks[:n])

        self._bid_depths.append(bid_depth)
        self._ask_depths.append(ask_depth)
        self._push_count += 1

        # Rolling means
        bid_mean = float(np.mean(self._bid_depths))
        ask_mean = float(np.mean(self._ask_depths))

        bid_depth_ratio = bid_depth / bid_mean if bid_mean > 0 else 1.0
        ask_depth_ratio = ask_depth / ask_mean if ask_mean > 0 else 1.0

        # Thinning detection (only after enough baseline)
        has_baseline = len(self._bid_depths) >= self.config.min_samples
        bid_thinning = has_baseline and bid_depth_ratio < self.config.thin_threshold
        ask_thinning = has_baseline and ask_depth_ratio < self.config.thin_threshold

        # Breakout lean
        if ask_thinning and not bid_thinning:
            breakout_lean: Literal["up", "down", "none"] = "up"
        elif bid_thinning and not ask_thinning:
            breakout_lean = "down"
        else:
            breakout_lean = "none"

        signal = DepthSignal(
            timestamp=snapshot.timestamp,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            bid_depth_ratio=bid_depth_ratio,
            ask_depth_ratio=ask_depth_ratio,
            bid_thinning=bid_thinning,
            ask_thinning=ask_thinning,
            breakout_lean=breakout_lean,
        )
        self._latest_signal = signal

        if self.persist:
            self._pending.append(signal)
            if self._push_count % self.config.persist_every_n == 0:
                self._flush_parquet()

        return signal

    def push_sync(self, snapshot: L2Snapshot) -> DepthSignal:
        """Synchronous push for backtesting (no Parquet persistence)."""
        n = self.config.depth_levels
        bid_depth = sum(size for _, size in snapshot.bids[:n])
        ask_depth = sum(size for _, size in snapshot.asks[:n])

        self._bid_depths.append(bid_depth)
        self._ask_depths.append(ask_depth)
        self._push_count += 1

        bid_mean = float(np.mean(self._bid_depths))
        ask_mean = float(np.mean(self._ask_depths))

        bid_depth_ratio = bid_depth / bid_mean if bid_mean > 0 else 1.0
        ask_depth_ratio = ask_depth / ask_mean if ask_mean > 0 else 1.0

        has_baseline = len(self._bid_depths) >= self.config.min_samples
        bid_thinning = has_baseline and bid_depth_ratio < self.config.thin_threshold
        ask_thinning = has_baseline and ask_depth_ratio < self.config.thin_threshold

        if ask_thinning and not bid_thinning:
            breakout_lean: Literal["up", "down", "none"] = "up"
        elif bid_thinning and not ask_thinning:
            breakout_lean = "down"
        else:
            breakout_lean = "none"

        signal = DepthSignal(
            timestamp=snapshot.timestamp,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            bid_depth_ratio=bid_depth_ratio,
            ask_depth_ratio=ask_depth_ratio,
            bid_thinning=bid_thinning,
            ask_thinning=ask_thinning,
            breakout_lean=breakout_lean,
        )
        self._latest_signal = signal
        return signal

    def is_thinning(self, side: Literal["bid", "ask"]) -> bool:
        """Check whether the specified side of the book is thinning.

        Args:
            side: "bid" or "ask".

        Returns:
            True when depth_ratio is below thin_threshold and baseline exists.
        """
        if self._latest_signal is None:
            return False
        if side == "bid":
            return self._latest_signal.bid_thinning
        return self._latest_signal.ask_thinning

    @property
    def buffer_size(self) -> int:
        """Current number of depth observations in buffer."""
        return len(self._bid_depths)

    @property
    def latest_signal(self) -> DepthSignal | None:
        """Most recently computed signal, or None if no data pushed."""
        return self._latest_signal

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_parquet(self) -> None:
        """Append pending signals to a date-partitioned Parquet file."""
        if not self._pending:
            return

        rows = [
            {
                "timestamp": s.timestamp,
                "bid_depth": s.bid_depth,
                "ask_depth": s.ask_depth,
                "bid_depth_ratio": s.bid_depth_ratio,
                "ask_depth_ratio": s.ask_depth_ratio,
                "bid_thinning": s.bid_thinning,
                "ask_thinning": s.ask_thinning,
                "breakout_lean": s.breakout_lean,
            }
            for s in self._pending
        ]
        df = pl.DataFrame(rows)

        date_str = self._pending[-1].timestamp.strftime("%Y-%m-%d")
        out_dir = Path(self.config.parquet_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"depth_signals_{date_str}.parquet"

        if out_path.exists():
            existing = pl.read_parquet(out_path)
            df = pl.concat([existing, df])

        df.write_parquet(out_path, compression="zstd")
        self._pending.clear()
