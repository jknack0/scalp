"""Iceberg and level absorption detection — hidden order and S/R signals.

Tracks top-of-book level persistence, size refreshes, and trade consumption
to detect iceberg orders. Monitors repeated price tests to identify absorption
events and breakout triggers. Flushes signals to date-partitioned Parquet files.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl

from src.filters import L2Snapshot

# MES tick size in index points
MES_TICK = 0.25


def _round_to_tick(price: float) -> float:
    """Round a price to the nearest MES tick."""
    return round(price / MES_TICK) * MES_TICK


@dataclass(frozen=True)
class TradeEvent:
    """Individual trade from the Rithmic trade feed."""

    timestamp: datetime
    price: float
    size: int
    aggressor: Literal["buy", "sell"]


@dataclass
class LevelActivity:
    """Tracks activity at a single price level."""

    price: float
    side: Literal["bid", "ask"]
    appearances: int = 0
    consecutive_appearances: int = 0
    total_consumed: int = 0
    last_seen: datetime | None = None
    refreshes: int = 0
    last_size: int = 0
    _absent_count: int = 0  # snapshots absent since last seen


@dataclass(frozen=True)
class IcebergSignal:
    """Signal emitted when an iceberg order is detected."""

    timestamp: datetime
    price: float
    side: Literal["bid", "ask"]
    estimated_total_size: int
    confidence: float
    trade_with: bool


@dataclass(frozen=True)
class AbsorptionSignal:
    """Signal emitted when absorption or breakout is detected at a level."""

    timestamp: datetime
    price: float
    side: Literal["bid", "ask"]
    test_count: int
    absorbed_size: int
    status: Literal["absorbing", "broken"]


@dataclass
class IcebergConfig:
    """Tuning knobs for iceberg and absorption detection."""

    min_appearances: int = 10
    consumed_threshold: int = 50
    test_threshold: int = 5
    window_seconds: float = 30.0
    top_n_levels: int = 3
    refresh_tolerance: float = 0.5  # size ratio tolerance for refresh detection
    absent_grace: int = 2  # snapshots a level can be absent before resetting
    persist: bool = False
    parquet_dir_iceberg: str = "data/iceberg_signals"
    parquet_dir_absorption: str = "data/absorption_signals"


@dataclass
class _AbsorptionTracker:
    """Internal tracker for absorption at a single price level."""

    price: float
    side: Literal["bid", "ask"]
    tests: list[datetime] = field(default_factory=list)
    total_absorbed: int = 0
    signaled_absorbing: bool = False


# ------------------------------------------------------------------
# Tick rule helper
# ------------------------------------------------------------------

def infer_aggressor(price: float, last_price: float | None) -> Literal["buy", "sell"]:
    """Infer trade aggressor via tick rule.

    Uptick or zero-uptick → buyer aggressor.
    Downtick or zero-downtick → seller aggressor.
    """
    if last_price is None or price >= last_price:
        return "buy"
    return "sell"


# ------------------------------------------------------------------
# Iceberg Detector
# ------------------------------------------------------------------


@dataclass
class IcebergDetector:
    """Detects iceberg orders from L2 level persistence and trade consumption.

    Args:
        config: Detection parameters.
    """

    config: IcebergConfig = field(default_factory=IcebergConfig)

    def __post_init__(self) -> None:
        self._levels: dict[tuple[float, str], LevelActivity] = {}
        self._pending_parquet: list[IcebergSignal] = []

    def push_l2(self, snapshot: L2Snapshot) -> list[IcebergSignal]:
        """Process an L2 snapshot and check for iceberg signals.

        Args:
            snapshot: Top-of-book depth snapshot.

        Returns:
            List of IcebergSignals detected on this update.
        """
        n = self.config.top_n_levels
        current_levels: set[tuple[float, str]] = set()

        # Extract top N bid and ask levels
        for price, size in snapshot.bids[:n]:
            key = (_round_to_tick(price), "bid")
            current_levels.add(key)
            self._update_level(key, size, snapshot.timestamp, "bid")

        for price, size in snapshot.asks[:n]:
            key = (_round_to_tick(price), "ask")
            current_levels.add(key)
            self._update_level(key, size, snapshot.timestamp, "ask")

        # Mark absent levels
        signals: list[IcebergSignal] = []
        keys_to_remove: list[tuple[float, str]] = []
        for key, activity in self._levels.items():
            if key not in current_levels:
                activity._absent_count += 1
                if activity._absent_count > self.config.absent_grace:
                    activity.consecutive_appearances = 0
                    # Don't remove — keep for refresh detection
            else:
                # Check for iceberg signal
                sig = self._check_iceberg(activity, snapshot.timestamp)
                if sig is not None:
                    signals.append(sig)

        # Clean up stale levels (absent for a long time)
        for key, activity in list(self._levels.items()):
            if activity._absent_count > self.config.min_appearances * 2:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._levels[key]

        if self.config.persist and signals:
            self._pending_parquet.extend(signals)
            self._flush_parquet()

        return signals

    def push_trade(self, trade: TradeEvent) -> None:
        """Update consumed totals when a trade occurs at a tracked level.

        Args:
            trade: Individual trade event.
        """
        price = _round_to_tick(trade.price)

        # A buy aggressor consumes ask liquidity, sell consumes bid
        if trade.aggressor == "buy":
            key = (price, "ask")
        else:
            key = (price, "bid")

        if key in self._levels:
            self._levels[key].total_consumed += trade.size

    def _update_level(
        self,
        key: tuple[float, str],
        size: int,
        timestamp: datetime,
        side: Literal["bid", "ask"],
    ) -> None:
        """Update or create a LevelActivity record."""
        if key not in self._levels:
            self._levels[key] = LevelActivity(price=key[0], side=side)

        activity = self._levels[key]

        # Detect refresh: level was briefly absent then reappeared with similar size
        if activity._absent_count > 0 and activity.last_size > 0:
            if activity._absent_count <= self.config.absent_grace:
                ratio = size / activity.last_size if activity.last_size > 0 else 0.0
                if abs(1.0 - ratio) <= self.config.refresh_tolerance:
                    activity.refreshes += 1

        activity._absent_count = 0
        activity.appearances += 1
        activity.consecutive_appearances += 1
        activity.last_seen = timestamp
        activity.last_size = size

    def _check_iceberg(
        self, activity: LevelActivity, timestamp: datetime
    ) -> IcebergSignal | None:
        """Check if a level qualifies as an iceberg."""
        if activity.consecutive_appearances < self.config.min_appearances:
            return None
        if activity.total_consumed < self.config.consumed_threshold:
            return None

        # Confidence based on refreshes and consumption
        refresh_score = min(activity.refreshes / 3.0, 1.0)
        consume_score = min(
            activity.total_consumed / (self.config.consumed_threshold * 3.0), 1.0
        )
        confidence = (refresh_score + consume_score) / 2.0
        confidence = min(max(confidence, 0.0), 1.0)

        # Estimate total size: consumed + current visible
        estimated_total = activity.total_consumed + activity.last_size

        # trade_with: True if iceberg is on bid side (support) or ask side (resistance)
        trade_with = True  # always a S/R signal when detected

        return IcebergSignal(
            timestamp=timestamp,
            price=activity.price,
            side=activity.side,
            estimated_total_size=estimated_total,
            confidence=confidence,
            trade_with=trade_with,
        )

    def _flush_parquet(self) -> None:
        """Flush pending iceberg signals to Parquet."""
        if not self._pending_parquet:
            return

        rows = [
            {
                "timestamp": s.timestamp,
                "price": s.price,
                "side": s.side,
                "estimated_total_size": s.estimated_total_size,
                "confidence": s.confidence,
                "trade_with": s.trade_with,
            }
            for s in self._pending_parquet
        ]
        df = pl.DataFrame(rows)

        date_str = self._pending_parquet[-1].timestamp.strftime("%Y-%m-%d")
        out_dir = Path(self.config.parquet_dir_iceberg)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"iceberg_signals_{date_str}.parquet"

        if out_path.exists():
            existing = pl.read_parquet(out_path)
            df = pl.concat([existing, df])

        df.write_parquet(out_path, compression="zstd")
        self._pending_parquet.clear()


# ------------------------------------------------------------------
# Absorption Detector
# ------------------------------------------------------------------


@dataclass
class AbsorptionDetector:
    """Detects absorption events and breakout triggers at price levels.

    Args:
        config: Detection parameters.
    """

    config: IcebergConfig = field(default_factory=IcebergConfig)

    def __post_init__(self) -> None:
        self._trackers: dict[tuple[float, str], _AbsorptionTracker] = {}
        self._last_trade_price: float | None = None
        self._current_book_levels: set[tuple[float, str]] = set()
        self._pending_parquet: list[AbsorptionSignal] = []

    def push_trade(self, trade: TradeEvent) -> None:
        """Record a trade as a test of the level it hits.

        Args:
            trade: Individual trade event.
        """
        price = _round_to_tick(trade.price)
        self._last_trade_price = trade.price

        # A buy aggressor tests ask levels, sell tests bid levels
        if trade.aggressor == "buy":
            side: Literal["bid", "ask"] = "ask"
        else:
            side = "bid"

        key = (price, side)

        # Only track if this level is currently in the book
        if key not in self._current_book_levels:
            return

        if key not in self._trackers:
            self._trackers[key] = _AbsorptionTracker(price=price, side=side)

        tracker = self._trackers[key]

        # Prune old tests outside the window
        cutoff = trade.timestamp.timestamp() - self.config.window_seconds
        tracker.tests = [
            t for t in tracker.tests if t.timestamp() >= cutoff
        ]

        tracker.tests.append(trade.timestamp)
        tracker.total_absorbed += trade.size

    def push_l2(self, snapshot: L2Snapshot) -> list[AbsorptionSignal]:
        """Process an L2 snapshot and check for absorption/breakout signals.

        Args:
            snapshot: Top-of-book depth snapshot.

        Returns:
            List of AbsorptionSignals detected on this update.
        """
        n = self.config.top_n_levels

        new_levels: set[tuple[float, str]] = set()
        for price, _ in snapshot.bids[:n]:
            new_levels.add((_round_to_tick(price), "bid"))
        for price, _ in snapshot.asks[:n]:
            new_levels.add((_round_to_tick(price), "ask"))

        signals: list[AbsorptionSignal] = []

        # Check for absorbing signals (level still present, enough tests)
        for key, tracker in list(self._trackers.items()):
            # Prune old tests
            if tracker.tests:
                cutoff = tracker.tests[-1].timestamp() - self.config.window_seconds
                tracker.tests = [
                    t for t in tracker.tests if t.timestamp() >= cutoff
                ]

            if key in new_levels and len(tracker.tests) >= self.config.test_threshold:
                if not tracker.signaled_absorbing:
                    signals.append(
                        AbsorptionSignal(
                            timestamp=snapshot.timestamp,
                            price=tracker.price,
                            side=tracker.side,
                            test_count=len(tracker.tests),
                            absorbed_size=tracker.total_absorbed,
                            status="absorbing",
                        )
                    )
                    tracker.signaled_absorbing = True

            # Check for breakout: level was being absorbed but now gone
            if (
                key not in new_levels
                and tracker.signaled_absorbing
            ):
                signals.append(
                    AbsorptionSignal(
                        timestamp=snapshot.timestamp,
                        price=tracker.price,
                        side=tracker.side,
                        test_count=len(tracker.tests),
                        absorbed_size=tracker.total_absorbed,
                        status="broken",
                    )
                )
                del self._trackers[key]

        self._current_book_levels = new_levels

        # Clean up trackers for levels no longer in book and not signaled
        keys_to_remove = [
            k for k, t in self._trackers.items()
            if k not in new_levels and not t.signaled_absorbing
        ]
        for key in keys_to_remove:
            del self._trackers[key]

        if self.config.persist and signals:
            self._pending_parquet.extend(signals)
            self._flush_parquet()

        return signals

    @property
    def last_trade_price(self) -> float | None:
        """Last observed trade price, for tick rule inference."""
        return self._last_trade_price

    def _flush_parquet(self) -> None:
        """Flush pending absorption signals to Parquet."""
        if not self._pending_parquet:
            return

        rows = [
            {
                "timestamp": s.timestamp,
                "price": s.price,
                "side": s.side,
                "test_count": s.test_count,
                "absorbed_size": s.absorbed_size,
                "status": s.status,
            }
            for s in self._pending_parquet
        ]
        df = pl.DataFrame(rows)

        date_str = self._pending_parquet[-1].timestamp.strftime("%Y-%m-%d")
        out_dir = Path(self.config.parquet_dir_absorption)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"absorption_signals_{date_str}.parquet"

        if out_path.exists():
            existing = pl.read_parquet(out_path)
            df = pl.concat([existing, df])

        df.write_parquet(out_path, compression="zstd")
        self._pending_parquet.clear()
