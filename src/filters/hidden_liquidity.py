"""Hidden liquidity map — persistent S/R overlay from invisible order detection.

Identifies price levels where trades consistently occur without visible book
presence, builds a strength-scored map with exponential recency decay, and
persists to date-partitioned Parquet files. Loads existing map on startup.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import polars as pl

from src.filters import L2Snapshot


def _round_to_tick(price: float, tick_size: float) -> float:
    """Round a price to the nearest tick."""
    return round(price / tick_size) * tick_size


@dataclass(frozen=True)
class HiddenTradeEvent:
    """A single trade that occurred at a level with near-zero visible depth."""

    timestamp: datetime
    price: float
    size: int


@dataclass
class HiddenLevel:
    """Aggregated hidden liquidity at a single price level."""

    price: float
    event_count: int
    total_hidden_volume: int
    last_seen: datetime
    strength: float


@dataclass
class HiddenLiquidityConfig:
    """Tuning knobs for hidden liquidity detection."""

    visibility_threshold: int = 10
    decay_days: float = 30.0
    min_strength: float = 0.3
    persist_interval_seconds: float = 300.0
    tick_size: float = 0.25
    radius_ticks: int = 4
    parquet_dir: str = "data/hidden_liquidity"


@dataclass
class _LevelRecord:
    """Internal mutable record for a price level."""

    events: list[HiddenTradeEvent] = field(default_factory=list)


@dataclass
class HiddenLiquidityDetector:
    """Detects trades at levels with near-zero visible book depth.

    Compares each incoming trade against the most recent L2 snapshot to
    determine if the trade price had visible depth below the threshold.

    Args:
        config: Detection parameters.
    """

    config: HiddenLiquidityConfig = field(default_factory=HiddenLiquidityConfig)

    def __post_init__(self) -> None:
        self._last_snapshot: L2Snapshot | None = None
        self._visible_sizes: dict[float, int] = {}

    def update_book(self, snapshot: L2Snapshot) -> None:
        """Cache the latest L2 snapshot for visibility checks.

        Args:
            snapshot: Top-of-book depth snapshot.
        """
        self._last_snapshot = snapshot
        self._visible_sizes.clear()
        for price, size in snapshot.bids:
            key = _round_to_tick(price, self.config.tick_size)
            self._visible_sizes[key] = self._visible_sizes.get(key, 0) + size
        for price, size in snapshot.asks:
            key = _round_to_tick(price, self.config.tick_size)
            self._visible_sizes[key] = self._visible_sizes.get(key, 0) + size

    def check_trade(
        self, price: float, size: int, timestamp: datetime
    ) -> HiddenTradeEvent | None:
        """Check if a trade occurred at a level with near-zero visible depth.

        Args:
            price: Trade price.
            size: Trade size.
            timestamp: Trade timestamp.

        Returns:
            HiddenTradeEvent if the level had low visibility, else None.
        """
        if self._last_snapshot is None:
            return None

        key = _round_to_tick(price, self.config.tick_size)
        visible = self._visible_sizes.get(key, 0)

        if visible < self.config.visibility_threshold:
            return HiddenTradeEvent(timestamp=timestamp, price=key, size=size)
        return None


@dataclass
class HiddenLiquidityMap:
    """Persistent map of hidden liquidity levels with recency-decayed strength.

    Loads existing events from Parquet on startup, records new hidden events,
    and provides spatial queries for strategy integration.

    Args:
        config: Map parameters.
        persist: Whether to flush events to Parquet files.
    """

    config: HiddenLiquidityConfig = field(default_factory=HiddenLiquidityConfig)
    persist: bool = False

    def __post_init__(self) -> None:
        self._levels: dict[float, _LevelRecord] = defaultdict(_LevelRecord)
        self._pending: list[HiddenTradeEvent] = []
        self._event_count: int = 0

    def load_from_parquet(self, parquet_dir: str | None = None) -> int:
        """Load existing hidden events from Parquet files.

        Args:
            parquet_dir: Directory to load from. Defaults to config.parquet_dir.

        Returns:
            Number of events loaded.
        """
        directory = Path(parquet_dir or self.config.parquet_dir)
        if not directory.exists():
            return 0

        total = 0
        for path in sorted(directory.glob("hidden_events_*.parquet")):
            df = pl.read_parquet(path)
            for row in df.iter_rows(named=True):
                ts = row["timestamp"]
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                event = HiddenTradeEvent(
                    timestamp=ts,
                    price=float(row["price"]),
                    size=int(row["size"]),
                )
                self._levels[event.price].events.append(event)
                total += 1

        return total

    def record_hidden_event(
        self, price: float, size: int, timestamp: datetime
    ) -> None:
        """Record a hidden trade event at a price level.

        Args:
            price: Tick-rounded price.
            size: Trade size.
            timestamp: Event timestamp.
        """
        key = _round_to_tick(price, self.config.tick_size)
        event = HiddenTradeEvent(timestamp=timestamp, price=key, size=size)
        self._levels[key].events.append(event)
        self._event_count += 1

        if self.persist:
            self._pending.append(event)

    def get_strength(
        self, price: float, now: datetime | None = None
    ) -> float:
        """Compute recency-decayed strength score for a price level.

        Args:
            price: Price level to query.
            now: Reference time for decay calculation. Defaults to UTC now.

        Returns:
            Strength score in [0.0, 1.0].
        """
        key = _round_to_tick(price, self.config.tick_size)
        record = self._levels.get(key)
        if record is None or not record.events:
            return 0.0

        if now is None:
            now = datetime.now(timezone.utc)

        return self._compute_strength(record.events, now)

    def get_nearby_levels(
        self,
        price: float,
        radius_ticks: int | None = None,
        now: datetime | None = None,
    ) -> list[HiddenLevel]:
        """Find hidden levels within a tick radius of a price, sorted by strength.

        Args:
            price: Center price.
            radius_ticks: Number of ticks in each direction. Defaults to config.
            now: Reference time for decay. Defaults to UTC now.

        Returns:
            List of HiddenLevel sorted by strength descending.
        """
        r = radius_ticks if radius_ticks is not None else self.config.radius_ticks
        if now is None:
            now = datetime.now(timezone.utc)

        tick = self.config.tick_size
        low = _round_to_tick(price - r * tick, tick)
        high = _round_to_tick(price + r * tick, tick)

        results: list[HiddenLevel] = []
        for level_price, record in self._levels.items():
            if not record.events:
                continue
            if low <= level_price <= high:
                strength = self._compute_strength(record.events, now)
                if strength > 0:
                    results.append(
                        HiddenLevel(
                            price=level_price,
                            event_count=len(record.events),
                            total_hidden_volume=sum(e.size for e in record.events),
                            last_seen=max(e.timestamp for e in record.events),
                            strength=strength,
                        )
                    )

        results.sort(key=lambda x: x.strength, reverse=True)
        return results

    def is_near_hidden_support(
        self,
        price: float,
        direction: Literal["long", "short"],
        now: datetime | None = None,
    ) -> bool:
        """Check if price is near a hidden liquidity level for confluence.

        Args:
            price: Current price.
            direction: Trade direction being considered.
            now: Reference time for decay. Defaults to UTC now.

        Returns:
            True if a qualifying hidden level exists in the relevant zone.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        tick = self.config.tick_size
        r = self.config.radius_ticks

        if direction == "long":
            # Look for hidden support below current price
            low = _round_to_tick(price - r * tick, tick)
            high = _round_to_tick(price, tick)
        else:
            # Look for hidden resistance above current price
            low = _round_to_tick(price, tick)
            high = _round_to_tick(price + r * tick, tick)

        for level_price, record in self._levels.items():
            if not record.events:
                continue
            if low <= level_price <= high:
                strength = self._compute_strength(record.events, now)
                if strength >= self.config.min_strength:
                    return True

        return False

    def flush_parquet(self) -> None:
        """Flush pending events to a date-partitioned Parquet file."""
        if not self._pending:
            return

        rows = [
            {
                "timestamp": e.timestamp,
                "price": e.price,
                "size": e.size,
            }
            for e in self._pending
        ]
        df = pl.DataFrame(rows)

        date_str = self._pending[-1].timestamp.strftime("%Y-%m-%d")
        out_dir = Path(self.config.parquet_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"hidden_events_{date_str}.parquet"

        if out_path.exists():
            existing = pl.read_parquet(out_path)
            df = pl.concat([existing, df])

        df.write_parquet(out_path, compression="zstd")
        self._pending.clear()

    @property
    def level_count(self) -> int:
        """Number of distinct price levels with hidden events."""
        return sum(1 for r in self._levels.values() if r.events)

    @property
    def total_events(self) -> int:
        """Total number of hidden events recorded."""
        return sum(len(r.events) for r in self._levels.values())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_strength(
        self, events: list[HiddenTradeEvent], now: datetime
    ) -> float:
        """Compute recency-decayed strength from a list of events.

        Uses exponential decay: each event contributes exp(-age_days / decay_days).
        Final score is normalized to [0.0, 1.0] by clamping the sum.
        """
        if not events:
            return 0.0

        decay_days = self.config.decay_days
        raw_score = 0.0

        for event in events:
            age_seconds = (now - event.timestamp).total_seconds()
            age_days = max(age_seconds / 86400.0, 0.0)
            raw_score += math.exp(-age_days / decay_days)

        # Normalize: a single fresh event ≈ 0.1, 10+ fresh events → 1.0
        return min(raw_score / 10.0, 1.0)
