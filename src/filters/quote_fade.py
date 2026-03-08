"""Quote fade detection — tracks how often best quotes vanish after order routing.

Monitors the time between an order being routed and the relevant best quote
disappearing (moving by more than a configurable tick threshold). Produces
an adaptive order-type recommendation based on rolling fade rate.
Flushes fade results to date-partitioned Parquet files.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl


# MES tick size in index points
MES_TICK = 0.25


@dataclass(frozen=True)
class QuoteEvent:
    """Single L1 quote update."""

    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int


@dataclass(frozen=True)
class OrderRouteEvent:
    """Notification that an order has been routed to the exchange."""

    timestamp: datetime
    direction: Literal["long", "short"]
    intended_price: float


@dataclass(frozen=True)
class FadeResult:
    """Outcome of a fade check for a single routed order."""

    order_timestamp: datetime
    fade_detected: bool
    ms_to_fade: float | None
    fade_rate: float


@dataclass
class FadeConfig:
    """Tuning knobs for the quote fade detector."""

    fade_window_ms: float = 200.0
    fade_threshold_ticks: int = 1
    lookback: int = 20
    persist_every_n: int = 1
    parquet_dir: str = "data/quote_fades"


@dataclass
class _PendingOrder:
    """Internal tracker for an order awaiting fade evaluation."""

    event: OrderRouteEvent
    routed_at: float  # monotonic time from event loop


@dataclass
class QuoteFadeDetector:
    """Detects quote fades after order routing and recommends order aggression.

    Args:
        config: Tuning parameters.
        persist: Whether to flush fade results to Parquet files.
    """

    config: FadeConfig = field(default_factory=FadeConfig)
    persist: bool = False

    def __post_init__(self) -> None:
        self._pending: list[_PendingOrder] = []
        self._results: deque[FadeResult] = deque(maxlen=self.config.lookback * 5)
        self._flush_pending: list[FadeResult] = []
        self._flush_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_order_routed(self, event: OrderRouteEvent) -> None:
        """Register a pending fade check for a routed order.

        Args:
            event: The order routing notification.
        """
        mono = asyncio.get_event_loop().time()
        self._pending.append(_PendingOrder(event=event, routed_at=mono))

    def on_quote(self, event: QuoteEvent) -> list[FadeResult]:
        """Process a quote update and check all pending orders for fades.

        Args:
            event: The L1 quote update.

        Returns:
            List of FadeResults for any orders resolved on this quote.
        """
        mono = asyncio.get_event_loop().time()
        threshold_price = self.config.fade_threshold_ticks * MES_TICK
        resolved: list[FadeResult] = []
        still_pending: list[_PendingOrder] = []

        for pending in self._pending:
            elapsed_ms = (mono - pending.routed_at) * 1000.0
            order = pending.event

            # Check if the relevant quote has faded
            if order.direction == "long":
                # Buying — watching the ask; fade = ask moved up
                price_move = event.ask - order.intended_price
            else:
                # Selling — watching the bid; fade = bid moved down
                price_move = order.intended_price - event.bid

            faded = price_move >= threshold_price

            if faded:
                result = FadeResult(
                    order_timestamp=order.timestamp,
                    fade_detected=True,
                    ms_to_fade=elapsed_ms,
                    fade_rate=self._compute_fade_rate_with(True),
                )
                self._record(result)
                resolved.append(result)
            elif elapsed_ms >= self.config.fade_window_ms:
                # Window expired — no fade
                result = FadeResult(
                    order_timestamp=order.timestamp,
                    fade_detected=False,
                    ms_to_fade=None,
                    fade_rate=self._compute_fade_rate_with(False),
                )
                self._record(result)
                resolved.append(result)
            else:
                still_pending.append(pending)

        self._pending = still_pending
        return resolved

    def get_fade_rate(self, lookback: int | None = None) -> float:
        """Return fade_count / total_routed for the last N resolved orders.

        Args:
            lookback: Number of recent results to consider. Defaults to config.lookback.

        Returns:
            Fade rate as a float in [0.0, 1.0], or 0.0 if no results.
        """
        n = lookback or self.config.lookback
        recent = list(self._results)[-n:]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.fade_detected) / len(recent)

    def recommend_order_type(self) -> Literal["limit", "ioc", "market"]:
        """Recommend order aggression based on current fade rate.

        Returns:
            "market" if fade_rate > 0.5, "ioc" if > 0.25, else "limit".
        """
        rate = self.get_fade_rate()
        if rate > 0.5:
            return "market"
        if rate > 0.25:
            return "ioc"
        return "limit"

    @property
    def pending_count(self) -> int:
        """Number of orders still awaiting fade evaluation."""
        return len(self._pending)

    @property
    def total_resolved(self) -> int:
        """Total number of resolved fade checks."""
        return len(self._results)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_fade_rate_with(self, include_fade: bool) -> float:
        """Compute fade rate including a not-yet-recorded result."""
        n = self.config.lookback
        recent = list(self._results)[-(n - 1):]
        fades = sum(1 for r in recent if r.fade_detected) + (1 if include_fade else 0)
        total = len(recent) + 1
        return fades / total

    def _record(self, result: FadeResult) -> None:
        """Store result and optionally queue for persistence."""
        self._results.append(result)
        if self.persist:
            self._flush_pending.append(result)
            self._flush_count += 1
            if self._flush_count % self.config.persist_every_n == 0:
                self._flush_parquet()

    def _flush_parquet(self) -> None:
        """Append pending results to a date-partitioned Parquet file."""
        if not self._flush_pending:
            return

        rows = [
            {
                "timestamp": r.order_timestamp,
                "fade_detected": r.fade_detected,
                "ms_to_fade": r.ms_to_fade,
                "fade_rate": r.fade_rate,
                "recommendation": (
                    "market" if r.fade_rate > 0.5
                    else "ioc" if r.fade_rate > 0.25
                    else "limit"
                ),
            }
            for r in self._flush_pending
        ]
        df = pl.DataFrame(rows)

        date_str = self._flush_pending[-1].order_timestamp.strftime("%Y-%m-%d")
        out_dir = Path(self.config.parquet_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"quote_fades_{date_str}.parquet"

        if out_path.exists():
            existing = pl.read_parquet(out_path)
            df = pl.concat([existing, df])

        df.write_parquet(out_path, compression="zstd")
        self._flush_pending.clear()
