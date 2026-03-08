"""Event dataclasses and async event bus for the bot pipeline."""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Coroutine

from src.core.logging import get_logger

logger = get_logger("events")

# Type alias for async subscriber callbacks
Callback = Callable[..., Coroutine[Any, Any, None]]


class EventType(Enum):
    """Canonical event types for the bus."""

    TICK = auto()
    BAR = auto()
    SIGNAL = auto()
    FILL = auto()
    RISK = auto()


@dataclass(frozen=True, slots=True)
class TickEvent:
    """Real-time quote/trade update."""

    symbol: str
    bid: float
    ask: float
    last_price: float
    last_size: int
    timestamp_ns: int

    @property
    def event_type(self) -> EventType:
        return EventType.TICK


@dataclass(frozen=True, slots=True)
class BarEvent:
    """Completed OHLCV bar, optionally enriched with L1 order book data."""

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    bar_type: str  # "1s", "1m", "5m", "dollar", etc.
    timestamp_ns: int
    # L1-enriched fields (0.0 when not available)
    avg_bid_size: float = 0.0
    avg_ask_size: float = 0.0
    aggressive_buy_vol: float = 0.0
    aggressive_sell_vol: float = 0.0

    @property
    def event_type(self) -> EventType:
        return EventType.BAR


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """Strategy trading signal."""

    strategy_id: str
    direction: str  # "BUY" or "SELL"
    strength: float  # 0.0 to 1.0
    reason: str
    timestamp_ns: int

    @property
    def event_type(self) -> EventType:
        return EventType.SIGNAL


@dataclass(frozen=True, slots=True)
class FillEvent:
    """Order fill confirmation."""

    order_id: str
    symbol: str
    direction: str  # "BUY" or "SELL"
    fill_price: float
    fill_size: int
    commission: float
    timestamp_ns: int

    @property
    def event_type(self) -> EventType:
        return EventType.FILL


@dataclass(frozen=True, slots=True)
class RiskEvent:
    """Risk management alert."""

    risk_type: str  # "HALT", "WARNING", "INFO"
    message: str
    timestamp_ns: int

    @property
    def event_type(self) -> EventType:
        return EventType.RISK


class EventBus:
    """Asyncio queue-based pub/sub event bus.

    Usage:
        bus = EventBus()
        bus.subscribe(EventType.TICK, my_handler)
        await bus.publish(TickEvent(...))
        await bus.run()  # blocks, dispatching events to subscribers
    """

    def __init__(self, maxsize: int = 10_000) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers: dict[EventType, list[Callback]] = {}
        self._event_counts: dict[EventType, int] = {}
        self._running: bool = False

    def subscribe(self, event_type: EventType, callback: Callback) -> None:
        """Register an async callback for an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    async def publish(self, event: Any) -> None:
        """Put event on queue (non-blocking). Drops if queue is full."""
        et = event.event_type
        self._event_counts[et] = self._event_counts.get(et, 0) + 1
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("event_queue_full", event_type=et.name, dropped=True)

    async def run(self) -> None:
        """Main dispatch loop. Blocks until stop() is called."""
        self._running = True
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            et = event.event_type
            for callback in self._subscribers.get(et, []):
                try:
                    await callback(event)
                except Exception:
                    logger.exception(
                        "subscriber_error",
                        event_type=et.name,
                        callback=callback.__name__,
                    )

    def stop(self) -> None:
        """Signal the run loop to exit."""
        self._running = False

    @property
    def event_counts(self) -> dict[EventType, int]:
        """Return copy of event counts for monitoring."""
        return dict(self._event_counts)
