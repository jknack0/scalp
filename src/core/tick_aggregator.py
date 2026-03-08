"""Tick-to-bar aggregator for live trading.

Accumulates TickEvents from the EventBus and emits BarEvents at configurable
intervals. Supports time-based bars (1s, 5s, 1m) with L1 approximation from
tick-level bid/ask data.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from src.core.events import BarEvent, EventBus, EventType, TickEvent
from src.core.logging import get_logger

logger = get_logger("tick_aggregator")


@dataclass
class _BarAccumulator:
    """Working state for a bar being built."""

    open: float = 0.0
    high: float = -float("inf")
    low: float = float("inf")
    close: float = 0.0
    volume: int = 0
    tick_count: int = 0
    start_ns: int = 0

    # L1 approximation from tick-level bid/ask
    bid_sum: float = 0.0
    ask_sum: float = 0.0
    bid_count: int = 0
    ask_count: int = 0
    buy_vol: int = 0   # ticks where last_price >= ask (aggressive buy)
    sell_vol: int = 0   # ticks where last_price <= bid (aggressive sell)

    def is_empty(self) -> bool:
        return self.tick_count == 0


class TickAggregator:
    """Aggregates ticks into time-based bars and publishes to EventBus.

    Subscribes to TICK events, accumulates OHLCV + L1 approximation,
    and emits BAR events at the configured interval.

    Usage:
        agg = TickAggregator(bus, symbol="MESM6", interval_seconds=5)
        bus.subscribe(EventType.TICK, agg.on_tick)
        asyncio.create_task(agg.run())
    """

    def __init__(
        self,
        event_bus: EventBus,
        symbol: str = "MESM6",
        interval_seconds: float = 5.0,
    ) -> None:
        self._bus = event_bus
        self._symbol = symbol
        self._interval = interval_seconds
        self._bar = _BarAccumulator()
        self._running = False

    async def on_tick(self, tick: TickEvent) -> None:
        """Accumulate a tick into the current bar."""
        if tick.symbol != self._symbol:
            return

        price = tick.last_price
        size = tick.last_size

        if price <= 0:
            return

        bar = self._bar
        if bar.is_empty():
            bar.open = price
            bar.start_ns = tick.timestamp_ns

        bar.high = max(bar.high, price)
        bar.low = min(bar.low, price)
        bar.close = price
        bar.volume += size
        bar.tick_count += 1

        # L1 approximation
        if tick.bid > 0:
            bar.bid_sum += tick.bid
            bar.bid_count += 1
        if tick.ask > 0:
            bar.ask_sum += tick.ask
            bar.ask_count += 1

        # Aggressive volume classification (Lee-Ready at BBO)
        if tick.ask > 0 and price >= tick.ask:
            bar.buy_vol += size
        elif tick.bid > 0 and price <= tick.bid:
            bar.sell_vol += size

    async def run(self) -> None:
        """Timer loop that flushes bars at the configured interval."""
        self._running = True
        logger.info(
            "aggregator_started",
            symbol=self._symbol,
            interval=self._interval,
        )

        while self._running:
            await asyncio.sleep(self._interval)
            await self._flush_bar()

    def stop(self) -> None:
        self._running = False

    async def _flush_bar(self) -> None:
        """Emit the current bar and reset accumulator."""
        bar = self._bar
        if bar.is_empty():
            return

        avg_bid = bar.bid_sum / bar.bid_count if bar.bid_count > 0 else 0.0
        avg_ask = bar.ask_sum / bar.ask_count if bar.ask_count > 0 else 0.0

        bar_type = f"{int(self._interval)}s" if self._interval < 60 else f"{int(self._interval // 60)}m"

        bar_event = BarEvent(
            symbol=self._symbol,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            bar_type=bar_type,
            timestamp_ns=bar.start_ns,
            avg_bid_size=avg_bid,
            avg_ask_size=avg_ask,
            aggressive_buy_vol=float(bar.buy_vol),
            aggressive_sell_vol=float(bar.sell_vol),
        )

        await self._bus.publish(bar_event)
        self._bar = _BarAccumulator()
