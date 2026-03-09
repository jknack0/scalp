"""Databento live market data feed.

Connects to Databento's Live Subscription Gateway for real-time CME
futures data (MBP-1 / top-of-book). Publishes TickEvents to the EventBus.

Uses the same BaseFeed interface as TradovateFeed, so it's a drop-in
replacement for market data while Tradovate handles order execution.

Usage:
    feed = DatabentoFeed(event_bus=bus, config=config)
    await feed.run()  # blocks, streaming ticks
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

import databento as db

from src.core.config import BotConfig
from src.core.events import EventBus, TickEvent
from src.core.logging import get_logger
from src.feeds.base import BaseFeed

logger = get_logger("feed.databento")

# CME MDP3 dataset for all CME Group futures
_DATASET = "GLBX.MDP3"


class DatabentoFeed(BaseFeed):
    """Databento live feed for CME futures via MBP-1 (top-of-book).

    Streams best bid/ask + last trade and converts to TickEvents.
    """

    def __init__(self, event_bus: EventBus, config: BotConfig) -> None:
        super().__init__(event_bus)
        self._config = config
        self._client: db.Live | None = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tick_count = 0
        self._last_tick_time = time.monotonic()
        self._latency_samples: deque[float] = deque(maxlen=10_000)
        self._stop_event = asyncio.Event()

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def latency_stats(self) -> dict[str, float]:
        if not self._latency_samples:
            return {"min_ms": 0.0, "avg_ms": 0.0, "max_ms": 0.0, "p99_ms": 0.0}
        samples = sorted(self._latency_samples)
        p99_idx = max(0, int(len(samples) * 0.99) - 1)
        return {
            "min_ms": samples[0],
            "avg_ms": sum(samples) / len(samples),
            "max_ms": samples[-1],
            "p99_ms": samples[p99_idx],
        }

    async def connect(self) -> None:
        """Create the Databento Live client."""
        self._client = db.Live(
            key=self._config.databento_api_key,
            reconnect_policy="reconnect",
        )
        logger.info("client_created")

    async def subscribe(self, symbol: str) -> None:
        """Subscribe to MBP-1 (top-of-book) for the symbol."""
        if not self._client:
            raise ConnectionError("Client not created — call connect() first")

        # Databento uses continuous front-month via stype_in="parent"
        # e.g. "MES.FUT" for micro E-mini S&P front month
        root = symbol[:3] if len(symbol) > 3 else symbol  # MES from MESH6

        self._client.subscribe(
            dataset=_DATASET,
            schema="mbp-1",
            symbols=f"{root}.FUT",
            stype_in="parent",
        )
        logger.info("subscribed", symbol=f"{root}.FUT", schema="mbp-1")

    async def disconnect(self) -> None:
        """Stop the live client."""
        if self._client:
            try:
                self._client.stop()
            except Exception:
                pass
            self._client = None
        logger.info("disconnected")

    def stop(self) -> None:
        """Signal the feed to stop."""
        self._running = False
        if self._client:
            try:
                self._client.stop()
            except Exception:
                pass
        # Unblock the wait in run()
        if self._loop:
            self._loop.call_soon_threadsafe(self._stop_event.set)

    async def _heartbeat(self) -> None:
        """Log feed stats every 30 seconds so we know it's alive."""
        last_count = 0
        while self._running:
            await asyncio.sleep(30)
            if not self._running:
                break
            delta = self._tick_count - last_count
            last_count = self._tick_count
            stale_sec = time.monotonic() - self._last_tick_time
            stats = self.latency_stats
            logger.info(
                "feed_heartbeat",
                total_ticks=self._tick_count,
                ticks_30s=delta,
                stale_sec=round(stale_sec, 1),
                avg_latency_ms=round(stats["avg_ms"], 1),
                p99_latency_ms=round(stats["p99_ms"], 1),
            )

    async def run(self) -> None:
        """Main loop: connect, subscribe, stream ticks via callback."""
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()
        symbol = self._config.symbol
        logger.info("feed_starting", symbol=symbol)

        heartbeat_task: asyncio.Task | None = None

        try:
            await self.connect()
            await self.subscribe(symbol)

            # Use callback API — Databento calls our function from its own thread
            self._client.add_callback(self._on_record, self._on_error)

            # Start heartbeat logging
            heartbeat_task = asyncio.create_task(self._heartbeat(), name="feed_heartbeat")

            # Start the client (begins receiving data — blocks in executor)
            logger.info("feed_client_starting")
            await self._loop.run_in_executor(None, self._client.start)

            # If start() returns, the session ended or connection was lost
            logger.warning(
                "feed_client_returned",
                tick_count=self._tick_count,
                hint="client.start() returned — connection may have dropped",
            )

            # Block until stop() is called (callbacks may still fire on reconnect)
            await self._stop_event.wait()

        except asyncio.CancelledError:
            logger.info("feed_cancelled")
        except Exception as e:
            if self._running:
                logger.error("feed_error", error=str(e), error_type=type(e).__name__)
        finally:
            if heartbeat_task:
                heartbeat_task.cancel()
            await self.disconnect()
            logger.info("feed_stopped", tick_count=self._tick_count)

    def _on_record(self, record: object) -> None:
        """Callback invoked by Databento for each record (from its thread)."""
        if not self._running:
            return

        # Only process MBP1Msg records (skip SystemMsg, SymbolMappingMsg, etc.)
        if type(record).__name__ != "MBP1Msg":
            return

        try:
            level = record.levels[0]

            bid = level.bid_px / 1e9  # Databento uses fixed-point (1e9)
            ask = level.ask_px / 1e9

            # Trade info from the record
            last_price = record.price / 1e9 if hasattr(record, "price") else 0.0
            last_size = record.size if hasattr(record, "size") else 0

            # Timestamp in nanoseconds (Databento native)
            ts_ns = record.ts_event

            # Latency calculation
            now_ns = time.time_ns()
            latency_ms = (now_ns - ts_ns) / 1_000_000
            if 0 <= latency_ms < 60_000:
                self._latency_samples.append(latency_ms)

            self._last_tick_time = time.monotonic()
            self._tick_count += 1

            # Use mid price as last if no trade price
            if last_price <= 0 and bid > 0 and ask > 0:
                last_price = (bid + ask) / 2

            tick = TickEvent(
                symbol=self._config.symbol,
                bid=bid,
                ask=ask,
                last_price=last_price,
                last_size=last_size,
                timestamp_ns=ts_ns,
            )

            # Thread-safe publish to the async event bus
            asyncio.run_coroutine_threadsafe(self._bus.publish(tick), self._loop)

        except Exception as e:
            logger.warning("tick_parse_error", error=str(e))

    def _on_error(self, exc: Exception) -> None:
        """Callback for Databento client errors."""
        if self._running:
            logger.error("databento_error", error=str(exc))
