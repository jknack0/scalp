"""Standalone CLI to test the Tradovate live feed.

Usage:
    python scripts/test_feed.py --seconds 30 --symbol MESM6

Connects to the Tradovate demo API, subscribes to quotes, prints each
tick, and shows a latency summary when done.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.core.config import BotConfig
from src.core.events import EventBus, EventType, TickEvent
from src.core.logging import configure_logging, get_logger
from src.feeds.tradovate import TradovateFeed


async def main(seconds: int, symbol: str, live: bool = False) -> None:
    configure_logging(log_level="INFO", log_file=None)
    logger = get_logger("test_feed")

    config = BotConfig.from_yaml()
    if live:
        config.tradovate_demo = False
    if not config.tradovate_username or not config.tradovate_password:
        logger.error("missing_credentials", hint="Set TRADOVATE_USERNAME and TRADOVATE_PASSWORD in .env")
        return

    # Override symbol if specified
    if symbol:
        config = BotConfig.from_yaml(symbol=symbol)
        if live:
            config.tradovate_demo = False

    bus = EventBus()
    feed = TradovateFeed(event_bus=bus, config=config)

    tick_count = 0

    async def on_tick(event: TickEvent) -> None:
        nonlocal tick_count
        tick_count += 1
        from datetime import datetime, timezone

        ts = datetime.fromtimestamp(event.timestamp_ns / 1e9, tz=timezone.utc)
        print(
            f"[{ts.strftime('%H:%M:%S.%f')[:-3]}] "
            f"bid={event.bid:<10.2f} ask={event.ask:<10.2f} "
            f"last={event.last_price:<10.2f} size={event.last_size}"
        )

    bus.subscribe(EventType.TICK, on_tick)

    # Run bus + feed concurrently, stop after timeout
    bus_task = asyncio.create_task(bus.run(), name="bus")
    feed_task = asyncio.create_task(feed.run(), name="feed")

    logger.info("test_starting", seconds=seconds, symbol=config.symbol)

    await asyncio.sleep(seconds)

    feed.stop()
    bus.stop()
    await asyncio.gather(bus_task, feed_task, return_exceptions=True)

    # Summary
    stats = feed.latency_stats
    print(f"\n--- Summary ---")
    print(f"Ticks received: {feed.tick_count}")
    print(f"Latency (ms): min={stats['min_ms']:.1f}  avg={stats['avg_ms']:.1f}  "
          f"max={stats['max_ms']:.1f}  p99={stats['p99_ms']:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Tradovate live feed")
    parser.add_argument("--seconds", type=int, default=30, help="Run duration in seconds")
    parser.add_argument("--symbol", type=str, default="", help="Symbol to subscribe (default: from config)")
    parser.add_argument("--live", action="store_true", help="Use live API instead of demo")
    args = parser.parse_args()
    asyncio.run(main(args.seconds, args.symbol, live=args.live))
