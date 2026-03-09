"""Quick test of the Databento live feed.

Usage:
    python scripts/test_databento_feed.py --seconds 15 --symbol MESH6
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.core.config import BotConfig
from src.core.events import EventBus, EventType, TickEvent
from src.core.logging import configure_logging, get_logger
from src.feeds.databento_feed import DatabentoFeed


async def main(seconds: int, symbol: str) -> None:
    configure_logging(log_level="INFO", log_file=None)
    logger = get_logger("test_feed")

    config = BotConfig.from_yaml()
    if symbol:
        config = BotConfig.from_yaml(symbol=symbol)

    if not config.databento_api_key:
        logger.error("missing_key", hint="Set DATABENTO_API_KEY in .env")
        return

    bus = EventBus()
    feed = DatabentoFeed(event_bus=bus, config=config)

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

    bus_task = asyncio.create_task(bus.run(), name="bus")
    feed_task = asyncio.create_task(feed.run(), name="feed")

    logger.info("test_starting", seconds=seconds, symbol=config.symbol)

    await asyncio.sleep(seconds)

    feed.stop()
    bus.stop()
    await asyncio.gather(bus_task, feed_task, return_exceptions=True)

    stats = feed.latency_stats
    print(f"\n--- Summary ---")
    print(f"Ticks received: {feed.tick_count}")
    print(f"Latency (ms): min={stats['min_ms']:.1f}  avg={stats['avg_ms']:.1f}  "
          f"max={stats['max_ms']:.1f}  p99={stats['p99_ms']:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Databento live feed")
    parser.add_argument("--seconds", type=int, default=15, help="Run duration")
    parser.add_argument("--symbol", type=str, default="", help="Symbol (default: from config)")
    args = parser.parse_args()
    asyncio.run(main(args.seconds, args.symbol))
