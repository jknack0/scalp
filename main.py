"""MES Scalping Bot -- Main entrypoint.

Modes:
    python main.py              # Paper trading (default)
    python main.py --live       # Live trading (Tradovate demo account)
    python main.py --strategy orb  # Run only ORB strategy
"""

import argparse
import asyncio
import signal
import sys

from dotenv import load_dotenv

load_dotenv()

from src.core.config import BotConfig
from src.core.events import EventBus
from src.core.logging import configure_logging, get_logger
from src.core.session import SessionManager
from src.core.signal_handler import SignalHandler
from src.core.tick_aggregator import TickAggregator
from src.filters.spread_monitor import SpreadConfig, SpreadMonitor
from src.feeds.databento_feed import DatabentoFeed
from src.feeds.tradovate import TradovateFeed
from src.features.feature_hub import FeatureHub
from src.monitoring.health import HealthMonitor
from src.oms.fill_monitor import FillMonitor
from src.oms.tradovate_oms import TradovateOMS
from src.risk.risk_manager import RiskManager


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MES Scalping Bot")
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live order submission (default: paper trading)",
    )
    parser.add_argument(
        "--strategy", nargs="*", default=None,
        help="Strategy names to run (default: all). Options: orb",
    )
    parser.add_argument(
        "--bar-interval", type=float, default=1.0,
        help="Bar aggregation interval in seconds (default: 1)",
    )
    return parser.parse_args()


def _build_strategies(names: list[str] | None) -> list:
    """Instantiate requested strategies with tuned configs."""
    from src.strategies.orb_strategy import ORBConfig, ORBStrategy
    from src.strategies.vwap_strategy import VWAPConfig, VWAPStrategy

    if names is None:
        names = ["orb", "vwap"]

    strategies = []
    for name in names:
        name = name.lower()
        hub = FeatureHub()
        if name == "orb":
            cfg = ORBConfig(
                require_hmm_states=[], min_confidence=0.0,
                target_multiplier=1.0, volume_multiplier=1.5,
                max_signal_time="10:30", expiry_minutes=60,
                require_vwap_alignment=True, max_signals_per_day=1,
            )
            strategies.append(ORBStrategy(cfg, hub))
        elif name == "vwap":
            cfg = VWAPConfig(
                reversion_hmm_states=[], pullback_hmm_states=[],
                min_confidence=0.3, entry_sd_reversion=1.5,
                stop_sd=2.0, pullback_entry_sd=0.5,
                mode_cooldown_bars=5, max_signals_per_day=4,
                expiry_minutes=30,
            )
            strategies.append(VWAPStrategy(cfg, hub))
        else:
            print(f"Unknown strategy: {name}. Available: orb, vwap")
            sys.exit(1)

    return strategies


async def main() -> None:
    args = _parse_args()
    config = BotConfig.from_yaml()
    configure_logging(log_level=config.log_level, log_file=config.log_file)
    logger = get_logger("main")

    paper = not args.live
    mode = "PAPER" if paper else ("DEMO" if config.tradovate_demo else "LIVE")

    logger.info(
        "bot_starting",
        symbol=config.symbol,
        mode=mode,
        max_daily_loss=config.max_daily_loss_usd,
    )

    # ── Core components ──────────────────────────────────────
    bus = EventBus()
    session = SessionManager(
        session_start=config.session_start,
        session_end=config.session_end,
        timezone=config.timezone,
    )
    risk = RiskManager(
        max_daily_loss_usd=config.max_daily_loss_usd,
        max_position_contracts=config.max_position_contracts,
        max_signals_per_day=config.max_signals_per_day,
    )
    health = HealthMonitor(event_bus=bus, risk_manager=risk, port=config.health_port)

    # ── OMS ──────────────────────────────────────────────────
    oms = TradovateOMS(bus, config, paper=paper)
    await oms.initialize()

    # ── Strategies ───────────────────────────────────────────
    strategies = _build_strategies(args.strategy)
    strategy_names = [s.config.strategy_id for s in strategies]
    logger.info("strategies_loaded", strategies=strategy_names)

    # ── Spread filter (ORB only — hurts VWAP) ───────────────
    spread_monitor = SpreadMonitor(config=SpreadConfig(z_threshold=2.0))

    # ── Signal handler (strategies -> risk -> OMS) ────────────
    handler = SignalHandler(bus, strategies, risk, oms, spread_monitor=spread_monitor)
    handler.wire()

    # ── Tick aggregator (ticks → bars) ───────────────────────
    aggregator = TickAggregator(
        bus,
        symbol=config.symbol,
        interval_seconds=args.bar_interval,
    )
    from src.core.events import EventType
    bus.subscribe(EventType.TICK, aggregator.on_tick)

    # ── Fill monitor ─────────────────────────────────────────
    fill_monitor = FillMonitor(bus, oms)

    # ── Feed (Databento for market data, Tradovate for orders only) ──
    feed: DatabentoFeed | None = None
    if config.databento_api_key:
        feed = DatabentoFeed(event_bus=bus, config=config)
    else:
        logger.warning("feed_skipped", reason="DATABENTO_API_KEY not configured")

    # ── Session callbacks ────────────────────────────────────
    async def _on_session_open() -> None:
        risk.reset_daily()
        for s in strategies:
            s.reset()

    async def _on_session_close() -> None:
        await handler.session_close()

    session.on_session_open(_on_session_open)
    session.on_session_close(_on_session_close)

    # ── Graceful shutdown ────────────────────────────────────
    stop_event = asyncio.Event()

    def _signal_handler(sig: int, frame: object) -> None:
        logger.info("shutdown_signal", signal=sig)
        stop_event.set()
        bus.stop()
        session.stop()
        aggregator.stop()
        fill_monitor.stop()
        if feed:
            feed.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _signal_handler)

    logger.info(
        "bot_ready",
        mode=mode,
        strategies=strategy_names,
        bar_interval=args.bar_interval,
        session_active=session.is_rth(),
    )

    # ── Run all components concurrently ──────────────────────
    tasks = [
        asyncio.create_task(bus.run(), name="event_bus"),
        asyncio.create_task(session.run(), name="session_manager"),
        asyncio.create_task(health.start(), name="health_monitor"),
        asyncio.create_task(aggregator.run(), name="tick_aggregator"),
        asyncio.create_task(fill_monitor.run(), name="fill_monitor"),
    ]
    if feed:
        tasks.append(asyncio.create_task(feed.run(), name="tradovate_feed"))

    # Wait for shutdown signal
    await stop_event.wait()

    # Cleanup
    logger.info("bot_shutting_down")
    await oms.close()

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("bot_stopped")


if __name__ == "__main__":
    asyncio.run(main())
