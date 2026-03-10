"""MES Scalping Bot -- Main entrypoint.

Modes:
    python main.py              # Paper trading (default)
    python main.py --live       # Live trading (Tradovate demo account)
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
from src.feeds.databento_feed import DatabentoFeed
from src.feeds.tradovate import TradovateFeed
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
        help="Strategy names to run (default: all). Options: vwap_band, gap, va",
    )
    parser.add_argument(
        "--bar-interval", type=float, default=1.0,
        help="Bar aggregation interval in seconds (default: 1)",
    )
    return parser.parse_args()


def _build_strategies(names: list[str] | None) -> list:
    """Instantiate requested strategies from YAML configs."""
    from src.strategies.vwap_band_reversion import VWAPBandReversionStrategy
    from src.strategies.gap_fill import GapFillStrategy
    from src.strategies.value_area_reversion import ValueAreaReversionStrategy

    if names is None:
        names = ["vwap_band", "gap", "va"]

    strategies = []
    for name in names:
        name = name.lower()
        if name == "vwap_band":
            strategies.append(VWAPBandReversionStrategy.from_yaml("config/strategies/vwap_band_reversion.yaml"))
        elif name == "gap":
            strategies.append(GapFillStrategy.from_yaml("config/strategies/gap_fill.yaml"))
        elif name == "va":
            strategies.append(ValueAreaReversionStrategy.from_yaml("config/strategies/value_area_reversion.yaml"))
        else:
            print(f"Unknown strategy: {name}. Available: vwap_band, gap, va")
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
    strategy_names = [s.strategy_id for s in strategies]
    logger.info("strategies_loaded", strategies=strategy_names)

    # ── Signal + Filter engines (loaded from strategy YAML) ───
    from config.loader import build_filter_engine, build_signal_engine, load_strategy_config
    from src.signals.signal_bundle import SignalEngine
    # Use first strategy's YAML for signal/filter config
    strategy_yaml_map = {
        "vwap_band": "vwap_band_reversion",
        "gap": "gap_fill",
        "va": "value_area_reversion",
    }
    first_strat = (args.strategy or ["vwap_band"])[0].lower()
    yaml_cfg = load_strategy_config(strategy_yaml_map.get(first_strat, first_strat))
    signal_engine = build_signal_engine(yaml_cfg)
    filter_engine = build_filter_engine(yaml_cfg)

    # ── Signal handler (strategies -> risk -> OMS) ────────────
    handler = SignalHandler(
        bus, strategies, risk, oms,
        signal_engine=signal_engine,
        filter_engine=filter_engine,
    )
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
