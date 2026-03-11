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
        help="Strategy names to run (default: vwap_band, gap, va). "
             "Options: vwap_band, gap, va, orb, cvd, ttm, macd, poc_va, "
             "stoch_bb, ema_ribbon, micro, regime, pdh_pdl, donchian, mfi_obv, ib",
    )
    parser.add_argument(
        "--bar-interval", type=float, default=1.0,
        help="Bar aggregation interval in seconds (default: 1)",
    )
    return parser.parse_args()


_STRATEGY_MAP: dict[str, tuple[str, str]] = {
    # name -> (module_path, yaml_path)
    "vwap_band": ("src.strategies.vwap_band_reversion:VWAPBandReversionStrategy", "config/strategies/vwap_band_reversion.yaml"),
    "gap": ("src.strategies.gap_fill:GapFillStrategy", "config/strategies/gap_fill.yaml"),
    "va": ("src.strategies.value_area_reversion:ValueAreaReversionStrategy", "config/strategies/value_area_reversion.yaml"),
    "orb": ("src.strategies.orb_breakout:ORBBreakoutStrategy", "config/strategies/orb_breakout.yaml"),
    "cvd": ("src.strategies.cvd_divergence:CVDDivergenceStrategy", "config/strategies/cvd_divergence.yaml"),
    "ttm": ("src.strategies.ttm_squeeze:TTMSqueezeStrategy", "config/strategies/ttm_squeeze.yaml"),
    "macd": ("src.strategies.macd_zero_line:MACDZeroLineStrategy", "config/strategies/macd_zero_line.yaml"),
    "poc_va": ("src.strategies.poc_va_bounce:POCVABounceStrategy", "config/strategies/poc_va_bounce.yaml"),
    "stoch_bb": ("src.strategies.stoch_bb_fade:StochBBFadeStrategy", "config/strategies/stoch_bb_fade.yaml"),
    "ema_ribbon": ("src.strategies.ema_ribbon_pullback:EmaRibbonPullbackStrategy", "config/strategies/ema_ribbon_pullback.yaml"),
    "micro": ("src.strategies.micro_pullback:MicroPullbackStrategy", "config/strategies/micro_pullback.yaml"),
    "regime": ("src.strategies.regime_switcher:RegimeSwitcherStrategy", "config/strategies/regime_switcher.yaml"),
    "pdh_pdl": ("src.strategies.pdh_pdl_fade:PDHPDLFadeStrategy", "config/strategies/pdh_pdl_fade.yaml"),
    "donchian": ("src.strategies.donchian_breakout:DonchianBreakoutStrategy", "config/strategies/donchian_breakout.yaml"),
    "mfi_obv": ("src.strategies.mfi_obv_divergence:MFIOBVDivergenceStrategy", "config/strategies/mfi_obv_divergence.yaml"),
    "ib": ("src.strategies.ib_fade:IBFadeStrategy", "config/strategies/ib_fade.yaml"),
}


def _build_strategies(names: list[str] | None) -> list:
    """Instantiate requested strategies from YAML configs."""
    import importlib

    if names is None:
        names = ["vwap_band", "gap", "va"]

    available = ", ".join(sorted(_STRATEGY_MAP.keys()))
    strategies = []
    for name in names:
        name = name.lower()
        if name not in _STRATEGY_MAP:
            print(f"Unknown strategy: {name}. Available: {available}")
            sys.exit(1)
        module_class, yaml_path = _STRATEGY_MAP[name]
        mod_path, cls_name = module_class.rsplit(":", 1)
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        strategies.append(cls.from_yaml(yaml_path))

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
    first_strat = (args.strategy or ["vwap_band"])[0].lower()
    _, yaml_path = _STRATEGY_MAP.get(first_strat, (None, None))
    yaml_name = yaml_path.replace("config/strategies/", "").replace(".yaml", "") if yaml_path else first_strat
    yaml_cfg = load_strategy_config(yaml_name)
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
