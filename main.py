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
# TickAggregator no longer needed — feed emits 1s bars directly via ohlcv-1s
from src.feeds.databento_feed import DatabentoFeed
from src.feeds.tradovate import TradovateFeed
from src.monitoring.health import HealthMonitor
from src.oms.fill_monitor import FillMonitor
from src.oms.tradovate_oms import TradovateOMS
from src.data.bar_store import BarStore
from src.data.trade_store import TradeStore
from src.risk.risk_manager import RiskManager


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MES Scalping Bot")
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live order submission (default: paper trading)",
    )
    parser.add_argument(
        "--strategy", nargs="*", default=None,
        help="Strategy names to run (default: vwap_band, donchian). "
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
        names = ["vwap_band", "donchian"]

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

    # ── Signal + Filter engines (loaded from strategy YAMLs) ───
    from config.loader import build_signal_engine, load_strategy_config
    from src.filters.filter_engine import FilterEngine
    from src.signals.signal_bundle import SignalEngine

    strat_names = args.strategy or ["vwap_band", "donchian"]
    all_signals: list[str] = []
    all_signal_configs: dict = {}
    yaml_cfgs: list[dict] = []
    for sname in strat_names:
        sname = sname.lower()
        _, ypath = _STRATEGY_MAP.get(sname, (None, None))
        if ypath is None:
            continue
        yname = ypath.replace("config/strategies/", "").replace(".yaml", "")
        ycfg = load_strategy_config(yname)
        yaml_cfgs.append(ycfg)
        for sig in ycfg.get("signals", []):
            if sig not in all_signals:
                all_signals.append(sig)
        all_signal_configs.update(ycfg.get("signal_configs", {}))

    signal_engine = SignalEngine(all_signals, all_signal_configs) if all_signals else None
    # Each strategy has its own FilterEngine — use empty handler-level filter
    filter_engine = FilterEngine()
    yaml_cfg = yaml_cfgs[0] if yaml_cfgs else {}
    logger.info("multi_strategy_signals", signals=all_signals)

    # ── Trade store (persist trades to Postgres) ─────────────
    import os
    trade_store: TradeStore | None = None
    if os.environ.get("DATABASE_URL"):
        trade_store = TradeStore()
        oms._trade_store = trade_store
        logger.info("trade_store_enabled")

    # ── Signal handler (strategies -> risk -> OMS) ────────────
    handler = SignalHandler(
        bus, strategies, risk, oms,
        signal_engine=signal_engine,
        filter_engine=filter_engine,
        trade_store=trade_store,
    )

    # ── Bar resampler (1s → strategy freq, e.g. 5m) ──────────
    from src.core.bar_resampler import BarResampler, _freq_to_seconds
    from src.core.events import EventType

    bar_cfg = yaml_cfg.get("bar", {})
    bar_freq = bar_cfg.get("freq", "1s")
    bar_freq_seconds = _freq_to_seconds(bar_freq)

    resampler: BarResampler | None = None
    if bar_freq_seconds > 1:
        resampler = BarResampler(freq_seconds=bar_freq_seconds, callback=handler.on_bar)
        handler.wire(subscribe_bar=False)  # resampler routes bars instead
        bus.subscribe(EventType.BAR, resampler.on_bar)
        logger.info("bar_resampler_enabled", freq=bar_freq, seconds=bar_freq_seconds)
    else:
        handler.wire()

    # Warm up signals from Databento historical bars
    if config.databento_api_key:
        handler.warmup_from_databento(
            symbol=config.symbol,
            api_key=config.databento_api_key,
            bars=350,
            bar_freq=bar_freq,
        )

    # ── Tick aggregator (disabled — feed now emits 1s bars directly) ──
    # TickAggregator was needed when using mbp-1 tick feed.
    # With ohlcv-1s, DatabentoFeed emits BarEvents directly.
    # TickEvents are still emitted (close price) for paper bracket fills.

    # ── Bar store (persist 1m bars to Postgres) ──────────────
    bar_store: BarStore | None = None
    if os.environ.get("DATABASE_URL"):
        bar_store = BarStore(event_bus=bus)
        bus.subscribe(EventType.BAR, bar_store.on_bar)
        logger.info("bar_store_enabled")

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
        asyncio.create_task(fill_monitor.run(), name="fill_monitor"),
    ]
    if feed:
        tasks.append(asyncio.create_task(feed.run(), name="tradovate_feed"))

    # Wait for shutdown signal
    await stop_event.wait()

    # Cleanup
    logger.info("bot_shutting_down")
    if bar_store:
        bar_store.close()
    if trade_store:
        trade_store.close()
    await oms.close()

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("bot_stopped")


if __name__ == "__main__":
    asyncio.run(main())
