"""Signal handler: bridges strategy signals to the OMS via risk checks.

Subscribes to BAR events, feeds them to strategies, takes any generated
signals through risk validation, and submits approved orders to the OMS.

Paper mode exit monitoring is handled entirely by TradovateOMS.on_tick(),
matching the backtest SimulatedOMS bracket model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.events import BarEvent, EventBus, EventType, FillEvent, TickEvent
from src.core.logging import get_logger
from src.data.trade_store import TradeStore
from src.risk.risk_manager import RiskManager
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle, SignalEngine
from src.filters.filter_engine import FilterEngine
from src.strategies.base import Signal, StrategyBase

if TYPE_CHECKING:
    from src.oms.tradovate_oms import TradovateOMS

logger = get_logger("signal_handler")


class SignalHandler:
    """Orchestrates strategies, risk checks, and order submission.

    Lifecycle per bar:
    1. BAR event arrives from TickAggregator
    2. SignalEngine computes signals, FilterEngine gates entry
    3. Each strategy processes the bar (may generate a Signal)
    4. Signal goes through RiskManager validation
    5. Approved signals are submitted to the OMS as bracket orders
    6. OMS manages the full bracket lifecycle (entry fill, target/stop/expiry)
    """

    def __init__(
        self,
        event_bus: EventBus,
        strategies: list[StrategyBase],
        risk_manager: RiskManager,
        oms: TradovateOMS,
        signal_engine: SignalEngine | None = None,
        filter_engine: FilterEngine | None = None,
        trade_store: TradeStore | None = None,
    ) -> None:
        self._bus = event_bus
        self._strategies = strategies
        self._risk = risk_manager
        self._oms = oms
        self._signal_engine = signal_engine
        self._filter_engine = filter_engine or FilterEngine()
        self._trade_store = trade_store
        self._bar_window: list[BarEvent] = []
        self._bar_count: int = 0

    def wire(self) -> None:
        """Subscribe to relevant events on the bus."""
        self._bus.subscribe(EventType.BAR, self.on_bar)
        self._bus.subscribe(EventType.TICK, self.on_tick)
        self._bus.subscribe(EventType.FILL, self.on_fill)
        logger.info(
            "signal_handler_wired",
            strategies=[getattr(s, 'strategy_id', getattr(getattr(s, 'config', None), 'strategy_id', '?')) for s in self._strategies],
            paper=self._oms.is_paper,
        )

    def warmup_from_postgres(self, symbol: str, bars: int = 500) -> None:
        """Pre-feed historical bars from Postgres to warm up signals.

        Reads the last N 1m bars from bars_1m, converts to BarEvents,
        and runs the signal engine on them so regime detector, ADX, ATR,
        VWAP, etc. all have context on first live bar.
        """
        import os
        dsn = os.environ.get("DATABASE_URL", "")
        if not dsn:
            return
        if self._signal_engine is None:
            return

        try:
            import psycopg2
            conn = psycopg2.connect(dsn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT timestamp, open, high, low, close, volume "
                    "FROM bars_1m WHERE symbol = %s "
                    "ORDER BY timestamp DESC LIMIT %s",
                    (symbol, bars),
                )
                rows = cur.fetchall()
            conn.close()
        except Exception as e:
            logger.warning("warmup_db_error", error=str(e))
            return

        if not rows:
            logger.info("warmup_no_data", symbol=symbol)
            return

        # Rows are DESC, reverse to chronological
        rows.reverse()

        # Convert to BarEvents and feed to signal engine
        for ts, o, h, l, c, vol in rows:
            bar = BarEvent(
                symbol=symbol,
                open=o, high=h, low=l, close=c,
                volume=vol,
                bar_type="1m",
                timestamp_ns=int(ts.timestamp() * 1e9),
            )
            self._bar_window.append(bar)

        # Trim to max window
        if len(self._bar_window) > 500:
            self._bar_window = self._bar_window[-500:]

        # Run signal engine once on the full window to warm up all signals
        self._signal_engine.compute(self._bar_window)

        logger.info("warmup_complete", bars_loaded=len(rows),
                     window_size=len(self._bar_window))

    async def on_tick(self, tick: TickEvent) -> None:
        """Forward ticks to OMS for paper bracket monitoring."""
        await self._oms.on_tick(tick)

    async def on_bar(self, bar: BarEvent) -> None:
        """Feed bar to all strategies and handle any signals."""
        self._bar_count += 1

        # Compute signals bundle
        bundle = EMPTY_BUNDLE
        if self._signal_engine is not None:
            self._bar_window.append(bar)
            if len(self._bar_window) > 500:
                self._bar_window = self._bar_window[-500:]
            bundle = self._signal_engine.compute(self._bar_window)

            # Evaluate filters
            filter_result = self._filter_engine.evaluate(bundle)

            # Log signal + filter snapshot every 60 bars (~1 min)
            if self._bar_count % 60 == 0:
                self._log_snapshot(bar, bundle, filter_result)

            if not filter_result.passes:
                return

        for strategy in self._strategies:
            try:
                signal = strategy.on_bar(bar, bundle)
            except TypeError:
                signal = strategy.on_bar(bar)
            except Exception:
                sid = getattr(strategy, 'strategy_id', getattr(getattr(strategy, 'config', None), 'strategy_id', '?'))
                logger.exception("strategy_error", strategy=sid)
                continue

            if signal is not None:
                await self._process_signal(signal, bar)

    async def on_fill(self, fill: FillEvent) -> None:
        """Update risk manager with fill information."""
        self._risk.record_fill(fill)

    async def _process_signal(self, signal: Signal, bar: BarEvent) -> None:
        """Risk-check and submit a signal as a bracket order."""
        # Convert Signal direction to risk manager format
        direction = "BUY" if signal.direction.value == "LONG" else "SELL"

        # Create a lightweight signal event for risk check
        from src.core.events import SignalEvent
        risk_signal = SignalEvent(
            strategy_id=signal.strategy_id,
            direction=direction,
            strength=signal.confidence,
            reason=f"entry={signal.entry_price} target={signal.target_price} stop={signal.stop_price}",
            timestamp_ns=bar.timestamp_ns,
        )

        # Risk check
        result = self._risk.check_order(
            risk_signal,
            self._oms.position,
            True,  # Assume valid if we're receiving bars during RTH
        )

        if not result.approved:
            logger.info(
                "signal_rejected",
                strategy=signal.strategy_id,
                direction=direction,
                reason=result.reason,
            )
            return

        # Submit bracket order to OMS (entry + target + stop + expiry)
        order_id = await self._oms.submit_order(signal)

        logger.info(
            "signal_submitted",
            strategy=signal.strategy_id,
            direction=direction,
            entry=signal.entry_price,
            target=signal.target_price,
            stop=signal.stop_price,
            confidence=round(signal.confidence, 3),
            order_id=order_id,
            rr=round(signal.risk_reward_ratio, 2),
        )

        # Record trade entry to Postgres
        if self._trade_store:
            self._trade_store.record_entry(
                order_id=order_id,
                strategy_id=signal.strategy_id,
                symbol=self._oms._config.symbol,
                direction=direction,
                entry_price=signal.entry_price,
                entry_time=signal.signal_time,
                target_price=signal.target_price,
                stop_price=signal.stop_price,
                confidence=signal.confidence,
                risk_reward=signal.risk_reward_ratio,
                regime=signal.regime_state.name if signal.regime_state else "UNKNOWN",
                metadata=signal.metadata,
            )

    def _log_snapshot(self, bar: BarEvent, bundle: SignalBundle, filter_result) -> None:
        """Log current signal values and filter status (periodic, every ~1 min)."""
        snap: dict = {"close": bar.close, "bars": self._bar_count}

        # Key signal values
        for name in ("adx", "atr", "relative_volume", "regime_v2", "vwap_session"):
            result = bundle.get(name)
            if result is None:
                continue
            if name == "regime_v2":
                meta = result.metadata
                snap["regime"] = meta.get("regime", "?")
                snap["regime_conf"] = round(meta.get("confidence", 0), 3)
                snap["regime_size"] = meta.get("position_size", "?")
                snap["regime_halt"] = meta.get("whipsaw_halt", False)
                snap["regime_passes"] = result.passes
            elif name == "vwap_session":
                meta = result.metadata
                snap["vwap_dev_sd"] = round(meta.get("deviation_sd", 0), 2)
                snap["vwap_slope"] = round(meta.get("slope", 0), 4)
                snap["vwap_age"] = meta.get("session_age_bars", 0)
            elif name == "adx":
                snap["adx"] = round(result.value, 1)
            elif name == "atr":
                snap["atr"] = round(result.metadata.get("atr_raw", 0), 2)
            elif name == "relative_volume":
                snap["rvol"] = round(result.value, 2)

        # Filter status
        snap["filters_pass"] = filter_result.passes
        if not filter_result.passes:
            snap["blocked_by"] = filter_result.block_reasons[:3]

        logger.info("signal_snapshot", **snap)

    async def session_close(self) -> None:
        """End-of-session cleanup: flatten position, reset strategies."""
        logger.info("session_closing", position=self._oms.position)

        # Flatten any open position
        if self._oms.position != 0:
            await self._oms.flatten()

        # Cancel working orders
        cancelled = await self._oms.cancel_all()
        if cancelled:
            logger.info("orders_cancelled", count=cancelled)

        # Reset strategies for next day
        for s in self._strategies:
            s.reset()

        logger.info("session_closed")
