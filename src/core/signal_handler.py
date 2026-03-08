"""Signal handler: bridges strategy signals to the OMS via risk checks.

Subscribes to BAR events, feeds them to strategies, takes any generated
signals through risk validation, and submits approved orders to the OMS.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.core.events import BarEvent, EventBus, EventType, FillEvent, TickEvent
from src.core.logging import get_logger
from src.filters.spread_monitor import SpreadConfig, SpreadMonitor, SpreadSnapshot
from src.filters.vpin_monitor import VPINMonitor
from src.risk.risk_manager import RiskManager
from src.strategies.base import Signal, StrategyBase

if TYPE_CHECKING:
    from src.oms.tradovate_oms import TradovateOMS

logger = get_logger("signal_handler")


class SignalHandler:
    """Orchestrates strategies, risk checks, and order submission.

    Lifecycle per bar:
    1. BAR event arrives from TickAggregator
    2. Each strategy processes the bar (may generate a Signal)
    3. Signal goes through RiskManager validation
    4. Approved signals are submitted to the OMS
    5. Fill events update the RiskManager

    Also handles position management: monitors open positions for
    target/stop exit (in paper mode), and session-end flattening.
    """

    def __init__(
        self,
        event_bus: EventBus,
        strategies: list[StrategyBase],
        risk_manager: RiskManager,
        oms: TradovateOMS,
        spread_monitor: SpreadMonitor | None = None,
        vpin_monitor: VPINMonitor | None = None,
    ) -> None:
        self._bus = event_bus
        self._strategies = strategies
        self._risk = risk_manager
        self._oms = oms
        self._spread = spread_monitor or SpreadMonitor()
        self._vpin = vpin_monitor

        # Track pending paper positions for target/stop management
        self._paper_entries: list[_PaperPosition] = []

    def wire(self) -> None:
        """Subscribe to relevant events on the bus."""
        self._bus.subscribe(EventType.BAR, self.on_bar)
        self._bus.subscribe(EventType.TICK, self.on_tick)
        self._bus.subscribe(EventType.FILL, self.on_fill)
        logger.info(
            "signal_handler_wired",
            strategies=[s.config.strategy_id for s in self._strategies],
            paper=self._oms.is_paper,
        )

    async def on_tick(self, tick: TickEvent) -> None:
        """Feed VPIN monitor and monitor open paper positions for target/stop hits."""
        # Feed tick to VPIN monitor
        if self._vpin is not None and tick.last_size > 0:
            self._vpin.on_tick(
                price=tick.last_price,
                size=tick.last_size,
                bid=tick.bid,
                ask=tick.ask,
            )

        if not self._oms.is_paper or not self._paper_entries:
            return

        price = tick.last_price
        if price <= 0:
            return

        closed = []
        for pp in self._paper_entries:
            hit = False
            if pp.direction == "Buy":
                if price >= pp.target:
                    pp.exit_price = pp.target
                    hit = True
                elif price <= pp.stop:
                    pp.exit_price = pp.stop
                    hit = True
            else:  # Sell
                if price <= pp.target:
                    pp.exit_price = pp.target
                    hit = True
                elif price >= pp.stop:
                    pp.exit_price = pp.stop
                    hit = True

            if hit:
                closed.append(pp)
                exit_dir = "Sell" if pp.direction == "Buy" else "Buy"
                fill = FillEvent(
                    order_id=f"{pp.order_id}-exit",
                    symbol=tick.symbol,
                    direction="SELL" if exit_dir == "Sell" else "BUY",
                    fill_price=pp.exit_price,
                    fill_size=1,
                    commission=0.35,
                    timestamp_ns=tick.timestamp_ns,
                )
                await self._bus.publish(fill)
                self._oms._position += -1 if pp.direction == "Buy" else 1

                pnl = (pp.exit_price - pp.entry) if pp.direction == "Buy" else (pp.entry - pp.exit_price)
                pnl_usd = pnl / 0.25 * 1.25  # convert points to dollars
                logger.info(
                    "paper_exit",
                    order_id=pp.order_id,
                    strategy=pp.strategy_id,
                    direction=pp.direction,
                    entry=pp.entry,
                    exit=pp.exit_price,
                    pnl_ticks=pnl / 0.25,
                    pnl_usd=round(pnl_usd, 2),
                )

        for pp in closed:
            self._paper_entries.remove(pp)

    async def on_bar(self, bar: BarEvent) -> None:
        """Feed bar to all strategies and handle any signals."""
        # Update spread monitor with bar's average bid/ask prices
        if bar.avg_bid_size > 0 and bar.avg_ask_size > 0:
            ts = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=timezone.utc)
            snap = SpreadSnapshot(timestamp=ts, bid=bar.avg_bid_size, ask=bar.avg_ask_size)
            await self._spread.push(snap)

        for strategy in self._strategies:
            try:
                strategy.on_bar(bar)
            except Exception:
                logger.exception(
                    "strategy_error",
                    strategy=strategy.config.strategy_id,
                )
                continue

            # Check if strategy generated a new signal
            if strategy._signals_generated and strategy._signals_generated[-1]:
                signal = strategy._signals_generated[-1]
                # Only process if it's from this bar (avoid reprocessing)
                if self._is_fresh_signal(signal, bar):
                    await self._process_signal(signal, bar)

    async def on_fill(self, fill: FillEvent) -> None:
        """Update risk manager with fill information."""
        self._risk.record_fill(fill)

    async def _process_signal(self, signal: Signal, bar: BarEvent) -> None:
        """Risk-check and submit a signal."""
        # Hard gate: spread filter
        spread_ok, spread_reason = self._spread.is_spread_normal()
        if not spread_ok:
            logger.info(
                "signal_blocked_spread",
                strategy=signal.strategy_id,
                reason=spread_reason,
            )
            return

        # Regime gate: VPIN filter
        if self._vpin is not None:
            vpin_blocked, vpin_reason = self._vpin.should_block(signal.strategy_id)
            if vpin_blocked:
                logger.info(
                    "signal_blocked_vpin",
                    strategy=signal.strategy_id,
                    reason=vpin_reason,
                )
                return

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
        from src.core.session import SessionManager
        session_valid = True  # Assume valid if we're receiving bars during RTH

        result = self._risk.check_order(
            risk_signal,
            self._oms.position,
            session_valid,
        )

        if not result.approved:
            logger.info(
                "signal_rejected",
                strategy=signal.strategy_id,
                direction=direction,
                reason=result.reason,
            )
            return

        # Submit to OMS
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

        # Track for paper target/stop monitoring
        if self._oms.is_paper:
            tv_dir = "Buy" if direction == "BUY" else "Sell"
            self._paper_entries.append(_PaperPosition(
                order_id=order_id,
                strategy_id=signal.strategy_id,
                direction=tv_dir,
                entry=signal.entry_price,
                target=signal.target_price,
                stop=signal.stop_price,
            ))

    def _is_fresh_signal(self, signal: Signal, bar: BarEvent) -> bool:
        """Check if signal was generated on this bar (not a stale reprocess)."""
        # Simple: compare signal time to bar time (within 2 bar intervals)
        if not hasattr(signal, 'signal_time'):
            return True
        signal_ns = int(signal.signal_time.timestamp() * 1e9)
        return abs(signal_ns - bar.timestamp_ns) < 120_000_000_000  # 2 minutes

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

        # Clear paper positions
        self._paper_entries.clear()

        # Reset strategies for next day
        for s in self._strategies:
            s.reset()

        logger.info("session_closed")


class _PaperPosition:
    """Tracks an open paper position for target/stop monitoring."""

    __slots__ = ("order_id", "strategy_id", "direction", "entry", "target", "stop", "exit_price")

    def __init__(
        self,
        order_id: str,
        strategy_id: str,
        direction: str,
        entry: float,
        target: float,
        stop: float,
    ) -> None:
        self.order_id = order_id
        self.strategy_id = strategy_id
        self.direction = direction
        self.entry = entry
        self.target = target
        self.stop = stop
        self.exit_price = 0.0
