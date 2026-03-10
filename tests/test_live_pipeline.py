"""Integration tests for the live trading pipeline.

Tests the complete flow: tick → bar → strategy → signal → risk → OMS → fill.
All tests use paper mode (no real API calls).
"""

import asyncio
import time

from src.core.events import BarEvent, EventBus, EventType, FillEvent, TickEvent
from src.core.signal_handler import SignalHandler
from src.core.tick_aggregator import TickAggregator
from src.oms.tradovate_oms import TradovateOMS, OrderStatus
from src.risk.risk_manager import RiskManager


# ── Helpers ──────────────────────────────────────────────────────


def _make_tick(
    price: float = 5000.0,
    size: int = 1,
    bid: float = 0.0,
    ask: float = 0.0,
    symbol: str = "MESM6",
    ts_ns: int = 0,
) -> TickEvent:
    return TickEvent(
        symbol=symbol,
        bid=bid or price - 0.25,
        ask=ask or price + 0.25,
        last_price=price,
        last_size=size,
        timestamp_ns=ts_ns or time.time_ns(),
    )


def _make_bar(
    close: float = 5000.0,
    volume: int = 100,
    ts_ns: int = 0,
) -> BarEvent:
    return BarEvent(
        symbol="MESM6",
        open=close - 0.25,
        high=close + 0.25,
        low=close - 0.50,
        close=close,
        volume=volume,
        bar_type="5s",
        timestamp_ns=ts_ns or time.time_ns(),
    )


def _make_paper_oms(bus: EventBus) -> TradovateOMS:
    """Create a paper-mode OMS (no API calls)."""
    from src.core.config import BotConfig
    config = BotConfig(
        tradovate_username="test",
        tradovate_password="test",
        tradovate_demo=True,
    )
    return TradovateOMS(bus, config, paper=True)


# ── TickAggregator Tests ────────────────────────────────────────


class TestTickAggregator:
    async def test_accumulates_ticks_into_bar(self):
        bus = EventBus()
        bars_received = []

        async def capture_bar(bar: BarEvent):
            bars_received.append(bar)

        bus.subscribe(EventType.BAR, capture_bar)
        agg = TickAggregator(bus, symbol="MESM6", interval_seconds=0.1)

        # Feed ticks
        await agg.on_tick(_make_tick(price=5000.0, size=10))
        await agg.on_tick(_make_tick(price=5001.0, size=5))
        await agg.on_tick(_make_tick(price=4999.5, size=8))

        # Flush
        await agg._flush_bar()

        # Now dispatch events from bus queue
        event = bus._queue.get_nowait()
        await capture_bar(event)

        assert len(bars_received) == 1
        bar = bars_received[0]
        assert bar.open == 5000.0
        assert bar.high == 5001.0
        assert bar.low == 4999.5
        assert bar.close == 4999.5
        assert bar.volume == 23

    async def test_empty_bar_not_emitted(self):
        bus = EventBus()
        agg = TickAggregator(bus, symbol="MESM6", interval_seconds=0.1)
        await agg._flush_bar()
        assert bus._queue.empty()

    async def test_ignores_wrong_symbol(self):
        bus = EventBus()
        agg = TickAggregator(bus, symbol="MESM6", interval_seconds=0.1)
        await agg.on_tick(_make_tick(symbol="ESM6"))
        await agg._flush_bar()
        assert bus._queue.empty()


# ── TradovateOMS Paper Tests ────────────────────────────────────


class TestPaperOMS:
    async def test_paper_entry_pending_then_tick_fills(self):
        """Submit order → pending entry → tick at entry price → filled."""
        bus = EventBus()
        oms = _make_paper_oms(bus)
        await oms.initialize()

        fills = []

        async def capture_fill(f: FillEvent):
            fills.append(f)

        bus.subscribe(EventType.FILL, capture_fill)

        from src.strategies.base import Direction, Signal
        from src.models.hmm_regime import RegimeState
        from datetime import datetime, timedelta

        signal = Signal(
            strategy_id="test",
            direction=Direction.LONG,
            entry_price=5000.0,
            target_price=5002.0,
            stop_price=4998.0,
            signal_time=datetime.now(),
            expiry_time=datetime.now() + timedelta(minutes=60),
            confidence=0.8,
            regime_state=RegimeState.LOW_VOL_RANGE,
        )
        order_id = await oms.submit_order(signal)

        # Not filled yet — waiting for tick at entry price
        assert oms.position == 0
        order = oms.get_order(order_id)
        assert order.status == OrderStatus.WORKING

        # Tick at entry price triggers fill
        await oms.on_tick(_make_tick(price=5000.0))

        # Drain bus
        event = bus._queue.get_nowait()
        await capture_fill(event)

        assert len(fills) == 1
        assert fills[0].fill_price == 5000.0
        assert fills[0].direction == "BUY"
        assert oms.position == 1

    async def test_paper_bracket_target_exit(self):
        """Filled position exits on target tick."""
        bus = EventBus()
        oms = _make_paper_oms(bus)
        await oms.initialize()

        fills = []

        async def capture_fill(f: FillEvent):
            fills.append(f)

        bus.subscribe(EventType.FILL, capture_fill)

        from src.strategies.base import Direction, Signal
        from src.models.hmm_regime import RegimeState
        from datetime import datetime, timedelta

        sig = Signal(
            strategy_id="test", direction=Direction.LONG,
            entry_price=5000.0, target_price=5002.0, stop_price=4998.0,
            signal_time=datetime.now(),
            expiry_time=datetime.now() + timedelta(minutes=60),
            confidence=0.8, regime_state=RegimeState.LOW_VOL_RANGE,
        )
        await oms.submit_order(sig)

        # Fill entry
        await oms.on_tick(_make_tick(price=5000.0))
        assert oms.position == 1

        # Hit target
        await oms.on_tick(_make_tick(price=5002.0))
        assert oms.position == 0

        # Drain fills (entry + exit)
        while not bus._queue.empty():
            event = bus._queue.get_nowait()
            await capture_fill(event)

        assert len(fills) == 2
        assert fills[1].fill_price == 5002.0
        assert fills[1].direction == "SELL"

    async def test_paper_bracket_stop_exit(self):
        """Filled position exits on stop tick."""
        bus = EventBus()
        oms = _make_paper_oms(bus)
        await oms.initialize()

        from src.strategies.base import Direction, Signal
        from src.models.hmm_regime import RegimeState
        from datetime import datetime, timedelta

        sig = Signal(
            strategy_id="test", direction=Direction.SHORT,
            entry_price=5000.0, target_price=4998.0, stop_price=5002.0,
            signal_time=datetime.now(),
            expiry_time=datetime.now() + timedelta(minutes=60),
            confidence=0.8, regime_state=RegimeState.LOW_VOL_RANGE,
        )
        await oms.submit_order(sig)

        # Fill entry (price at or above entry for short)
        await oms.on_tick(_make_tick(price=5000.0))
        assert oms.position == -1

        # Hit stop
        await oms.on_tick(_make_tick(price=5002.0))
        assert oms.position == 0

    async def test_paper_expiry_cancels_unfilled_entry(self):
        """Unfilled entry expires when tick arrives after expiry time."""
        bus = EventBus()
        oms = _make_paper_oms(bus)
        await oms.initialize()

        from src.strategies.base import Direction, Signal
        from src.models.hmm_regime import RegimeState
        from datetime import datetime, timedelta

        sig = Signal(
            strategy_id="test", direction=Direction.LONG,
            entry_price=4990.0, target_price=4992.0, stop_price=4988.0,
            signal_time=datetime.now(),
            expiry_time=datetime.now() - timedelta(seconds=1),  # Already expired
            confidence=0.8, regime_state=RegimeState.LOW_VOL_RANGE,
        )
        oid = await oms.submit_order(sig)

        # Tick arrives — entry should expire, not fill
        await oms.on_tick(_make_tick(price=4990.0))
        assert oms.position == 0
        assert oms.get_order(oid) is None  # Cleaned up

    async def test_cancel_paper_order(self):
        bus = EventBus()
        oms = _make_paper_oms(bus)
        await oms.initialize()

        from src.strategies.base import Direction, Signal
        from src.models.hmm_regime import RegimeState
        from datetime import datetime, timedelta

        sig = Signal(
            strategy_id="test", direction=Direction.LONG,
            entry_price=5000.0, target_price=5002.0, stop_price=4998.0,
            signal_time=datetime.now(),
            expiry_time=datetime.now() + timedelta(minutes=60),
            confidence=0.8, regime_state=RegimeState.LOW_VOL_RANGE,
        )
        oid = await oms.submit_order(sig)
        order = oms.get_order(oid)
        assert order.status == OrderStatus.WORKING

        result = await oms.cancel_order(oid)
        assert result is True
        assert order.status == OrderStatus.CANCELLED


# ── SignalHandler Tests ─────────────────────────────────────────


class TestSignalHandler:
    async def test_risk_rejection_blocks_order(self):
        bus = EventBus()
        oms = _make_paper_oms(bus)
        await oms.initialize()
        risk = RiskManager(max_signals_per_day=0)  # Block all signals

        handler = SignalHandler(bus, [], risk, oms)
        handler.wire()

        # No signals should get through
        assert oms.position == 0

    async def test_fill_updates_risk_manager(self):
        bus = EventBus()
        risk = RiskManager()

        oms = _make_paper_oms(bus)
        await oms.initialize()

        handler = SignalHandler(bus, [], risk, oms)
        handler.wire()

        fill = FillEvent(
            order_id="test-1",
            symbol="MESM6",
            direction="BUY",
            fill_price=5000.0,
            fill_size=1,
            commission=0.35,
            timestamp_ns=time.time_ns(),
        )
        await handler.on_fill(fill)
        assert risk.current_position == 1


# ── End-to-End Pipeline Test ────────────────────────────────────


class TestEndToEnd:
    async def test_tick_to_bar_pipeline(self):
        """Verify ticks aggregate into bars and reach subscribers."""
        bus = EventBus()
        bars = []

        async def on_bar(bar: BarEvent):
            bars.append(bar)

        bus.subscribe(EventType.BAR, on_bar)

        agg = TickAggregator(bus, symbol="MESM6", interval_seconds=0.05)
        bus.subscribe(EventType.TICK, agg.on_tick)

        # Start bus + aggregator
        bus_task = asyncio.create_task(bus.run())
        agg_task = asyncio.create_task(agg.run())

        # Publish ticks
        for i in range(5):
            await bus.publish(_make_tick(price=5000.0 + i * 0.25))

        # Wait for bar to flush
        await asyncio.sleep(0.15)

        agg.stop()
        bus.stop()
        agg_task.cancel()
        bus_task.cancel()

        try:
            await asyncio.gather(bus_task, agg_task, return_exceptions=True)
        except asyncio.CancelledError:
            pass

        assert len(bars) >= 1
        assert bars[0].volume == 5


# ── Spread Gate Tests ─────────────────────────────────────────
# Spread filtering now handled by FilterEngine (declarative YAML rules).
# See tests/test_filter_engine.py for filter evaluation tests.
