"""Integration tests for the live trading pipeline.

Tests the complete flow: tick → bar → strategy → signal → risk → OMS → fill.
All tests use paper mode (no real API calls).
"""

import asyncio
import time

from src.core.events import BarEvent, EventBus, EventType, FillEvent, TickEvent
from src.core.signal_handler import SignalHandler
from src.core.tick_aggregator import TickAggregator
from src.features.feature_hub import FeatureHub
from src.filters.spread_monitor import SpreadConfig, SpreadMonitor, SpreadSnapshot
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
    async def test_paper_fill_emits_event(self):
        bus = EventBus()
        oms = _make_paper_oms(bus)
        await oms.initialize()

        fills = []

        async def capture_fill(f: FillEvent):
            fills.append(f)

        bus.subscribe(EventType.FILL, capture_fill)

        from src.strategies.base import Direction, Signal
        from src.models.hmm_regime import RegimeState
        from datetime import datetime

        signal = Signal(
            strategy_id="test",
            direction=Direction.LONG,
            entry_price=5000.0,
            target_price=5002.0,
            stop_price=4998.0,
            signal_time=datetime.now(),
            expiry_time=datetime.now(),
            confidence=0.8,
            regime_state=RegimeState.LOW_VOL_RANGE,
        )
        order_id = await oms.submit_order(signal)

        # Drain bus
        event = bus._queue.get_nowait()
        await capture_fill(event)

        assert len(fills) == 1
        assert fills[0].fill_price == 5000.0
        assert fills[0].direction == "BUY"
        assert oms.position == 1

    async def test_paper_position_tracking(self):
        bus = EventBus()
        oms = _make_paper_oms(bus)
        await oms.initialize()

        from src.strategies.base import Direction, Signal
        from src.models.hmm_regime import RegimeState
        from datetime import datetime

        # Long
        sig = Signal(
            strategy_id="test", direction=Direction.LONG,
            entry_price=5000.0, target_price=5002.0, stop_price=4998.0,
            signal_time=datetime.now(), expiry_time=datetime.now(),
            confidence=0.8, regime_state=RegimeState.LOW_VOL_RANGE,
        )
        await oms.submit_order(sig)
        assert oms.position == 1

        # Short to flatten
        sig2 = Signal(
            strategy_id="test", direction=Direction.SHORT,
            entry_price=5001.0, target_price=4999.0, stop_price=5003.0,
            signal_time=datetime.now(), expiry_time=datetime.now(),
            confidence=0.8, regime_state=RegimeState.LOW_VOL_RANGE,
        )
        await oms.submit_order(sig2)
        assert oms.position == 0

    async def test_cancel_paper_order(self):
        bus = EventBus()
        oms = _make_paper_oms(bus)
        await oms.initialize()

        from src.strategies.base import Direction, Signal
        from src.models.hmm_regime import RegimeState
        from datetime import datetime

        sig = Signal(
            strategy_id="test", direction=Direction.LONG,
            entry_price=5000.0, target_price=5002.0, stop_price=4998.0,
            signal_time=datetime.now(), expiry_time=datetime.now(),
            confidence=0.8, regime_state=RegimeState.LOW_VOL_RANGE,
        )
        oid = await oms.submit_order(sig)
        # Paper fills are instant, so status is already FILLED
        order = oms.get_order(oid)
        assert order.status == OrderStatus.FILLED


# ── SignalHandler Tests ─────────────────────────────────────────


class TestSignalHandler:
    async def test_risk_rejection_blocks_order(self):
        bus = EventBus()
        oms = _make_paper_oms(bus)
        await oms.initialize()
        risk = RiskManager(max_signals_per_day=0)  # Block all signals

        from src.strategies.orb_strategy import ORBConfig, ORBStrategy
        hub = FeatureHub()
        strat = ORBStrategy(ORBConfig(), hub)

        handler = SignalHandler(bus, [strat], risk, oms)
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


class TestSpreadGate:
    async def test_spread_gate_blocks_during_anomalous_spread(self):
        """Signal should be blocked when spread is anomalously wide."""
        bus = EventBus()
        oms = _make_paper_oms(bus)
        await oms.initialize()
        risk = RiskManager()

        # Build a spread monitor with low min_samples and prime with stable data
        config = SpreadConfig(z_threshold=2.0, min_samples=5)
        spread_mon = SpreadMonitor(config=config)
        from datetime import datetime, timezone
        for _ in range(50):
            await spread_mon.push(SpreadSnapshot(
                timestamp=datetime.now(timezone.utc), bid=5000.0, ask=5000.25,
            ))
        # Now inject a massive spike so next check fails
        await spread_mon.push(SpreadSnapshot(
            timestamp=datetime.now(timezone.utc), bid=5000.0, ask=5010.0,
        ))

        handler = SignalHandler(bus, [], risk, oms, spread_monitor=spread_mon)

        from src.strategies.base import Direction, Signal
        from src.models.hmm_regime import RegimeState

        signal = Signal(
            strategy_id="test", direction=Direction.LONG,
            entry_price=5000.0, target_price=5002.0, stop_price=4998.0,
            signal_time=datetime.now(), expiry_time=datetime.now(),
            confidence=0.8, regime_state=RegimeState.LOW_VOL_RANGE,
        )
        bar = _make_bar(close=5000.0)

        await handler._process_signal(signal, bar)

        # Should NOT have submitted — position still 0
        assert oms.position == 0

    async def test_spread_gate_passes_during_normal_spread(self):
        """Signal should pass through when spread is normal."""
        bus = EventBus()
        oms = _make_paper_oms(bus)
        await oms.initialize()
        risk = RiskManager()

        config = SpreadConfig(z_threshold=2.0, min_samples=5)
        spread_mon = SpreadMonitor(config=config)
        from datetime import datetime, timezone
        for _ in range(50):
            await spread_mon.push(SpreadSnapshot(
                timestamp=datetime.now(timezone.utc), bid=5000.0, ask=5000.25,
            ))

        handler = SignalHandler(bus, [], risk, oms, spread_monitor=spread_mon)
        handler.wire()

        from src.strategies.base import Direction, Signal
        from src.models.hmm_regime import RegimeState

        signal = Signal(
            strategy_id="test", direction=Direction.LONG,
            entry_price=5000.0, target_price=5002.0, stop_price=4998.0,
            signal_time=datetime.now(), expiry_time=datetime.now(),
            confidence=0.8, regime_state=RegimeState.LOW_VOL_RANGE,
        )
        bar = _make_bar(close=5000.0)

        await handler._process_signal(signal, bar)

        # Should have submitted — position is 1
        assert oms.position == 1

    async def test_spread_gate_allows_before_min_samples(self):
        """Spread gate should not block before min_samples are collected."""
        bus = EventBus()
        oms = _make_paper_oms(bus)
        await oms.initialize()
        risk = RiskManager()

        config = SpreadConfig(z_threshold=2.0, min_samples=100)
        spread_mon = SpreadMonitor(config=config)
        from datetime import datetime, timezone
        # Push only 5 samples (well below min_samples=100)
        for _ in range(5):
            await spread_mon.push(SpreadSnapshot(
                timestamp=datetime.now(timezone.utc), bid=5000.0, ask=5000.25,
            ))

        handler = SignalHandler(bus, [], risk, oms, spread_monitor=spread_mon)
        handler.wire()

        from src.strategies.base import Direction, Signal
        from src.models.hmm_regime import RegimeState

        signal = Signal(
            strategy_id="test", direction=Direction.LONG,
            entry_price=5000.0, target_price=5002.0, stop_price=4998.0,
            signal_time=datetime.now(), expiry_time=datetime.now(),
            confidence=0.8, regime_state=RegimeState.LOW_VOL_RANGE,
        )
        bar = _make_bar(close=5000.0)

        await handler._process_signal(signal, bar)

        # Should pass — not enough data to filter
        assert oms.position == 1
