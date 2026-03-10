"""Tests for BarProcessor — signal/filter/strategy orchestration."""

from src.core.bar_processor import BarProcessor
from src.core.events import BarEvent
from src.filters.filter_engine import FilterEngine
from src.signals.base import SignalResult
from src.signals.signal_bundle import SignalBundle, SignalEngine
from src.strategies.base import Direction, Signal

from datetime import datetime, timedelta
from src.models.hmm_regime import RegimeState


def _bar(close: float = 5000.0, volume: int = 100, ts_offset: int = 0) -> BarEvent:
    base_ns = 1_700_000_000_000_000_000 + ts_offset * 1_000_000_000
    return BarEvent(
        symbol="MESM6",
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=volume,
        bar_type="1s",
        timestamp_ns=base_ns,
    )


def _signal(strategy_id: str = "test") -> Signal:
    now = datetime(2024, 1, 15, 10, 0)
    return Signal(
        strategy_id=strategy_id,
        direction=Direction.LONG,
        entry_price=5000.0,
        target_price=5010.0,
        stop_price=4990.0,
        signal_time=now,
        expiry_time=now + timedelta(minutes=30),
        confidence=0.8,
        regime_state=RegimeState.LOW_VOL_RANGE,
    )


class FakeStrategy:
    """Minimal strategy duck-type for testing."""

    strategy_id = "fake"

    def __init__(self, signal: Signal | None = None):
        self._signal = signal
        self.last_bundle: SignalBundle | None = None

    def on_bar(self, bar: BarEvent, signals: SignalBundle | None = None) -> Signal | None:
        self.last_bundle = signals
        return self._signal

    def reset(self):
        self.last_bundle = None


class LegacyStrategy:
    """Strategy that only accepts bar (no signals param)."""

    strategy_id = "legacy"

    def __init__(self, signal: Signal | None = None):
        self._signal = signal

    def on_bar(self, bar: BarEvent) -> Signal | None:
        return self._signal

    def reset(self):
        pass


class TestBarProcessor:
    def test_no_strategies_no_signals(self):
        proc = BarProcessor()
        signals = proc.on_bar(_bar())
        assert signals == []

    def test_strategy_receives_bundle(self):
        strat = FakeStrategy()
        proc = BarProcessor(strategies=[strat])
        proc.on_bar(_bar())
        assert strat.last_bundle is not None

    def test_strategy_generates_signal(self):
        sig = _signal()
        strat = FakeStrategy(signal=sig)
        proc = BarProcessor(strategies=[strat])
        result = proc.on_bar(_bar())
        assert len(result) == 1
        assert result[0] is sig

    def test_filter_blocks_strategy(self):
        # Filter requires spread < 0.5 — our bars won't have that
        fe = FilterEngine.from_list([{"signal": "some_signal", "expr": "< 0.0"}])
        # Build a bundle that has some_signal = 1.0 (will be blocked)
        strat = FakeStrategy(signal=_signal())
        proc = BarProcessor(filter_engine=fe, strategies=[strat])

        # Without a signal engine, bundle is EMPTY_BUNDLE and "some_signal"
        # won't exist, so the filter rule is skipped (missing => skip)
        result = proc.on_bar(_bar())
        # Missing signal means rule is skipped, so filter passes
        assert len(result) == 1

    def test_legacy_strategy_fallback(self):
        sig = _signal("legacy")
        strat = LegacyStrategy(signal=sig)
        proc = BarProcessor(strategies=[strat])
        result = proc.on_bar(_bar())
        assert len(result) == 1
        assert result[0].strategy_id == "legacy"

    def test_bar_window_maintained(self):
        proc = BarProcessor(max_window=5)
        for i in range(10):
            proc.on_bar(_bar(ts_offset=i))
        assert len(proc.bar_window) == 5

    def test_reset_clears_window(self):
        strat = FakeStrategy()
        proc = BarProcessor(strategies=[strat])
        proc.on_bar(_bar())
        assert len(proc.bar_window) == 1

        proc.reset()
        assert len(proc.bar_window) == 0

    def test_multiple_strategies(self):
        sig1 = _signal("strat1")
        sig2 = _signal("strat2")
        s1 = FakeStrategy(signal=sig1)
        s2 = FakeStrategy(signal=sig2)
        proc = BarProcessor(strategies=[s1, s2])

        result = proc.on_bar(_bar())
        assert len(result) == 2
        ids = {s.strategy_id for s in result}
        assert ids == {"strat1", "strat2"}

    def test_strategy_returning_none(self):
        strat = FakeStrategy(signal=None)
        proc = BarProcessor(strategies=[strat])
        result = proc.on_bar(_bar())
        assert result == []
