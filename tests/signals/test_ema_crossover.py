"""Tests for EMA crossover signal."""

from src.core.events import BarEvent
from src.signals.ema_crossover import EMACrossoverConfig, EMACrossoverSignal


def _bar(close: float, ts_ns: int = 0) -> BarEvent:
    return BarEvent(
        symbol="MESM6", open=close, high=close + 0.5,
        low=close - 0.5, close=close, volume=100,
        bar_type="5m", timestamp_ns=ts_ns,
    )


class TestEMACrossover:
    def test_insufficient_bars(self):
        sig = EMACrossoverSignal(EMACrossoverConfig(fast_period=3, slow_period=5))
        result = sig.compute([_bar(100.0)] * 3)
        assert not result.passes
        assert result.metadata.get("reason") == "insufficient_bars"

    def test_bullish_crossover(self):
        # Build a series where fast EMA crosses above slow EMA
        sig = EMACrossoverSignal(EMACrossoverConfig(fast_period=3, slow_period=8))
        # Start low, then ramp up sharply
        bars = [_bar(100.0)] * 15
        for i in range(5):
            bars.append(_bar(100.0 + (i + 1) * 2.0))
        result = sig.compute(bars)
        assert result.direction == "long"
        assert result.metadata["spread"] > 0

    def test_bearish_crossover(self):
        sig = EMACrossoverSignal(EMACrossoverConfig(fast_period=3, slow_period=8))
        bars = [_bar(110.0)] * 15
        for i in range(5):
            bars.append(_bar(110.0 - (i + 1) * 2.0))
        result = sig.compute(bars)
        assert result.direction == "short"
        assert result.metadata["spread"] < 0

    def test_no_crossover_flat(self):
        sig = EMACrossoverSignal(EMACrossoverConfig(fast_period=3, slow_period=8))
        bars = [_bar(100.0)] * 30
        result = sig.compute(bars)
        assert not result.passes
        assert result.metadata["crossed"] == "none"
