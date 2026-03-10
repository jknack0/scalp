"""Tests for ADX signal."""

import random

from src.core.events import BarEvent
from src.signals.adx import ADXConfig, ADXSignal


def _trending_bars(start: float, n: int, step: float = 0.5) -> list[BarEvent]:
    """Generate bars with a clear uptrend."""
    bars = []
    for i in range(n):
        c = start + i * step
        bars.append(BarEvent(
            symbol="MESM6", open=c - 0.1, high=c + 0.5,
            low=c - 0.3, close=c, volume=100,
            bar_type="5m", timestamp_ns=i * 300_000_000_000,
        ))
    return bars


def _choppy_bars(center: float, n: int) -> list[BarEvent]:
    """Generate sideways choppy bars."""
    bars = []
    rng = random.Random(42)
    for i in range(n):
        c = center + rng.uniform(-0.5, 0.5)
        bars.append(BarEvent(
            symbol="MESM6", open=c + 0.1, high=c + 0.8,
            low=c - 0.8, close=c, volume=100,
            bar_type="5m", timestamp_ns=i * 300_000_000_000,
        ))
    return bars


class TestADX:
    def test_insufficient_bars(self):
        sig = ADXSignal(ADXConfig(period=14, threshold=25.0))
        result = sig.compute([_trending_bars(100, 1)[0]] * 10)
        assert not result.passes
        assert result.metadata.get("reason") == "insufficient_bars"

    def test_trending_market_high_adx(self):
        sig = ADXSignal(ADXConfig(period=7, threshold=20.0))
        bars = _trending_bars(100.0, 50, step=1.0)
        result = sig.compute(bars)
        assert result.value > 0
        assert result.direction == "long"
        assert "adx" in result.metadata

    def test_direction_short_in_downtrend(self):
        sig = ADXSignal(ADXConfig(period=7, threshold=20.0))
        bars = _trending_bars(150.0, 50, step=-1.0)
        result = sig.compute(bars)
        assert result.direction == "short"

    def test_choppy_market_lower_adx(self):
        sig = ADXSignal(ADXConfig(period=7, threshold=50.0))
        bars = _choppy_bars(100.0, 50)
        result = sig.compute(bars)
        # Choppy market should have lower ADX than a strong trend
        assert result.value < 50.0
