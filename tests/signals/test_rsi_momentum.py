"""Tests for RSI momentum signal."""

from src.core.events import BarEvent
from src.signals.rsi_momentum import RSIMomentumConfig, RSIMomentumSignal


def _bar(close: float) -> BarEvent:
    return BarEvent(
        symbol="MESM6", open=close, high=close + 0.5,
        low=close - 0.5, close=close, volume=100,
        bar_type="5m", timestamp_ns=0,
    )


class TestRSIMomentum:
    def test_insufficient_bars(self):
        sig = RSIMomentumSignal(RSIMomentumConfig(period=14))
        result = sig.compute([_bar(100.0)] * 5)
        assert not result.passes

    def test_strong_uptrend_high_rsi(self):
        sig = RSIMomentumSignal(RSIMomentumConfig(period=7, long_threshold=50.0))
        # Steadily rising prices
        bars = [_bar(100.0 + i * 0.5) for i in range(30)]
        result = sig.compute(bars)
        assert result.value > 50.0
        assert result.direction == "long"
        assert result.passes

    def test_strong_downtrend_low_rsi(self):
        sig = RSIMomentumSignal(RSIMomentumConfig(period=7, short_threshold=50.0))
        bars = [_bar(120.0 - i * 0.5) for i in range(30)]
        result = sig.compute(bars)
        assert result.value < 50.0
        assert result.direction == "short"
        assert result.passes

    def test_flat_market_neutral(self):
        sig = RSIMomentumSignal(RSIMomentumConfig(period=7))
        # Alternating up/down
        bars = [_bar(100.0 + (i % 2) * 0.25) for i in range(30)]
        result = sig.compute(bars)
        # RSI should be near 50 for alternating moves
        assert 40.0 < result.value < 60.0

    def test_metadata_contains_components(self):
        sig = RSIMomentumSignal()
        bars = [_bar(100.0 + i * 0.3) for i in range(30)]
        result = sig.compute(bars)
        assert "rsi" in result.metadata
        assert "avg_gain" in result.metadata
        assert "avg_loss" in result.metadata
