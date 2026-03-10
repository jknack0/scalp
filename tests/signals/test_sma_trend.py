"""Tests for SMA trend alignment signal."""

import pytest

from src.core.events import BarEvent
from src.signals.sma_trend import SMATrendConfig, SMATrendSignal


def _make_bars(prices: list[float]) -> list[BarEvent]:
    return [
        BarEvent(
            symbol="MESM6", open=p - 0.25, high=p + 0.25,
            low=p - 0.25, close=p, volume=100, bar_type="5m",
            timestamp_ns=i * 1_000_000_000,
        )
        for i, p in enumerate(prices)
    ]


class TestSMATrendSignal:
    def test_insufficient_bars(self):
        sig = SMATrendSignal(SMATrendConfig(period=200))
        result = sig.compute(_make_bars([5600.0] * 50))
        assert result.direction == "none"

    def test_price_above_sma_long(self):
        """Price above SMA → direction = long."""
        # 200 bars + current bar well above
        prices = [5600.0] * 200 + [5620.0]
        sig = SMATrendSignal(SMATrendConfig(period=200))
        result = sig.compute(_make_bars(prices))
        assert result.direction == "long"
        assert result.value > 0

    def test_price_below_sma_short(self):
        """Price below SMA → direction = short."""
        prices = [5600.0] * 200 + [5580.0]
        sig = SMATrendSignal(SMATrendConfig(period=200))
        result = sig.compute(_make_bars(prices))
        assert result.direction == "short"
        assert result.value < 0

    def test_sma_in_metadata(self):
        prices = [5600.0] * 200 + [5600.0]
        sig = SMATrendSignal(SMATrendConfig(period=200))
        result = sig.compute(_make_bars(prices))
        assert "sma" in result.metadata
        assert result.metadata["sma"] == pytest.approx(5600.0)
