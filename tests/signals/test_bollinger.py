"""Tests for Bollinger Bands signal."""

import numpy as np
import pytest

from src.core.events import BarEvent
from src.signals.bollinger import BollingerConfig, BollingerSignal


def _make_bars(prices: list[float]) -> list[BarEvent]:
    return [
        BarEvent(
            symbol="MESM6", open=p - 0.25, high=p + 0.25,
            low=p - 0.25, close=p, volume=100, bar_type="5m",
            timestamp_ns=i * 1_000_000_000,
        )
        for i, p in enumerate(prices)
    ]


class TestBollingerSignal:
    def test_insufficient_bars(self):
        sig = BollingerSignal(BollingerConfig(period=20))
        result = sig.compute(_make_bars([5600.0] * 10))
        assert result.passes is False

    def test_price_at_sma_no_signal(self):
        """Price at the SMA should not pass."""
        prices = [5600.0 + np.sin(i * 0.5) * 2 for i in range(25)]
        prices[-1] = np.mean(prices[-20:])  # Force close to SMA
        sig = BollingerSignal(BollingerConfig(period=20, num_std=2.0))
        result = sig.compute(_make_bars(prices))
        assert result.passes is False
        assert result.direction == "none"

    def test_price_below_lower_band_long(self):
        """Price far below lower band → direction = long."""
        prices = [5600.0] * 20 + [5580.0]  # Big drop
        sig = BollingerSignal(BollingerConfig(period=20, num_std=2.0))
        result = sig.compute(_make_bars(prices))
        assert result.passes is True
        assert result.direction == "long"
        assert result.value < -2.0  # Beyond 2 SD

    def test_price_above_upper_band_short(self):
        """Price far above upper band → direction = short."""
        prices = [5600.0] * 20 + [5620.0]
        sig = BollingerSignal(BollingerConfig(period=20, num_std=2.0))
        result = sig.compute(_make_bars(prices))
        assert result.passes is True
        assert result.direction == "short"
        assert result.value > 2.0

    def test_metadata_keys(self):
        prices = [5600.0 + i * 0.1 for i in range(25)]
        sig = BollingerSignal()
        result = sig.compute(_make_bars(prices))
        assert "sma" in result.metadata
        assert "upper" in result.metadata
        assert "lower" in result.metadata
        assert "bandwidth" in result.metadata
        assert "pct_b" in result.metadata
        assert result.metadata["upper"] > result.metadata["sma"]
        assert result.metadata["lower"] < result.metadata["sma"]

    def test_pct_b_range(self):
        """% B should be near 0-1 for prices within bands."""
        prices = [5600.0 + np.sin(i * 0.3) * 1 for i in range(25)]
        sig = BollingerSignal()
        result = sig.compute(_make_bars(prices))
        assert 0.0 <= result.metadata["pct_b"] <= 1.0
