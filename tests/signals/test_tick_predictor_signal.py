"""Tests for TickPredictorSignal integration."""

import numpy as np
import pytest

from src.core.events import BarEvent
from src.signals.base import SignalResult
from src.signals.tick_predictor.signal import TickPredictorSignal


def _make_bar(close: float, volume: int = 100, ts_ns: int = 0) -> BarEvent:
    return BarEvent(
        symbol="MES", open=close, high=close + 0.5, low=close - 0.5,
        close=close, volume=volume, bar_type="1s", timestamp_ns=ts_ns,
    )


def _make_bar_sequence(n: int, base_price: float = 5600.0) -> list[BarEvent]:
    """Generate n bars with a random walk."""
    rng = np.random.default_rng(42)
    bars = []
    price = base_price
    for i in range(n):
        price += rng.normal(0, 0.5)
        vol = int(rng.integers(50, 200))
        bars.append(_make_bar(price, volume=vol, ts_ns=i * 1_000_000_000))
    return bars


class TestTickPredictorSignal:
    def test_compute_returns_signal_result_type(self):
        signal = TickPredictorSignal(config={})
        bars = _make_bar_sequence(5)
        result = signal.compute(bars)
        assert isinstance(result, SignalResult)

    def test_neutral_result_before_warmup(self):
        signal = TickPredictorSignal(config={})
        bars = _make_bar_sequence(10)
        result = signal.compute(bars)
        assert result.metadata.get("is_warm") is False

    def test_neutral_when_no_model(self):
        """Without a trained model, should return neutral result."""
        signal = TickPredictorSignal(config={})
        bars = _make_bar_sequence(200)
        # Feed all bars through
        for i in range(len(bars)):
            result = signal.compute(bars[: i + 1])
        # After warmup, should still be neutral (no model file)
        assert result.metadata.get("reason") == "no_model"

    def test_reset_clears_warmup_state(self):
        signal = TickPredictorSignal(config={})
        bars = _make_bar_sequence(100)
        for i in range(len(bars)):
            signal.compute(bars[: i + 1])
        signal.reset()
        result = signal.compute([bars[0]])
        assert result.metadata.get("is_warm") is False

    def test_value_encoding_invariants(self):
        """Value should encode direction + confidence as signed float."""
        # We can't test with a real model, but we can test the encoding logic
        # by checking the neutral case
        signal = TickPredictorSignal(config={})
        result = signal.compute([_make_bar(5600.0)])
        # Neutral → value should be 0
        assert result.value == 0.0

    def test_feature_builder_processes_bars(self):
        """Verify feature builder is actually processing bars."""
        signal = TickPredictorSignal(config={})
        bars = _make_bar_sequence(60)
        for i in range(len(bars)):
            signal.compute(bars[: i + 1])
        # Feature builder should have processed all bars
        assert signal._feature_builder._bar_count == 60
