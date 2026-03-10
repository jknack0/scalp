"""Tests for the stateless signal framework and individual signals."""

from __future__ import annotations

import pytest

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.spread import SpreadConfig, SpreadSignal
from src.signals.registry import SignalRegistry


def _make_bar(
    close: float = 5000.0,
    open_: float = 4999.0,
    high: float = 5001.0,
    low: float = 4998.0,
    volume: int = 100,
    avg_bid_price: float = 0.0,
    avg_ask_price: float = 0.0,
    timestamp_ns: int = 1_000_000_000,
) -> BarEvent:
    return BarEvent(
        symbol="MESM6",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bar_type="1s",
        timestamp_ns=timestamp_ns,
        avg_bid_price=avg_bid_price,
        avg_ask_price=avg_ask_price,
    )


class TestSignalResult:
    def test_frozen(self):
        r = SignalResult(value=1.0, passes=True)
        with pytest.raises(AttributeError):
            r.value = 2.0  # type: ignore[misc]

    def test_defaults(self):
        r = SignalResult(value=0.5, passes=False)
        assert r.direction == "none"
        assert r.metadata == {}


class TestSpreadSignal:
    def test_passes_with_normal_spread(self):
        """Tight spread produces z~0, passes=True."""
        bars = [
            _make_bar(avg_bid_price=5000.0, avg_ask_price=5000.25, timestamp_ns=i * 1_000_000_000)
            for i in range(50)
        ]
        sig = SpreadSignal(SpreadConfig(z_threshold=2.0, min_bars=30))
        result = sig.compute(bars)
        assert result.passes is True
        assert abs(result.value) < 2.0

    def test_blocks_with_wide_spread(self):
        """One abnormally wide spread at the end should fail."""
        bars = [
            _make_bar(avg_bid_price=5000.0, avg_ask_price=5000.25, timestamp_ns=i * 1_000_000_000)
            for i in range(49)
        ]
        # Giant spread on the last bar
        bars.append(_make_bar(avg_bid_price=5000.0, avg_ask_price=5010.0, timestamp_ns=50_000_000_000))
        sig = SpreadSignal(SpreadConfig(z_threshold=2.0, min_bars=30))
        result = sig.compute(bars)
        assert not result.passes
        assert result.value > 2.0

    def test_insufficient_data_passes(self):
        """Fewer than min_bars should pass (no data to judge)."""
        bars = [_make_bar(avg_bid_price=5000.0, avg_ask_price=5000.25) for _ in range(5)]
        sig = SpreadSignal(SpreadConfig(z_threshold=2.0, min_bars=30))
        result = sig.compute(bars)
        assert result.passes is True
        assert result.metadata.get("reason") == "insufficient_data"

    def test_empty_bars(self):
        sig = SpreadSignal()
        result = sig.compute([])
        assert result.passes is True

    def test_no_spread_data_passes(self):
        """Bars without bid/ask should pass (no spread to compute)."""
        bars = [_make_bar() for _ in range(50)]  # no avg_bid/ask_price
        sig = SpreadSignal(SpreadConfig(min_bars=5))
        result = sig.compute(bars)
        assert result.passes is True


class TestSignalRegistry:
    def test_spread_registered(self):
        # Import triggers registration
        import src.signals  # noqa: F401
        assert "spread" in SignalRegistry.available()

    def test_get_known_signal(self):
        import src.signals  # noqa: F401
        cls = SignalRegistry.get("spread")
        assert cls is SpreadSignal

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown signal"):
            SignalRegistry.get("nonexistent_signal_xyz")

    def test_build_with_defaults(self):
        import src.signals  # noqa: F401
        sig = SignalRegistry.build("spread")
        assert isinstance(sig, SpreadSignal)
        assert sig.config.z_threshold == 2.0

    def test_build_with_kwargs(self):
        import src.signals  # noqa: F401
        sig = SignalRegistry.build("spread", z_threshold=3.0, min_bars=10)
        assert isinstance(sig, SpreadSignal)
        assert sig.config.z_threshold == 3.0
        assert sig.config.min_bars == 10
