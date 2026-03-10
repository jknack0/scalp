"""Tests for RelativeVolumeSignal."""

from src.core.events import BarEvent
from src.signals.relative_volume import RelativeVolumeConfig, RelativeVolumeSignal


def _make_bar(
    close: float = 5000.0,
    open_: float = 4999.0,
    high: float = 5001.0,
    low: float = 4998.0,
    volume: int = 100,
    timestamp_ns: int = 1_000_000_000,
    **kwargs,
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
        **kwargs,
    )


def test_insufficient_bars():
    signal = RelativeVolumeSignal(RelativeVolumeConfig(lookback_bars=20))
    bars = [_make_bar(volume=100) for _ in range(5)]
    result = signal.compute(bars)
    assert not result.passes


def test_normal_volume():
    """Volume equal to average should not pass high threshold."""
    signal = RelativeVolumeSignal(
        RelativeVolumeConfig(lookback_bars=5, high_threshold=1.5)
    )
    bars = [_make_bar(volume=100) for _ in range(5)]
    result = signal.compute(bars)
    assert not result.passes
    assert result.metadata["regime"] == "normal"


def test_high_rvol_passes():
    """Last bar with much higher volume should pass."""
    signal = RelativeVolumeSignal(
        RelativeVolumeConfig(lookback_bars=5, high_threshold=1.5)
    )
    bars = [_make_bar(volume=100) for _ in range(4)]
    bars.append(_make_bar(volume=300))
    result = signal.compute(bars)
    assert result.passes is True
    assert result.metadata["regime"] == "high"


def test_low_rvol_does_not_pass():
    """Very low volume should not pass (passes only for high rvol)."""
    signal = RelativeVolumeSignal(
        RelativeVolumeConfig(lookback_bars=5, high_threshold=1.5, low_threshold=0.5)
    )
    bars = [_make_bar(volume=100) for _ in range(4)]
    bars.append(_make_bar(volume=10))
    result = signal.compute(bars)
    assert not result.passes
    assert result.metadata["regime"] == "low"
