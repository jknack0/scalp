"""Tests for VolumeExhaustionSignal."""

from src.models.dollar_bar import DollarBar
from src.signals.volume_exhaustion import VolumeExhaustionConfig, VolumeExhaustionSignal


def _make_dollar_bar(
    close: float = 5000.0,
    open_: float = 4999.0,
    high: float = 5001.0,
    low: float = 4998.0,
    volume: int = 100,
    timestamp_ns: int = 1_000_000_000,
    session_vwap: float = 5000.0,
    prior_day_vwap: float = 4990.0,
    buy_volume: int = 60,
    sell_volume: int = 40,
    session_open_time=None,
    **kwargs,
) -> DollarBar:
    return DollarBar(
        symbol="MESM6",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bar_type="1s",
        timestamp_ns=timestamp_ns,
        session_vwap=session_vwap,
        prior_day_vwap=prior_day_vwap,
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        session_open_time=session_open_time,
        **kwargs,
    )


def test_insufficient_bars():
    signal = VolumeExhaustionSignal(VolumeExhaustionConfig(lookback_bars=20))
    bars = [_make_dollar_bar(volume=100) for _ in range(5)]
    result = signal.compute(bars)
    assert not result.passes


def test_normal_volume_does_not_pass():
    """Consistent volume should not trigger exhaustion."""
    signal = VolumeExhaustionSignal(
        VolumeExhaustionConfig(lookback_bars=5, exhaustion_threshold=0.4)
    )
    bars = [_make_dollar_bar(volume=100) for _ in range(5)]
    result = signal.compute(bars)
    assert not result.passes


def test_very_low_last_bar_passes():
    """Last bar with very low volume relative to prior bars should pass."""
    signal = VolumeExhaustionSignal(
        VolumeExhaustionConfig(lookback_bars=5, exhaustion_threshold=0.4)
    )
    bars = [_make_dollar_bar(volume=200) for _ in range(4)]
    bars.append(_make_dollar_bar(volume=10, buy_volume=2, sell_volume=8))
    result = signal.compute(bars)
    assert result.passes is True


def test_direction_from_buy_sell_split():
    """When buy volume is much less than sell volume, direction is short (buy exhaustion)."""
    signal = VolumeExhaustionSignal(
        VolumeExhaustionConfig(lookback_bars=5, exhaustion_threshold=0.4)
    )
    bars = [_make_dollar_bar(volume=200) for _ in range(4)]
    # Last bar: very low volume, buy_volume << sell_volume -> buy exhaustion -> short
    bars.append(_make_dollar_bar(volume=10, buy_volume=1, sell_volume=9))
    result = signal.compute(bars)
    assert result.passes is True
    assert result.direction == "short"


def test_direction_sell_exhaustion():
    """When sell volume is much less than buy volume, direction is long (sell exhaustion)."""
    signal = VolumeExhaustionSignal(
        VolumeExhaustionConfig(lookback_bars=5, exhaustion_threshold=0.4)
    )
    bars = [_make_dollar_bar(volume=200) for _ in range(4)]
    bars.append(_make_dollar_bar(volume=10, buy_volume=9, sell_volume=1))
    result = signal.compute(bars)
    assert result.passes is True
    assert result.direction == "long"
