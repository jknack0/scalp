"""Tests for PriorDayBiasSignal."""

from src.models.dollar_bar import DollarBar
from src.signals.prior_day_bias import PriorDayBiasConfig, PriorDayBiasSignal


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
    signal = PriorDayBiasSignal(PriorDayBiasConfig(lookback_bars=5))
    bars = [_make_dollar_bar() for _ in range(2)]
    result = signal.compute(bars)
    assert not result.passes


def test_no_prior_day_vwap():
    signal = PriorDayBiasSignal(PriorDayBiasConfig(lookback_bars=5))
    bars = [_make_dollar_bar(prior_day_vwap=0.0) for _ in range(5)]
    result = signal.compute(bars)
    assert not result.passes
    assert result.metadata.get("reason") == "no_prior_day_vwap"


def test_gap_above_gives_long():
    """Price consistently above prior day VWAP should give long."""
    signal = PriorDayBiasSignal(
        PriorDayBiasConfig(min_gap_ticks=2, lookback_bars=5)
    )
    # Prior day VWAP at 4990, all closes at 5000 -> gap = 10 / 0.25 = 40 ticks
    bars = [
        _make_dollar_bar(close=5000.0, prior_day_vwap=4990.0)
        for _ in range(5)
    ]
    result = signal.compute(bars)
    assert result.passes is True
    assert result.direction == "long"


def test_gap_below_gives_short():
    """Price consistently below prior day VWAP should give short."""
    signal = PriorDayBiasSignal(
        PriorDayBiasConfig(min_gap_ticks=2, lookback_bars=5)
    )
    # Prior day VWAP at 5010, all closes at 5000 -> gap = -10 / 0.25 = -40 ticks
    bars = [
        _make_dollar_bar(close=5000.0, prior_day_vwap=5010.0)
        for _ in range(5)
    ]
    result = signal.compute(bars)
    assert result.passes is True
    assert result.direction == "short"


def test_small_gap_does_not_pass():
    """Gap smaller than min_gap_ticks should not pass."""
    signal = PriorDayBiasSignal(
        PriorDayBiasConfig(min_gap_ticks=10, lookback_bars=5)
    )
    # Prior day VWAP at 4999.75, close at 5000 -> gap = 0.25 / 0.25 = 1 tick < 10
    bars = [
        _make_dollar_bar(close=5000.0, prior_day_vwap=4999.75)
        for _ in range(5)
    ]
    result = signal.compute(bars)
    assert not result.passes
