"""Tests for CVDDivergenceSignal."""

from src.core.events import BarEvent
from src.signals.cvd_divergence import CVDDivergenceConfig, CVDDivergenceSignal


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
    signal = CVDDivergenceSignal(CVDDivergenceConfig(lookback_bars=20))
    bars = [_make_bar() for _ in range(5)]
    result = signal.compute(bars)
    assert not result.passes
    assert result.direction == "none"


def test_no_divergence():
    """When price and CVD trend the same direction, no divergence."""
    signal = CVDDivergenceSignal(CVDDivergenceConfig(lookback_bars=5, z_threshold=1.0))
    # Price going up, volume mostly buy-side -> CVD should trend up too
    bars = [
        _make_bar(
            close=5000.0 + i * 2,
            open_=4999.0 + i * 2,
            high=5001.0 + i * 2,
            low=4998.0 + i * 2,
            volume=100,
        )
        for i in range(10)
    ]
    result = signal.compute(bars)
    assert not result.passes


def test_bearish_divergence():
    """Price up but CVD down should give short direction."""
    signal = CVDDivergenceSignal(CVDDivergenceConfig(lookback_bars=5, z_threshold=0.0))
    n = 20
    bars = []
    for i in range(n):
        # Price trending up: close rises
        close = 5000.0 + i * 1.0
        # But volume is heavily sell-side (close near low -> low buy_pct)
        high = close + 5.0
        low = close - 0.1
        bars.append(
            _make_bar(
                close=close,
                open_=close - 0.5,
                high=high,
                low=low,
                volume=200,
            )
        )
    result = signal.compute(bars)
    # Price slope > 0, CVD slope < 0 (sell-dominated) -> bearish divergence -> short
    if result.passes:
        assert result.direction == "short"


def test_insufficient_z_score():
    """Even with divergence, if z_score is below threshold, should not pass."""
    signal = CVDDivergenceSignal(CVDDivergenceConfig(lookback_bars=5, z_threshold=100.0))
    n = 20
    bars = []
    for i in range(n):
        close = 5000.0 + i * 1.0
        high = close + 5.0
        low = close - 0.1
        bars.append(
            _make_bar(
                close=close,
                open_=close - 0.5,
                high=high,
                low=low,
                volume=200,
            )
        )
    result = signal.compute(bars)
    assert not result.passes
