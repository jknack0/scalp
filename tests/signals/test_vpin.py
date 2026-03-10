"""Tests for VPINSignal."""

from src.core.events import BarEvent
from src.signals.vpin import VPINConfig, VPINSignal


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


def test_empty_bars():
    signal = VPINSignal(VPINConfig())
    result = signal.compute([])
    assert not result.passes
    assert result.metadata["bucket_count"] == 0


def test_insufficient_buckets():
    """Not enough volume to fill n_buckets should not pass."""
    signal = VPINSignal(VPINConfig(bucket_size=100, n_buckets=50))
    # Only 5 bars of 100 volume each = 500 total -> 5 buckets, need 50
    bars = [_make_bar(volume=100) for _ in range(5)]
    result = signal.compute(bars)
    assert not result.passes
    assert result.metadata["bucket_count"] < 50


def test_compute_with_enough_data():
    """Enough volume to fill all buckets should pass and produce valid VPIN."""
    signal = VPINSignal(VPINConfig(bucket_size=10, n_buckets=5))
    # 100 bars x 100 volume = 10000 total, bucket_size=10 -> 1000 buckets >> 5
    bars = [
        _make_bar(
            close=5000.0 + (i % 3),
            high=5002.0 + (i % 3),
            low=4998.0 + (i % 3),
            volume=100,
        )
        for i in range(100)
    ]
    result = signal.compute(bars)
    assert result.passes is True
    assert result.metadata["bucket_count"] >= 5
    assert 0.0 <= result.value <= 1.0
