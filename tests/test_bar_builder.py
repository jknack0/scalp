"""Tests for per-strategy bar builders."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.core.events import BarEvent, TickEvent
from src.core.bar_builder import (
    BarBuilderFactory,
    DollarBarBuilder,
    TickBarBuilder,
    TimeBarBuilder,
    VolumeBarBuilder,
)
from src.models.dollar_bar import DollarBar


def _tick(
    price: float = 5000.0,
    size: int = 1,
    bid: float = 4999.75,
    ask: float = 5000.25,
    timestamp_ns: int = 1_000_000_000,
) -> TickEvent:
    return TickEvent(
        symbol="MESM6",
        bid=bid,
        ask=ask,
        last_price=price,
        last_size=size,
        timestamp_ns=timestamp_ns,
    )


class TestTimeBarBuilder:
    def test_emits_bar_at_interval(self):
        builder = TimeBarBuilder(interval_seconds=1.0)
        # First tick at t=0
        assert builder.on_tick(_tick(timestamp_ns=0)) is None
        assert builder.on_tick(_tick(price=5001.0, timestamp_ns=500_000_000)) is None
        # Tick at t=1s should trigger bar emission
        bar = builder.on_tick(_tick(price=5002.0, timestamp_ns=1_000_000_000))
        assert bar is not None
        assert isinstance(bar, BarEvent)
        assert bar.open == 5000.0
        assert bar.high == 5001.0
        assert bar.close == 5001.0
        assert bar.volume == 2

    def test_emits_dollar_bar_when_enriched(self):
        builder = TimeBarBuilder(
            interval_seconds=1.0,
            emit_dollar_bar=True,
            prior_day_vwap=4990.0,
        )
        builder.on_tick(_tick(price=5000.0, size=10, timestamp_ns=0))
        bar = builder.on_tick(_tick(price=5001.0, size=5, timestamp_ns=1_000_000_000))
        assert bar is not None
        assert isinstance(bar, DollarBar)
        assert bar.prior_day_vwap == 4990.0
        assert bar.session_vwap > 0
        assert bar.buy_volume + bar.sell_volume >= 0

    def test_flush_emits_partial_bar(self):
        builder = TimeBarBuilder(interval_seconds=10.0)
        builder.on_tick(_tick(price=5000.0, timestamp_ns=0))
        builder.on_tick(_tick(price=5002.0, timestamp_ns=100_000_000))
        bar = builder.flush()
        assert bar is not None
        assert bar.open == 5000.0
        assert bar.close == 5002.0

    def test_flush_returns_none_when_empty(self):
        builder = TimeBarBuilder(interval_seconds=1.0)
        assert builder.flush() is None

    def test_reset_clears_state(self):
        builder = TimeBarBuilder(interval_seconds=1.0)
        builder.on_tick(_tick(timestamp_ns=0))
        builder.reset()
        assert builder.flush() is None

    def test_bar_type_label(self):
        assert TimeBarBuilder(interval_seconds=5.0).bar_type_label == "5s"
        assert TimeBarBuilder(interval_seconds=60.0).bar_type_label == "1m"
        assert TimeBarBuilder(interval_seconds=300.0).bar_type_label == "5m"

    def test_skips_zero_price_ticks(self):
        builder = TimeBarBuilder(interval_seconds=1.0)
        assert builder.on_tick(_tick(price=0.0)) is None


class TestDollarBarBuilder:
    def test_emits_at_threshold(self):
        # MES: dollar vol = price * size * 5.0
        # 5000 * 10 * 5 = 250,000 per tick
        builder = DollarBarBuilder(dollar_threshold=250_000)
        # One tick should cross the threshold
        bar = builder.on_tick(_tick(price=5000.0, size=10))
        assert bar is not None
        assert isinstance(bar, DollarBar)
        assert bar.session_vwap > 0

    def test_accumulates_until_threshold(self):
        builder = DollarBarBuilder(dollar_threshold=100_000)
        # 5000 * 1 * 5 = 25,000 per tick. Need 4 ticks.
        assert builder.on_tick(_tick(price=5000.0, size=1, timestamp_ns=0)) is None
        assert builder.on_tick(_tick(price=5000.0, size=1, timestamp_ns=100)) is None
        assert builder.on_tick(_tick(price=5000.0, size=1, timestamp_ns=200)) is None
        bar = builder.on_tick(_tick(price=5000.0, size=1, timestamp_ns=300))
        assert bar is not None
        assert bar.volume == 4

    def test_bar_type_label(self):
        assert DollarBarBuilder(dollar_threshold=50_000).bar_type_label == "dollar_50k"
        assert DollarBarBuilder(dollar_threshold=1_000_000).bar_type_label == "dollar_1M"

    def test_session_vwap_tracks_across_bars(self):
        builder = DollarBarBuilder(dollar_threshold=25_000)
        # First bar: one tick
        bar1 = builder.on_tick(_tick(price=5000.0, size=1))
        assert bar1 is not None
        vwap1 = bar1.session_vwap
        # Second bar
        bar2 = builder.on_tick(_tick(price=5010.0, size=1))
        assert bar2 is not None
        # Session VWAP should now be average of 5000 and 5010
        assert bar2.session_vwap != vwap1


class TestVolumeBarBuilder:
    def test_emits_at_threshold(self):
        builder = VolumeBarBuilder(volume_threshold=10)
        for i in range(9):
            assert builder.on_tick(_tick(size=1, timestamp_ns=i)) is None
        bar = builder.on_tick(_tick(size=1, timestamp_ns=10))
        assert bar is not None
        assert bar.volume == 10

    def test_bar_type_label(self):
        assert VolumeBarBuilder(volume_threshold=500).bar_type_label == "500vol"

    def test_enriched_emits_dollar_bar(self):
        builder = VolumeBarBuilder(volume_threshold=5, emit_dollar_bar=True)
        for i in range(5):
            bar = builder.on_tick(_tick(size=1, timestamp_ns=i))
        assert isinstance(bar, DollarBar)


class TestTickBarBuilder:
    def test_emits_every_n_ticks(self):
        builder = TickBarBuilder(tick_count=3)
        assert builder.on_tick(_tick(price=5000.0, timestamp_ns=0)) is None
        assert builder.on_tick(_tick(price=5001.0, timestamp_ns=1)) is None
        bar = builder.on_tick(_tick(price=5002.0, timestamp_ns=2))
        assert bar is not None
        assert bar.open == 5000.0
        assert bar.close == 5002.0
        assert bar.volume == 3

    def test_bar_type_label(self):
        assert TickBarBuilder(tick_count=100).bar_type_label == "100tick"


class TestBarBuilderFactory:
    def test_creates_time_builder(self):
        builder = BarBuilderFactory.from_config({
            "type": "time",
            "interval_seconds": 5,
        })
        assert isinstance(builder, TimeBarBuilder)
        assert builder.bar_type_label == "5s"

    def test_creates_dollar_builder(self):
        builder = BarBuilderFactory.from_config({
            "type": "dollar",
            "dollar_threshold": 50000,
        })
        assert isinstance(builder, DollarBarBuilder)

    def test_creates_volume_builder(self):
        builder = BarBuilderFactory.from_config({
            "type": "volume",
            "volume_threshold": 500,
        })
        assert isinstance(builder, VolumeBarBuilder)

    def test_creates_tick_builder(self):
        builder = BarBuilderFactory.from_config({
            "type": "tick",
            "tick_count": 100,
        })
        assert isinstance(builder, TickBarBuilder)

    def test_default_is_time(self):
        builder = BarBuilderFactory.from_config({})
        assert isinstance(builder, TimeBarBuilder)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown bar type"):
            BarBuilderFactory.from_config({"type": "range"})

    def test_enrich_flag_passes_through(self):
        builder = BarBuilderFactory.from_config({
            "type": "time",
            "interval_seconds": 1,
            "enrich": True,
        })
        assert isinstance(builder, TimeBarBuilder)
        # Verify enrichment by feeding ticks and checking output type
        builder.on_tick(_tick(timestamp_ns=0))
        bar = builder.on_tick(_tick(timestamp_ns=1_000_000_000))
        assert isinstance(bar, DollarBar)

    def test_available_types(self):
        types = BarBuilderFactory.available_types()
        assert "time" in types
        assert "dollar" in types
        assert "volume" in types
        assert "tick" in types
