"""Tests for the quote fade detection module."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from src.filters.quote_fade import (
    FadeConfig,
    OrderRouteEvent,
    QuoteEvent,
    QuoteFadeDetector,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _quote(bid: float, ask: float, bid_size: int = 10, ask_size: int = 10) -> QuoteEvent:
    return QuoteEvent(timestamp=_now(), bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)


def _order(direction: str, price: float) -> OrderRouteEvent:
    return OrderRouteEvent(timestamp=_now(), direction=direction, intended_price=price)


class _FakeLoop:
    """Fake event loop with controllable monotonic time."""

    def __init__(self, start: float = 1000.0):
        self._time = start

    def time(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


@pytest.fixture
def fake_loop():
    loop = _FakeLoop()
    with patch("src.filters.quote_fade.asyncio") as mock_asyncio:
        mock_asyncio.get_event_loop.return_value = loop
        yield loop


async def test_fade_detected_when_quote_moves(fake_loop):
    """Fade should be detected when ask moves beyond threshold after long order."""
    config = FadeConfig(fade_window_ms=200.0, fade_threshold_ticks=1)
    detector = QuoteFadeDetector(config=config)

    # Route a long order at ask=100.25
    detector.on_order_routed(_order("long", 100.25))

    # Advance 50ms, ask moves up by 1 tick (0.25)
    fake_loop.advance(0.050)
    results = detector.on_quote(_quote(bid=100.0, ask=100.50))

    assert len(results) == 1
    assert results[0].fade_detected is True
    assert results[0].ms_to_fade is not None
    assert results[0].ms_to_fade == pytest.approx(50.0, abs=1.0)


async def test_no_fade_when_quote_stable(fake_loop):
    """No fade when quote stays at intended price through the window."""
    config = FadeConfig(fade_window_ms=100.0, fade_threshold_ticks=1)
    detector = QuoteFadeDetector(config=config)

    detector.on_order_routed(_order("long", 100.25))

    # Quote stable, advance past window
    fake_loop.advance(0.150)
    results = detector.on_quote(_quote(bid=100.0, ask=100.25))

    assert len(results) == 1
    assert results[0].fade_detected is False
    assert results[0].ms_to_fade is None


async def test_fade_detected_short_direction(fake_loop):
    """Fade on short side — bid drops beyond threshold."""
    config = FadeConfig(fade_window_ms=200.0, fade_threshold_ticks=1)
    detector = QuoteFadeDetector(config=config)

    # Route a short order at bid=100.00
    detector.on_order_routed(_order("short", 100.00))

    # Bid drops by 1 tick
    fake_loop.advance(0.030)
    results = detector.on_quote(_quote(bid=99.75, ask=100.25))

    assert len(results) == 1
    assert results[0].fade_detected is True


async def test_pending_orders_expire_after_window(fake_loop):
    """Orders should be resolved as no-fade after fade_window_ms expires."""
    config = FadeConfig(fade_window_ms=100.0, fade_threshold_ticks=1)
    detector = QuoteFadeDetector(config=config)

    detector.on_order_routed(_order("long", 100.25))
    assert detector.pending_count == 1

    # Still within window — no resolution
    fake_loop.advance(0.050)
    results = detector.on_quote(_quote(bid=100.0, ask=100.25))
    assert len(results) == 0
    assert detector.pending_count == 1

    # Past window — should expire
    fake_loop.advance(0.060)
    results = detector.on_quote(_quote(bid=100.0, ask=100.25))
    assert len(results) == 1
    assert results[0].fade_detected is False
    assert detector.pending_count == 0


async def test_recommend_limit_low_fade_rate(fake_loop):
    """Should recommend limit orders when fade rate is low."""
    config = FadeConfig(fade_window_ms=100.0, fade_threshold_ticks=1, lookback=10)
    detector = QuoteFadeDetector(config=config)

    # Route 10 orders, all stable (no fades)
    for _ in range(10):
        detector.on_order_routed(_order("long", 100.25))
        fake_loop.advance(0.150)
        detector.on_quote(_quote(bid=100.0, ask=100.25))

    assert detector.get_fade_rate() == 0.0
    assert detector.recommend_order_type() == "limit"


async def test_recommend_ioc_medium_fade_rate(fake_loop):
    """Should recommend IOC when fade rate is between 0.25 and 0.5."""
    config = FadeConfig(fade_window_ms=100.0, fade_threshold_ticks=1, lookback=10)
    detector = QuoteFadeDetector(config=config)

    # 3 fades out of 10 = 0.3
    for i in range(10):
        detector.on_order_routed(_order("long", 100.25))
        fake_loop.advance(0.050)
        if i < 3:
            # Fade — ask jumps
            detector.on_quote(_quote(bid=100.0, ask=100.75))
        else:
            # No fade — expire
            fake_loop.advance(0.060)
            detector.on_quote(_quote(bid=100.0, ask=100.25))

    rate = detector.get_fade_rate(lookback=10)
    assert 0.25 < rate <= 0.5
    assert detector.recommend_order_type() == "ioc"


async def test_recommend_market_high_fade_rate(fake_loop):
    """Should recommend market orders when fade rate exceeds 0.5."""
    config = FadeConfig(fade_window_ms=100.0, fade_threshold_ticks=1, lookback=10)
    detector = QuoteFadeDetector(config=config)

    # 8 fades out of 10 = 0.8
    for i in range(10):
        detector.on_order_routed(_order("long", 100.25))
        fake_loop.advance(0.050)
        if i < 8:
            detector.on_quote(_quote(bid=100.0, ask=100.75))
        else:
            fake_loop.advance(0.060)
            detector.on_quote(_quote(bid=100.0, ask=100.25))

    rate = detector.get_fade_rate(lookback=10)
    assert rate > 0.5
    assert detector.recommend_order_type() == "market"


async def test_fade_rate_with_no_results():
    """Fade rate should be 0.0 when no orders have been resolved."""
    detector = QuoteFadeDetector()
    assert detector.get_fade_rate() == 0.0
    assert detector.recommend_order_type() == "limit"


async def test_persist_flushes_to_parquet(fake_loop, tmp_path: Path):
    """Fade results should flush to Parquet."""
    config = FadeConfig(
        fade_window_ms=100.0,
        fade_threshold_ticks=1,
        persist_every_n=1,
        parquet_dir=str(tmp_path),
    )
    detector = QuoteFadeDetector(config=config, persist=True)

    # Generate 3 fade events
    for _ in range(3):
        detector.on_order_routed(_order("long", 100.25))
        fake_loop.advance(0.050)
        detector.on_quote(_quote(bid=100.0, ask=100.75))

    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(parquet_files) == 1

    df = pl.read_parquet(parquet_files[0])
    assert len(df) == 3
    assert set(df.columns) == {
        "timestamp",
        "fade_detected",
        "ms_to_fade",
        "fade_rate",
        "recommendation",
    }
    assert df["fade_detected"].to_list() == [True, True, True]
