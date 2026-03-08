"""Tests for the event bus pub/sub system."""

import asyncio
import time

import pytest

from src.core.events import (
    EventBus,
    EventType,
    TickEvent,
    BarEvent,
    SignalEvent,
    FillEvent,
    RiskEvent,
)


def _make_tick(**kwargs) -> TickEvent:
    defaults = dict(
        symbol="MESM6", bid=5500.0, ask=5500.25,
        last_price=5500.0, last_size=1, timestamp_ns=time.time_ns(),
    )
    defaults.update(kwargs)
    return TickEvent(**defaults)


def _make_bar(**kwargs) -> BarEvent:
    defaults = dict(
        symbol="MESM6", open=5500.0, high=5501.0, low=5499.0,
        close=5500.5, volume=100, bar_type="1s", timestamp_ns=time.time_ns(),
    )
    defaults.update(kwargs)
    return BarEvent(**defaults)


async def test_subscribe_and_receive(event_bus: EventBus):
    """Subscriber receives published events of its type."""
    received = []

    async def handler(event):
        received.append(event)
        event_bus.stop()

    event_bus.subscribe(EventType.TICK, handler)
    tick = _make_tick()
    await event_bus.publish(tick)

    await asyncio.wait_for(event_bus.run(), timeout=2.0)
    assert len(received) == 1
    assert received[0] is tick


async def test_multiple_subscribers(event_bus: EventBus):
    """Multiple subscribers for same event type all receive the event."""
    received_a = []
    received_b = []
    call_count = 0

    async def handler_a(event):
        nonlocal call_count
        received_a.append(event)
        call_count += 1
        if call_count >= 2:
            event_bus.stop()

    async def handler_b(event):
        nonlocal call_count
        received_b.append(event)
        call_count += 1
        if call_count >= 2:
            event_bus.stop()

    event_bus.subscribe(EventType.TICK, handler_a)
    event_bus.subscribe(EventType.TICK, handler_b)
    await event_bus.publish(_make_tick())

    await asyncio.wait_for(event_bus.run(), timeout=2.0)
    assert len(received_a) == 1
    assert len(received_b) == 1


async def test_subscriber_isolation(event_bus: EventBus):
    """Subscriber for TICK does not receive BAR events."""
    tick_received = []
    bar_received = []

    async def tick_handler(event):
        tick_received.append(event)

    async def bar_handler(event):
        bar_received.append(event)
        event_bus.stop()

    event_bus.subscribe(EventType.TICK, tick_handler)
    event_bus.subscribe(EventType.BAR, bar_handler)

    # Only publish a BAR
    await event_bus.publish(_make_bar())

    await asyncio.wait_for(event_bus.run(), timeout=2.0)
    assert len(tick_received) == 0
    assert len(bar_received) == 1


async def test_event_counts_tracked(event_bus: EventBus):
    """EventBus tracks count of published events by type."""
    await event_bus.publish(_make_tick())
    await event_bus.publish(_make_tick())
    await event_bus.publish(_make_bar())

    counts = event_bus.event_counts
    assert counts[EventType.TICK] == 2
    assert counts[EventType.BAR] == 1


async def test_publish_to_empty_bus(event_bus: EventBus):
    """Publishing when no subscribers does not error."""
    await event_bus.publish(_make_tick())
    assert event_bus.event_counts[EventType.TICK] == 1


async def test_subscriber_exception_does_not_crash_bus(event_bus: EventBus):
    """A failing subscriber does not prevent other subscribers from running."""
    received = []

    async def bad_handler(event):
        raise ValueError("boom")

    async def good_handler(event):
        received.append(event)
        event_bus.stop()

    event_bus.subscribe(EventType.TICK, bad_handler)
    event_bus.subscribe(EventType.TICK, good_handler)
    await event_bus.publish(_make_tick())

    await asyncio.wait_for(event_bus.run(), timeout=2.0)
    assert len(received) == 1


async def test_event_types():
    """Each event dataclass reports the correct EventType."""
    assert _make_tick().event_type == EventType.TICK
    assert _make_bar().event_type == EventType.BAR
    assert SignalEvent("orb", "BUY", 0.8, "test", time.time_ns()).event_type == EventType.SIGNAL
    assert FillEvent("o1", "MESM6", "BUY", 5500.0, 1, 0.35, time.time_ns()).event_type == EventType.FILL
    assert RiskEvent("HALT", "test", time.time_ns()).event_type == EventType.RISK
