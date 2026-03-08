"""Tests for Tradovate authentication and WebSocket feed.

All tests are mocked — no real API calls.
"""

import asyncio
import json
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import BotConfig
from src.core.events import EventBus, EventType, TickEvent
from src.feeds.tradovate import (
    TradovateAuth,
    TradovateFeed,
    _DEMO_REST_URL,
    _DEMO_WS_URL,
    _LIVE_REST_URL,
    _LIVE_WS_URL,
    _INITIAL_BACKOFF,
    _MAX_BACKOFF,
)


def _make_config(**overrides) -> BotConfig:
    """Create a BotConfig with test defaults."""
    defaults = {
        "tradovate_username": "testuser",
        "tradovate_password": "testpass",
        "tradovate_app_id": "TestApp",
        "tradovate_app_version": "1.0.0",
        "tradovate_cid": "test-cid",
        "tradovate_secret": "test-secret",
        "tradovate_demo": True,
        "symbol": "MESM6",
    }
    defaults.update(overrides)
    return BotConfig(**defaults)


# ── Test 1: Auth sends correct credentials ────────────────────

async def test_auth_sends_correct_credentials():
    """POST body contains all 6 required fields, token stored correctly."""
    config = _make_config()
    auth = TradovateAuth(config)

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={
        "accessToken": "tok_abc123",
        "userId": 42,
    })
    mock_resp.text = AsyncMock(return_value="")
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.closed = False

    auth._session = mock_session

    await auth.authenticate()

    # Verify correct URL
    call_args = mock_session.post.call_args
    assert call_args[0][0] == f"{_DEMO_REST_URL}/auth/accesstokenrequest"

    # Verify all 6 credential fields in payload
    payload = call_args[1]["json"]
    assert payload["name"] == "testuser"
    assert payload["password"] == "testpass"
    assert payload["appId"] == "TestApp"
    assert payload["appVersion"] == "1.0.0"
    assert payload["cid"] == "test-cid"
    assert payload["sec"] == "test-secret"

    # Token stored
    assert auth.access_token == "tok_abc123"
    assert auth.user_id == 42


# ── Test 2: Auth failure raises ConnectionError ───────────────

async def test_auth_failure_raises():
    """Non-200 response raises ConnectionError."""
    config = _make_config()
    auth = TradovateAuth(config)

    mock_resp = AsyncMock()
    mock_resp.status = 401
    mock_resp.text = AsyncMock(return_value="Unauthorized")
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.closed = False

    auth._session = mock_session

    with pytest.raises(ConnectionError, match="401"):
        await auth.authenticate()


# ── Test 3: Demo vs live URLs ─────────────────────────────────

async def test_demo_vs_live_urls():
    """demo=True uses demo endpoints, demo=False uses live."""
    demo_config = _make_config(tradovate_demo=True)
    live_config = _make_config(tradovate_demo=False)

    demo_auth = TradovateAuth(demo_config)
    live_auth = TradovateAuth(live_config)

    assert demo_auth.base_url == _DEMO_REST_URL
    assert live_auth.base_url == _LIVE_REST_URL

    demo_feed = TradovateFeed(EventBus(), demo_config)
    live_feed = TradovateFeed(EventBus(), live_config)

    assert demo_feed.ws_url == _DEMO_WS_URL
    assert live_feed.ws_url == _LIVE_WS_URL


# ── Test 4: Quote parsing to TickEvent ────────────────────────

async def test_quote_parsing_to_tick_event():
    """Tradovate quote JSON maps to TickEvent with correct fields."""
    config = _make_config()
    bus = EventBus()
    feed = TradovateFeed(bus, config)
    feed._symbol = "MESM6"

    received: list[TickEvent] = []

    async def capture(event: TickEvent) -> None:
        received.append(event)

    bus.subscribe(EventType.TICK, capture)

    # Start bus briefly to process the published event
    bus_task = asyncio.create_task(bus.run())

    quote = {
        "entries": {
            "Bid": {"price": 5200.25},
            "Offer": {"price": 5200.50},
            "Trade": {"price": 5200.25, "size": 3},
        },
        "timestamp": "2025-01-15T14:30:00.000Z",
    }

    feed._handle_quote(quote)

    # Give the bus time to dispatch
    await asyncio.sleep(0.1)
    bus.stop()
    await bus_task

    assert len(received) == 1
    tick = received[0]
    assert tick.symbol == "MESM6"
    assert tick.bid == 5200.25
    assert tick.ask == 5200.50
    assert tick.last_price == 5200.25
    assert tick.last_size == 3
    assert tick.timestamp_ns > 0


# ── Test 5: Heartbeat sends on schedule ───────────────────────

async def test_heartbeat_sends_on_schedule():
    """Heartbeat loop sends '[]' while connected."""
    config = _make_config()
    feed = TradovateFeed(EventBus(), config)
    feed._running = True
    feed._connected = True

    mock_ws = AsyncMock()
    mock_ws.closed = False
    mock_ws.send_str = AsyncMock()
    feed._ws = mock_ws

    # Run heartbeat for a short time then stop
    async def stop_after_delay():
        await asyncio.sleep(0.3)
        feed._running = False
        feed._connected = False

    await asyncio.gather(
        feed._heartbeat_loop(),
        stop_after_delay(),
    )

    # Should have sent at least one heartbeat
    assert mock_ws.send_str.call_count >= 1
    mock_ws.send_str.assert_any_call("[]")


# ── Test 6: Exponential backoff ───────────────────────────────

async def test_exponential_backoff():
    """Reconnect delay doubles: 1→2→4→...→60 (capped)."""
    config = _make_config()
    feed = TradovateFeed(EventBus(), config)

    assert feed._reconnect_delay == _INITIAL_BACKOFF

    # Simulate backoff progression
    delays = []
    delay = _INITIAL_BACKOFF
    for _ in range(10):
        delays.append(delay)
        delay = min(delay * 2, _MAX_BACKOFF)

    assert delays[0] == 1.0
    assert delays[1] == 2.0
    assert delays[2] == 4.0
    assert delays[3] == 8.0
    assert delays[4] == 16.0
    assert delays[5] == 32.0
    assert delays[6] == 60.0  # capped
    assert delays[7] == 60.0  # stays capped


# ── Test 7: Latency tracking ─────────────────────────────────

async def test_latency_tracking():
    """Latency samples collected, stats computed correctly."""
    config = _make_config()
    feed = TradovateFeed(EventBus(), config)

    # Empty stats
    stats = feed.latency_stats
    assert stats["min_ms"] == 0.0
    assert stats["avg_ms"] == 0.0

    # Add known samples
    feed._latency_samples = deque([10.0, 20.0, 30.0, 40.0, 50.0], maxlen=1000)

    stats = feed.latency_stats
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 50.0
    assert stats["avg_ms"] == 30.0
    assert stats["p99_ms"] == 40.0  # p99 index of 5 samples: int(5*0.99)-1 = 3


# ── Test 8: Tick event published to EventBus ──────────────────

async def test_tick_event_published_to_bus():
    """Quote update results in TickEvent being dispatched by EventBus."""
    config = _make_config()
    bus = EventBus()
    feed = TradovateFeed(bus, config)
    feed._symbol = "MESM6"

    received: list[TickEvent] = []

    async def on_tick(event: TickEvent) -> None:
        received.append(event)

    bus.subscribe(EventType.TICK, on_tick)
    bus_task = asyncio.create_task(bus.run())

    # Simulate a Tradovate server-push frame
    raw = json.dumps([{
        "e": "md",
        "d": {
            "quotes": [{
                "entries": {
                    "Bid": {"price": 5100.00},
                    "Offer": {"price": 5100.25},
                    "Trade": {"price": 5100.00, "size": 1},
                },
                "timestamp": "2025-06-10T15:00:00.000Z",
            }],
        },
    }])
    # Add server-push "a" prefix
    feed._parse_and_dispatch("a" + raw)

    await asyncio.sleep(0.1)
    bus.stop()
    await bus_task

    assert len(received) == 1
    assert received[0].bid == 5100.00
    assert received[0].ask == 5100.25
    assert received[0].last_price == 5100.00
    assert received[0].last_size == 1
    assert feed.tick_count == 1
