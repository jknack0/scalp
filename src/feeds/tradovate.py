"""Tradovate WebSocket market data feed.

Tradovate API endpoints:
- Demo REST:  https://demo.tradovateapi.com/v1/
- Live REST:  https://live.tradovateapi.com/v1/
- Demo WS:    wss://md-demo.tradovateapi.com/v1/websocket
- Live WS:    wss://md.tradovateapi.com/v1/websocket

Authentication flow:
1. POST /auth/accesstokenrequest with credentials -> access_token
2. Connect WebSocket, send authorize frame with token
3. Subscribe: md/subscribeQuote for real-time quotes
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import aiohttp

from src.core.config import BotConfig
from src.core.events import EventBus, TickEvent
from src.core.logging import get_logger
from src.feeds.base import BaseFeed

logger = get_logger("feed.tradovate")

# Tradovate API base URLs
_DEMO_REST_URL = "https://demo.tradovateapi.com/v1"
_LIVE_REST_URL = "https://live.tradovateapi.com/v1"
_DEMO_WS_URL = "wss://md-demo.tradovateapi.com/v1/websocket"
_LIVE_WS_URL = "wss://md.tradovateapi.com/v1/websocket"

# Reconnect backoff limits
_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 60.0

# Heartbeat and stale detection intervals (seconds)
_HEARTBEAT_INTERVAL = 2.5
_STALE_WARN_SECONDS = 30.0
_STALE_RECONNECT_SECONDS = 60.0
_TOKEN_CHECK_INTERVAL = 60.0
_TOKEN_REFRESH_BUFFER = 300  # 5 minutes before expiry


class TradovateAuth:
    """Handles Tradovate REST authentication and token lifecycle."""

    def __init__(self, config: BotConfig) -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None
        self._access_token: str = ""
        self._user_id: int | None = None
        self._expiry: float = 0.0  # Unix timestamp when token expires

    @property
    def base_url(self) -> str:
        return _DEMO_REST_URL if self._config.tradovate_demo else _LIVE_REST_URL

    @property
    def access_token(self) -> str:
        return self._access_token

    @property
    def user_id(self) -> int | None:
        return self._user_id

    @property
    def is_expired(self) -> bool:
        return time.time() >= (self._expiry - _TOKEN_REFRESH_BUFFER)

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def authenticate(self) -> None:
        """POST /auth/accesstokenrequest to get an access token."""
        session = await self._ensure_session()
        url = f"{self.base_url}/auth/accesstokenrequest"
        payload = {
            "name": self._config.tradovate_username,
            "password": self._config.tradovate_password,
            "appId": self._config.tradovate_app_id,
            "appVersion": self._config.tradovate_app_version,
            "cid": self._config.tradovate_cid,
            "sec": self._config.tradovate_secret,
        }

        logger.info("auth_requesting", url=url)
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise ConnectionError(
                    f"Tradovate auth failed (HTTP {resp.status}): {body}"
                )
            data = await resp.json()

        self._access_token = data["accessToken"]
        self._user_id = data.get("userId")
        # Tradovate tokens last ~80 minutes; expirationTime is ISO string
        # Use a conservative 70-minute window from now
        self._expiry = time.time() + 4200
        logger.info("auth_success", user_id=self._user_id)

    async def refresh_token(self) -> None:
        """POST /auth/renewAccessToken. Falls back to full re-auth on failure."""
        try:
            session = await self._ensure_session()
            url = f"{self.base_url}/auth/renewaccesstoken"
            headers = {"Authorization": f"Bearer {self._access_token}"}
            async with session.post(url, headers=headers) as resp:
                if resp.status != 200:
                    raise ConnectionError(f"Token refresh failed (HTTP {resp.status})")
                data = await resp.json()
            self._access_token = data["accessToken"]
            self._expiry = time.time() + 4200
            logger.info("token_refreshed")
        except Exception:
            logger.warning("token_refresh_failed_reauthing")
            await self.authenticate()

    async def ensure_valid_token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        if not self._access_token:
            await self.authenticate()
        elif self.is_expired:
            await self.refresh_token()
        return self._access_token

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


class TradovateFeed(BaseFeed):
    """Tradovate WebSocket market data feed.

    Connects to the Tradovate market data WebSocket, subscribes to
    quote updates, and publishes TickEvents to the EventBus.
    """

    def __init__(self, event_bus: EventBus, config: BotConfig) -> None:
        super().__init__(event_bus)
        self._config = config
        self._auth = TradovateAuth(config)

        # WebSocket state
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_session: aiohttp.ClientSession | None = None
        self._running = False
        self._connected = False

        # Reconnection
        self._reconnect_delay = _INITIAL_BACKOFF

        # Metrics
        self._latency_samples: deque[float] = deque(maxlen=1000)
        self._last_tick_time: float = time.monotonic()
        self._tick_count: int = 0
        self._request_id: int = 0

        # Subscribed symbol
        self._symbol: str = ""

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    @property
    def ws_url(self) -> str:
        return _DEMO_WS_URL if self._config.tradovate_demo else _LIVE_WS_URL

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def latency_stats(self) -> dict[str, float]:
        """Return latency statistics in milliseconds."""
        if not self._latency_samples:
            return {"min_ms": 0.0, "avg_ms": 0.0, "max_ms": 0.0, "p99_ms": 0.0}

        samples = sorted(self._latency_samples)
        p99_idx = max(0, int(len(samples) * 0.99) - 1)
        return {
            "min_ms": samples[0],
            "avg_ms": sum(samples) / len(samples),
            "max_ms": samples[-1],
            "p99_ms": samples[p99_idx],
        }

    async def connect(self) -> None:
        """Authenticate via REST, then open WebSocket and authorize."""
        # REST authentication
        token = await self._auth.ensure_valid_token()

        # Open WebSocket
        self._ws_session = aiohttp.ClientSession()
        self._ws = await self._ws_session.ws_connect(self.ws_url)

        # Tradovate uses SockJS framing — consume the 'o' (open) frame first
        open_resp = await self._ws.receive(timeout=10)
        if open_resp.type == aiohttp.WSMsgType.TEXT:
            frame = open_resp.data.strip()
            if frame == "o":
                logger.debug("ws_sockjs_open")
            else:
                logger.warning("ws_unexpected_open_frame", data=frame[:100])
        else:
            raise ConnectionError(f"Expected SockJS open frame, got: {open_resp.type}")

        # Authorize on the WebSocket
        req_id = self._next_request_id()
        auth_msg = f"authorize\n{req_id}\n\n{token}"
        await self._ws.send_str(auth_msg)

        # Wait for authorization response — skip any 'h' heartbeat frames
        for _ in range(10):
            resp = await self._ws.receive(timeout=10)
            if resp.type != aiohttp.WSMsgType.TEXT:
                raise ConnectionError(f"Unexpected WS response type: {resp.type}")
            body = resp.data.strip()
            if body == "h":
                continue  # SockJS heartbeat, skip
            if '"s":200' in body or '"i":' in body:
                self._connected = True
                self._reconnect_delay = _INITIAL_BACKOFF
                logger.info("ws_authorized", url=self.ws_url)
                return
            else:
                raise ConnectionError(f"WS authorization failed: {body}")

        raise ConnectionError("WS authorization timed out — too many heartbeat frames")

    async def subscribe(self, symbol: str) -> None:
        """Subscribe to quote updates for a symbol."""
        if not self._ws or self._ws.closed:
            raise ConnectionError("WebSocket not connected")

        self._symbol = symbol
        req_id = self._next_request_id()
        sub_msg = f'md/subscribeQuote\n{req_id}\n\n{{"symbol":"{symbol}"}}'
        await self._ws.send_str(sub_msg)
        logger.info("subscribed", symbol=symbol, request_id=req_id)

    async def disconnect(self) -> None:
        """Unsubscribe and close WebSocket + auth session."""
        self._connected = False

        if self._ws and not self._ws.closed:
            # Unsubscribe if we have a symbol
            if self._symbol:
                try:
                    req_id = self._next_request_id()
                    unsub = f'md/unsubscribeQuote\n{req_id}\n\n{{"symbol":"{self._symbol}"}}'
                    await self._ws.send_str(unsub)
                except Exception:
                    pass
            try:
                await self._ws.close()
            except Exception:
                pass

        if self._ws_session and not self._ws_session.closed:
            await self._ws_session.close()
            self._ws_session = None

        await self._auth.close()
        logger.info("disconnected")

    def stop(self) -> None:
        """Signal the run loop to exit."""
        self._running = False
        self._connected = False

    async def run(self) -> None:
        """Main loop with automatic reconnection."""
        self._running = True
        logger.info("feed_starting", symbol=self._config.symbol)

        while self._running:
            try:
                await self.connect()
                await self.subscribe(self._config.symbol)

                # Run 4 concurrent tasks; if any fails, all cancel
                await asyncio.gather(
                    self._message_loop(),
                    self._heartbeat_loop(),
                    self._stale_check_loop(),
                    self._token_refresh_loop(),
                )
            except asyncio.CancelledError:
                logger.info("feed_cancelled")
                break
            except Exception as e:
                if not self._running:
                    break
                logger.error(
                    "feed_connection_error",
                    error=str(e),
                    reconnect_delay=self._reconnect_delay,
                )
                await self._cleanup_ws()
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, _MAX_BACKOFF
                )

        await self.disconnect()
        logger.info("feed_stopped", tick_count=self._tick_count)

    async def _cleanup_ws(self) -> None:
        """Close WS resources without closing auth session."""
        self._connected = False
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._ws_session and not self._ws_session.closed:
            try:
                await self._ws_session.close()
            except Exception:
                pass
        self._ws = None
        self._ws_session = None

    # ── Internal concurrent tasks ──────────────────────────────

    async def _message_loop(self) -> None:
        """Read and route WebSocket messages."""
        assert self._ws is not None
        async for msg in self._ws:
            if not self._running:
                return

            if msg.type == aiohttp.WSMsgType.TEXT:
                data = msg.data.strip()
                if not data or data == "[]" or data in ("h", "o"):
                    # SockJS heartbeat/open frame or empty — ignore
                    continue
                self._parse_and_dispatch(data)

            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.ERROR,
            ):
                logger.warning("ws_closed", type=str(msg.type))
                return

    async def _heartbeat_loop(self) -> None:
        """Send heartbeat frames every 2.5 seconds."""
        while self._running and self._connected:
            if self._ws and not self._ws.closed:
                try:
                    await self._ws.send_str("[]")
                except Exception:
                    return
            await asyncio.sleep(_HEARTBEAT_INTERVAL)

    async def _stale_check_loop(self) -> None:
        """Warn on stale data during RTH, force reconnect after 60s."""
        from src.core.session import SessionManager

        session = SessionManager(
            session_start=self._config.session_start,
            session_end=self._config.session_end,
            timezone=self._config.timezone,
        )

        while self._running and self._connected:
            await asyncio.sleep(10)
            if not session.is_rth():
                continue

            elapsed = time.monotonic() - self._last_tick_time
            if elapsed > _STALE_RECONNECT_SECONDS:
                logger.error("stale_data_reconnecting", seconds=elapsed)
                # Force close to trigger reconnection in the outer loop
                if self._ws and not self._ws.closed:
                    await self._ws.close()
                return
            elif elapsed > _STALE_WARN_SECONDS:
                logger.warning("stale_data_warning", seconds=elapsed)

    async def _token_refresh_loop(self) -> None:
        """Periodically check and refresh the access token."""
        while self._running and self._connected:
            await asyncio.sleep(_TOKEN_CHECK_INTERVAL)
            if self._auth.is_expired:
                try:
                    await self._auth.refresh_token()
                except Exception:
                    logger.exception("token_refresh_error")

    # ── Message parsing ────────────────────────────────────────

    def _parse_and_dispatch(self, raw: str) -> None:
        """Parse Tradovate WS frames and dispatch quote updates.

        Tradovate WS protocol frames look like:
            a[{"e":"md","d":{"quotes":[{...}]}}]

        The outer "a" prefix and brackets wrap server-push events.
        Quote entries contain nested dicts with Bid/Offer/Trade data.
        """
        try:
            # Strip the leading "a" if present (server-push prefix)
            text = raw
            if text.startswith("a"):
                text = text[1:]

            data = json.loads(text)

            # Handle array of events
            if isinstance(data, list):
                for item in data:
                    self._handle_event(item)
            elif isinstance(data, dict):
                self._handle_event(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Not all frames are quote data; some are responses to our requests
            pass

    def _handle_event(self, event: dict[str, Any]) -> None:
        """Route a single event dict."""
        # Server-push market data events have e="md"
        if event.get("e") == "md":
            d = event.get("d", {})
            quotes = d.get("quotes", [])
            for quote in quotes:
                self._handle_quote(quote)
        # Also handle direct quote data in subscription responses
        elif "entries" in event:
            self._handle_quote(event)

    def _handle_quote(self, quote: dict[str, Any]) -> None:
        """Convert a Tradovate quote dict to a TickEvent and publish."""
        entries = quote.get("entries", {})

        bid = entries.get("Bid", {}).get("price", 0.0)
        ask = entries.get("Offer", {}).get("price", 0.0)
        trade_entry = entries.get("Trade", {})
        last_price = trade_entry.get("price", 0.0)
        last_size = trade_entry.get("size", 0)

        # Timestamp from the quote (ISO string → nanoseconds)
        ts_str = quote.get("timestamp", "")
        if ts_str:
            try:
                from datetime import datetime, timezone

                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                ts_ns = int(dt.timestamp() * 1_000_000_000)
            except (ValueError, TypeError):
                ts_ns = time.time_ns()
        else:
            ts_ns = time.time_ns()

        # Calculate latency
        exchange_time_s = ts_ns / 1_000_000_000
        latency_ms = (time.time() - exchange_time_s) * 1000
        if latency_ms >= 0:
            self._latency_samples.append(latency_ms)

        self._last_tick_time = time.monotonic()
        self._tick_count += 1

        tick = TickEvent(
            symbol=self._symbol or self._config.symbol,
            bid=bid,
            ask=ask,
            last_price=last_price,
            last_size=last_size,
            timestamp_ns=ts_ns,
        )

        # Publish to bus (fire-and-forget; bus.publish is sync-safe with put_nowait)
        asyncio.get_event_loop().create_task(self._bus.publish(tick))
