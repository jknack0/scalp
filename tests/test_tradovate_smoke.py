"""Smoke tests for Tradovate integration (live-mode paths).

Tests the live OMS submit/cancel/flatten/bracket and FillMonitor polling,
all with mocked HTTP — no real API calls.
"""

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import BotConfig
from src.core.events import EventBus, EventType, FillEvent
from src.models.hmm_regime import RegimeState
from src.oms.fill_monitor import FillMonitor
from src.oms.tradovate_oms import (
    ManagedOrder,
    OrderStatus,
    TradovateOMS,
)
from src.strategies.base import Direction, Signal


# ── Helpers ──────────────────────────────────────────────────────


def _make_config(**overrides) -> BotConfig:
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


def _make_signal(**overrides) -> Signal:
    defaults = {
        "strategy_id": "test",
        "direction": Direction.LONG,
        "entry_price": 5000.0,
        "target_price": 5002.0,
        "stop_price": 4998.0,
        "signal_time": datetime.now(),
        "expiry_time": datetime.now(),
        "confidence": 0.8,
        "regime_state": RegimeState.LOW_VOL_RANGE,
    }
    defaults.update(overrides)
    return Signal(**defaults)


def _mock_response(status=200, json_data=None, text=""):
    """Create a mock aiohttp response usable as async context manager."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=text)
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_live_oms(bus: EventBus) -> TradovateOMS:
    """Create a live-mode OMS with mocked auth."""
    config = _make_config()
    oms = TradovateOMS(bus, config, paper=False)
    return oms


async def _init_live_oms(bus: EventBus) -> TradovateOMS:
    """Create and initialize a live-mode OMS with all API calls mocked."""
    oms = _make_live_oms(bus)

    # Mock auth
    oms._auth.authenticate = AsyncMock()
    oms._auth._access_token = "tok_test"
    oms._auth.access_token = "tok_test"
    oms._auth.ensure_valid_token = AsyncMock(return_value="tok_test")
    oms._auth.base_url = "https://demo.tradovateapi.com/v1"

    # Mock session for initialize() — account/list + contract/find
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = {}

    def mock_headers_update(d):
        mock_session.headers.update(d)

    mock_session.headers = MagicMock()
    mock_session.headers.update = mock_headers_update

    account_resp = _mock_response(json_data=[{"id": 12345, "name": "DEMO12345"}])
    contract_resp = _mock_response(json_data={"id": 99, "name": "MESM6"})

    mock_session.get = MagicMock(side_effect=[account_resp, contract_resp])

    with patch("aiohttp.ClientSession", return_value=mock_session):
        await oms.initialize()

    assert oms._account_id == 12345
    assert oms._contract_id == 99

    return oms


# ── Live OMS Initialize ────────────────────────────────────────


async def test_live_oms_initialize_resolves_account_and_contract():
    """initialize() calls /account/list and /contract/find, stores IDs."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    assert oms._account_id == 12345
    assert oms._account_name == "DEMO12345"
    assert oms._contract_id == 99
    assert not oms.is_paper


async def test_live_oms_initialize_no_accounts_raises():
    """initialize() raises if /account/list returns empty."""
    bus = EventBus()
    oms = _make_live_oms(bus)
    oms._auth.authenticate = AsyncMock()
    oms._auth._access_token = "tok_test"
    oms._auth.access_token = "tok_test"
    oms._auth.base_url = "https://demo.tradovateapi.com/v1"

    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    account_resp = _mock_response(json_data=[])
    mock_session.get = MagicMock(return_value=account_resp)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(RuntimeError, match="No Tradovate accounts"):
            await oms.initialize()


# ── Live Order Submission ──────────────────────────────────────


async def test_live_submit_posts_correct_payload():
    """submit_order() in live mode POSTs to /order/placeorder with correct fields."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    # Mock session for submit
    place_resp = _mock_response(json_data={"orderId": 777})
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.post = MagicMock(return_value=place_resp)
    oms._session = mock_session

    signal = _make_signal(target_price=0.0, stop_price=0.0)  # no bracket
    order_id = await oms.submit_order(signal)

    # Verify order tracked
    order = oms.get_order(order_id)
    assert order is not None
    assert order.tradovate_order_id == 777
    assert order.status == OrderStatus.WORKING
    assert order.direction == "Buy"

    # Verify POST payload
    call_args = mock_session.post.call_args
    assert "/order/placeorder" in call_args[0][0]
    payload = call_args[1]["json"]
    assert payload["action"] == "Buy"
    assert payload["symbol"] == "MESM6"
    assert payload["orderQty"] == 1
    assert payload["orderType"] == "Market"
    assert payload["isAutomated"] is True
    assert payload["accountId"] == 12345


async def test_live_submit_short_direction():
    """SHORT signal maps to 'Sell' direction in the order payload."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    place_resp = _mock_response(json_data={"orderId": 888})
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.post = MagicMock(return_value=place_resp)
    oms._session = mock_session

    signal = _make_signal(direction=Direction.SHORT, target_price=0.0, stop_price=0.0)
    await oms.submit_order(signal)

    payload = mock_session.post.call_args[1]["json"]
    assert payload["action"] == "Sell"


async def test_live_submit_rejection():
    """Non-200 response sets order status to REJECTED."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    reject_resp = _mock_response(status=400, json_data={"errorText": "Insufficient margin"})
    # override raise_for_status so it doesn't raise
    reject_resp.raise_for_status = MagicMock()
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.post = MagicMock(return_value=reject_resp)
    oms._session = mock_session

    signal = _make_signal(target_price=0.0, stop_price=0.0)
    order_id = await oms.submit_order(signal)

    order = oms.get_order(order_id)
    assert order.status == OrderStatus.REJECTED


# ── Bracket Orders ─────────────────────────────────────────────


async def test_live_submit_with_bracket():
    """When target and stop are set, bracket orders are placed after entry."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    # Entry response + 2 bracket responses
    entry_resp = _mock_response(json_data={"orderId": 100})
    tp_resp = _mock_response(json_data={"orderId": 101})
    sl_resp = _mock_response(json_data={"orderId": 102})

    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.post = MagicMock(side_effect=[entry_resp, tp_resp, sl_resp])
    oms._session = mock_session

    signal = _make_signal(
        entry_price=5000.0,
        target_price=5002.0,
        stop_price=4998.0,
    )
    order_id = await oms.submit_order(signal)

    order = oms.get_order(order_id)
    assert order.target_order_id == 101
    assert order.stop_order_id == 102

    # 3 POSTs total: entry + TP + SL
    assert mock_session.post.call_count == 3

    # TP payload check
    tp_payload = mock_session.post.call_args_list[1][1]["json"]
    assert tp_payload["action"] == "Sell"  # opposite of Buy entry
    assert tp_payload["orderType"] == "Limit"
    assert tp_payload["price"] == 5002.0

    # SL payload check
    sl_payload = mock_session.post.call_args_list[2][1]["json"]
    assert sl_payload["action"] == "Sell"
    assert sl_payload["orderType"] == "Stop"
    assert sl_payload["stopPrice"] == 4998.0


# ── Cancel Order ───────────────────────────────────────────────


async def test_live_cancel_order():
    """cancel_order() POSTs to /order/cancelorder."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    # Create a working order
    order = ManagedOrder(
        order_id="test-1",
        strategy_id="test",
        symbol="MESM6",
        direction="Buy",
        qty=1,
        order_type="Market",
        price=5000.0,
        status=OrderStatus.WORKING,
        tradovate_order_id=555,
    )
    oms._orders["test-1"] = order

    cancel_resp = _mock_response(status=200)
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.post = MagicMock(return_value=cancel_resp)
    oms._session = mock_session

    result = await oms.cancel_order("test-1")

    assert result is True
    assert order.status == OrderStatus.CANCELLED

    payload = mock_session.post.call_args[1]["json"]
    assert payload["orderId"] == 555


async def test_cancel_nonexistent_order():
    """cancel_order() returns False for unknown order_id."""
    bus = EventBus()
    oms = await _init_live_oms(bus)
    assert await oms.cancel_order("nonexistent-99") is False


# ── Get Position ───────────────────────────────────────────────


async def test_live_get_position():
    """get_position() queries /position/list and returns netPos."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    pos_resp = _mock_response(json_data=[
        {"contractId": 99, "netPos": 2},
        {"contractId": 50, "netPos": -1},
    ])
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.get = MagicMock(return_value=pos_resp)
    oms._session = mock_session

    pos = await oms.get_position("MESM6")
    assert pos == 2


async def test_live_get_position_no_match():
    """get_position() returns 0 when contract not in position list."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    pos_resp = _mock_response(json_data=[{"contractId": 50, "netPos": -1}])
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.get = MagicMock(return_value=pos_resp)
    oms._session = mock_session

    pos = await oms.get_position("MESM6")
    assert pos == 0


# ── Flatten ────────────────────────────────────────────────────


async def test_live_flatten_long_position():
    """flatten() sends a Sell market order to close a long position."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    # Mock get_position to return +2
    pos_resp = _mock_response(json_data=[{"contractId": 99, "netPos": 2}])
    flatten_resp = _mock_response(status=200)

    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.get = MagicMock(return_value=pos_resp)
    mock_session.post = MagicMock(return_value=flatten_resp)
    oms._session = mock_session

    await oms.flatten()

    payload = mock_session.post.call_args[1]["json"]
    assert payload["action"] == "Sell"
    assert payload["orderQty"] == 2
    assert payload["orderType"] == "Market"


async def test_live_flatten_short_position():
    """flatten() sends a Buy market order to close a short position."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    pos_resp = _mock_response(json_data=[{"contractId": 99, "netPos": -3}])
    flatten_resp = _mock_response(status=200)

    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.get = MagicMock(return_value=pos_resp)
    mock_session.post = MagicMock(return_value=flatten_resp)
    oms._session = mock_session

    await oms.flatten()

    payload = mock_session.post.call_args[1]["json"]
    assert payload["action"] == "Buy"
    assert payload["orderQty"] == 3


async def test_flatten_no_position_is_noop():
    """flatten() does nothing when position is 0."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    pos_resp = _mock_response(json_data=[])
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.get = MagicMock(return_value=pos_resp)
    mock_session.post = MagicMock()
    oms._session = mock_session

    await oms.flatten()
    mock_session.post.assert_not_called()


# ── Cancel All ─────────────────────────────────────────────────


async def test_cancel_all():
    """cancel_all() cancels all PENDING/WORKING orders."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    for i in range(3):
        order = ManagedOrder(
            order_id=f"test-{i}",
            strategy_id="test",
            symbol="MESM6",
            direction="Buy",
            qty=1,
            order_type="Market",
            price=5000.0,
            status=OrderStatus.WORKING,
            tradovate_order_id=100 + i,
        )
        oms._orders[f"test-{i}"] = order

    # Also add a filled order — should NOT be cancelled
    filled = ManagedOrder(
        order_id="test-filled",
        strategy_id="test",
        symbol="MESM6",
        direction="Buy",
        qty=1,
        order_type="Market",
        price=5000.0,
        status=OrderStatus.FILLED,
        tradovate_order_id=200,
    )
    oms._orders["test-filled"] = filled

    cancel_resp = _mock_response(status=200)
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.post = MagicMock(return_value=cancel_resp)
    oms._session = mock_session

    count = await oms.cancel_all()
    assert count == 3
    assert filled.status == OrderStatus.FILLED  # not touched


# ── FillMonitor ────────────────────────────────────────────────


async def test_fill_monitor_detects_fill():
    """FillMonitor polls /order/item and emits FillEvent on fill."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    # Create a WORKING order
    order = ManagedOrder(
        order_id="mon-1",
        strategy_id="test",
        symbol="MESM6",
        direction="Buy",
        qty=1,
        order_type="Market",
        price=5000.0,
        status=OrderStatus.WORKING,
        tradovate_order_id=300,
    )
    oms._orders["mon-1"] = order

    # Mock order/item response
    item_resp = _mock_response(json_data={
        "ordStatus": "Filled",
        "avgPx": 5000.25,
    })
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.get = MagicMock(return_value=item_resp)
    oms._session = mock_session

    fills = []

    async def capture_fill(f: FillEvent):
        fills.append(f)

    bus.subscribe(EventType.FILL, capture_fill)
    bus_task = asyncio.create_task(bus.run())

    monitor = FillMonitor(bus, oms)
    await monitor._check_orders()

    await asyncio.sleep(0.1)
    bus.stop()
    await bus_task

    assert order.status == OrderStatus.FILLED
    assert order.fill_price == 5000.25
    assert len(fills) == 1
    assert fills[0].fill_price == 5000.25
    assert fills[0].direction == "BUY"


async def test_fill_monitor_detects_cancellation():
    """FillMonitor updates order status on Cancelled response."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    order = ManagedOrder(
        order_id="mon-2",
        strategy_id="test",
        symbol="MESM6",
        direction="Sell",
        qty=1,
        order_type="Limit",
        price=5005.0,
        status=OrderStatus.WORKING,
        tradovate_order_id=301,
    )
    oms._orders["mon-2"] = order

    item_resp = _mock_response(json_data={"ordStatus": "Cancelled"})
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.get = MagicMock(return_value=item_resp)
    oms._session = mock_session

    monitor = FillMonitor(bus, oms)
    await monitor._check_orders()

    assert order.status == OrderStatus.CANCELLED


async def test_fill_monitor_skips_paper_mode():
    """FillMonitor.run() is a no-op in paper mode."""
    bus = EventBus()
    config = _make_config()
    oms = TradovateOMS(bus, config, paper=True)
    await oms.initialize()

    monitor = FillMonitor(bus, oms)

    # Run briefly then stop
    async def stop_soon():
        await asyncio.sleep(0.1)
        monitor.stop()

    await asyncio.gather(monitor.run(), stop_soon())
    # If we got here without hanging, paper skip works


async def test_fill_monitor_skips_orders_without_tv_id():
    """FillMonitor skips orders that have no tradovate_order_id."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    order = ManagedOrder(
        order_id="mon-3",
        strategy_id="test",
        symbol="MESM6",
        direction="Buy",
        qty=1,
        order_type="Market",
        price=5000.0,
        status=OrderStatus.WORKING,
        tradovate_order_id=None,  # No TV ID yet
    )
    oms._orders["mon-3"] = order

    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = MagicMock()
    mock_session.get = MagicMock()
    oms._session = mock_session

    monitor = FillMonitor(bus, oms)
    await monitor._check_orders()

    # Should not have queried the API
    mock_session.get.assert_not_called()
    assert order.status == OrderStatus.WORKING


# ── Token Refresh / Session Handling ───────────────────────────


async def test_ensure_session_refreshes_token():
    """_ensure_session() updates Authorization header with fresh token."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    oms._auth.ensure_valid_token = AsyncMock(return_value="tok_refreshed")

    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.headers = {}

    def mock_update(d):
        mock_session.headers.update(d)

    mock_session.headers = {"Authorization": "Bearer tok_old"}
    mock_session.headers.update = mock_update
    oms._session = mock_session

    await oms._ensure_session()

    assert mock_session.headers["Authorization"] == "Bearer tok_refreshed"


async def test_ensure_session_creates_new_on_closed():
    """_ensure_session() creates new ClientSession if current is closed."""
    bus = EventBus()
    oms = await _init_live_oms(bus)

    oms._auth.ensure_valid_token = AsyncMock(return_value="tok_new")
    oms._session = None  # simulate closed/missing

    with patch("aiohttp.ClientSession") as MockSession:
        mock_instance = AsyncMock()
        mock_instance.closed = False
        MockSession.return_value = mock_instance

        await oms._ensure_session()

        MockSession.assert_called_once()
        assert oms._session is mock_instance
