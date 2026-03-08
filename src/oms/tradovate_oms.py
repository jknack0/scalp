"""Tradovate order management system.

Places, monitors, and cancels orders via Tradovate REST API.
Supports both live and paper trading modes.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum

import aiohttp

from src.core.config import BotConfig
from src.core.events import EventBus, EventType, FillEvent
from src.core.logging import get_logger
from src.feeds.tradovate import TradovateAuth
from src.oms.base import BaseOMS

logger = get_logger("oms.tradovate")

# MES tick size and multiplier
TICK_SIZE = 0.25
TICK_VALUE = 1.25  # $5 per point × 0.25 = $1.25 per tick
COMMISSION_PER_SIDE = 0.35  # Tradovate Free plan


class OrderStatus(str, Enum):
    PENDING = "Pending"
    WORKING = "Working"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"
    EXPIRED = "Expired"
    UNKNOWN = "Unknown"


@dataclass
class ManagedOrder:
    """An order being tracked by the OMS."""

    order_id: str
    strategy_id: str
    symbol: str
    direction: str  # "Buy" or "Sell"
    qty: int
    order_type: str  # "Limit", "Market", "Stop"
    price: float  # limit/stop price (0 for market)
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float = 0.0
    tradovate_order_id: int | None = None
    created_at: float = field(default_factory=time.time)
    # Bracket orders
    target_price: float = 0.0
    stop_price: float = 0.0
    target_order_id: int | None = None
    stop_order_id: int | None = None


class TradovateOMS(BaseOMS):
    """Tradovate REST-based order management.

    Supports two modes:
    - paper=True: logs orders, simulates fills at entry price (no API calls)
    - paper=False: submits real orders to Tradovate demo/live

    Usage:
        oms = TradovateOMS(bus, config, paper=True)
        await oms.initialize()
        order_id = await oms.place_order(...)
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: BotConfig,
        paper: bool = True,
    ) -> None:
        super().__init__(event_bus)
        self._config = config
        self._paper = paper
        self._auth = TradovateAuth(config)
        self._session: aiohttp.ClientSession | None = None

        # Account info (populated on initialize)
        self._account_id: int | None = None
        self._account_name: str = ""
        self._contract_id: int | None = None

        # Order tracking
        self._orders: dict[str, ManagedOrder] = {}
        self._position: int = 0  # net position (signed)
        self._next_id: int = 0

    @property
    def is_paper(self) -> bool:
        return self._paper

    @property
    def position(self) -> int:
        return self._position

    async def initialize(self) -> None:
        """Authenticate and resolve account + contract IDs."""
        if self._paper:
            logger.info("oms_initialized", mode="PAPER")
            return

        await self._auth.authenticate()
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self._auth.access_token}"}
        )

        base = self._auth.base_url

        # Resolve account
        async with self._session.get(f"{base}/account/list") as resp:
            resp.raise_for_status()
            accounts = await resp.json()
        if not accounts:
            raise RuntimeError("No Tradovate accounts found")
        self._account_id = accounts[0]["id"]
        self._account_name = accounts[0].get("name", "")

        # Resolve contract
        async with self._session.get(
            f"{base}/contract/find",
            params={"name": self._config.symbol},
        ) as resp:
            resp.raise_for_status()
            contract = await resp.json()
        self._contract_id = contract["id"]

        logger.info(
            "oms_initialized",
            mode="LIVE" if not self._config.tradovate_demo else "DEMO",
            account=self._account_name,
            contract=self._config.symbol,
            contract_id=self._contract_id,
        )

    async def submit_order(self, signal) -> str:
        """Submit an order from a strategy Signal.

        Adapts from Strategy Signal format to Tradovate order API.
        Returns a local order_id string.
        """
        from src.strategies.base import Direction, Signal

        if isinstance(signal, Signal):
            direction = "Buy" if signal.direction == Direction.LONG else "Sell"
            price = signal.entry_price
            target = signal.target_price
            stop = signal.stop_price
            strategy_id = signal.strategy_id
        else:
            # Legacy SignalEvent
            direction = signal.direction
            price = getattr(signal, "price", 0.0)
            target = 0.0
            stop = 0.0
            strategy_id = signal.strategy_id

        order_id = self._gen_id(strategy_id)
        order = ManagedOrder(
            order_id=order_id,
            strategy_id=strategy_id,
            symbol=self._config.symbol,
            direction=direction,
            qty=1,
            order_type="Market",
            price=price,
            target_price=target,
            stop_price=stop,
        )
        self._orders[order_id] = order

        if self._paper:
            return await self._paper_fill(order)

        return await self._live_submit(order)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        order = self._orders.get(order_id)
        if not order:
            return False

        if self._paper:
            order.status = OrderStatus.CANCELLED
            logger.info("order_cancelled", order_id=order_id, mode="PAPER")
            return True

        if order.tradovate_order_id is None:
            return False

        try:
            base = self._auth.base_url
            await self._ensure_session()
            async with self._session.post(
                f"{base}/order/cancelorder",
                json={"orderId": order.tradovate_order_id},
            ) as resp:
                if resp.status == 200:
                    order.status = OrderStatus.CANCELLED
                    logger.info("order_cancelled", order_id=order_id)
                    return True
                logger.warning(
                    "cancel_failed",
                    order_id=order_id,
                    status=resp.status,
                    body=await resp.text(),
                )
                return False
        except Exception:
            logger.exception("cancel_error", order_id=order_id)
            return False

    async def get_position(self, symbol: str) -> int:
        """Get current net position."""
        if self._paper:
            return self._position

        try:
            base = self._auth.base_url
            await self._ensure_session()
            async with self._session.get(f"{base}/position/list") as resp:
                if resp.status != 200:
                    return self._position
                positions = await resp.json()
            for pos in positions:
                if pos.get("contractId") == self._contract_id:
                    self._position = pos.get("netPos", 0)
                    return self._position
            return 0
        except Exception:
            logger.exception("position_query_error")
            return self._position

    async def cancel_all(self) -> int:
        """Cancel all working orders. Returns count cancelled."""
        cancelled = 0
        for oid, order in list(self._orders.items()):
            if order.status in (OrderStatus.PENDING, OrderStatus.WORKING):
                if await self.cancel_order(oid):
                    cancelled += 1
        return cancelled

    async def flatten(self) -> None:
        """Close any open position with a market order."""
        pos = await self.get_position(self._config.symbol)
        if pos == 0:
            return

        direction = "Sell" if pos > 0 else "Buy"
        qty = abs(pos)

        logger.info("flattening", direction=direction, qty=qty)

        if self._paper:
            self._position = 0
            logger.info("flattened", mode="PAPER")
            return

        try:
            base = self._auth.base_url
            await self._ensure_session()
            payload = {
                "accountSpec": self._account_name,
                "accountId": self._account_id,
                "action": direction,
                "symbol": self._config.symbol,
                "orderQty": qty,
                "orderType": "Market",
                "timeInForce": "Day",
                "isAutomated": True,
            }
            async with self._session.post(
                f"{base}/order/placeorder", json=payload
            ) as resp:
                if resp.status == 200:
                    logger.info("flatten_order_placed")
                else:
                    logger.error("flatten_failed", body=await resp.text())
        except Exception:
            logger.exception("flatten_error")

    async def close(self) -> None:
        """Cleanup: cancel orders, flatten, close session."""
        await self.cancel_all()
        if self._session and not self._session.closed:
            await self._session.close()
        await self._auth.close()

    # ── Internal ─────────────────────────────────────────────

    def _gen_id(self, strategy_id: str) -> str:
        self._next_id += 1
        return f"{strategy_id}-{self._next_id}"

    async def _ensure_session(self) -> None:
        """Ensure HTTP session has valid token."""
        token = await self._auth.ensure_valid_token()
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {token}"}
            )
        else:
            self._session.headers.update({"Authorization": f"Bearer {token}"})

    async def _paper_fill(self, order: ManagedOrder) -> str:
        """Simulate an immediate fill for paper trading."""
        order.status = OrderStatus.FILLED
        order.fill_price = order.price

        # Update position
        delta = order.qty if order.direction == "Buy" else -order.qty
        self._position += delta

        logger.info(
            "paper_fill",
            order_id=order.order_id,
            direction=order.direction,
            price=order.price,
            target=order.target_price,
            stop=order.stop_price,
            position=self._position,
            strategy=order.strategy_id,
        )

        # Emit fill event
        fill = FillEvent(
            order_id=order.order_id,
            symbol=order.symbol,
            direction="BUY" if order.direction == "Buy" else "SELL",
            fill_price=order.price,
            fill_size=order.qty,
            commission=COMMISSION_PER_SIDE,
            timestamp_ns=time.time_ns(),
        )
        await self._bus.publish(fill)

        return order.order_id

    async def _live_submit(self, order: ManagedOrder) -> str:
        """Submit a real order to Tradovate."""
        try:
            base = self._auth.base_url
            await self._ensure_session()

            payload = {
                "accountSpec": self._account_name,
                "accountId": self._account_id,
                "action": order.direction,
                "symbol": order.symbol,
                "orderQty": order.qty,
                "orderType": order.order_type,
                "timeInForce": "Day",
                "isAutomated": True,
            }

            if order.order_type == "Limit":
                payload["price"] = order.price

            async with self._session.post(
                f"{base}/order/placeorder", json=payload
            ) as resp:
                body = await resp.json()
                if resp.status != 200:
                    order.status = OrderStatus.REJECTED
                    logger.error(
                        "order_rejected",
                        order_id=order.order_id,
                        body=body,
                    )
                    return order.order_id

            tv_order_id = body.get("orderId") or body.get("id")
            order.tradovate_order_id = tv_order_id
            order.status = OrderStatus.WORKING

            logger.info(
                "order_placed",
                order_id=order.order_id,
                tv_order_id=tv_order_id,
                direction=order.direction,
                type=order.order_type,
                price=order.price,
            )

            # Place bracket (target + stop) if specified
            if order.target_price > 0 and order.stop_price > 0:
                await self._place_bracket(order)

            return order.order_id

        except Exception:
            order.status = OrderStatus.REJECTED
            logger.exception("order_submit_error", order_id=order.order_id)
            return order.order_id

    async def _place_bracket(self, parent: ManagedOrder) -> None:
        """Place OCO bracket (take-profit + stop-loss) for a filled entry."""
        base = self._auth.base_url
        exit_dir = "Sell" if parent.direction == "Buy" else "Buy"

        # Take profit (limit)
        tp_payload = {
            "accountSpec": self._account_name,
            "accountId": self._account_id,
            "action": exit_dir,
            "symbol": parent.symbol,
            "orderQty": parent.qty,
            "orderType": "Limit",
            "price": parent.target_price,
            "timeInForce": "Day",
            "isAutomated": True,
        }

        # Stop loss
        sl_payload = {
            "accountSpec": self._account_name,
            "accountId": self._account_id,
            "action": exit_dir,
            "symbol": parent.symbol,
            "orderQty": parent.qty,
            "orderType": "Stop",
            "stopPrice": parent.stop_price,
            "timeInForce": "Day",
            "isAutomated": True,
        }

        try:
            async with self._session.post(
                f"{base}/order/placeorder", json=tp_payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    parent.target_order_id = data.get("orderId") or data.get("id")
                    logger.info("bracket_tp_placed", price=parent.target_price)

            async with self._session.post(
                f"{base}/order/placeorder", json=sl_payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    parent.stop_order_id = data.get("orderId") or data.get("id")
                    logger.info("bracket_sl_placed", price=parent.stop_price)
        except Exception:
            logger.exception("bracket_placement_error")

    def get_order(self, order_id: str) -> ManagedOrder | None:
        return self._orders.get(order_id)

    @property
    def active_orders(self) -> list[ManagedOrder]:
        return [
            o for o in self._orders.values()
            if o.status in (OrderStatus.PENDING, OrderStatus.WORKING)
        ]
