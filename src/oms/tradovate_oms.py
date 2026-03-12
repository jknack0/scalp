"""Tradovate order management system.

Places, monitors, and cancels orders via Tradovate REST API.
Supports both live and paper trading modes.

Paper mode mirrors the backtest SimulatedOMS bracket model:
  Signal → PendingOrder (limit entry wait) → target/stop/expiry exits on tick
This ensures paper results match backtest assumptions.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import aiohttp

from src.core.config import BotConfig
from src.core.events import EventBus, EventType, FillEvent, TickEvent
from src.core.logging import get_logger
from src.data.trade_store import TradeStore
from src.feeds.tradovate import TradovateAuth
from src.oms.base import BaseOMS
from src.strategies.base import Direction, Signal

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
    expiry_time: datetime | None = None


class TradovateOMS(BaseOMS):
    """Tradovate REST-based order management.

    Supports two modes:
    - paper=True: bracket order simulation (limit entry, target/stop/expiry exits)
    - paper=False: submits real orders to Tradovate demo/live

    Paper mode works identically to BacktestEngine's SimulatedOMS:
    Signal → pending limit entry → tick fills entry → tick monitors target/stop/expiry.
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

        # Trade persistence (set externally via set_trade_store)
        self._trade_store: TradeStore | None = None

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

        Paper mode: queues a pending limit entry (filled on tick).
        Live mode: submits market order + bracket to Tradovate.
        Returns a local order_id string.
        """
        if isinstance(signal, Signal):
            direction = "Buy" if signal.direction == Direction.LONG else "Sell"
            price = signal.entry_price
            target = signal.target_price
            stop = signal.stop_price
            strategy_id = signal.strategy_id
            expiry = signal.expiry_time
        else:
            # Legacy SignalEvent
            direction = signal.direction
            price = getattr(signal, "price", 0.0)
            target = 0.0
            stop = 0.0
            strategy_id = signal.strategy_id
            expiry = None

        order_id = self._gen_id(strategy_id)
        order = ManagedOrder(
            order_id=order_id,
            strategy_id=strategy_id,
            symbol=self._config.symbol,
            direction=direction,
            qty=1,
            order_type="Limit" if self._paper else "Market",
            price=price,
            target_price=target,
            stop_price=stop,
            expiry_time=expiry,
        )
        self._orders[order_id] = order

        if self._paper:
            # Don't fill immediately — wait for tick to reach entry price
            order.status = OrderStatus.WORKING
            logger.info(
                "paper_order_pending",
                order_id=order.order_id,
                direction=order.direction,
                entry=order.price,
                target=order.target_price,
                stop=order.stop_price,
                expiry=str(order.expiry_time),
                strategy=order.strategy_id,
            )
            return order_id

        return await self._live_submit(order)

    async def on_tick(self, tick: TickEvent) -> None:
        """Process a tick for paper bracket orders.

        Mirrors SimulatedOMS behavior:
        1. Pending entries: fill if tick reaches entry price (limit)
        2. Open positions: exit if tick hits target/stop/expiry
        """
        if not self._paper:
            return

        price = tick.last_price
        if price <= 0:
            return

        now = datetime.fromtimestamp(tick.timestamp_ns / 1e9)
        closed_ids: list[str] = []

        for oid, order in self._orders.items():
            # --- Pending entry: check for limit fill ---
            if order.status == OrderStatus.WORKING and order.fill_price == 0.0:
                # Check expiry before fill
                if order.expiry_time and now >= order.expiry_time:
                    order.status = OrderStatus.EXPIRED
                    closed_ids.append(oid)
                    logger.info(
                        "paper_entry_expired",
                        order_id=oid,
                        strategy=order.strategy_id,
                    )
                    continue

                # Limit entry: fill when price reaches entry level
                filled = False
                if order.direction == "Buy" and price <= order.price:
                    filled = True
                elif order.direction == "Sell" and price >= order.price:
                    filled = True

                if filled:
                    order.fill_price = order.price  # Limit fill, no slippage
                    order.status = OrderStatus.FILLED
                    delta = order.qty if order.direction == "Buy" else -order.qty
                    self._position += delta

                    logger.info(
                        "paper_entry_fill",
                        order_id=oid,
                        direction=order.direction,
                        fill_price=order.fill_price,
                        target=order.target_price,
                        stop=order.stop_price,
                        position=self._position,
                        strategy=order.strategy_id,
                    )

                    fill = FillEvent(
                        order_id=oid,
                        symbol=order.symbol,
                        direction="BUY" if order.direction == "Buy" else "SELL",
                        fill_price=order.fill_price,
                        fill_size=order.qty,
                        commission=COMMISSION_PER_SIDE,
                        timestamp_ns=tick.timestamp_ns,
                    )
                    await self._bus.publish(fill)

            # --- Open position: check target/stop/expiry ---
            elif order.status == OrderStatus.FILLED and order.target_price > 0:
                hit = False
                exit_price = 0.0
                exit_reason = ""

                if order.direction == "Buy":
                    if price >= order.target_price:
                        exit_price = order.target_price
                        exit_reason = "target"
                        hit = True
                    elif price <= order.stop_price:
                        exit_price = order.stop_price
                        exit_reason = "stop"
                        hit = True
                else:  # Sell
                    if price <= order.target_price:
                        exit_price = order.target_price
                        exit_reason = "target"
                        hit = True
                    elif price >= order.stop_price:
                        exit_price = order.stop_price
                        exit_reason = "stop"
                        hit = True

                # Check expiry
                if not hit and order.expiry_time and now >= order.expiry_time:
                    exit_price = price  # Market exit at current price
                    exit_reason = "expiry"
                    hit = True

                if hit:
                    # Close position
                    delta = -order.qty if order.direction == "Buy" else order.qty
                    self._position += delta
                    order.status = OrderStatus.CANCELLED  # Mark as done

                    pnl = (exit_price - order.fill_price) if order.direction == "Buy" else (order.fill_price - exit_price)
                    pnl_usd = pnl / TICK_SIZE * TICK_VALUE

                    logger.info(
                        "paper_exit",
                        order_id=oid,
                        strategy=order.strategy_id,
                        direction=order.direction,
                        entry=order.fill_price,
                        exit=exit_price,
                        reason=exit_reason,
                        pnl_ticks=pnl / TICK_SIZE,
                        pnl_usd=round(pnl_usd, 2),
                    )

                    # Record exit to Postgres
                    if self._trade_store:
                        from datetime import datetime, timezone
                        exit_dt = datetime.fromtimestamp(
                            tick.timestamp_ns / 1e9, tz=timezone.utc,
                        )
                        entry_dt = datetime.fromtimestamp(
                            order.created_at, tz=timezone.utc,
                        )
                        duration = (exit_dt - entry_dt).total_seconds()
                        self._trade_store.record_exit(
                            order_id=oid,
                            exit_price=exit_price,
                            exit_time=exit_dt,
                            exit_reason=exit_reason,
                            pnl_ticks=pnl / TICK_SIZE,
                            pnl_usd=round(pnl_usd - COMMISSION_PER_SIDE * 2, 2),
                            commission=COMMISSION_PER_SIDE * 2,
                            duration_seconds=duration,
                        )

                    exit_dir = "SELL" if order.direction == "Buy" else "BUY"
                    fill = FillEvent(
                        order_id=f"{oid}-exit",
                        symbol=order.symbol,
                        direction=exit_dir,
                        fill_price=exit_price,
                        fill_size=order.qty,
                        commission=COMMISSION_PER_SIDE,
                        timestamp_ns=tick.timestamp_ns,
                    )
                    await self._bus.publish(fill)
                    closed_ids.append(oid)

        # Clean up closed orders
        for oid in closed_ids:
            self._orders.pop(oid, None)

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
            # Close all open paper positions
            for oid, order in list(self._orders.items()):
                if order.status == OrderStatus.FILLED:
                    order.status = OrderStatus.CANCELLED
                    self._orders.pop(oid, None)
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

    @property
    def pending_entry_count(self) -> int:
        """Number of pending entry orders (not yet filled)."""
        return sum(
            1 for o in self._orders.values()
            if o.status == OrderStatus.WORKING and o.fill_price == 0.0
        )

    @property
    def open_position_count(self) -> int:
        """Number of open bracket positions being monitored."""
        return sum(
            1 for o in self._orders.values()
            if o.status == OrderStatus.FILLED and o.target_price > 0
        )

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
            if o.status in (OrderStatus.PENDING, OrderStatus.WORKING, OrderStatus.FILLED)
        ]
