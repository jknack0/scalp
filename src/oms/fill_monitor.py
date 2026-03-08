"""Fill monitor: polls Tradovate for order status updates.

In paper mode this is a no-op — the paper fills are handled
synchronously in TradovateOMS and SignalHandler.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from src.core.events import EventBus, FillEvent
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.oms.tradovate_oms import ManagedOrder, TradovateOMS

logger = get_logger("fill_monitor")

_POLL_INTERVAL = 2.0  # seconds between order status checks


class FillMonitor:
    """Polls Tradovate order status and emits FillEvents.

    Only active in live mode — paper fills are instant.
    """

    def __init__(
        self,
        event_bus: EventBus,
        oms: TradovateOMS,
    ) -> None:
        self._bus = event_bus
        self._oms = oms
        self._running = False

    async def run(self) -> None:
        """Poll loop: check working orders for fills."""
        if self._oms.is_paper:
            logger.info("fill_monitor_skipped", reason="paper mode")
            # Just sleep forever — paper fills handled elsewhere
            self._running = True
            while self._running:
                await asyncio.sleep(60)
            return

        self._running = True
        logger.info("fill_monitor_started")

        while self._running:
            try:
                await self._check_orders()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("fill_monitor_error")
            await asyncio.sleep(_POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False

    async def _check_orders(self) -> None:
        """Check status of all working orders."""
        from src.oms.tradovate_oms import OrderStatus

        active = self._oms.active_orders
        if not active:
            return

        for order in active:
            if order.tradovate_order_id is None:
                continue

            try:
                base = self._oms._auth.base_url
                await self._oms._ensure_session()
                async with self._oms._session.get(
                    f"{base}/order/item",
                    params={"id": order.tradovate_order_id},
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()

                status = data.get("ordStatus", "Unknown")

                if status == "Filled" and order.status != OrderStatus.FILLED:
                    order.status = OrderStatus.FILLED
                    order.fill_price = data.get("avgPx", order.price)

                    fill = FillEvent(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        direction="BUY" if order.direction == "Buy" else "SELL",
                        fill_price=order.fill_price,
                        fill_size=order.qty,
                        commission=0.35,
                        timestamp_ns=time.time_ns(),
                    )
                    await self._bus.publish(fill)

                    logger.info(
                        "live_fill",
                        order_id=order.order_id,
                        price=order.fill_price,
                        direction=order.direction,
                    )

                elif status in ("Cancelled", "Rejected", "Expired"):
                    order.status = OrderStatus(status)
                    logger.info(
                        "order_terminal",
                        order_id=order.order_id,
                        status=status,
                    )

            except Exception:
                logger.exception(
                    "order_check_error",
                    order_id=order.order_id,
                )
