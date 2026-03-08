"""Abstract base class for order management."""

from abc import ABC, abstractmethod

from src.core.events import EventBus, SignalEvent


class BaseOMS(ABC):
    """Interface for order management system adapters.

    Subclasses: TradovateOMS (Phase 1 Task 3), PaperOMS (backtesting).
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus

    @abstractmethod
    async def submit_order(self, signal: SignalEvent) -> str:
        """Submit an order based on a signal. Returns order_id."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if successful."""

    @abstractmethod
    async def get_position(self, symbol: str) -> int:
        """Get current position (signed: +long, -short)."""
