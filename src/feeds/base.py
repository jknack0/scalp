"""Abstract base class for market data feeds."""

from abc import ABC, abstractmethod

from src.core.events import EventBus


class BaseFeed(ABC):
    """Interface that all feed adapters must implement.

    Subclasses: TradovateFeed (Phase 1 Task 3), ReplayFeed (backtesting).
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""

    @abstractmethod
    async def subscribe(self, symbol: str) -> None:
        """Subscribe to market data for a symbol."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up connection."""

    @abstractmethod
    async def run(self) -> None:
        """Main loop: read data, publish TickEvent/BarEvent to bus."""
