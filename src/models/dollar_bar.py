"""Dollar bar data model.

DollarBar extends BarEvent with fields provided by the Databento ingestion
layer: session VWAP, prior-day VWAP, buy/sell volume split, and session
open time.  Frozen and slotted for performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.core.events import BarEvent


@dataclass(frozen=True, slots=True)
class DollarBar(BarEvent):
    """Pre-built dollar bar with Databento-enriched fields.

    Inherits all BarEvent fields (symbol, OHLCV, timestamp_ns, L1 enrichment)
    and adds session-level context computed upstream.
    """

    session_vwap: float = 0.0
    prior_day_vwap: float = 0.0
    buy_volume: int = 0
    sell_volume: int = 0
    session_open_time: datetime | None = None

    @property
    def timestamp(self) -> datetime:
        """UTC datetime from timestamp_ns."""
        return datetime.utcfromtimestamp(self.timestamp_ns / 1_000_000_000)
