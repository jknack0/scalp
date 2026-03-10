from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Literal
from src.core.events import BarEvent

# BarEvent fields that are always present in any bar
OHLCV_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "timestamp_ns"})


@dataclass(frozen=True)
class SignalResult:
    value: float
    passes: bool
    direction: Literal["long", "short", "none"] = "none"
    metadata: dict = field(default_factory=dict)


# Sentinel result for signals that cannot compute due to missing data
UNAVAILABLE_RESULT = SignalResult(
    value=0.0,
    passes=True,
    direction="none",
    metadata={"unavailable": True, "reason": "missing_data"},
)


class SignalBase(ABC):
    name: str

    # BarEvent fields required beyond basic OHLCV.
    # Signals that only need OHLCV leave this empty.
    # Override in subclass to declare L1/enriched field requirements.
    required_columns: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def compute(self, bars: list[BarEvent]) -> SignalResult:
        ...

    def has_required_data(self, bars: list[BarEvent]) -> bool:
        """Check if bars contain the data this signal needs.

        Checks the last bar for non-zero values on required columns.
        """
        if not self.required_columns or not bars:
            return True
        bar = bars[-1]
        for col in self.required_columns:
            val = getattr(bar, col, 0.0)
            if val == 0.0:
                return False
        return True
