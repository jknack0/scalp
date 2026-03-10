"""Stateless spread z-score signal.

Computes the bid-ask spread z-score from a list of BarEvent objects.
When the spread deviates beyond a configurable z-score threshold from the
rolling mean, the signal blocks (passes=False).  Non-directional.

Ported from src/filters/spread_monitor.py into the stateless signal framework.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class SpreadConfig:
    z_threshold: float = 2.0
    min_bars: int = 30


def _extract_spread(bar: BarEvent) -> float | None:
    """Extract spread from a bar, handling the naming inconsistency.

    TickAggregator stores average bid/ask *prices* in avg_bid_size / avg_ask_size.
    If avg_bid_price / avg_ask_price are available (>0), prefer those.
    Returns None when valid spread cannot be computed.
    """
    if bar.avg_bid_price > 0 and bar.avg_ask_price > 0:
        return bar.avg_ask_price - bar.avg_bid_price
    if bar.avg_bid_size > 0 and bar.avg_ask_size > 0:
        return bar.avg_ask_size - bar.avg_bid_size
    return None


@SignalRegistry.register
class SpreadSignal(SignalBase):
    """Spread z-score signal: blocks when spread is abnormally wide."""

    name = "spread"
    required_columns = frozenset({"avg_bid_price", "avg_ask_price"})

    def __init__(self, config: SpreadConfig | None = None) -> None:
        self.config = config or SpreadConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        """Compute spread z-score from the full bar history.

        Args:
            bars: List of BarEvent objects (oldest first). The last bar is the
                  current bar whose spread is evaluated.

        Returns:
            SignalResult with value=z_score, passes=True when spread is normal.
        """
        if not bars:
            return SignalResult(value=0.0, passes=True, direction="none", metadata={})

        # Extract valid spreads from all bars
        spreads: list[float] = []
        for bar in bars:
            s = _extract_spread(bar)
            if s is not None and s > 0:
                spreads.append(s)

        # Not enough data to judge -- let trades through
        if len(spreads) < self.config.min_bars:
            return SignalResult(
                value=0.0,
                passes=True,
                direction="none",
                metadata={
                    "reason": "insufficient_data",
                    "valid_bars": len(spreads),
                    "min_bars": self.config.min_bars,
                },
            )

        arr = np.array(spreads, dtype=np.float64)
        current_spread = arr[-1]
        rolling_mean = float(np.mean(arr))
        rolling_std = float(np.std(arr, ddof=1))

        if rolling_std > 0:
            z_score = (current_spread - rolling_mean) / rolling_std
        else:
            z_score = 0.0

        passes = abs(z_score) < self.config.z_threshold

        return SignalResult(
            value=z_score,
            passes=passes,
            direction="none",
            metadata={
                "current_spread": current_spread,
                "rolling_mean": rolling_mean,
                "rolling_std": rolling_std,
                "z_score": z_score,
                "z_threshold": self.config.z_threshold,
                "valid_bars": len(spreads),
            },
        )
