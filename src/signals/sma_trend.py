"""SMA trend alignment signal.

Reports whether price is above or below a long-period SMA,
used as a trend filter (only take longs above SMA, shorts below).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class SMATrendConfig:
    period: int = 200


@SignalRegistry.register
class SMATrendSignal(SignalBase):
    """SMA trend alignment signal.

    value = (price - sma) / sma * 100 (percent deviation).
    passes = True always (informational).
    direction = "long" if price > SMA, "short" if below.
    """

    name = "sma_trend"

    def __init__(self, config: SMATrendConfig | None = None) -> None:
        self.config = config or SMATrendConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        period = self.config.period
        if len(bars) < period:
            return SignalResult(
                value=0.0, passes=True, direction="none",
                metadata={"reason": "insufficient_bars", "sma": 0.0},
            )

        closes = np.array([b.close for b in bars], dtype=np.float64)
        sma = float(np.mean(closes[-period:]))
        price = closes[-1]

        pct_dev = (price - sma) / sma * 100 if sma > 0 else 0.0

        direction = "long" if price > sma else ("short" if price < sma else "none")

        return SignalResult(
            value=pct_dev,
            passes=True,
            direction=direction,
            metadata={"sma": sma, "price": price, "pct_deviation": pct_dev},
        )
