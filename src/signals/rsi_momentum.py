"""RSI momentum signal.

Confirms directional momentum: RSI above 50 for longs, below 50 for shorts.
Unlike classic overbought/oversold usage, this treats RSI as a momentum
confirmation filter.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class RSIMomentumConfig:
    period: int = 14
    long_threshold: float = 50.0   # RSI above this for long confirmation
    short_threshold: float = 50.0  # RSI below this for short confirmation


@SignalRegistry.register
class RSIMomentumSignal(SignalBase):
    """RSI momentum confirmation signal.

    value = current RSI reading (0-100).
    passes = True when RSI confirms directional momentum.
    direction = "long" if RSI > long_threshold, "short" if RSI < short_threshold.
    """

    name = "rsi_momentum"

    def __init__(self, config: RSIMomentumConfig | None = None) -> None:
        self.config = config or RSIMomentumConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        period = self.config.period
        min_bars = period + 2
        if len(bars) < min_bars:
            return SignalResult(value=50.0, passes=False, direction="none",
                               metadata={"reason": "insufficient_bars"})

        closes = np.array([b.close for b in bars], dtype=np.float64)
        deltas = np.diff(closes)

        # Wilder-smoothed average gains and losses
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Initial averages (SMA of first `period` values)
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))

        # Wilder smoothing for remaining values
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss < 1e-10:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)

        if rsi > self.config.long_threshold:
            direction = "long"
            passes = True
        elif rsi < self.config.short_threshold:
            direction = "short"
            passes = True
        else:
            direction = "none"
            passes = False

        return SignalResult(
            value=rsi,
            passes=passes,
            direction=direction,
            metadata={
                "rsi": rsi,
                "avg_gain": avg_gain,
                "avg_loss": avg_loss,
            },
        )
