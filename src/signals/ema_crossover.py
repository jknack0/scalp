"""EMA crossover signal.

Detects when a fast EMA crosses above/below a slow EMA on bar closes.
Directional: long when fast > slow, short when fast < slow.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class EMACrossoverConfig:
    fast_period: int = 8
    slow_period: int = 21


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Compute exponential moving average."""
    alpha = 2.0 / (period + 1)
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


@SignalRegistry.register
class EMACrossoverSignal(SignalBase):
    """EMA crossover signal.

    value = fast_ema - slow_ema (positive = bullish).
    passes = True when a crossover occurred on the most recent bar.
    direction = "long" if fast crossed above slow, "short" if below.
    """

    name = "ema_crossover"

    def __init__(self, config: EMACrossoverConfig | None = None) -> None:
        self.config = config or EMACrossoverConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        min_bars = self.config.slow_period + 2  # need at least 2 bars after warmup
        if len(bars) < min_bars:
            return SignalResult(value=0.0, passes=False, direction="none",
                               metadata={"reason": "insufficient_bars"})

        closes = np.array([b.close for b in bars], dtype=np.float64)

        fast = _ema(closes, self.config.fast_period)
        slow = _ema(closes, self.config.slow_period)

        diff_now = float(fast[-1] - slow[-1])
        diff_prev = float(fast[-2] - slow[-2])

        # Crossover: sign change between prev and current
        crossed_up = diff_prev <= 0 and diff_now > 0
        crossed_down = diff_prev >= 0 and diff_now < 0

        passes = crossed_up or crossed_down
        if crossed_up:
            direction = "long"
        elif crossed_down:
            direction = "short"
        else:
            # No crossover, but report current alignment
            direction = "long" if diff_now > 0 else ("short" if diff_now < 0 else "none")

        return SignalResult(
            value=diff_now,
            passes=passes,
            direction=direction,
            metadata={
                "fast_ema": float(fast[-1]),
                "slow_ema": float(slow[-1]),
                "spread": diff_now,
                "crossed": "up" if crossed_up else ("down" if crossed_down else "none"),
            },
        )
