"""EMA Ribbon signal.

Computes 5 EMAs (8, 13, 21, 34, 55) and detects fanned/trending state.
Reports proximity to the ribbon for pullback entry detection.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry

MES_TICK_SIZE = 0.25


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA over an array using the standard multiplier."""
    result = np.empty_like(values)
    alpha = 2.0 / (period + 1)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result


@dataclass(frozen=True)
class EmaRibbonConfig:
    periods: list[int] = field(default_factory=lambda: [8, 13, 21, 34, 55])
    atr_proximity: float = 1.0
    atr_period: int = 14


@SignalRegistry.register
class EmaRibbonSignal(SignalBase):
    """EMA Ribbon signal.

    value = distance from ema_8 (positive = above, negative = below).
    passes = True when ribbon is fanned AND price is within atr_proximity * ATR
             of ema_8 or ema_13 (pullback zone).
    direction = "long" if trend=="up", "short" if trend=="down", "none" if flat.
    metadata includes all 5 EMA values, fanned state, and trend label.
    """

    name = "ema_ribbon"

    def __init__(self, config: EmaRibbonConfig | None = None) -> None:
        self.config = config or EmaRibbonConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        max_period = max(self.config.periods)
        # Need enough bars for the longest EMA to stabilize
        min_bars = max_period * 2
        if len(bars) < min_bars:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        closes = np.array([b.close for b in bars], dtype=np.float64)
        price = float(closes[-1])

        # Compute all EMAs
        emas: dict[int, float] = {}
        for period in self.config.periods:
            ema_series = _ema(closes, period)
            emas[period] = float(ema_series[-1])

        ema_8 = emas[self.config.periods[0]]
        ema_13 = emas[self.config.periods[1]]
        ema_21 = emas[self.config.periods[2]]
        ema_34 = emas[self.config.periods[3]]
        ema_55 = emas[self.config.periods[4]]

        # Fanned detection: all 5 EMAs in strict order
        fanned_up = ema_8 > ema_13 > ema_21 > ema_34 > ema_55
        fanned_down = ema_8 < ema_13 < ema_21 < ema_34 < ema_55
        fanned = fanned_up or fanned_down

        # Trend detection: top 3 EMAs in order
        if ema_8 > ema_13 > ema_21:
            trend = "up"
        elif ema_8 < ema_13 < ema_21:
            trend = "down"
        else:
            trend = "flat"

        # Compute ATR(14) for proximity threshold
        atr = self._compute_atr(bars)

        # Distance from ema_8
        distance_from_ema8 = price - ema_8

        # Proximity check: price within atr_proximity * ATR of ema_8 or ema_13
        proximity_threshold = self.config.atr_proximity * atr if atr > 0 else 0.0
        near_ema8 = abs(price - ema_8) <= proximity_threshold
        near_ema13 = abs(price - ema_13) <= proximity_threshold
        near_ribbon = near_ema8 or near_ema13

        passes = fanned and near_ribbon

        if trend == "up":
            direction = "long"
        elif trend == "down":
            direction = "short"
        else:
            direction = "none"

        return SignalResult(
            value=distance_from_ema8,
            passes=passes,
            direction=direction,
            metadata={
                "ema_8": ema_8,
                "ema_13": ema_13,
                "ema_21": ema_21,
                "ema_34": ema_34,
                "ema_55": ema_55,
                "fanned": fanned,
                "trend": trend,
                "atr": atr,
            },
        )

    def _compute_atr(self, bars: list[BarEvent]) -> float:
        """Compute Wilder-smoothed ATR(14) from bar history."""
        n = len(bars)
        if n < 2:
            return 0.0

        true_ranges = np.empty(n, dtype=np.float64)
        true_ranges[0] = bars[0].high - bars[0].low

        for i in range(1, n):
            h = bars[i].high
            low = bars[i].low
            pc = bars[i - 1].close
            true_ranges[i] = max(h - low, abs(h - pc), abs(low - pc))

        period = self.config.atr_period
        atr = true_ranges[0]
        for i in range(1, n):
            if i < period:
                atr = (atr * i + true_ranges[i]) / (i + 1)
            else:
                atr = atr + (true_ranges[i] - atr) / period

        return float(atr)
