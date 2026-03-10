"""ADX (Average Directional Index) signal.

Measures trend strength regardless of direction.  High ADX confirms that
a directional move is genuine, not noise.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class ADXConfig:
    period: int = 14
    threshold: float = 25.0  # ADX above this = strong trend


def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing (equivalent to EMA with alpha = 1/period)."""
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = result[i - 1] + (values[i] - result[i - 1]) / period
    return result


@SignalRegistry.register
class ADXSignal(SignalBase):
    """ADX trend strength signal.

    value = current ADX reading.
    passes = True when ADX >= threshold (strong trend confirmed).
    direction = "long" if +DI > -DI, "short" if -DI > +DI.
    """

    name = "adx"

    def __init__(self, config: ADXConfig | None = None) -> None:
        self.config = config or ADXConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        period = self.config.period
        min_bars = period * 2 + 1  # need warmup for smoothing
        if len(bars) < min_bars:
            return SignalResult(value=0.0, passes=False, direction="none",
                               metadata={"reason": "insufficient_bars"})

        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)
        closes = np.array([b.close for b in bars], dtype=np.float64)

        n = len(bars)

        # True Range
        tr = np.empty(n - 1)
        for i in range(1, n):
            tr[i - 1] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        # Directional movement
        plus_dm = np.empty(n - 1)
        minus_dm = np.empty(n - 1)
        for i in range(1, n):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            plus_dm[i - 1] = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm[i - 1] = down_move if (down_move > up_move and down_move > 0) else 0.0

        # Wilder smooth
        atr = _wilder_smooth(tr, period)
        plus_di_smooth = _wilder_smooth(plus_dm, period)
        minus_di_smooth = _wilder_smooth(minus_dm, period)

        # +DI and -DI (avoid division by zero)
        plus_di = np.where(atr > 0, 100.0 * plus_di_smooth / atr, 0.0)
        minus_di = np.where(atr > 0, 100.0 * minus_di_smooth / atr, 0.0)

        # DX
        di_sum = plus_di + minus_di
        dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)

        # ADX = Wilder smooth of DX
        adx = _wilder_smooth(dx, period)

        current_adx = float(adx[-1])
        current_plus_di = float(plus_di[-1])
        current_minus_di = float(minus_di[-1])

        passes = current_adx >= self.config.threshold

        if current_plus_di > current_minus_di:
            direction = "long"
        elif current_minus_di > current_plus_di:
            direction = "short"
        else:
            direction = "none"

        return SignalResult(
            value=current_adx,
            passes=passes,
            direction=direction,
            metadata={
                "adx": current_adx,
                "plus_di": current_plus_di,
                "minus_di": current_minus_di,
            },
        )
