"""Keltner Channel signal.

Computes EMA-based mid line with ATR-based upper/lower bands.
Used for TTM Squeeze detection (Bollinger inside Keltner = squeeze).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class KeltnerChannelConfig:
    period: int = 20
    atr_period: int = 20
    multiplier: float = 1.5


@SignalRegistry.register
class KeltnerChannelSignal(SignalBase):
    """Keltner Channel signal.

    value = channel width (upper - lower).
    passes = True always (informational signal).
    direction = "none".
    metadata includes upper, lower, mid, width.
    """

    name = "keltner_channel"

    def __init__(self, config: KeltnerChannelConfig | None = None) -> None:
        self.config = config or KeltnerChannelConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        period = self.config.period
        atr_period = self.config.atr_period
        # Need enough bars for both EMA and ATR (ATR needs atr_period + 1)
        min_bars = max(period, atr_period + 1)

        if len(bars) < min_bars:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        closes = np.array([b.close for b in bars], dtype=np.float64)
        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)

        # EMA of close over period
        mid = self._ema(closes, period)

        # Wilder-smoothed ATR
        atr_val = self._wilder_atr(highs, lows, closes, atr_period)

        upper = mid + self.config.multiplier * atr_val
        lower = mid - self.config.multiplier * atr_val
        width = upper - lower

        return SignalResult(
            value=float(width),
            passes=True,
            direction="none",
            metadata={
                "upper": float(upper),
                "lower": float(lower),
                "mid": float(mid),
                "width": float(width),
            },
        )

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Compute EMA and return the last value."""
        alpha = 2.0 / (period + 1)
        ema = data[0]
        for i in range(1, len(data)):
            ema = alpha * data[i] + (1.0 - alpha) * ema
        return float(ema)

    @staticmethod
    def _wilder_atr(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
    ) -> float:
        """Compute Wilder-smoothed ATR and return the last value."""
        # True Range series
        n = len(highs)
        tr = np.empty(n - 1, dtype=np.float64)
        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i - 1] = max(hl, hc, lc)

        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0

        # Initial ATR = SMA of first `period` TRs
        atr = float(np.mean(tr[:period]))

        # Wilder smoothing for remaining TRs
        for i in range(period, len(tr)):
            atr = (atr * (period - 1) + tr[i]) / period

        return atr
