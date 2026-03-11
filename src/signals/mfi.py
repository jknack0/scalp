"""Money Flow Index (MFI) signal.

Oscillator (0-100) combining price and volume to identify overbought/oversold
conditions. MFI < 20 = oversold (long), MFI > 80 = overbought (short).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class MFIConfig:
    period: int = 14


@SignalRegistry.register
class MFISignal(SignalBase):
    """Money Flow Index signal.

    value = MFI (0-100).
    passes = True when MFI < 20 (oversold) or MFI > 80 (overbought).
    direction = "long" if < 20, "short" if > 80.
    metadata includes mfi_value, positive_flow, negative_flow.
    """

    name = "mfi"

    def __init__(self, config: MFIConfig | None = None) -> None:
        self.config = config or MFIConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        period = self.config.period
        # Need period + 1 bars to compute period changes
        if len(bars) < period + 1:
            return SignalResult(
                value=50.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)
        closes = np.array([b.close for b in bars], dtype=np.float64)
        volumes = np.array([b.volume for b in bars], dtype=np.float64)

        # Typical price for each bar
        typical = (highs + lows + closes) / 3.0

        # Raw money flow
        raw_mf = typical * volumes

        # Compare typical price to previous typical price over last period+1 bars
        tp_window = typical[-(period + 1):]
        mf_window = raw_mf[-(period + 1):]

        positive_flow = 0.0
        negative_flow = 0.0
        for i in range(1, period + 1):
            if tp_window[i] > tp_window[i - 1]:
                positive_flow += mf_window[i]
            elif tp_window[i] < tp_window[i - 1]:
                negative_flow += mf_window[i]

        money_ratio = positive_flow / max(negative_flow, 1e-10)
        mfi = 100.0 - (100.0 / (1.0 + money_ratio))

        # Direction and passes
        if mfi < 20.0:
            direction = "long"
            passes = True
        elif mfi > 80.0:
            direction = "short"
            passes = True
        else:
            direction = "none"
            passes = False

        return SignalResult(
            value=float(mfi),
            passes=passes,
            direction=direction,
            metadata={
                "mfi_value": float(mfi),
                "positive_flow": float(positive_flow),
                "negative_flow": float(negative_flow),
            },
        )
