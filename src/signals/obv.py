"""On-Balance Volume (OBV) signal.

Running cumulative volume sum weighted by price direction. The slope of OBV
over a lookback window reveals accumulation (rising) or distribution (falling).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class OBVConfig:
    lookback: int = 10


@SignalRegistry.register
class OBVSignal(SignalBase):
    """On-Balance Volume signal.

    value = linear regression slope of OBV over lookback bars.
    passes = True when enough bars are available (abs(slope) > 0).
    direction = "long" if slope > 0, "short" if slope < 0.
    metadata includes obv_value (last OBV) and obv_slope.
    """

    name = "obv"

    def __init__(self, config: OBVConfig | None = None) -> None:
        self.config = config or OBVConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        lookback = self.config.lookback
        # Need at least lookback + 1 bars (1 for initial OBV, lookback for slope)
        if len(bars) < lookback + 1:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        closes = np.array([b.close for b in bars], dtype=np.float64)
        volumes = np.array([b.volume for b in bars], dtype=np.float64)

        # Build OBV series for the full bar window
        obv = np.zeros(len(bars), dtype=np.float64)
        for i in range(1, len(bars)):
            if closes[i] > closes[i - 1]:
                obv[i] = obv[i - 1] + volumes[i]
            elif closes[i] < closes[i - 1]:
                obv[i] = obv[i - 1] - volumes[i]
            else:
                obv[i] = obv[i - 1]

        # Linear regression slope of OBV over last lookback bars
        obv_window = obv[-lookback:]
        x = np.arange(lookback, dtype=np.float64)
        x_mean = x.mean()
        obv_mean = obv_window.mean()

        numerator = float(np.sum((x - x_mean) * (obv_window - obv_mean)))
        denominator = float(np.sum((x - x_mean) ** 2))

        if denominator < 1e-10:
            slope = 0.0
        else:
            slope = numerator / denominator

        obv_value = float(obv[-1])

        if slope > 0:
            direction = "long"
        elif slope < 0:
            direction = "short"
        else:
            direction = "none"

        passes = abs(slope) > 0

        return SignalResult(
            value=float(slope),
            passes=passes,
            direction=direction,
            metadata={
                "obv_value": obv_value,
                "obv_slope": float(slope),
            },
        )
