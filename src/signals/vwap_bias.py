"""VWAP bias confirmation signal.

Checks whether price is consistently on one side of VWAP, confirming
directional bias.  Uses session_vwap from DollarBar.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class VWAPBiasConfig:
    lookback_bars: int = 10
    consistency_threshold: float = 0.7  # fraction of bars on same side


@SignalRegistry.register
class VWAPBiasSignal(SignalBase):
    """Directional bias from price-vs-VWAP consistency.

    value = fraction of lookback bars where close is on the dominant side.
    passes = True when consistency >= threshold.
    direction = "long" if mostly above VWAP, "short" if mostly below.
    """

    name = "vwap_bias"

    def __init__(self, config: VWAPBiasConfig | None = None) -> None:
        self.config = config or VWAPBiasConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        lookback = self.config.lookback_bars

        if len(bars) < lookback:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        tail = bars[-lookback:]
        closes = np.array([b.close for b in tail], dtype=np.float64)
        vwaps = np.array(
            [getattr(b, "session_vwap", 0.0) for b in tail], dtype=np.float64
        )

        # If no VWAP data, can't compute
        if np.all(vwaps == 0.0):
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "no_vwap_data"},
            )

        above = np.sum(closes > vwaps)
        below = np.sum(closes < vwaps)
        total = len(tail)

        above_frac = float(above / total)
        below_frac = float(below / total)

        if above_frac >= below_frac:
            consistency = above_frac
            direction = "long"
        else:
            consistency = below_frac
            direction = "short"

        passes = consistency >= self.config.consistency_threshold

        return SignalResult(
            value=consistency,
            passes=passes,
            direction=direction if passes else "none",
            metadata={
                "above_frac": above_frac,
                "below_frac": below_frac,
                "consistency": consistency,
            },
        )
