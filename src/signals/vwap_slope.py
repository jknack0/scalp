from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class VWAPSlopeConfig:
    lookback_bars: int = 10
    slope_threshold: float = 0.01


@SignalRegistry.register
class VWAPSlopeSignal(SignalBase):
    name = "vwap_slope"

    def __init__(self, config: VWAPSlopeConfig | None = None) -> None:
        self.config = config or VWAPSlopeConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        lookback = self.config.lookback_bars

        if len(bars) < lookback:
            return SignalResult(value=0.0, passes=True, direction="none", metadata={"slope": 0.0})

        tail = bars[-lookback:]

        vwap_now = getattr(tail[-1], "session_vwap", 0.0)
        vwap_ago = getattr(tail[0], "session_vwap", 0.0)

        highs = np.array([b.high for b in tail], dtype=np.float64)
        lows = np.array([b.low for b in tail], dtype=np.float64)
        bar_ranges = highs - lows
        mean_bar_range = float(np.mean(bar_ranges))

        # If mean bar range is zero or near-zero, slope is undefined -> treat as flat
        if mean_bar_range < 1e-12:
            return SignalResult(
                value=0.0,
                passes=True,
                direction="none",
                metadata={"slope": 0.0},
            )

        slope = (vwap_now - vwap_ago) / (lookback * mean_bar_range)

        passes = abs(slope) < self.config.slope_threshold

        return SignalResult(
            value=slope,
            passes=passes,
            direction="none",
            metadata={"slope": slope},
        )
