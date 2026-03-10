from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class VWAPDeviationConfig:
    lookback_bars: int = 20
    std_threshold: float = 1.5


@SignalRegistry.register
class VWAPDeviationSignal(SignalBase):
    name = "vwap_deviation"

    def __init__(self, config: VWAPDeviationConfig | None = None) -> None:
        self.config = config or VWAPDeviationConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        lookback = self.config.lookback_bars
        threshold = self.config.std_threshold

        if len(bars) < lookback:
            return SignalResult(value=0.0, passes=False, direction="none", metadata={})

        # Extract session_vwap from DollarBar instances (safe fallback to 0.0)
        tail = bars[-lookback:]
        vwaps = np.array(
            [getattr(b, "session_vwap", 0.0) for b in tail], dtype=np.float64
        )

        # If all vwaps are zero, DollarBar fields are unavailable
        if np.all(vwaps == 0.0):
            return SignalResult(value=0.0, passes=False, direction="none", metadata={})

        closes = np.array([b.close for b in tail], dtype=np.float64)

        # Deviation of close from precomputed session VWAP at each bar
        deviations = closes - vwaps

        # Rolling std dev of deviations over the lookback window
        std_dev = float(np.std(deviations, ddof=1))

        current_close = float(closes[-1])
        current_vwap = float(vwaps[-1])
        current_deviation = current_close - current_vwap

        if std_dev > 0.0:
            deviation_sigmas = current_deviation / std_dev
        else:
            deviation_sigmas = 0.0

        passes = abs(deviation_sigmas) >= threshold

        if current_deviation < 0.0 and passes:
            direction: str = "long"
        elif current_deviation > 0.0 and passes:
            direction = "short"
        else:
            direction = "none"

        return SignalResult(
            value=deviation_sigmas,
            passes=passes,
            direction=direction,
            metadata={
                "vwap": current_vwap,
                "deviation_sigmas": deviation_sigmas,
                "std_dev": std_dev,
            },
        )
