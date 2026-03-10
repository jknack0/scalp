"""Relative volume signal.

Compares current bar volume to the rolling average to detect abnormal
activity.  High relative volume confirms breakouts; low relative volume
suggests false moves.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class RelativeVolumeConfig:
    lookback_bars: int = 20
    high_threshold: float = 1.5  # rvol above this = high relative volume
    low_threshold: float = 0.5   # rvol below this = low relative volume


@SignalRegistry.register
class RelativeVolumeSignal(SignalBase):
    """Relative volume (RVOL) signal.

    value = current_volume / mean(prior volumes).
    passes = True when RVOL >= high_threshold (confirms move).
    direction = "none" (informational — direction comes from other signals).
    """

    name = "relative_volume"

    def __init__(self, config: RelativeVolumeConfig | None = None) -> None:
        self.config = config or RelativeVolumeConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        lookback = self.config.lookback_bars

        if len(bars) < lookback:
            return SignalResult(
                value=1.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        tail = bars[-lookback:]
        volumes = np.array([b.volume for b in tail], dtype=np.float64)

        # Mean of prior bars (exclude current)
        prior_mean = float(np.mean(volumes[:-1]))

        if prior_mean < 1e-9:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "zero_mean_volume"},
            )

        current_vol = float(volumes[-1])
        rvol = current_vol / prior_mean

        passes = rvol >= self.config.high_threshold

        return SignalResult(
            value=rvol,
            passes=passes,
            direction="none",
            metadata={
                "rvol": rvol,
                "current_vol": current_vol,
                "prior_mean": prior_mean,
                "regime": "high" if rvol >= self.config.high_threshold
                    else ("low" if rvol <= self.config.low_threshold else "normal"),
            },
        )
