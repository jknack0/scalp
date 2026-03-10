"""Prior day VWAP gap bias signal.

Measures the gap between current price and prior day's VWAP to establish
overnight directional bias.  Uses prior_day_vwap from DollarBar.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


MES_TICK_SIZE = 0.25


@dataclass(frozen=True)
class PriorDayBiasConfig:
    min_gap_ticks: int = 2  # minimum gap to establish bias
    lookback_bars: int = 5  # bars to confirm gap direction holds


@SignalRegistry.register
class PriorDayBiasSignal(SignalBase):
    """Prior day VWAP gap bias.

    value = gap in ticks (positive = above prior VWAP, negative = below).
    passes = True when |gap| >= min_gap_ticks and recent bars confirm.
    direction = "long" if above prior VWAP, "short" if below.
    """

    name = "prior_day_bias"

    def __init__(self, config: PriorDayBiasConfig | None = None) -> None:
        self.config = config or PriorDayBiasConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        lookback = self.config.lookback_bars

        if len(bars) < lookback:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        tail = bars[-lookback:]

        # Get prior_day_vwap from DollarBar
        prior_vwaps = np.array(
            [getattr(b, "prior_day_vwap", 0.0) for b in tail], dtype=np.float64
        )

        # Use the most recent non-zero prior_day_vwap
        prior_vwap = 0.0
        for v in reversed(prior_vwaps):
            if v > 0.0:
                prior_vwap = float(v)
                break

        if prior_vwap == 0.0:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "no_prior_day_vwap"},
            )

        current_close = tail[-1].close
        gap_ticks = (current_close - prior_vwap) / MES_TICK_SIZE

        # Confirm: check if majority of recent bars are on the same side
        closes = np.array([b.close for b in tail], dtype=np.float64)
        if gap_ticks > 0:
            confirmed = float(np.mean(closes > prior_vwap)) >= 0.6
        elif gap_ticks < 0:
            confirmed = float(np.mean(closes < prior_vwap)) >= 0.6
        else:
            confirmed = False

        passes = abs(gap_ticks) >= self.config.min_gap_ticks and confirmed

        if gap_ticks > 0 and passes:
            direction = "long"
        elif gap_ticks < 0 and passes:
            direction = "short"
        else:
            direction = "none"

        return SignalResult(
            value=gap_ticks,
            passes=passes,
            direction=direction,
            metadata={
                "prior_day_vwap": prior_vwap,
                "gap_ticks": gap_ticks,
                "confirmed": confirmed,
            },
        )
