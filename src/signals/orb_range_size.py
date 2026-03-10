"""ORB range size signal.

Compares the opening range size to recent ATR to determine if the range
is tradeable.  Too-small ranges produce false breakouts; too-large ranges
have poor risk/reward.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class ORBRangeSizeConfig:
    orb_minutes: int = 15
    atr_lookback: int = 14
    min_range_atr_ratio: float = 0.3  # range must be >= 30% of ATR
    max_range_atr_ratio: float = 1.5  # range must be <= 150% of ATR


@SignalRegistry.register
class ORBRangeSizeSignal(SignalBase):
    """Evaluates ORB range relative to ATR.

    value = range / ATR ratio.
    passes = True when ratio is within [min_range_atr_ratio, max_range_atr_ratio].
    direction = "none" (informational).
    """

    name = "orb_range_size"

    def __init__(self, config: ORBRangeSizeConfig | None = None) -> None:
        self.config = config or ORBRangeSizeConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        if len(bars) < self.config.atr_lookback + 1:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        # Find session_open_time
        session_open = None
        for b in bars:
            sot = getattr(b, "session_open_time", None)
            if sot is not None:
                session_open = sot
                break

        if session_open is None:
            from datetime import datetime
            session_open = datetime.utcfromtimestamp(bars[0].timestamp_ns / 1_000_000_000)

        orb_cutoff_ns = int(
            (session_open.timestamp() + self.config.orb_minutes * 60) * 1_000_000_000
        )

        # Collect ORB bars
        orb_highs = []
        orb_lows = []
        for b in bars:
            if b.timestamp_ns <= orb_cutoff_ns:
                orb_highs.append(b.high)
                orb_lows.append(b.low)

        if not orb_highs:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "no_orb_bars"},
            )

        orb_range = max(orb_highs) - min(orb_lows)

        # Compute Wilder ATR
        n = len(bars)
        period = self.config.atr_lookback
        atr = bars[0].high - bars[0].low
        for i in range(1, n):
            h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
            tr = max(h - l, abs(h - pc), abs(l - pc))
            if i < period:
                atr = (atr * i + tr) / (i + 1)
            else:
                atr = atr + (tr - atr) / period

        if atr < 1e-9:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "zero_atr"},
            )

        ratio = orb_range / atr

        passes = (
            self.config.min_range_atr_ratio <= ratio <= self.config.max_range_atr_ratio
        )

        return SignalResult(
            value=ratio,
            passes=passes,
            direction="none",
            metadata={
                "orb_range": orb_range,
                "atr": atr,
                "range_atr_ratio": ratio,
            },
        )
