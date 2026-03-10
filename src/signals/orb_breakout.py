"""ORB breakout detection signal.

Identifies when price breaks above/below the opening range.  Uses
session_open_time from DollarBar to determine which bars form the
opening range (first 15 minutes by default).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class ORBBreakoutConfig:
    orb_minutes: int = 15
    require_close_beyond: bool = True  # require close (not just wick) beyond range
    min_bars_in_range: int = 3


@SignalRegistry.register
class ORBBreakoutSignal(SignalBase):
    """Detects price breaking out of the opening range.

    Scans bars for session_open_time, collects bars within the first
    orb_minutes to define the range, then checks if the latest bar
    closes beyond that range.

    value = distance beyond range in ticks (0.25 tick size).
    passes = True when a breakout is detected.
    direction = "long" for upside breakout, "short" for downside.
    """

    name = "orb_breakout"

    def __init__(self, config: ORBBreakoutConfig | None = None) -> None:
        self.config = config or ORBBreakoutConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        if len(bars) < self.config.min_bars_in_range + 1:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        # Find session_open_time from first DollarBar that has it
        session_open = None
        for b in bars:
            sot = getattr(b, "session_open_time", None)
            if sot is not None:
                session_open = sot
                break

        if session_open is None:
            # Fallback: use first bar's timestamp as session open
            from datetime import datetime
            session_open = datetime.utcfromtimestamp(bars[0].timestamp_ns / 1_000_000_000)

        # ORB cutoff in nanoseconds
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

        if len(orb_highs) < self.config.min_bars_in_range:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_orb_bars", "orb_bar_count": len(orb_highs)},
            )

        range_high = max(orb_highs)
        range_low = min(orb_lows)

        last_bar = bars[-1]

        # Only check for breakout after ORB period
        if last_bar.timestamp_ns <= orb_cutoff_ns:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={
                    "reason": "still_in_orb_period",
                    "range_high": range_high,
                    "range_low": range_low,
                },
            )

        tick_size = 0.25
        price = last_bar.close if self.config.require_close_beyond else last_bar.high

        if last_bar.close > range_high:
            distance = (last_bar.close - range_high) / tick_size
            return SignalResult(
                value=distance, passes=True, direction="long",
                metadata={
                    "range_high": range_high,
                    "range_low": range_low,
                    "breakout_distance_ticks": distance,
                },
            )

        if last_bar.close < range_low:
            distance = (range_low - last_bar.close) / tick_size
            return SignalResult(
                value=distance, passes=True, direction="short",
                metadata={
                    "range_high": range_high,
                    "range_low": range_low,
                    "breakout_distance_ticks": distance,
                },
            )

        # No breakout
        return SignalResult(
            value=0.0, passes=False, direction="none",
            metadata={
                "range_high": range_high,
                "range_low": range_low,
            },
        )
