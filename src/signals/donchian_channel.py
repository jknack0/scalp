"""Donchian Channel signal.

Computes entry and exit channels from highest highs / lowest lows over
configurable lookback periods. Used for breakout detection and trailing stops.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class DonchianChannelConfig:
    entry_period: int = 20
    exit_period: int = 10


@SignalRegistry.register
class DonchianChannelSignal(SignalBase):
    """Donchian Channel signal.

    value = channel width (entry_upper - entry_lower).
    passes = True when close breaks above entry_upper or below entry_lower.
    direction = "long" if close > entry_upper, "short" if close < entry_lower.
    metadata includes entry_upper, entry_lower, exit_upper, exit_lower, mid, width.
    """

    name = "donchian_channel"

    def __init__(self, config: DonchianChannelConfig | None = None) -> None:
        self.config = config or DonchianChannelConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        entry_period = self.config.entry_period
        exit_period = self.config.exit_period
        min_bars = max(entry_period, exit_period) + 1  # +1: current bar excluded from channel

        if len(bars) < min_bars:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)
        close = float(bars[-1].close)

        # Channels are computed from the N bars BEFORE the current bar
        # (exclude the current bar to avoid look-ahead on breakout detection)
        entry_upper = float(np.max(highs[-(entry_period + 1):-1]))
        entry_lower = float(np.min(lows[-(entry_period + 1):-1]))
        exit_upper = float(np.max(highs[-(exit_period + 1):-1]))
        exit_lower = float(np.min(lows[-(exit_period + 1):-1]))

        mid = (entry_upper + entry_lower) / 2.0
        width = entry_upper - entry_lower

        # Breakout detection
        breakout_long = close > entry_upper
        breakout_short = close < entry_lower

        if breakout_long:
            direction = "long"
        elif breakout_short:
            direction = "short"
        else:
            direction = "none"

        return SignalResult(
            value=float(width),
            passes=bool(breakout_long or breakout_short),
            direction=direction,
            metadata={
                "entry_upper": entry_upper,
                "entry_lower": entry_lower,
                "exit_upper": exit_upper,
                "exit_lower": exit_lower,
                "mid": mid,
                "width": width,
            },
        )
