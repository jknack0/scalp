"""Donchian Channel signal — enhanced for all 4 strategy flavors.

Computes entry and exit channels from highest highs / lowest lows over
configurable lookback periods. Provides rich metadata for breakout,
fade, midline pullback, and squeeze strategies:

- Breakout detection (upper/lower break)
- Band touch detection (for mean reversion fades)
- Midline proximity and direction (for pullback entries)
- Width percentile ranking (for squeeze detection)
- Channel trend/slope (for directional bias)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry

TICK_SIZE = 0.25


@dataclass(frozen=True)
class DonchianChannelConfig:
    entry_period: int = 20
    exit_period: int = 10
    width_percentile_lookback: int = 100
    trend_lookback: int = 5
    touch_tolerance_ticks: int = 2


@SignalRegistry.register
class DonchianChannelSignal(SignalBase):
    """Donchian Channel signal with metadata for all 4 flavors.

    value = channel width (entry_upper - entry_lower).
    passes = True when close breaks above entry_upper or below entry_lower.
    direction = "long" if breakout up, "short" if breakout down.
    metadata: see compute() for full list.
    """

    name = "donchian_channel"

    def __init__(self, config: DonchianChannelConfig | None = None) -> None:
        self.config = config or DonchianChannelConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        entry_period = self.config.entry_period
        exit_period = self.config.exit_period
        min_bars = max(entry_period, exit_period) + 1

        if len(bars) < min_bars:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)
        close = float(bars[-1].close)

        # Channels exclude current bar (no look-ahead)
        entry_upper = float(np.max(highs[-(entry_period + 1):-1]))
        entry_lower = float(np.min(lows[-(entry_period + 1):-1]))
        exit_upper = float(np.max(highs[-(exit_period + 1):-1]))
        exit_lower = float(np.min(lows[-(exit_period + 1):-1]))

        mid = (entry_upper + entry_lower) / 2.0
        width = entry_upper - entry_lower

        # ── Breakout detection ──────────────────────────────────────
        breakout_long = close > entry_upper
        breakout_short = close < entry_lower

        if breakout_long:
            direction = "long"
        elif breakout_short:
            direction = "short"
        else:
            direction = "none"

        # ── Band touch detection (for fades) ────────────────────────
        tolerance = self.config.touch_tolerance_ticks * TICK_SIZE
        upper_touch = (not breakout_long) and close >= entry_upper - tolerance
        lower_touch = (not breakout_short) and close <= entry_lower + tolerance

        # ── Midline fields (for pullback strategy) ──────────────────
        above_midline = close > mid
        midline_distance = abs(close - mid)
        # "at midline" = within 5% of channel width or 2 ticks, whichever is larger
        midline_tol = max(2 * TICK_SIZE, width * 0.05) if width > 0 else 2 * TICK_SIZE
        at_midline = midline_distance <= midline_tol

        # ── Width percentile (for squeeze detection) ────────────────
        width_percentile = 50.0
        channel_expanding = False
        lookback = self.config.width_percentile_lookback
        # Need at least (entry_period + lookback) bars to compute rolling widths
        if len(highs) >= entry_period + lookback + 1:
            # Use sliding windows on the bars BEFORE current bar
            h_prior = highs[:-1]
            l_prior = lows[:-1]
            n_prior = len(h_prior)
            # Compute rolling widths for the last `lookback` positions
            start = max(0, n_prior - lookback - entry_period + 1)
            rolling_widths = []
            for i in range(start + entry_period, n_prior + 1):
                w = float(np.max(h_prior[i - entry_period:i]) -
                          np.min(l_prior[i - entry_period:i]))
                rolling_widths.append(w)
            if rolling_widths:
                rolling_arr = np.array(rolling_widths)
                width_percentile = float(np.sum(rolling_arr < width) / len(rolling_arr) * 100.0)
                # Channel expanding: current width > width from trend_lookback bars ago
                tl = self.config.trend_lookback
                if len(rolling_widths) > tl:
                    channel_expanding = width > rolling_widths[-tl]

        # ── Channel trend (midline slope) ───────────────────────────
        trend_slope = 0.0
        tl = self.config.trend_lookback
        if len(highs) >= entry_period + tl + 2:
            # Midline from tl bars ago
            prior_upper = float(np.max(highs[-(entry_period + tl + 1):-(tl + 1)]))
            prior_lower = float(np.min(lows[-(entry_period + tl + 1):-(tl + 1)]))
            prior_mid = (prior_upper + prior_lower) / 2.0
            trend_slope = mid - prior_mid

        return SignalResult(
            value=float(width),
            passes=bool(breakout_long or breakout_short),
            direction=direction,
            metadata={
                # Core channel levels
                "entry_upper": entry_upper,
                "entry_lower": entry_lower,
                "exit_upper": exit_upper,
                "exit_lower": exit_lower,
                "mid": mid,
                "width": width,
                # Band touch (fade entries)
                "upper_touch": upper_touch,
                "lower_touch": lower_touch,
                # Midline (pullback entries)
                "above_midline": above_midline,
                "at_midline": at_midline,
                "midline_distance": midline_distance,
                # Squeeze detection
                "width_percentile": width_percentile,
                "channel_expanding": channel_expanding,
                # Trend
                "trend_slope": trend_slope,
            },
        )
