"""Session VWAP calculator with standard deviation bands.

Maintains running accumulators for VWAP and variance, updated incrementally
on each bar. Exposes deviation bands, slope, and first-kiss detection for
the VWAP Reversion strategy.
"""

from __future__ import annotations

import math
from collections import deque


class VWAPCalculator:
    """Incremental session VWAP with SD bands and trend detection.

    Call on_bar() for each bar during the session. Call reset() at session open.

    Properties expose VWAP, deviation in standard deviations, 1/2/3 SD bands,
    a 20-bar slope of the VWAP line, and flat/trending classification.
    """

    def __init__(self, flat_threshold: float = 0.001, deviation_lookback: int = 20) -> None:
        self._flat_threshold = flat_threshold
        self._sum_pv: float = 0.0        # sum(price * volume)
        self._sum_vol: float = 0.0       # sum(volume)
        self._sum_pv2: float = 0.0       # sum(volume * price^2)
        self._last_price: float = 0.0
        self._vwap_history: deque[float] = deque(maxlen=20)
        self._deviation_history: deque[float] = deque(maxlen=deviation_lookback)
        self._bar_count: int = 0
        self._cached_slope: float = 0.0

    def on_bar(self, price: float, volume: int) -> None:
        """Update VWAP accumulators with a new bar."""
        if volume <= 0:
            return
        self._sum_pv += price * volume
        self._sum_vol += volume
        self._sum_pv2 += volume * price * price
        self._last_price = price
        self._bar_count += 1
        self._vwap_history.append(self.vwap)
        # Track deviation history for first-kiss detection
        sd = self._sd
        if sd > 0:
            self._deviation_history.append(abs(price - self.vwap) / sd)
        else:
            self._deviation_history.append(0.0)
        # Cache slope (avoids recomputing OLS on every property access)
        self._cached_slope = self._compute_slope()

    def reset(self) -> None:
        """Reset all state for a new session."""
        self._sum_pv = 0.0
        self._sum_vol = 0.0
        self._sum_pv2 = 0.0
        self._last_price = 0.0
        self._vwap_history.clear()
        self._deviation_history.clear()
        self._bar_count = 0
        self._cached_slope = 0.0

    @property
    def vwap(self) -> float:
        """Current session VWAP."""
        if self._sum_vol == 0:
            return 0.0
        return self._sum_pv / self._sum_vol

    @property
    def _variance(self) -> float:
        """Volume-weighted variance of prices around VWAP."""
        if self._sum_vol == 0:
            return 0.0
        mean = self.vwap
        # Var = E[X^2] - E[X]^2 (volume-weighted)
        return max(0.0, self._sum_pv2 / self._sum_vol - mean * mean)

    @property
    def _sd(self) -> float:
        """Standard deviation of prices around VWAP."""
        return math.sqrt(self._variance)

    @property
    def deviation_sd(self) -> float:
        """Current price deviation from VWAP in standard deviations."""
        sd = self._sd
        if sd == 0:
            return 0.0
        return (self._last_price - self.vwap) / sd

    @property
    def band_upper_1sd(self) -> float:
        return self.vwap + self._sd

    @property
    def band_lower_1sd(self) -> float:
        return self.vwap - self._sd

    @property
    def band_upper_2sd(self) -> float:
        return self.vwap + 2.0 * self._sd

    @property
    def band_lower_2sd(self) -> float:
        return self.vwap - 2.0 * self._sd

    @property
    def band_upper_3sd(self) -> float:
        return self.vwap + 3.0 * self._sd

    @property
    def band_lower_3sd(self) -> float:
        return self.vwap - 3.0 * self._sd

    def _compute_slope(self) -> float:
        """Linear regression slope of last 20 VWAP values (OLS)."""
        n = len(self._vwap_history)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(self._vwap_history) / n
        num = 0.0
        den = 0.0
        for i, y in enumerate(self._vwap_history):
            dx = i - x_mean
            num += dx * (y - y_mean)
            den += dx * dx
        if den == 0:
            return 0.0
        return num / den

    @property
    def slope_20bar(self) -> float:
        """Linear regression slope of last 20 VWAP values (cached)."""
        return self._cached_slope

    @property
    def is_flat(self) -> bool:
        """True if VWAP slope magnitude is below flat_threshold."""
        return abs(self._cached_slope) < self._flat_threshold

    @property
    def is_trending(self) -> bool:
        """True if VWAP is not flat."""
        return not self.is_flat

    def first_kiss_detected(
        self,
        current_price: float,
        lookback_bars: int = 6,
        sd_threshold: float = 2.0,
    ) -> bool:
        """Detect first-kiss pattern: price was far from VWAP, now returning.

        Returns True if price was >sd_threshold SDs away within the last
        lookback_bars and is now within 0.5 SD of VWAP.

        Args:
            current_price: Current market price.
            lookback_bars: Number of recent bars to check for extreme deviation.
            sd_threshold: SD distance that qualifies as "far away".
        """
        sd = self._sd
        if sd == 0 or self._bar_count < 2:
            return False

        vwap_val = self.vwap
        current_dev = abs(current_price - vwap_val) / sd

        # Must currently be near VWAP (within 0.5 SD)
        if current_dev > 0.5:
            return False

        # Check if any bar in the recent lookback was >sd_threshold SDs away
        n = len(self._deviation_history)
        if n < 2:
            return False
        recent = list(self._deviation_history)[-lookback_bars:]
        return any(d >= sd_threshold for d in recent)
