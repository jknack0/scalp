"""Cumulative Volume Delta calculator with trade classification.

Classifies trades as buyer- or seller-initiated using bid/ask comparison
(L1 data) or tick rule fallback. Tracks running CVD, per-bar deltas,
slope, z-score, and price-CVD divergence.
"""

from __future__ import annotations

import math
from collections import deque


class CVDCalculator:
    """Incremental CVD with trade classification and divergence detection.

    For L1 tick data: call on_tick() per trade.
    For 1s bar fallback: call on_bar_approx() per bar.
    Call on_bar_close() to mark bar boundaries for per-bar delta tracking.
    Call reset() at session open.
    """

    def __init__(self, slope_window: int = 20, zscore_window: int = 20) -> None:
        self._slope_window = slope_window
        self._zscore_window = zscore_window

        self._cvd: float = 0.0
        self._bar_delta: float = 0.0     # delta accumulated in current bar
        self._prev_price: float | None = None

        # Per-bar delta history for slope and z-score
        self._bar_deltas: deque[float] = deque(maxlen=max(slope_window, zscore_window))
        self._cached_slope: float = 0.0
        self._cached_zscore: float = 0.0

    def on_tick(self, price: float, size: int, bid: float, ask: float) -> None:
        """Classify a single trade and update CVD.

        Classification:
        - price >= ask → buy (add to CVD)
        - price <= bid → sell (subtract from CVD)
        - mid-spread → tick rule (compare to previous price)
        """
        if price >= ask:
            delta = size
        elif price <= bid:
            delta = -size
        else:
            # Tick rule fallback
            if self._prev_price is not None and price > self._prev_price:
                delta = size
            elif self._prev_price is not None and price < self._prev_price:
                delta = -size
            else:
                delta = 0  # unchanged price, ambiguous

        self._cvd += delta
        self._bar_delta += delta
        self._prev_price = price

    def on_bar_approx(self, open_: float, close: float, volume: int) -> None:
        """Approximate CVD from 1s bar data (no L1 ticks).

        If close > open → net buy; close < open → net sell.
        Less accurate than tick-level classification but testable on existing data.
        """
        if close > open_:
            delta = volume
        elif close < open_:
            delta = -volume
        else:
            delta = 0

        self._cvd += delta
        self._bar_delta += delta

    def on_bar_close(self) -> None:
        """Mark end of current bar. Stores bar delta and resets accumulator."""
        self._bar_deltas.append(self._bar_delta)
        self._bar_delta = 0.0
        # Cache slope and zscore (avoids recomputing on every property access)
        self._cached_slope = self._compute_slope()
        self._cached_zscore = self._compute_zscore()

    def reset(self) -> None:
        """Reset all state for a new session."""
        self._cvd = 0.0
        self._bar_delta = 0.0
        self._prev_price = None
        self._bar_deltas.clear()
        self._cached_slope = 0.0
        self._cached_zscore = 0.0

    @property
    def cvd(self) -> float:
        """Running cumulative volume delta."""
        return self._cvd

    @property
    def cvd_delta_per_bar(self) -> float:
        """CVD change accumulated in the current (open) bar."""
        return self._bar_delta

    def _compute_slope(self) -> float:
        """Linear regression slope of recent per-bar deltas (OLS)."""
        n = min(len(self._bar_deltas), self._slope_window)
        if n < 2:
            return 0.0
        # Iterate deque directly (avoid list conversion)
        vals = list(self._bar_deltas)[-n:] if n < len(self._bar_deltas) else list(self._bar_deltas)
        x_mean = (n - 1) / 2.0
        y_mean = sum(vals) / n
        num = 0.0
        den = 0.0
        for i, y in enumerate(vals):
            dx = i - x_mean
            num += dx * (y - y_mean)
            den += dx * dx
        if den == 0:
            return 0.0
        return num / den

    def _compute_zscore(self) -> float:
        """Current CVD normalized by rolling std of bar deltas."""
        n = min(len(self._bar_deltas), self._zscore_window)
        if n < 2:
            return 0.0
        vals = list(self._bar_deltas)[-n:] if n < len(self._bar_deltas) else list(self._bar_deltas)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(var)
        if std == 0:
            return 0.0
        return (self._cvd - mean) / std

    @property
    def cvd_slope_20bar(self) -> float:
        """Linear regression slope of recent per-bar deltas (cached)."""
        return self._cached_slope

    @property
    def cvd_zscore(self) -> float:
        """Current CVD normalized by rolling standard deviation of bar deltas (cached)."""
        return self._cached_zscore

    def divergence_from_price(self, price_change: float) -> float:
        """Score divergence between price movement and CVD.

        Returns 0.0 when price and CVD agree in direction,
        up to 1.0 when they maximally disagree.

        Args:
            price_change: Price change over the measurement period.
        """
        if len(self._bar_deltas) == 0:
            return 0.0

        # Use recent CVD change (sum of last few bar deltas)
        recent_cvd_change = sum(list(self._bar_deltas)[-5:])

        # Both zero → no divergence
        if price_change == 0 and recent_cvd_change == 0:
            return 0.0

        # Normalize to unit vectors and compute divergence
        # If signs disagree → divergence; if agree → convergence
        if price_change == 0 or recent_cvd_change == 0:
            return 0.5  # one moving, other not

        # Signs: same = 0, opposite = 1
        if (price_change > 0) != (recent_cvd_change > 0):
            return 1.0
        return 0.0
