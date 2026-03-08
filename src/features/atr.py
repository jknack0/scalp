"""Rolling ATR calculator with volatility regime classification.

Uses Wilder smoothing (exponential moving average) for ATR computation.
Tracks a rolling window of ATR values for percentile-based regime
classification, and computes semi-variance for directional bias.
"""

from __future__ import annotations

import math
from collections import deque


class ATRCalculator:
    """Incremental ATR with vol regime and semi-variance.

    Call on_bar() with each bar's high/low/close. Call reset() at session open.
    """

    def __init__(
        self,
        period: int = 14,
        tick_size: float = 0.25,
        point_value: float = 5.0,
        regime_window: int = 100,
    ) -> None:
        self._period = period
        self._tick_size = tick_size
        self._point_value = point_value
        self._prev_close: float | None = None
        self._atr: float = 0.0
        self._bar_count: int = 0

        # Rolling window for regime classification
        self._atr_history: deque[float] = deque(maxlen=regime_window)

        # Semi-variance tracking
        self._up_trs: deque[float] = deque(maxlen=regime_window)
        self._down_trs: deque[float] = deque(maxlen=regime_window)
        self._cached_vol_regime: str = "NORMAL"

    def on_bar(self, high: float, low: float, close: float) -> None:
        """Update ATR with a new bar."""
        if self._prev_close is None:
            # First bar: TR = high - low
            tr = high - low
            self._atr = tr
        else:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
            # Wilder smoothing
            if self._bar_count < self._period:
                # Initial: simple average until we have enough bars
                self._atr = (self._atr * self._bar_count + tr) / (self._bar_count + 1)
            else:
                self._atr = self._atr * (self._period - 1) / self._period + tr / self._period

            # Track semi-variance: classify TR by direction
            if close > self._prev_close:
                self._up_trs.append(tr)
            elif close < self._prev_close:
                self._down_trs.append(tr)

        self._prev_close = close
        self._bar_count += 1
        self._atr_history.append(self._atr)
        # Cache vol regime (avoids O(n) percentile scan on every access)
        self._cached_vol_regime = self._compute_vol_regime()

    def reset(self) -> None:
        """Reset all state for a new session."""
        self._prev_close = None
        self._atr = 0.0
        self._bar_count = 0
        self._atr_history.clear()
        self._up_trs.clear()
        self._down_trs.clear()
        self._cached_vol_regime = "NORMAL"

    @property
    def atr(self) -> float:
        """Rolling ATR in index points."""
        return self._atr

    @property
    def atr_ticks(self) -> float:
        """ATR expressed in tick units."""
        if self._tick_size == 0:
            return 0.0
        return self._atr / self._tick_size

    @property
    def atr_dollars(self) -> float:
        """ATR expressed in dollar value per contract."""
        return self._atr * self._point_value

    def _compute_vol_regime(self) -> str:
        """Compute volatility regime from ATR percentile rank."""
        if len(self._atr_history) < 2:
            return "NORMAL"
        current = self._atr
        below = sum(1 for v in self._atr_history if v < current)
        percentile = below / len(self._atr_history)
        if percentile < 0.25:
            return "LOW"
        elif percentile > 0.75:
            return "HIGH"
        return "NORMAL"

    @property
    def vol_regime(self) -> str:
        """Volatility regime: LOW / NORMAL / HIGH based on percentile rank (cached)."""
        return self._cached_vol_regime

    @property
    def semi_variance_up(self) -> float:
        """Variance of true ranges on up-move bars."""
        return _variance(self._up_trs)

    @property
    def semi_variance_down(self) -> float:
        """Variance of true ranges on down-move bars."""
        return _variance(self._down_trs)

    @property
    def dominant_direction(self) -> str:
        """UP / DOWN / NEUTRAL based on semi-variance ratio.

        DOWN if semi_var_down > 1.5x semi_var_up, UP if opposite, else NEUTRAL.
        """
        sv_up = self.semi_variance_up
        sv_down = self.semi_variance_down

        if sv_up == 0 and sv_down == 0:
            return "NEUTRAL"
        if sv_down > 1.5 * sv_up:
            return "DOWN"
        if sv_up > 1.5 * sv_down:
            return "UP"
        return "NEUTRAL"


def _variance(values: deque[float]) -> float:
    """Compute variance of a deque of floats."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)
