"""MACD signal with zero-line rejection detection.

Computes MACD line, signal line, and histogram from EMA crossover.
Detects zero-line rejections: histogram approached zero (abs < 0.5 within
last 3 bars) then bounced away in the original trend direction.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class MACDConfig:
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    zero_approach_threshold: float = 0.5
    zero_approach_lookback: int = 3


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Compute exponential moving average over an array.

    Returns an array of the same length as values, with the first
    (period - 1) entries being NaN.
    """
    result = np.full_like(values, np.nan)
    if len(values) < period:
        return result

    # Seed with SMA of first `period` values
    result[period - 1] = np.mean(values[:period])
    alpha = 2.0 / (period + 1)
    for i in range(period, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


@SignalRegistry.register
class MACDSignal(SignalBase):
    """MACD zero-line rejection signal.

    value = current histogram value.
    passes = True when a zero-line rejection is detected.
    direction = "long" if histogram bouncing positive, "short" if bouncing negative.
    metadata includes macd_line, signal_line, histogram, prev_histogram.
    """

    name = "macd"

    def __init__(self, config: MACDConfig | None = None) -> None:
        self.config = config or MACDConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        min_bars = self.config.slow_period + self.config.signal_period
        if len(bars) < min_bars:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        closes = np.array([b.close for b in bars], dtype=np.float64)

        # Compute MACD components
        fast_ema = _ema(closes, self.config.fast_period)
        slow_ema = _ema(closes, self.config.slow_period)

        # MACD line starts where slow EMA is valid
        macd_line = fast_ema - slow_ema

        # Signal line = EMA of MACD line (only valid portion)
        # Find first valid MACD index
        first_valid = self.config.slow_period - 1
        macd_valid = macd_line[first_valid:]
        signal_ema = _ema(macd_valid, self.config.signal_period)

        # Map signal line back to full array
        signal_line = np.full_like(closes, np.nan)
        signal_line[first_valid:] = signal_ema

        # Histogram
        histogram = macd_line - signal_line

        # Current values
        cur_macd = float(macd_line[-1])
        cur_signal = float(signal_line[-1])
        cur_hist = float(histogram[-1])

        if np.isnan(cur_hist):
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars_for_signal_line"},
            )

        prev_hist = float(histogram[-2]) if len(histogram) >= 2 and not np.isnan(histogram[-2]) else 0.0

        # Zero-line rejection detection:
        # 1. Histogram approached zero (abs < threshold) within last N bars
        # 2. Histogram now moving AWAY from zero in the original trend direction
        lookback = self.config.zero_approach_lookback
        threshold = self.config.zero_approach_threshold

        # Need enough valid histogram bars for lookback + 1 (current)
        # Gather recent histogram values (current + lookback prior bars)
        recent_count = lookback + 1  # e.g. 4 bars total for lookback=3
        valid_hist = histogram[~np.isnan(histogram)]

        rejection = False
        direction = "none"

        if len(valid_hist) >= recent_count + 1:
            # recent = last (lookback+1) bars including current
            recent = valid_hist[-recent_count:]
            # prior = the bar just before the recent window (to establish original direction)
            prior_bar = valid_hist[-(recent_count + 1)]

            # Check if histogram approached zero within the lookback window
            # (excluding current bar — we check the prior `lookback` bars)
            lookback_bars = recent[:-1]  # the lookback bars before current
            approached_zero = bool(np.any(np.abs(lookback_bars) < threshold))

            if approached_zero:
                current = recent[-1]
                # Determine original trend direction from prior bar
                # (the bar before the lookback window)
                if prior_bar > 0 and current > 0:
                    # Was positive, approached zero, now bouncing positive
                    # Confirm it's moving away: current > minimum of lookback window
                    min_in_window = float(np.min(lookback_bars))
                    if current > min_in_window and abs(current) > threshold:
                        rejection = True
                        direction = "long"
                elif prior_bar < 0 and current < 0:
                    # Was negative, approached zero, now bouncing negative
                    max_in_window = float(np.max(lookback_bars))
                    if current < max_in_window and abs(current) > threshold:
                        rejection = True
                        direction = "short"

        return SignalResult(
            value=cur_hist,
            passes=rejection,
            direction=direction,
            metadata={
                "macd_line": cur_macd,
                "signal_line": cur_signal,
                "histogram": cur_hist,
                "prev_histogram": prev_hist,
            },
        )
