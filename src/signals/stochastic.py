"""Stochastic oscillator signal.

Computes %K and %D with configurable periods, detects bullish/bearish
crossovers in extreme zones (oversold < 20, overbought > 80).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class StochasticConfig:
    k_period: int = 14
    k_smooth: int = 3
    d_smooth: int = 3


@SignalRegistry.register
class StochasticSignal(SignalBase):
    """Stochastic oscillator signal.

    value = %K
    passes = True when crossover detected in extreme zone.
    direction = "long" (bullish crossover in oversold) or "short" (bearish crossover in overbought).
    metadata includes k, d, prev_k, prev_d.
    """

    name = "stochastic"

    def __init__(self, config: StochasticConfig | None = None) -> None:
        self.config = config or StochasticConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        cfg = self.config
        # Need enough bars for k_period + k_smooth + d_smooth + 1 (for prev values)
        min_bars = cfg.k_period + cfg.k_smooth + cfg.d_smooth
        if len(bars) < min_bars:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)
        closes = np.array([b.close for b in bars], dtype=np.float64)

        n = len(bars)

        # Compute raw %K for each bar where we have enough lookback
        raw_k = np.full(n, np.nan)
        for i in range(cfg.k_period - 1, n):
            highest = np.max(highs[i - cfg.k_period + 1 : i + 1])
            lowest = np.min(lows[i - cfg.k_period + 1 : i + 1])
            rng = highest - lowest
            if rng < 1e-10:
                raw_k[i] = 50.0  # flat market
            else:
                raw_k[i] = 100.0 * (closes[i] - lowest) / rng

        # %K = SMA(raw_k, k_smooth)
        pct_k = np.full(n, np.nan)
        start_k = cfg.k_period - 1 + cfg.k_smooth - 1
        for i in range(start_k, n):
            window = raw_k[i - cfg.k_smooth + 1 : i + 1]
            if np.any(np.isnan(window)):
                continue
            pct_k[i] = float(np.mean(window))

        # %D = SMA(%K, d_smooth)
        pct_d = np.full(n, np.nan)
        start_d = start_k + cfg.d_smooth - 1
        for i in range(start_d, n):
            window = pct_k[i - cfg.d_smooth + 1 : i + 1]
            if np.any(np.isnan(window)):
                continue
            pct_d[i] = float(np.mean(window))

        # Current and previous values
        k = pct_k[-1]
        d = pct_d[-1]
        prev_k = pct_k[-2]
        prev_d = pct_d[-2]

        if np.isnan(k) or np.isnan(d) or np.isnan(prev_k) or np.isnan(prev_d):
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_computed_values"},
            )

        k = float(k)
        d = float(d)
        prev_k = float(prev_k)
        prev_d = float(prev_d)

        # Crossover detection in extreme zones
        bullish_cross = prev_k < prev_d and k > d and k < 20.0
        bearish_cross = prev_k > prev_d and k < d and k > 80.0

        if bullish_cross:
            passes = True
            direction = "long"
        elif bearish_cross:
            passes = True
            direction = "short"
        else:
            passes = False
            direction = "none"

        return SignalResult(
            value=k,
            passes=passes,
            direction=direction,
            metadata={
                "k": k,
                "d": d,
                "prev_k": prev_k,
                "prev_d": prev_d,
            },
        )
