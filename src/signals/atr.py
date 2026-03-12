"""Stateless ATR signal with volatility regime classification.

Computes True Range, Wilder-smoothed ATR, ATR in ticks, and a percentile-based
volatility regime from a list of BarEvent objects.  Informational only --
always passes.

Ported from src/features/atr.py into the stateless signal framework.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry

_logger = logging.getLogger(__name__)

MES_TICK_SIZE = 0.25


@dataclass(frozen=True)
class ATRConfig:
    lookback_bars: int = 14
    vol_regime_lookback: int = 100


@SignalRegistry.register
class ATRSignal(SignalBase):
    """ATR signal: informational volatility measure, always passes."""

    name = "atr"

    def __init__(self, config: ATRConfig | None = None) -> None:
        self.config = config or ATRConfig()
        self._compute_count: int = 0

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        """Compute Wilder-smoothed ATR and vol regime from bar history.

        Args:
            bars: List of BarEvent objects (oldest first).

        Returns:
            SignalResult with value=atr_ticks, passes=True always.
            Metadata includes atr_ticks, vol_regime, atr_percentile.
        """
        if len(bars) < 2:
            return SignalResult(
                value=0.0,
                passes=True,
                direction="none",
                metadata={
                    "atr_ticks": 0.0,
                    "vol_regime": "NORMAL",
                    "atr_percentile": 50.0,
                },
            )

        # ----------------------------------------------------------
        # 1. Compute True Range for every bar
        # ----------------------------------------------------------
        n = len(bars)
        true_ranges = np.empty(n, dtype=np.float64)

        # Compute gap threshold: 3× median inter-bar interval
        # Bars separated by more than this are across a session gap
        if n > 2:
            deltas = np.array([
                bars[i].timestamp_ns - bars[i - 1].timestamp_ns
                for i in range(1, min(n, 50))  # sample first 50 pairs
            ])
            self._max_gap_ns = int(np.median(deltas) * 3)
        else:
            self._max_gap_ns = 10 * 60 * 1_000_000_000  # fallback 10 min

        # First bar: TR = high - low (no previous close)
        true_ranges[0] = bars[0].high - bars[0].low

        for i in range(1, n):
            h = bars[i].high
            l = bars[i].low
            pc = bars[i - 1].close
            # Detect session gaps: if bars are more than 3× the median
            # interval apart, use H-L only (prevClose across gap inflates TR)
            gap_ns = bars[i].timestamp_ns - bars[i - 1].timestamp_ns
            if gap_ns > self._max_gap_ns:
                true_ranges[i] = h - l
            else:
                true_ranges[i] = max(h - l, abs(h - pc), abs(l - pc))

        # ----------------------------------------------------------
        # 2. Wilder-smoothed ATR (incremental)
        # ----------------------------------------------------------
        period = self.config.lookback_bars
        atr_series = np.empty(n, dtype=np.float64)
        atr_series[0] = true_ranges[0]

        for i in range(1, n):
            if i < period:
                # Build-up phase: simple running average
                atr_series[i] = (atr_series[i - 1] * i + true_ranges[i]) / (i + 1)
            else:
                # Wilder smoothing: atr = prev + (tr - prev) / period
                atr_series[i] = atr_series[i - 1] + (true_ranges[i] - atr_series[i - 1]) / period

        current_atr = float(atr_series[-1])
        atr_ticks = current_atr / MES_TICK_SIZE

        # Debug logging for first 3 computes to diagnose inflated ATR
        self._compute_count += 1
        if self._compute_count <= 3:
            gaps_detected = sum(
                1 for i in range(1, n)
                if bars[i].timestamp_ns - bars[i - 1].timestamp_ns > self._max_gap_ns
            )
            _logger.info(
                "ATR_DEBUG n_bars=%d gap_threshold_min=%.1f gaps_detected=%d "
                "atr_raw=%.4f atr_ticks=%.1f max_tr=%.4f median_tr=%.4f",
                n,
                self._max_gap_ns / 60e9,
                gaps_detected,
                current_atr,
                atr_ticks,
                float(np.max(true_ranges)),
                float(np.median(true_ranges)),
            )

        # ----------------------------------------------------------
        # 3. Vol regime via percentile rank
        # ----------------------------------------------------------
        lookback = self.config.vol_regime_lookback
        regime_window = atr_series[-lookback:] if n >= lookback else atr_series
        below = float(np.sum(regime_window < current_atr))
        percentile = (below / len(regime_window)) * 100.0

        if percentile < 25.0:
            vol_regime = "LOW"
        elif percentile > 75.0:
            vol_regime = "HIGH"
        else:
            vol_regime = "NORMAL"

        return SignalResult(
            value=atr_ticks,
            passes=True,
            direction="none",
            metadata={
                "atr_ticks": atr_ticks,
                "atr_raw": current_atr,
                "vol_regime": vol_regime,
                "atr_percentile": percentile,
            },
        )
