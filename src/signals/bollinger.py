"""Bollinger Bands signal.

Computes 20-period SMA with configurable SD multiplier bands.
Reports deviation from the bands and band width (squeeze detection).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class BollingerConfig:
    period: int = 20
    num_std: float = 2.0


@SignalRegistry.register
class BollingerSignal(SignalBase):
    """Bollinger Bands signal.

    value = deviation in SDs from SMA (negative = below lower band territory).
    passes = True when price is outside the bands.
    direction = "long" if below lower band, "short" if above upper band.
    metadata includes sma, upper, lower, bandwidth, %b.
    """

    name = "bollinger"

    def __init__(self, config: BollingerConfig | None = None) -> None:
        self.config = config or BollingerConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        period = self.config.period
        if len(bars) < period:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        closes = np.array([b.close for b in bars], dtype=np.float64)
        window = closes[-period:]

        sma = float(np.mean(window))
        sd = float(np.std(window, ddof=1))

        if sd < 1e-10:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"sma": sma, "sd": 0.0, "upper": sma, "lower": sma,
                          "bandwidth": 0.0, "pct_b": 0.5},
            )

        upper = sma + self.config.num_std * sd
        lower = sma - self.config.num_std * sd
        price = closes[-1]

        # Deviation in SDs from SMA
        deviation_sd = (price - sma) / sd

        # %B: 0 = at lower band, 1 = at upper band
        band_width = upper - lower
        pct_b = (price - lower) / band_width if band_width > 0 else 0.5

        # Bandwidth relative to SMA (squeeze detection)
        bandwidth = band_width / sma if sma > 0 else 0.0

        # Band state: stable vs expanding
        if len(closes) >= period + 5:
            prev_window = closes[-(period + 5):-5]
            prev_sd = float(np.std(prev_window, ddof=1))
            band_expanding = sd > prev_sd * 1.1
        else:
            band_expanding = False

        price_f = float(price)
        outside = price_f < lower or price_f > upper
        if price_f < lower:
            direction = "long"
        elif price_f > upper:
            direction = "short"
        else:
            direction = "none"

        return SignalResult(
            value=float(deviation_sd),
            passes=bool(outside),
            direction=direction,
            metadata={
                "sma": sma,
                "sd": sd,
                "upper": upper,
                "lower": lower,
                "bandwidth": bandwidth,
                "pct_b": pct_b,
                "band_expanding": band_expanding,
            },
        )
