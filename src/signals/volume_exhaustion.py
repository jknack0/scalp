"""Volume exhaustion signal.

Detects volume drying up relative to recent history — a sign that a move
may be running out of steam.  Uses buy_volume / sell_volume from DollarBar
when available, falls back to total volume.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class VolumeExhaustionConfig:
    lookback_bars: int = 20
    exhaustion_threshold: float = 0.4  # current vol / mean vol below this = exhausted


@SignalRegistry.register
class VolumeExhaustionSignal(SignalBase):
    """Flags when current bar volume is abnormally low vs recent average.

    value = current_vol / mean_vol (ratio).
    passes = True when ratio < exhaustion_threshold (volume is drying up).
    direction = side that is exhausting (uses buy/sell split if available).
    """

    name = "volume_exhaustion"

    def __init__(self, config: VolumeExhaustionConfig | None = None) -> None:
        self.config = config or VolumeExhaustionConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        lookback = self.config.lookback_bars

        if len(bars) < lookback:
            return SignalResult(
                value=1.0, passes=False, direction="none",
                metadata={"reason": "insufficient_bars"},
            )

        tail = bars[-lookback:]
        volumes = np.array([b.volume for b in tail], dtype=np.float64)
        mean_vol = float(np.mean(volumes[:-1]))  # mean of prior bars

        if mean_vol < 1e-9:
            return SignalResult(
                value=1.0, passes=False, direction="none",
                metadata={"reason": "zero_mean_volume"},
            )

        current_vol = float(volumes[-1])
        ratio = current_vol / mean_vol

        passes = ratio < self.config.exhaustion_threshold

        # Direction: which side is exhausting?
        last_bar = tail[-1]
        buy_vol = float(getattr(last_bar, "buy_volume", 0))
        sell_vol = float(getattr(last_bar, "sell_volume", 0))

        if buy_vol + sell_vol > 0:
            if buy_vol < sell_vol * 0.5:
                direction = "short"  # buy exhaustion -> bearish
            elif sell_vol < buy_vol * 0.5:
                direction = "long"  # sell exhaustion -> bullish
            else:
                direction = "none"
        else:
            direction = "none"

        return SignalResult(
            value=ratio,
            passes=passes,
            direction=direction,
            metadata={
                "vol_ratio": ratio,
                "current_vol": current_vol,
                "mean_vol": mean_vol,
            },
        )
