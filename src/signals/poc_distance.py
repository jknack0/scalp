"""Stateless Point of Control (POC) distance signal.

Builds a developing session volume profile from bars, finds the POC
(price bucket with highest cumulative volume), and measures how far
the current price is from it in ticks.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class POCDistanceConfig:
    min_bars: int = 30
    proximity_ticks: int = 4
    bucket_size: float = 0.25


@SignalRegistry.register
class POCDistanceSignal(SignalBase):
    """Measures distance from the developing session POC in ticks.

    Computation:
        1. Round each bar's close to nearest bucket_size and accumulate volume.
        2. POC = price bucket with the highest cumulative volume.
        3. distance_ticks = (close - poc) / tick_size.
        4. passes = abs(distance_ticks) <= proximity_ticks.
        5. direction = "long" if above POC, "short" if below, "none" if at POC.
    """

    name = "poc_distance"

    def __init__(self, config: POCDistanceConfig | None = None) -> None:
        self.config = config or POCDistanceConfig()

    def _round_to_bucket(self, price: float) -> float:
        """Round price to the nearest bucket_size."""
        bs = self.config.bucket_size
        return round(price / bs) * bs

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        cfg = self.config

        # Not enough bars to form a meaningful profile
        if len(bars) < cfg.min_bars:
            return SignalResult(
                value=0.0,
                passes=False,
                direction="none",
                metadata={"reason": "insufficient_bars", "bar_count": len(bars)},
            )

        # Build volume profile: price bucket -> cumulative volume
        profile: dict[float, int] = defaultdict(int)
        for bar in bars:
            level = self._round_to_bucket(bar.close)
            profile[level] += bar.volume

        if not profile:
            return SignalResult(
                value=0.0,
                passes=False,
                direction="none",
                metadata={"reason": "empty_profile"},
            )

        # POC = bucket with highest volume
        poc_price = max(profile, key=profile.__getitem__)

        # Distance from current close to POC in ticks
        current_close = bars[-1].close
        distance_ticks = (current_close - poc_price) / cfg.bucket_size

        # Gate: price must be within proximity_ticks of POC
        passes = abs(distance_ticks) <= cfg.proximity_ticks

        # Direction relative to POC
        if distance_ticks > 0:
            direction = "long"
        elif distance_ticks < 0:
            direction = "short"
        else:
            direction = "none"

        price_above_poc = current_close > poc_price

        return SignalResult(
            value=distance_ticks,
            passes=passes,
            direction=direction,
            metadata={
                "poc_price": poc_price,
                "distance_ticks": distance_ticks,
                "price_above_poc": price_above_poc,
            },
        )
