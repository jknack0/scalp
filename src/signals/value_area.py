"""Value Area signal — prior session VAH/VAL/POC with proximity detection.

Maintains state across bars to track session boundaries and compute
the prior session's volume profile. Reports distance to nearest VA
boundary and whether price is within proximity of VAH or VAL.

This is DIFFERENT from poc_distance (which tracks developing session POC)
and from value_area_reversion (which uses the 80% rule for VA re-entry).
This signal detects when price APPROACHES VAH/VAL from outside, for
fade setups.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from zoneinfo import ZoneInfo

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry

_ET = ZoneInfo("US/Eastern")


@dataclass(frozen=True)
class ValueAreaConfig:
    price_step: float = 0.25
    va_pct: float = 0.70
    proximity_points: float = 2.0


@SignalRegistry.register
class ValueAreaSignal(SignalBase):
    """Prior session Value Area proximity signal.

    Maintains state across compute() calls to track session boundaries
    and build volume profiles from prior session bars.

    value = distance to nearest VA boundary (VAH or VAL)
    passes = True when price within proximity_points of VAH or VAL
    direction = "long" if near VAL (bounce up), "short" if near VAH (bounce down)
    metadata: vah, val, poc, va_width, price_position, distance_to_vah, distance_to_val
    """

    name = "value_area"

    def __init__(self, config: ValueAreaConfig | None = None) -> None:
        self.config = config or ValueAreaConfig()
        # State across calls
        self._prior_vah: float = 0.0
        self._prior_val: float = 0.0
        self._prior_poc: float = 0.0
        self._current_session_date = None
        self._session_bars: list[BarEvent] = []

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        if not bars:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "no_bars"},
            )

        last_bar = bars[-1]
        now = datetime.fromtimestamp(last_bar.timestamp_ns / 1e9, tz=_ET)
        today = now.date()

        # Detect new session day
        if today != self._current_session_date:
            # Compute VA from accumulated prior session bars
            if self._session_bars:
                self._compute_value_area()
            self._current_session_date = today
            self._session_bars = []

        # Track current session bars
        # Only add the latest bar (avoid duplicates on re-compute)
        if not self._session_bars or self._session_bars[-1].timestamp_ns != last_bar.timestamp_ns:
            self._session_bars.append(last_bar)

        # Need prior session VA to produce a meaningful result
        if self._prior_vah == 0.0 or self._prior_val == 0.0:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "no_prior_session"},
            )

        price = last_bar.close
        vah = self._prior_vah
        val = self._prior_val
        poc = self._prior_poc
        va_width = vah - val

        distance_to_vah = price - vah
        distance_to_val = price - val

        # Price position relative to VA
        if price > vah:
            price_position = "above"
        elif price < val:
            price_position = "below"
        else:
            price_position = "inside"

        # Distance to nearest boundary
        abs_dist_vah = abs(distance_to_vah)
        abs_dist_val = abs(distance_to_val)
        nearest_dist = min(abs_dist_vah, abs_dist_val)

        proximity = self.config.proximity_points
        near_vah = abs_dist_vah <= proximity
        near_val = abs_dist_val <= proximity

        passes = near_vah or near_val

        # Direction: fade towards POC
        if near_val and not near_vah:
            direction = "long"   # near VAL, expect bounce up
        elif near_vah and not near_val:
            direction = "short"  # near VAH, expect bounce down
        elif near_val and near_vah:
            # Both near (very narrow VA) — use position relative to POC
            direction = "long" if price < poc else "short"
        else:
            direction = "none"

        return SignalResult(
            value=nearest_dist,
            passes=passes,
            direction=direction,
            metadata={
                "vah": vah,
                "val": val,
                "poc": poc,
                "va_width": va_width,
                "price_position": price_position,
                "distance_to_vah": distance_to_vah,
                "distance_to_val": distance_to_val,
            },
        )

    def _compute_value_area(self) -> None:
        """Compute VAH, VAL, POC from accumulated session bars."""
        if not self._session_bars:
            return

        step = self.config.price_step

        # Build volume profile: price_level -> total_volume
        profile: dict[float, float] = defaultdict(float)
        for bar in self._session_bars:
            low = round(bar.low / step) * step
            high = round(bar.high / step) * step
            levels = int((high - low) / step) + 1
            if levels < 1:
                levels = 1
            vol_per_level = bar.volume / levels
            price = low
            while price <= high + step / 2:
                profile[round(price, 2)] += vol_per_level
                price += step

        if not profile:
            return

        total_vol = sum(profile.values())
        if total_vol == 0:
            return

        # POC = price level with highest volume
        self._prior_poc = max(profile, key=profile.get)  # type: ignore[arg-type]

        # Value Area: expand from POC until va_pct of volume captured
        sorted_levels = sorted(profile.keys())
        poc_idx = sorted_levels.index(self._prior_poc)

        va_vol = profile[self._prior_poc]
        lo_idx = poc_idx
        hi_idx = poc_idx

        while va_vol / total_vol < self.config.va_pct:
            can_go_up = hi_idx < len(sorted_levels) - 1
            can_go_down = lo_idx > 0

            if not can_go_up and not can_go_down:
                break

            up_vol = profile[sorted_levels[hi_idx + 1]] if can_go_up else -1
            down_vol = profile[sorted_levels[lo_idx - 1]] if can_go_down else -1

            if up_vol >= down_vol:
                hi_idx += 1
                va_vol += profile[sorted_levels[hi_idx]]
            else:
                lo_idx -= 1
                va_vol += profile[sorted_levels[lo_idx]]

        self._prior_vah = sorted_levels[hi_idx]
        self._prior_val = sorted_levels[lo_idx]
