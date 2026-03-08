"""Live developing volume profile with prior session reference.

Maintains a running price→volume map for the current session and
computes developing POC/VAH/VAL. Also holds prior session levels
for distance calculations used by strategy entry filters.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


class VolumeProfileTracker:
    """Incremental volume profile tracker.

    Call on_bar() with each bar to build the developing session profile.
    Call set_prior_session() to set reference levels from yesterday.
    Call reset() at session open.
    """

    def __init__(self, tick_size: float = 0.25) -> None:
        self._tick_size = tick_size
        self._price_volumes: dict[float, int] = defaultdict(int)
        self._last_price: float = 0.0

        # Prior session reference levels
        self._prior_poc: float = 0.0
        self._prior_vah: float = 0.0
        self._prior_val: float = 0.0

    def on_bar(self, close: float, volume: int) -> None:
        """Add volume at the price level (rounded to tick size)."""
        if volume <= 0:
            return
        level = self._round_to_tick(close)
        self._price_volumes[level] += volume
        self._last_price = close

    def set_prior_session(self, poc: float, vah: float, val: float) -> None:
        """Set prior session reference levels."""
        self._prior_poc = poc
        self._prior_vah = vah
        self._prior_val = val

    def reset(self) -> None:
        """Reset developing profile for a new session. Prior session levels are kept."""
        self._price_volumes.clear()
        self._last_price = 0.0

    def _round_to_tick(self, price: float) -> float:
        """Round price to nearest tick size."""
        return round(price / self._tick_size) * self._tick_size

    # ── Prior session properties ────────────────────────────────────

    @property
    def prior_poc(self) -> float:
        return self._prior_poc

    @property
    def prior_vah(self) -> float:
        return self._prior_vah

    @property
    def prior_val(self) -> float:
        return self._prior_val

    @property
    def poc_distance_ticks(self) -> float:
        """|current_price - prior_poc| in tick units."""
        if self._last_price == 0 or self._prior_poc == 0 or self._tick_size == 0:
            return 0.0
        return abs(self._last_price - self._prior_poc) / self._tick_size

    @property
    def price_above_poc(self) -> bool:
        """True if current price is above prior session POC."""
        return self._last_price > self._prior_poc

    @property
    def poc_proximity(self) -> bool:
        """True if current price is within 4 ticks of prior POC."""
        return self.poc_distance_ticks <= 4.0

    # ── Developing session properties ───────────────────────────────

    @property
    def live_poc(self) -> float:
        """Current session's developing Point of Control (max volume price)."""
        if not self._price_volumes:
            return 0.0
        return max(self._price_volumes, key=self._price_volumes.get)

    @property
    def live_vah(self) -> float:
        """Current session's developing Value Area High."""
        _, vah, _ = self._compute_live_value_area()
        return vah

    @property
    def live_val(self) -> float:
        """Current session's developing Value Area Low."""
        _, _, val = self._compute_live_value_area()
        return val

    def _compute_live_value_area(self, pct: float = 0.70) -> tuple[float, float, float]:
        """Compute POC, VAH, VAL from developing profile.

        Uses the same expand-from-POC algorithm as
        src/analysis/intraday_profile.compute_value_area().
        """
        if not self._price_volumes:
            return 0.0, 0.0, 0.0

        prices = sorted(self._price_volumes.keys())
        volumes = [self._price_volumes[p] for p in prices]
        total_vol = sum(volumes)

        if total_vol == 0:
            return 0.0, 0.0, 0.0

        # POC
        poc_idx = int(np.argmax(volumes))
        poc = prices[poc_idx]

        # Expand outward from POC to capture pct of volume
        target = total_vol * pct
        lo = poc_idx
        hi = poc_idx
        accumulated = volumes[poc_idx]

        while accumulated < target and (lo > 0 or hi < len(prices) - 1):
            vol_below = volumes[lo - 1] if lo > 0 else -1
            vol_above = volumes[hi + 1] if hi < len(prices) - 1 else -1

            if vol_below >= vol_above:
                lo -= 1
                accumulated += volumes[lo]
            else:
                hi += 1
                accumulated += volumes[hi]

        return poc, prices[hi], prices[lo]
