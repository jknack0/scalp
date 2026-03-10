"""Strategy 3: Value Area Reversion (80% Rule).

When price opens outside prior session's Value Area and re-enters,
there's ~70% probability it traverses the full Value Area.

Entry: price re-enters VA and holds inside for 2 consecutive 30-min periods.
Target: POC (primary) or opposite VA edge (full).
Stop: just outside the VA boundary where price re-entered.
Time stop: exit if no progress after 2 hours.

Expected: ~70% WR, 1 setup/day when conditions align, balanced markets.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.core.events import BarEvent
from src.core.logging import get_logger
from src.models.hmm_regime import RegimeState
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle
from src.strategies.base import Direction, Signal

from zoneinfo import ZoneInfo

logger = get_logger("value_area_reversion")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25
POINT_VALUE = 5.0

# Value Area = 70% of total volume
VA_PCT = 0.70


class VAState(Enum):
    WAITING = "WAITING"           # Waiting for session to start
    CHECKING_OPEN = "CHECKING_OPEN"  # Checking if open is outside prior VA
    WATCHING = "WATCHING"         # Watching for re-entry into VA
    CONFIRMING = "CONFIRMING"     # Re-entered, waiting for 2× 30-min acceptance
    DONE = "DONE"                 # Signal generated or conditions invalidated


class ValueAreaReversionStrategy:
    """Value Area reversion strategy — standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "value_area_reversion")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 1)

        va_cfg = config.get("value_area", {})
        self._confirmation_minutes: int = va_cfg.get("confirmation_minutes", 60)
        self._price_step: float = va_cfg.get("price_step", 0.25)
        self._adx_max: float = va_cfg.get("adx_max", 20.0)
        self._stop_buffer_ticks: int = va_cfg.get("stop_buffer_ticks", 4)
        self._min_va_width: float = va_cfg.get("min_va_width", 5.0)

        exit_cfg = config.get("exit", {})
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 120)

        # State
        self._state = VAState.WAITING
        self._signals_today = 0
        self._current_date = None
        self._current_regime = RegimeState.LOW_VOL_RANGE

        # Prior session profile
        self._prior_vah: float = 0.0
        self._prior_val: float = 0.0
        self._prior_poc: float = 0.0

        # Current session tracking
        self._session_bars: list[BarEvent] = []
        self._prior_session_bars: list[BarEvent] = []
        self._reentry_direction: Direction | None = None
        self._reentry_time: datetime | None = None
        self._inside_va_bars: int = 0

    @classmethod
    def from_yaml(cls, path: str) -> ValueAreaReversionStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)
        today = now.date()

        # New day detection
        if today != self._current_date:
            self._new_day(bar, today, now)

        self._session_bars.append(bar)

        if self._state == VAState.DONE:
            return None

        if self._state == VAState.CHECKING_OPEN:
            return self._check_open(bar, now, bundle)

        if self._state == VAState.WATCHING:
            return self._watch_reentry(bar, now, bundle)

        if self._state == VAState.CONFIRMING:
            return self._confirm_acceptance(bar, now, bundle)

        return None

    def _new_day(self, bar: BarEvent, today, now: datetime) -> None:
        """Reset for new day, compute prior session's value area."""
        if self._session_bars:
            self._prior_session_bars = list(self._session_bars)
            self._compute_value_area()

        self._current_date = today
        self._session_bars = []
        self._signals_today = 0
        self._reentry_direction = None
        self._reentry_time = None
        self._inside_va_bars = 0

        if self._prior_vah > 0 and self._prior_val > 0:
            self._state = VAState.CHECKING_OPEN
        else:
            self._state = VAState.WAITING

    def _compute_value_area(self) -> None:
        """Compute VAH, VAL, POC from prior session's bars using TPO-style volume profile."""
        if not self._prior_session_bars:
            return

        step = self._price_step

        # Build volume profile: price_level → total_volume
        profile: dict[float, float] = defaultdict(float)
        for bar in self._prior_session_bars:
            # Distribute volume across bar range
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
        self._prior_poc = max(profile, key=profile.get)

        # Value Area: expand from POC until 70% of volume captured
        sorted_levels = sorted(profile.keys())
        poc_idx = sorted_levels.index(self._prior_poc)

        va_vol = profile[self._prior_poc]
        lo_idx = poc_idx
        hi_idx = poc_idx

        while va_vol / total_vol < VA_PCT:
            # Look one level above and below, add whichever has more volume
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

        logger.info(
            "value_area_computed",
            component=self.strategy_id,
            poc=self._prior_poc,
            vah=self._prior_vah,
            val=self._prior_val,
            width=round(self._prior_vah - self._prior_val, 2),
        )

    def _check_open(self, bar: BarEvent, now: datetime, bundle: SignalBundle) -> Signal | None:
        """Check if session opens outside the prior VA."""
        va_width = self._prior_vah - self._prior_val
        if va_width < self._min_va_width:
            self._state = VAState.DONE
            return None

        price = bar.close

        if price > self._prior_vah:
            # Opened above VA — watch for re-entry down into VA
            self._reentry_direction = Direction.SHORT
            self._state = VAState.WATCHING
        elif price < self._prior_val:
            # Opened below VA — watch for re-entry up into VA
            self._reentry_direction = Direction.LONG
            self._state = VAState.WATCHING
        else:
            # Opened inside VA — no setup today
            self._state = VAState.DONE

        return None

    def _watch_reentry(self, bar: BarEvent, now: datetime, bundle: SignalBundle) -> Signal | None:
        """Watch for price to re-enter the Value Area."""
        # Filter: ADX must be low (balanced market)
        adx_result = bundle.get("adx")
        if adx_result is not None and adx_result.value >= self._adx_max:
            return None  # Keep watching, don't kill

        price = bar.close

        inside_va = self._prior_val <= price <= self._prior_vah

        if inside_va:
            self._reentry_time = now
            self._inside_va_bars = 1
            self._state = VAState.CONFIRMING

        # If price moves further away from VA, give up after 2 hours
        session_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if (now - session_start).total_seconds() > 7200:
            self._state = VAState.DONE

        return None

    def _confirm_acceptance(self, bar: BarEvent, now: datetime, bundle: SignalBundle) -> Signal | None:
        """Confirm price stays inside VA for confirmation period."""
        if self._signals_today >= self._max_signals_per_day:
            self._state = VAState.DONE
            return None

        price = bar.close
        inside_va = self._prior_val <= price <= self._prior_vah

        if not inside_va:
            # Price broke back out — reset to watching
            self._state = VAState.WATCHING
            self._inside_va_bars = 0
            return None

        self._inside_va_bars += 1

        # Check if enough time has passed since re-entry
        if self._reentry_time is None:
            return None

        elapsed = (now - self._reentry_time).total_seconds() / 60.0
        if elapsed < self._confirmation_minutes:
            return None

        # Filter: ADX must still be low
        adx_result = bundle.get("adx")
        if adx_result is not None and adx_result.value >= self._adx_max:
            return None

        # All conditions met — generate signal
        direction = self._reentry_direction
        if direction is None:
            self._state = VAState.DONE
            return None

        entry_price = bar.close

        # Target: POC (primary target)
        target_price = self._prior_poc

        # Stop: just outside the VA boundary where price re-entered
        buffer = self._stop_buffer_ticks * TICK_SIZE
        if direction == Direction.LONG:
            stop_price = self._prior_val - buffer
        else:
            stop_price = self._prior_vah + buffer

        # Geometry check
        if direction == Direction.LONG:
            if not (stop_price < entry_price < target_price):
                # Entry already past POC — use opposite VA edge as target
                target_price = self._prior_vah
                if not (stop_price < entry_price < target_price):
                    self._state = VAState.DONE
                    return None
        else:
            if not (stop_price > entry_price > target_price):
                target_price = self._prior_val
                if not (stop_price > entry_price > target_price):
                    self._state = VAState.DONE
                    return None

        expiry = now + timedelta(minutes=self._time_stop_minutes)

        signal = Signal(
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            signal_time=now,
            expiry_time=expiry,
            confidence=0.70,
            regime_state=self._current_regime,
            metadata={
                "prior_poc": self._prior_poc,
                "prior_vah": self._prior_vah,
                "prior_val": self._prior_val,
                "va_width": round(self._prior_vah - self._prior_val, 2),
                "confirmation_bars": self._inside_va_bars,
                "confirmation_minutes": elapsed,
            },
        )

        self._signals_today += 1
        self._state = VAState.DONE
        logger.info(
            "signal_generated",
            component=self.strategy_id,
            direction=direction.value,
            entry=entry_price,
            target=target_price,
            stop=stop_price,
            poc=self._prior_poc,
            vah=self._prior_vah,
            val=self._prior_val,
            confidence=0.70,
            signal_id=signal.id,
        )
        return signal

    def reset(self) -> None:
        self._state = VAState.WAITING
        self._signals_today = 0
        self._reentry_direction = None
        self._reentry_time = None
        self._inside_va_bars = 0
