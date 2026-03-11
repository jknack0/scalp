"""Initial Balance signal.

Tracks the IB range (first 60 minutes of RTH, 9:30-10:30 ET) and reports
proximity to IB high/low after the IB period completes. Used by the IB Fade
strategy to detect mean-reversion opportunities at IB extremes.

Stateful: maintains _ib_high, _ib_low, _ib_complete across compute() calls.
Resets automatically on new trading day detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

from zoneinfo import ZoneInfo

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry

_ET = ZoneInfo("US/Eastern")
_IB_START = time(9, 30)
_IB_BREAK_BUFFER = 0.50  # points beyond IB to count as a breakout


@dataclass(frozen=True)
class InitialBalanceConfig:
    ib_minutes: int = 60
    proximity_points: float = 1.5


@SignalRegistry.register
class InitialBalanceSignal(SignalBase):
    """Initial Balance signal.

    value = min distance to IB high or IB low.
    passes = near IB extreme AND no breakout beyond IB.
    direction = "long" if near IB low, "short" if near IB high.
    metadata includes ib_high, ib_low, ib_mid, ib_range, proximity flags.
    """

    name = "initial_balance"

    def __init__(self, config: InitialBalanceConfig | None = None) -> None:
        self.config = config or InitialBalanceConfig()
        # Stateful: reset per session
        self._ib_high: float = -1e18
        self._ib_low: float = 1e18
        self._ib_complete: bool = False
        self._current_date: int | None = None  # ordinal
        self._session_bars: int = 0

    def _reset_session(self) -> None:
        """Reset IB state for a new trading day."""
        self._ib_high = -1e18
        self._ib_low = 1e18
        self._ib_complete = False
        self._session_bars = 0

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        if not bars:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "no_bars"},
            )

        bar = bars[-1]
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)
        bar_date = now.date().toordinal()

        # Detect new session — reset IB state
        if self._current_date is None or bar_date != self._current_date:
            self._reset_session()
            self._current_date = bar_date

        bar_time = now.time()

        # Compute IB end time from config
        ib_end_hour = 9 + (30 + self.config.ib_minutes) // 60
        ib_end_minute = (30 + self.config.ib_minutes) % 60
        ib_end = time(ib_end_hour, ib_end_minute)

        # During IB period: accumulate high/low
        if _IB_START <= bar_time < ib_end:
            self._ib_high = max(self._ib_high, bar.high)
            self._ib_low = min(self._ib_low, bar.low)
            self._session_bars += 1
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={
                    "reason": "ib_forming",
                    "ib_high": self._ib_high if self._ib_high > -1e17 else 0.0,
                    "ib_low": self._ib_low if self._ib_low < 1e17 else 0.0,
                    "session_bars": self._session_bars,
                },
            )

        # At IB end time: mark complete (also accumulate this bar)
        if bar_time >= ib_end and not self._ib_complete:
            # If we haven't seen any IB bars, IB is invalid
            if self._ib_high < -1e17:
                return SignalResult(
                    value=0.0, passes=False, direction="none",
                    metadata={"reason": "no_ib_data"},
                )
            self._ib_complete = True

        # Before IB complete (shouldn't reach here, but safety)
        if not self._ib_complete:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "ib_not_complete"},
            )

        # IB complete — compute proximity
        close = bar.close
        ib_high = self._ib_high
        ib_low = self._ib_low
        ib_mid = (ib_high + ib_low) / 2.0
        ib_range = ib_high - ib_low

        dist_high = abs(close - ib_high)
        dist_low = abs(close - ib_low)
        min_dist = min(dist_high, dist_low)

        near_ib_high = dist_high <= self.config.proximity_points
        near_ib_low = dist_low <= self.config.proximity_points

        ib_break_up = close > ib_high + _IB_BREAK_BUFFER
        ib_break_down = close < ib_low - _IB_BREAK_BUFFER

        passes = (near_ib_high or near_ib_low) and not ib_break_up and not ib_break_down

        if near_ib_high:
            direction = "short"
        elif near_ib_low:
            direction = "long"
        else:
            direction = "none"

        return SignalResult(
            value=float(min_dist),
            passes=bool(passes),
            direction=direction,
            metadata={
                "ib_high": ib_high,
                "ib_low": ib_low,
                "ib_mid": ib_mid,
                "ib_range": ib_range,
                "near_ib_high": near_ib_high,
                "near_ib_low": near_ib_low,
                "ib_break_up": ib_break_up,
                "ib_break_down": ib_break_down,
            },
        )
