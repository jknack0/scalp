"""Session time signal — returns minutes since midnight in US/Eastern.

Allows time-based filters in YAML:
    - signal: session_time
      expr: "< 630"        # before 10:30 AM ET (10*60 + 30 = 630)
      seq: 1

The value is fractional minutes (e.g. 10:30:15 = 630.25).
"""

from __future__ import annotations

from datetime import datetime, timezone

from zoneinfo import ZoneInfo

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry

_ET = ZoneInfo("US/Eastern")


@SignalRegistry.register
class SessionTimeSignal(SignalBase):
    """Returns current bar time as minutes since midnight in US/Eastern."""

    name = "session_time"

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        if not bars:
            return SignalResult(value=0.0, passes=True, direction="none")

        bar = bars[-1]
        dt = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=timezone.utc)
        et = dt.astimezone(_ET)
        minutes = et.hour * 60 + et.minute + et.second / 60.0

        return SignalResult(
            value=minutes,
            passes=True,
            direction="none",
            metadata={"time_et": et.strftime("%H:%M:%S")},
        )
