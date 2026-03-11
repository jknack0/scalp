"""Prior day levels signal (PDH / PDL / PDC).

Tracks RTH session boundaries and captures prior session's high, low, and close.
Reports proximity to PDH/PDL for level-based fade strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from zoneinfo import ZoneInfo

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry

_ET = ZoneInfo("US/Eastern")

# RTH session start: 9:30 AM ET
_RTH_START_HOUR = 9
_RTH_START_MINUTE = 30


@dataclass(frozen=True)
class PriorDayLevelsConfig:
    proximity_points: float = 1.5


@SignalRegistry.register
class PriorDayLevelsSignal(SignalBase):
    """Prior day high/low/close proximity signal.

    Maintains session state across compute() calls to detect session boundaries
    and capture prior session's high, low, and close.

    value = minimum distance to PDH or PDL (whichever is closer).
    passes = True when price is within proximity_points of PDH or PDL.
    direction = "short" if near PDH, "long" if near PDL, "none" otherwise.
    metadata: pdh, pdl, pdc, pd_range, near_pdh, near_pdl.
    """

    name = "prior_day_levels"

    def __init__(self, config: PriorDayLevelsConfig | None = None) -> None:
        self.config = config or PriorDayLevelsConfig()

        # Session tracking state
        self._current_session_date: int | None = None  # date ordinal
        self._session_high: float = 0.0
        self._session_low: float = float("inf")
        self._session_close: float = 0.0

        # Prior session levels (populated after first session boundary)
        self._pdh: float = 0.0
        self._pdl: float = 0.0
        self._pdc: float = 0.0

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        if not bars:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "no_bars"},
            )

        bar = bars[-1]
        dt = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=timezone.utc)
        et = dt.astimezone(_ET)

        # Determine which RTH session this bar belongs to.
        # Bars before 9:30 AM belong to the previous trading day's session
        # (pre-market), but we only track RTH bars for PDH/PDL.
        bar_date = et.date()
        bar_hour = et.hour
        bar_minute = et.minute

        # Check if bar is at or after RTH start
        is_rth = (bar_hour > _RTH_START_HOUR or
                  (bar_hour == _RTH_START_HOUR and bar_minute >= _RTH_START_MINUTE))

        if is_rth:
            session_ordinal = bar_date.toordinal()
        else:
            # Pre-RTH bars belong to prior session (or no session yet)
            session_ordinal = bar_date.toordinal() - 1

        # Detect session boundary
        if self._current_session_date is not None and session_ordinal > self._current_session_date:
            # New session started — save prior session levels
            if self._session_high > 0.0 and self._session_low < float("inf"):
                self._pdh = self._session_high
                self._pdl = self._session_low
                self._pdc = self._session_close

            # Reset current session tracking
            self._session_high = bar.high
            self._session_low = bar.low
            self._session_close = bar.close
            self._current_session_date = session_ordinal
        elif self._current_session_date is None:
            # First bar ever — initialize session tracking
            self._current_session_date = session_ordinal
            self._session_high = bar.high
            self._session_low = bar.low
            self._session_close = bar.close

            # Scan all bars to build prior session levels from history
            self._build_history(bars[:-1])
        else:
            # Same session — update running high/low/close
            if bar.high > self._session_high:
                self._session_high = bar.high
            if bar.low < self._session_low:
                self._session_low = bar.low
            self._session_close = bar.close

        # Check if we have valid prior day levels
        if self._pdh == 0.0 or self._pdl == 0.0:
            return SignalResult(
                value=0.0, passes=False, direction="none",
                metadata={"reason": "no_prior_session_data"},
            )

        close = bar.close
        proximity = self.config.proximity_points

        dist_pdh = abs(close - self._pdh)
        dist_pdl = abs(close - self._pdl)
        min_dist = min(dist_pdh, dist_pdl)

        near_pdh = dist_pdh <= proximity
        near_pdl = dist_pdl <= proximity
        pd_range = self._pdh - self._pdl

        if near_pdh:
            direction = "short"
        elif near_pdl:
            direction = "long"
        else:
            direction = "none"

        passes = near_pdh or near_pdl

        return SignalResult(
            value=min_dist,
            passes=passes,
            direction=direction,
            metadata={
                "pdh": self._pdh,
                "pdl": self._pdl,
                "pdc": self._pdc,
                "pd_range": pd_range,
                "near_pdh": near_pdh,
                "near_pdl": near_pdl,
            },
        )

    def _build_history(self, bars: list[BarEvent]) -> None:
        """Scan historical bars to find the most recent prior session levels.

        Called once on first compute() to bootstrap PDH/PDL/PDC from
        the bar window history.
        """
        if not bars:
            return

        # Group bars by session date
        sessions: dict[int, list[BarEvent]] = {}
        for b in bars:
            dt = datetime.fromtimestamp(b.timestamp_ns / 1e9, tz=timezone.utc)
            et = dt.astimezone(_ET)
            bd = et.date()
            bh = et.hour
            bm = et.minute

            is_rth = (bh > _RTH_START_HOUR or
                      (bh == _RTH_START_HOUR and bm >= _RTH_START_MINUTE))
            if is_rth:
                ordinal = bd.toordinal()
            else:
                ordinal = bd.toordinal() - 1

            if ordinal not in sessions:
                sessions[ordinal] = []
            sessions[ordinal].append(b)

        # Find the session just before the current one
        prior_ordinals = sorted(
            o for o in sessions if o < self._current_session_date
        )
        if not prior_ordinals:
            return

        prior_session = sessions[prior_ordinals[-1]]
        self._pdh = max(b.high for b in prior_session)
        self._pdl = min(b.low for b in prior_session)
        self._pdc = prior_session[-1].close

        # Also update current session from any bars we've seen
        if self._current_session_date in sessions:
            curr = sessions[self._current_session_date]
            self._session_high = max(b.high for b in curr)
            self._session_low = min(b.low for b in curr)
            self._session_close = curr[-1].close
