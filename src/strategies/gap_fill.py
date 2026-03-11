"""Strategy 2: Opening Gap Fill with size scaling.

Fades overnight gaps between 0.10% and 0.50% during RTH session.
Waits for the first 15-minute candle to form, then enters in the
gap-fill direction on a breakout of that candle.

Entry: after first 15 minutes, fade the gap direction.
Target: prior day close (100% gap fill).
Stop: 75% of first 15-min range beyond entry, or half gap size — tighter wins.
Time stop: exit by noon if unfilled.

Expected: 60-69% WR, 1 trade/day max, R:R 1.7-2.3:1.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from src.core.events import BarEvent
from src.core.logging import get_logger
from src.models.hmm_regime import RegimeState
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle
from src.strategies.base import Direction, Signal

from zoneinfo import ZoneInfo

logger = get_logger("gap_fill")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25
POINT_VALUE = 5.0


class GapFillState(Enum):
    WAITING_FOR_OPEN = "WAITING_FOR_OPEN"
    COLLECTING_IB = "COLLECTING_IB"
    READY = "READY"
    DONE = "DONE"


class GapFillStrategy:
    """Opening gap fill strategy — standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "gap_fill")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 1)

        gap_cfg = config.get("gap", {})
        self._min_gap_pct: float = gap_cfg.get("min_gap_pct", 0.10)
        self._max_gap_pct: float = gap_cfg.get("max_gap_pct", 0.50)
        self._ib_minutes: int = gap_cfg.get("ib_minutes", 15)
        self._adx_max: float = gap_cfg.get("adx_max", 25.0)
        self._noon_cutoff: bool = gap_cfg.get("noon_cutoff", True)

        exit_cfg = config.get("exit", {})
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 150)
        self._stop_ib_pct: float = exit_cfg.get("stop_ib_pct", 0.75)

        # State
        self._state = GapFillState.WAITING_FOR_OPEN
        self._signals_today = 0
        self._prior_close: float = 0.0
        self._session_open: float = 0.0
        self._ib_high: float = 0.0
        self._ib_low: float = float("inf")
        self._ib_bars: int = 0
        self._ib_bar_count: int = 0
        self._current_date = None
        self._session_bars: list[BarEvent] = []
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> GapFillStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)
        today = now.date()

        # New day detection
        if today != self._current_date:
            self._new_day(bar, today)

        self._session_bars.append(bar)

        if self._state == GapFillState.DONE:
            return None

        if self._state == GapFillState.COLLECTING_IB:
            return self._collect_ib(bar, now, bundle)

        if self._state == GapFillState.READY:
            return self._check_entry(bar, now, bundle)

        return None

    def _new_day(self, bar: BarEvent, today) -> None:
        """Reset for new trading day, capture prior close."""
        if self._session_bars:
            self._prior_close = self._session_bars[-1].close

        self._current_date = today
        self._session_open = bar.open
        self._ib_high = bar.high
        self._ib_low = bar.low
        self._ib_bar_count = 1
        self._session_bars = []
        self._signals_today = 0

        if self._prior_close > 0:
            self._state = GapFillState.COLLECTING_IB
        else:
            self._state = GapFillState.WAITING_FOR_OPEN

    def _collect_ib(self, bar: BarEvent, now: datetime, bundle: SignalBundle) -> Signal | None:
        """Collect initial balance (first N minutes)."""
        self._ib_high = max(self._ib_high, bar.high)
        self._ib_low = min(self._ib_low, bar.low)
        self._ib_bar_count += 1

        session_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        elapsed = (now - session_start).total_seconds() / 60.0

        if elapsed >= self._ib_minutes:
            self._state = GapFillState.READY
            return self._check_entry(bar, now, bundle)

        return None

    def _check_entry(self, bar: BarEvent, now: datetime, bundle: SignalBundle) -> Signal | None:
        """Check gap fill entry conditions."""
        if self._signals_today >= self._max_signals_per_day:
            self._state = GapFillState.DONE
            return None

        # Noon cutoff
        if self._noon_cutoff and now.hour >= 12:
            self._state = GapFillState.DONE
            return None

        # Calculate gap
        gap = self._session_open - self._prior_close
        gap_pct = abs(gap / self._prior_close * 100) if self._prior_close > 0 else 0.0

        # Filter: gap must be in range
        if gap_pct < self._min_gap_pct or gap_pct > self._max_gap_pct:
            self._state = GapFillState.DONE
            return None

        # Filter: ADX < threshold (avoid strong trends that won't fill)
        adx_result = bundle.get("adx")
        if adx_result is not None and adx_result.value >= self._adx_max:
            return None

        # Direction: fade the gap
        if gap > 0:
            # Gap up → go short (expect fill back to prior close)
            direction = Direction.SHORT
        else:
            # Gap down → go long (expect fill back to prior close)
            direction = Direction.LONG

        entry_price = bar.close
        target_price = self._prior_close

        # Stop: 75% of IB range beyond entry, or half gap — tighter wins
        ib_range = self._ib_high - self._ib_low
        stop_ib = ib_range * self._stop_ib_pct
        stop_gap = abs(gap) * 0.5

        stop_distance = min(stop_ib, stop_gap)
        # Ensure minimum stop distance of 2 points
        stop_distance = max(stop_distance, 2.0)

        if direction == Direction.LONG:
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance

        # Geometry sanity check
        if direction == Direction.LONG:
            if not (stop_price < entry_price < target_price):
                self._state = GapFillState.DONE
                return None
        else:
            if not (stop_price > entry_price > target_price):
                self._state = GapFillState.DONE
                return None

        # Time stop: noon cutoff
        if self._noon_cutoff:
            expiry = now.replace(hour=12, minute=0, second=0, microsecond=0)
            if expiry <= now:
                self._state = GapFillState.DONE
                return None
        else:
            expiry = now + timedelta(minutes=self._time_stop_minutes)

        # Confidence based on gap size (smaller gaps fill more reliably)
        if gap_pct <= 0.30:
            confidence = 0.75
        else:
            confidence = 0.65

        signal = Signal(
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            signal_time=now,
            expiry_time=expiry,
            confidence=confidence,
            regime_state=self._current_regime,
            metadata={
                "gap_pct": gap_pct,
                "gap_points": gap,
                "prior_close": self._prior_close,
                "session_open": self._session_open,
                "ib_high": self._ib_high,
                "ib_low": self._ib_low,
                "ib_range": ib_range,
            },
        )

        self._signals_today += 1
        self._state = GapFillState.DONE
        logger.info(
            "signal_generated",
            component=self.strategy_id,
            direction=direction.value,
            entry=entry_price,
            target=target_price,
            stop=stop_price,
            gap_pct=round(gap_pct, 3),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def reset(self) -> None:
        self._state = GapFillState.WAITING_FOR_OPEN
        self._signals_today = 0
        self._ib_high = 0.0
        self._ib_low = float("inf")
        self._ib_bar_count = 0
