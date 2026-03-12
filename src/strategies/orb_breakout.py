"""Strategy: ORB Breakout.

First 15 min of RTH (9:30-9:45 ET) establish the opening range. Breakout
with above-average volume extends because institutional order flow creates
directional momentum.

All entry gates are declarative filters in the YAML config, evaluated by
FilterEngine before the strategy runs.

The strategy only handles:
1. Direction determination (long/short from orb_breakout signal)
2. OR range size validation (reject too-narrow or too-wide ranges)
3. Per-direction daily signal limiting (one long + one short max)
4. Exit geometry computation via ExitBuilder (or_width target, first_break stop)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from src.core.events import BarEvent
from src.core.logging import get_logger
from src.exits.exit_builder import ExitBuilder, ExitContext
from src.filters.filter_engine import FilterEngine
from src.models.hmm_regime import RegimeState
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle
from src.strategies.base import Direction, Signal

from zoneinfo import ZoneInfo

logger = get_logger("orb_breakout")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25

# OR range limits in index points
_MIN_OR_RANGE_POINTS = 2.0
_MAX_OR_RANGE_POINTS = 15.0


class ORBBreakoutStrategy:
    """ORB breakout strategy -- standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "orb_breakout")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 2)

        exit_cfg = config.get("exit", {})
        self._exit_builder = ExitBuilder.from_yaml(exit_cfg)
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 60)

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # State -- reset daily
        self._signals_today = 0
        self._longs_today = 0
        self._shorts_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> ORBBreakoutStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)

        if self._signals_today >= self._max_signals_per_day:
            logger.debug("blocked_daily_limit", time=now.strftime("%H:%M"),
                         signals_today=self._signals_today)
            return None

        # Run all declarative filters
        filter_result = self._filter_engine.evaluate(bundle)
        if not filter_result.passes:
            logger.debug("blocked_filters", time=now.strftime("%H:%M"),
                         close=bar.close,
                         reasons=filter_result.block_reasons[:3])
            return None

        # Need orb_breakout signal for direction and OR boundaries
        orb_result = bundle.get("orb_breakout")
        if orb_result is None:
            logger.debug("blocked_no_orb_breakout", time=now.strftime("%H:%M"))
            return None

        orb_meta = orb_result.metadata
        range_high = orb_meta.get("range_high", 0.0)
        range_low = orb_meta.get("range_low", 0.0)
        or_width = range_high - range_low

        # Validate OR range size in absolute points
        if or_width < _MIN_OR_RANGE_POINTS:
            logger.debug("blocked_or_too_narrow", time=now.strftime("%H:%M"),
                         or_width=round(or_width, 2),
                         min_required=_MIN_OR_RANGE_POINTS)
            return None

        if or_width > _MAX_OR_RANGE_POINTS:
            logger.debug("blocked_or_too_wide", time=now.strftime("%H:%M"),
                         or_width=round(or_width, 2),
                         max_allowed=_MAX_OR_RANGE_POINTS)
            return None

        # Direction from orb_breakout signal
        orb_direction = orb_result.direction
        if orb_direction == "long":
            direction = Direction.LONG
        elif orb_direction == "short":
            direction = Direction.SHORT
        else:
            logger.debug("blocked_no_direction", time=now.strftime("%H:%M"))
            return None

        # Only one signal per direction per day
        if direction == Direction.LONG and self._longs_today >= 1:
            logger.debug("blocked_long_already_taken", time=now.strftime("%H:%M"))
            return None
        if direction == Direction.SHORT and self._shorts_today >= 1:
            logger.debug("blocked_short_already_taken", time=now.strftime("%H:%M"))
            return None

        # Log that filters passed
        adx_result = bundle.get("adx")
        rvol_result = bundle.get("relative_volume")
        vpin_result = bundle.get("vpin")
        adx_val = adx_result.value if adx_result else 0.0
        rvol_val = rvol_result.value if rvol_result else 0.0
        vpin_val = vpin_result.value if vpin_result else 0.0

        logger.info("filters_passed", time=now.strftime("%H:%M"),
                    close=bar.close, direction=direction.value,
                    or_width=round(or_width, 2),
                    range_high=round(range_high, 2),
                    range_low=round(range_low, 2),
                    adx=round(adx_val, 1),
                    rvol=round(rvol_val, 1),
                    vpin=round(vpin_val, 3))

        # Entry at current bar close
        entry_price = bar.close

        # ATR for early exit computation
        atr_raw = 0.0
        atr_result = bundle.get("atr")
        if atr_result is not None:
            atr_raw = atr_result.metadata.get("atr_raw", 0.0)

        # first_break_extreme = opposite OR boundary
        # LONG: stop based on range_low; SHORT: stop based on range_high
        if direction == Direction.LONG:
            first_break_extreme = range_low
        else:
            first_break_extreme = range_high

        # Compute exit via ExitBuilder
        ctx = ExitContext(
            entry_price=entry_price,
            direction=direction.value,
            or_width=or_width,
            first_break_extreme=first_break_extreme,
            atr=atr_raw,
        )
        geo = self._exit_builder.compute(ctx)

        target = geo.target_price
        stop = geo.stop_price

        # Geometry sanity check
        if direction == Direction.LONG:
            if not (stop < entry_price < target):
                logger.info("blocked_geometry",
                            time=now.strftime("%H:%M"),
                            direction="LONG", entry=entry_price,
                            target=round(target, 2), stop=round(stop, 2),
                            reason="stop >= entry or entry >= target")
                return None
        else:
            if not (stop > entry_price > target):
                logger.info("blocked_geometry",
                            time=now.strftime("%H:%M"),
                            direction="SHORT", entry=entry_price,
                            target=round(target, 2), stop=round(stop, 2),
                            reason="stop <= entry or entry <= target")
                return None

        expiry = now + timedelta(minutes=self._time_stop_minutes)

        # Confidence: base 0.6, boosted by breakout distance and volume
        breakout_ticks = orb_meta.get("breakout_distance_ticks", 0.0)
        confidence = min(0.6 + breakout_ticks * 0.02 + max(0, rvol_val - 1.0) * 0.05, 0.9)

        signal = Signal(
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=entry_price,
            target_price=target,
            stop_price=stop,
            signal_time=now,
            expiry_time=expiry,
            confidence=confidence,
            regime_state=self._current_regime,
            metadata={
                "or_width": or_width,
                "range_high": range_high,
                "range_low": range_low,
                "breakout_distance_ticks": breakout_ticks,
                "atr": atr_raw,
                "adx": adx_val,
                "rvol": rvol_val,
                "vpin": vpin_val,
            },
        )

        self._signals_today += 1
        if direction == Direction.LONG:
            self._longs_today += 1
        else:
            self._shorts_today += 1

        logger.info(
            "signal_generated",
            component=self.strategy_id,
            direction=direction.value,
            entry=entry_price,
            target=round(target, 2),
            stop=round(stop, 2),
            or_width=round(or_width, 2),
            breakout_ticks=round(breakout_ticks, 1),
            adx=round(adx_val, 1),
            rvol=round(rvol_val, 1),
            atr=round(atr_raw, 2),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def reset(self) -> None:
        self._signals_today = 0
        self._longs_today = 0
        self._shorts_today = 0
