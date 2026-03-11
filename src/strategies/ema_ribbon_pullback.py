"""Strategy 8: EMA Ribbon Pullback.

Trend-following pullback strategy using a 5-EMA ribbon (8, 13, 21, 34, 55).
Enters on confirmed pullbacks to the fast EMAs during fanned ribbon states.

The strategy only handles:
1. Direction from ema_ribbon signal (confirmed by sma_trend agreement)
2. Pullback state tracking (was_beyond_ema8 -> returns near ribbon)
3. Bounce confirmation (close > open for long, close < open for short)
4. Manual stop/target geometry (stop at ema_21 +/- buffer, target at 2x ATR)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from src.core.events import BarEvent
from src.core.logging import get_logger
from src.filters.filter_engine import FilterEngine
from src.models.hmm_regime import RegimeState
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle
from src.strategies.base import Direction, Signal

from zoneinfo import ZoneInfo

logger = get_logger("ema_ribbon_pullback")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class EmaRibbonPullbackStrategy:
    """EMA Ribbon Pullback strategy -- standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "ema_ribbon_pullback")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 4)

        exit_cfg = config.get("exit", {})
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 30)
        self._target_atr_multiple: float = exit_cfg.get("target", {}).get("multiplier", 2.0)
        self._stop_buffer: float = exit_cfg.get("stop", {}).get("buffer", 0.50)

        # Parse early exit conditions from YAML
        self._early_exits = exit_cfg.get("early_exit", [])

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND
        self._was_beyond_ema8 = False
        self._last_direction: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> EmaRibbonPullbackStrategy:
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
            # Reset pullback state when filters fail
            self._was_beyond_ema8 = False
            self._last_direction = None
            return None

        # Get ema_ribbon signal data
        ribbon_result = bundle.get("ema_ribbon")
        if ribbon_result is None:
            logger.debug("blocked_no_ema_ribbon", time=now.strftime("%H:%M"))
            return None
        ribbon_meta = ribbon_result.metadata

        ema_8 = ribbon_meta.get("ema_8", 0.0)
        ema_13 = ribbon_meta.get("ema_13", 0.0)
        ema_21 = ribbon_meta.get("ema_21", 0.0)
        ema_34 = ribbon_meta.get("ema_34", 0.0)
        fanned = ribbon_meta.get("fanned", False)
        trend = ribbon_meta.get("trend", "flat")
        atr = ribbon_meta.get("atr", 0.0)

        if ema_8 == 0.0 or atr == 0.0:
            logger.debug("blocked_zero_ema_atr", time=now.strftime("%H:%M"))
            return None

        # Direction from ema_ribbon
        if ribbon_result.direction == "long":
            direction = Direction.LONG
        elif ribbon_result.direction == "short":
            direction = Direction.SHORT
        else:
            self._was_beyond_ema8 = False
            self._last_direction = None
            return None

        # sma_trend must agree with direction
        sma_result = bundle.get("sma_trend")
        if sma_result is not None and sma_result.direction != "none":
            if sma_result.direction != ribbon_result.direction:
                logger.debug("blocked_sma_disagrees",
                             time=now.strftime("%H:%M"),
                             ribbon_dir=ribbon_result.direction,
                             sma_dir=sma_result.direction)
                self._was_beyond_ema8 = False
                self._last_direction = None
                return None

        # Reset pullback tracking if direction changed
        if self._last_direction != ribbon_result.direction:
            self._was_beyond_ema8 = False
            self._last_direction = ribbon_result.direction

        # Pullback state tracking
        price = bar.close
        if direction == Direction.LONG:
            # Track if price was above ema_8 (trending)
            if price > ema_8:
                self._was_beyond_ema8 = True
        else:
            # Track if price was below ema_8 (trending)
            if price < ema_8:
                self._was_beyond_ema8 = True

        # Reject if price crosses beyond ema_21 (pullback too deep)
        if direction == Direction.LONG and price < ema_21:
            logger.debug("blocked_pullback_too_deep",
                         time=now.strftime("%H:%M"),
                         price=price, ema_21=round(ema_21, 2),
                         direction="LONG")
            self._was_beyond_ema8 = False
            return None
        if direction == Direction.SHORT and price > ema_21:
            logger.debug("blocked_pullback_too_deep",
                         time=now.strftime("%H:%M"),
                         price=price, ema_21=round(ema_21, 2),
                         direction="SHORT")
            self._was_beyond_ema8 = False
            return None

        # Trigger: was_beyond_ema8 AND now within 1.0*ATR of ema_8
        near_ema8 = abs(price - ema_8) <= 1.0 * atr
        if not (self._was_beyond_ema8 and near_ema8):
            return None

        # Bounce confirmation: LONG -> close > open; SHORT -> close < open
        if direction == Direction.LONG and bar.close <= bar.open:
            logger.debug("blocked_no_bounce", time=now.strftime("%H:%M"),
                         direction="LONG", close=bar.close, open=bar.open)
            return None
        if direction == Direction.SHORT and bar.close >= bar.open:
            logger.debug("blocked_no_bounce", time=now.strftime("%H:%M"),
                         direction="SHORT", close=bar.close, open=bar.open)
            return None

        # Log filter pass with signal values
        adx_result = bundle.get("adx")
        rvol_result = bundle.get("relative_volume")
        adx_val = adx_result.value if adx_result else 0.0
        rvol_val = rvol_result.value if rvol_result else 0.0

        logger.info("pullback_triggered", time=now.strftime("%H:%M"),
                    close=bar.close, direction=direction.value,
                    ema_8=round(ema_8, 2), ema_13=round(ema_13, 2),
                    ema_21=round(ema_21, 2), ema_34=round(ema_34, 2),
                    fanned=fanned, trend=trend, atr=round(atr, 2),
                    adx=round(adx_val, 1), rvol=round(rvol_val, 1))

        # Entry at current price
        entry_price = bar.close

        # Compute target: entry +/- target_atr_multiple * ATR
        if direction == Direction.LONG:
            target = entry_price + self._target_atr_multiple * atr
            stop = ema_21 - self._stop_buffer
        else:
            target = entry_price - self._target_atr_multiple * atr
            stop = ema_21 + self._stop_buffer

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

        # Confidence from ribbon strength
        confidence = 0.7 if fanned else 0.5

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
                "ema_8": ema_8,
                "ema_13": ema_13,
                "ema_21": ema_21,
                "ema_34": ema_34,
                "ema_55": ribbon_meta.get("ema_55", 0.0),
                "fanned": fanned,
                "trend": trend,
                "atr": atr,
                "adx": adx_val,
                "rvol": rvol_val,
            },
        )

        self._signals_today += 1
        self._was_beyond_ema8 = False  # Reset after signal

        logger.info(
            "signal_generated",
            component=self.strategy_id,
            direction=direction.value,
            entry=entry_price,
            target=round(target, 2),
            stop=round(stop, 2),
            ema_8=round(ema_8, 2),
            ema_21=round(ema_21, 2),
            atr=round(atr, 2),
            adx=round(adx_val, 1),
            rvol=round(rvol_val, 1),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def check_early_exit(
        self,
        bar: BarEvent,
        bundle: SignalBundle,
        bars_in_trade: int,
        direction: Direction,
        fill_price: float,
    ) -> str | None:
        """Check if any early exit condition fires (OR logic).

        Called by the backtest engine on each bar while a position is open.
        Returns an exit reason string or None.
        """
        for cond in self._early_exits:
            reason = self._eval_early_exit(cond, bar, bundle, bars_in_trade, direction, fill_price)
            if reason is not None:
                return reason
        return None

    def _eval_early_exit(
        self,
        cond: dict,
        bar: BarEvent,
        bundle: SignalBundle,
        bars_in_trade: int,
        direction: Direction,
        fill_price: float,
    ) -> str | None:
        """Evaluate a single early exit condition."""
        exit_type = cond.get("type", "")

        if exit_type == "ribbon_unfanned":
            return self._check_ribbon_unfanned_exit(cond, bundle)

        if exit_type == "beyond_ema34":
            return self._check_beyond_ema34_exit(cond, bar, bundle, direction)

        return None

    def _check_ribbon_unfanned_exit(
        self, cond: dict, bundle: SignalBundle
    ) -> str | None:
        """Exit if EMA ribbon is no longer fanned."""
        ribbon_result = bundle.get("ema_ribbon")
        if ribbon_result is None:
            return None
        fanned = ribbon_result.metadata.get("fanned", False)
        if not fanned:
            logger.info("early_exit_ribbon_unfanned")
            return "early:ribbon_unfanned"
        return None

    def _check_beyond_ema34_exit(
        self, cond: dict, bar: BarEvent, bundle: SignalBundle, direction: Direction
    ) -> str | None:
        """Exit if price closes beyond ema_34 against direction."""
        ribbon_result = bundle.get("ema_ribbon")
        if ribbon_result is None:
            return None
        ema_34 = ribbon_result.metadata.get("ema_34", 0.0)
        if ema_34 == 0.0:
            return None

        if direction == Direction.LONG and bar.close < ema_34:
            logger.info("early_exit_beyond_ema34", direction="LONG",
                        close=bar.close, ema_34=round(ema_34, 2))
            return "early:beyond_ema34"
        if direction == Direction.SHORT and bar.close > ema_34:
            logger.info("early_exit_beyond_ema34", direction="SHORT",
                        close=bar.close, ema_34=round(ema_34, 2))
            return "early:beyond_ema34"
        return None

    def reset(self) -> None:
        self._signals_today = 0
        self._was_beyond_ema8 = False
        self._last_direction = None
