"""Strategy 9: Micro-Pullback Scalp.

On strong trend days, 1-minute bars show repeated micro-pullbacks to the 8 EMA.
Each touch with a rejection candle is a scalp entry. Hold for 4-8 ticks.

All entry gates are declarative filters in the YAML config, evaluated by
FilterEngine before the strategy runs.

The strategy only handles:
1. Direction determination (from ema_crossover alignment + sma_trend confirmation)
2. Pullback-to-EMA detection (bar low/high within 0.50 points of 8 EMA)
3. Rejection candle check (close vs open)
4. Exit geometry computation (fixed ticks target/stop, 5-min time stop)
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

logger = get_logger("micro_pullback")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25

# Pullback proximity threshold: bar must come within this many points of 8 EMA
_PULLBACK_PROXIMITY = 0.50


class MicroPullbackStrategy:
    """Micro-pullback scalp strategy — standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "micro_pullback")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 10)

        exit_cfg = config.get("exit", {})
        self._exit_builder = ExitBuilder.from_yaml(exit_cfg)
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 5)

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> MicroPullbackStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)

        if self._signals_today >= self._max_signals_per_day:
            logger.debug("blocked_daily_limit", time=now.strftime("%H:%M"),
                         signals_today=self._signals_today)
            return None

        # Run all declarative filters (session_time, adx, relative_volume)
        filter_result = self._filter_engine.evaluate(bundle)
        if not filter_result.passes:
            logger.debug("blocked_filters", time=now.strftime("%H:%M"),
                         close=bar.close,
                         reasons=filter_result.block_reasons[:3])
            return None

        # Need ema_crossover for direction + 8 EMA value
        ema_result = bundle.get("ema_crossover")
        if ema_result is None:
            logger.debug("blocked_no_ema", time=now.strftime("%H:%M"))
            return None

        ema_direction = ema_result.direction
        if ema_direction == "none":
            logger.debug("blocked_ema_no_direction", time=now.strftime("%H:%M"))
            return None

        # EMAs must be aligned, not actively crossing
        if ema_result.passes:
            logger.debug("blocked_ema_crossing", time=now.strftime("%H:%M"),
                         crossed=ema_result.metadata.get("crossed"))
            return None

        # sma_trend must confirm the same direction
        sma_result = bundle.get("sma_trend")
        if sma_result is None:
            logger.debug("blocked_no_sma", time=now.strftime("%H:%M"))
            return None

        sma_direction = sma_result.direction
        if sma_direction != ema_direction:
            logger.debug("blocked_direction_mismatch", time=now.strftime("%H:%M"),
                         ema_dir=ema_direction, sma_dir=sma_direction)
            return None

        # Set trade direction from EMA alignment
        direction = Direction.LONG if ema_direction == "long" else Direction.SHORT

        # Get 8 EMA value from ema_crossover metadata
        fast_ema = ema_result.metadata.get("fast_ema", 0.0)
        if fast_ema == 0.0:
            logger.debug("blocked_no_fast_ema", time=now.strftime("%H:%M"))
            return None

        # Pullback-to-EMA detection + rejection candle
        if direction == Direction.LONG:
            # Bar low must come within proximity of 8 EMA AND close > open (bullish rejection)
            distance_to_ema = abs(bar.low - fast_ema)
            if distance_to_ema > _PULLBACK_PROXIMITY:
                return None
            if bar.close <= bar.open:
                logger.debug("blocked_no_rejection", time=now.strftime("%H:%M"),
                             direction="LONG", close=bar.close, open=bar.open)
                return None
        else:
            # Bar high must come within proximity of 8 EMA AND close < open (bearish rejection)
            distance_to_ema = abs(bar.high - fast_ema)
            if distance_to_ema > _PULLBACK_PROXIMITY:
                return None
            if bar.close >= bar.open:
                logger.debug("blocked_no_rejection", time=now.strftime("%H:%M"),
                             direction="SHORT", close=bar.close, open=bar.open)
                return None

        # Log that all conditions passed
        adx_result = bundle.get("adx")
        rvol_result = bundle.get("relative_volume")
        atr_result = bundle.get("atr")
        adx_val = adx_result.value if adx_result else 0.0
        rvol_val = rvol_result.value if rvol_result else 0.0
        atr_raw = atr_result.metadata.get("atr_raw", 0.0) if atr_result else 0.0

        logger.info("filters_passed", time=now.strftime("%H:%M"),
                    close=bar.close, direction=direction.value,
                    fast_ema=round(fast_ema, 2),
                    distance_to_ema=round(distance_to_ema, 2),
                    adx=round(adx_val, 1), rvol=round(rvol_val, 1))

        # Entry at bar close
        entry_price = bar.close

        # Compute exit via ExitBuilder (fixed_ticks for both target and stop)
        ctx = ExitContext(
            entry_price=entry_price,
            direction=direction.value,
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

        # Confidence: base 0.7, boost for stronger ADX
        confidence = min(0.7 + (adx_val - 25.0) * 0.005, 0.9) if adx_val > 25.0 else 0.7

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
                "fast_ema": fast_ema,
                "distance_to_ema": distance_to_ema,
                "ema_spread": ema_result.value,
                "adx": adx_val,
                "rvol": rvol_val,
                "atr": atr_raw,
            },
        )

        self._signals_today += 1
        logger.info(
            "signal_generated",
            component=self.strategy_id,
            direction=direction.value,
            entry=entry_price,
            target=round(target, 2),
            stop=round(stop, 2),
            fast_ema=round(fast_ema, 2),
            distance=round(distance_to_ema, 2),
            adx=round(adx_val, 1),
            rvol=round(rvol_val, 1),
            atr=round(atr_raw, 2),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def reset(self) -> None:
        self._signals_today = 0
