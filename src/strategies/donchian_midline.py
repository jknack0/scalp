"""Donchian Flavor 3: Midline pullback in trending regimes.

Enters on pullback to Donchian midline when trend is confirmed by
EMA crossover alignment and regime_v2 TRENDING state. Targets the
trend-side band; stops below the counter-trend band.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import yaml

from src.core.events import BarEvent
from src.core.logging import get_logger
from src.filters.filter_engine import FilterEngine
from src.models.hmm_regime import RegimeState
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle
from src.strategies.base import Direction, Signal

from zoneinfo import ZoneInfo

logger = get_logger("donchian_midline")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class DonchianMidlineStrategy:
    """Donchian midline pullback — duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "donchian_midline")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 3)

        exit_cfg = config.get("exit", {})
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 45)

        midline_cfg = config.get("midline", {})
        self._stop_buffer_atr: float = midline_cfg.get("stop_buffer_atr", 0.5)
        self._trailing_atr: float = midline_cfg.get("trailing_atr", 1.25)
        self._min_channel_width_atr: float = midline_cfg.get("min_channel_width_atr", 1.5)

        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> DonchianMidlineStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)

        if self._signals_today >= self._max_signals_per_day:
            return None

        # Universal filters: session_time, regime_v2 TRENDING, adx
        filter_result = self._filter_engine.evaluate(bundle)
        if not filter_result.passes:
            return None

        # Donchian midline proximity
        donchian = bundle.get("donchian_channel")
        if donchian is None:
            return None
        meta = donchian.metadata

        at_midline = meta.get("at_midline", False)
        if not at_midline:
            return None

        # ATR for geometry + channel width validation
        atr_result = bundle.get("atr")
        atr_raw = atr_result.metadata.get("atr_raw", 0.0) if atr_result else 0.0
        if atr_raw <= 0:
            return None

        width = meta.get("width", 0.0)
        # Channel too narrow = squeeze territory, midline is meaningless
        if atr_raw > 0 and width < self._min_channel_width_atr * atr_raw:
            return None

        # Direction from EMA crossover alignment (not crossover event, just bias)
        ema = bundle.get("ema_crossover")
        if ema is None:
            return None

        # EMA spread direction determines pullback direction
        if ema.value > 0:
            direction = Direction.LONG
        elif ema.value < 0:
            direction = Direction.SHORT
        else:
            return None

        # Midline trend slope must agree with direction
        trend_slope = meta.get("trend_slope", 0.0)
        if direction == Direction.LONG and trend_slope < -TICK_SIZE:
            return None
        if direction == Direction.SHORT and trend_slope > TICK_SIZE:
            return None

        entry_price = bar.close
        entry_upper = meta.get("entry_upper", entry_price)
        entry_lower = meta.get("entry_lower", entry_price)

        # Target: trend-side band. Stop: counter-trend band + buffer
        if direction == Direction.LONG:
            target = entry_upper
            stop = entry_lower - self._stop_buffer_atr * atr_raw
        else:
            target = entry_lower
            stop = entry_upper + self._stop_buffer_atr * atr_raw

        # Geometry sanity
        if direction == Direction.LONG:
            if not (stop < entry_price < target):
                return None
        else:
            if not (stop > entry_price > target):
                return None

        # Confidence: ADX + EMA spread strength + trend slope
        adx_result = bundle.get("adx")
        adx_val = adx_result.value if adx_result else 0.0
        ema_spread = abs(ema.value)

        confidence = 0.5
        confidence += min((adx_val - 20.0) * 0.012, 0.15)
        confidence += min(ema_spread * 0.05, 0.1)
        confidence += min(abs(trend_slope) * 0.5, 0.1)
        confidence = max(0.5, min(confidence, 0.9))

        expiry = now + timedelta(minutes=self._time_stop_minutes)

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
                "flavor": "midline",
                "mid": meta.get("mid", 0.0),
                "entry_upper": entry_upper,
                "entry_lower": entry_lower,
                "width": width,
                "trend_slope": trend_slope,
                "ema_spread": ema.value,
                "adx": adx_val,
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
            width=round(width, 2),
            trend_slope=round(trend_slope, 3),
            adx=round(adx_val, 1),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def reset(self) -> None:
        self._signals_today = 0
