"""Donchian Flavor 1: Breakout trend-following.

Enters on Donchian channel breakout when regime is TRENDING, with ADX,
relative volume, and VWAP bias confirmation. Targets 2:1 R:R via ATR
multiples. Session-gated to avoid IB and close.
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

logger = get_logger("donchian_breakout_trend")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class DonchianBreakoutTrendStrategy:
    """Donchian breakout in trending regimes — duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "donchian_breakout_trend")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 3)

        exit_cfg = config.get("exit", {})
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 45)

        donchian_cfg = config.get("donchian", {})
        self._target_atr: float = donchian_cfg.get("target_atr", 3.5)
        self._stop_atr: float = donchian_cfg.get("stop_atr", 1.75)
        self._width_min: float = donchian_cfg.get("width_min", 2.0)
        self._width_max: float = donchian_cfg.get("width_max", 30.0)

        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> DonchianBreakoutTrendStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)

        if self._signals_today >= self._max_signals_per_day:
            return None

        # Declarative filters: session_time, donchian passes, adx, rvol, regime_v2
        filter_result = self._filter_engine.evaluate(bundle)
        if not filter_result.passes:
            return None

        # Donchian breakout direction + width gate
        donchian = bundle.get("donchian_channel")
        if donchian is None or not donchian.passes:
            return None

        meta = donchian.metadata
        width = meta.get("width", 0.0)
        if width < self._width_min or width > self._width_max:
            return None

        # Direction from breakout
        if donchian.direction == "long":
            direction = Direction.LONG
        elif donchian.direction == "short":
            direction = Direction.SHORT
        else:
            return None

        # VWAP bias alignment (strategy-level check: direction must agree)
        vwap_bias = bundle.get("vwap_bias")
        if vwap_bias is not None and vwap_bias.direction != "none":
            if direction == Direction.LONG and vwap_bias.direction != "long":
                return None
            if direction == Direction.SHORT and vwap_bias.direction != "short":
                return None

        # ATR for exit geometry
        atr_result = bundle.get("atr")
        atr_raw = atr_result.metadata.get("atr_raw", 0.0) if atr_result else 0.0
        if atr_raw <= 0:
            return None

        entry_price = bar.close
        if direction == Direction.LONG:
            target = entry_price + self._target_atr * atr_raw
            stop = entry_price - self._stop_atr * atr_raw
        else:
            target = entry_price - self._target_atr * atr_raw
            stop = entry_price + self._stop_atr * atr_raw

        # Geometry sanity
        if direction == Direction.LONG:
            if not (stop < entry_price < target):
                return None
        else:
            if not (stop > entry_price > target):
                return None

        # Confidence: ADX strength + RVOL
        adx_result = bundle.get("adx")
        rvol_result = bundle.get("relative_volume")
        adx_val = adx_result.value if adx_result else 0.0
        rvol_val = rvol_result.value if rvol_result else 0.0

        confidence = 0.5
        confidence += min((adx_val - 20.0) * 0.015, 0.2)
        confidence += min((rvol_val - 1.0) * 0.1, 0.15)
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
                "width": width,
                "entry_upper": meta.get("entry_upper", 0.0),
                "entry_lower": meta.get("entry_lower", 0.0),
                "atr": atr_raw,
                "adx": adx_val,
                "rvol": rvol_val,
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
            adx=round(adx_val, 1),
            rvol=round(rvol_val, 1),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def reset(self) -> None:
        self._signals_today = 0
