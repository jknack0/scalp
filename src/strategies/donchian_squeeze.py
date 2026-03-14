"""Donchian Flavor 4: Squeeze breakout on volatility expansion.

Detects Donchian channel width compression (percentile < threshold),
then enters on breakout with high volume confirmation. Wider stops and
targets to capture the volatility expansion move.
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

logger = get_logger("donchian_squeeze")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class DonchianSqueezeStrategy:
    """Donchian squeeze breakout — duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "donchian_squeeze")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 2)

        exit_cfg = config.get("exit", {})
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 60)

        squeeze_cfg = config.get("squeeze", {})
        self._target_atr: float = squeeze_cfg.get("target_atr", 5.0)
        self._stop_atr: float = squeeze_cfg.get("stop_atr", 2.5)
        self._width_percentile_max: float = squeeze_cfg.get("width_percentile_max", 10.0)
        self._rvol_min: float = squeeze_cfg.get("rvol_min", 2.0)

        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # Track squeeze state: we need to have BEEN in squeeze before breakout fires
        self._squeeze_active = False
        self._squeeze_bars = 0
        self._min_squeeze_bars: int = squeeze_cfg.get("min_squeeze_bars", 5)

        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> DonchianSqueezeStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)

        if self._signals_today >= self._max_signals_per_day:
            return None

        # Universal filters: session_time
        filter_result = self._filter_engine.evaluate(bundle)
        if not filter_result.passes:
            return None

        donchian = bundle.get("donchian_channel")
        if donchian is None:
            return None
        meta = donchian.metadata

        width_pct = meta.get("width_percentile", 50.0)

        # Track squeeze state: channel width in bottom percentile
        if width_pct <= self._width_percentile_max:
            if not self._squeeze_active:
                self._squeeze_active = True
                self._squeeze_bars = 0
            self._squeeze_bars += 1
            # Still in squeeze — no breakout yet
            return None
        else:
            # Width expanded out of squeeze zone
            if not self._squeeze_active or self._squeeze_bars < self._min_squeeze_bars:
                # Never was in squeeze or too brief — reset and skip
                self._squeeze_active = False
                self._squeeze_bars = 0
                return None

            # Squeeze has fired! Channel expanding from compression
            self._squeeze_active = False
            self._squeeze_bars = 0

        # Breakout must have occurred
        if not donchian.passes:
            return None

        # Volume confirmation: need elevated RVOL for squeeze breakout
        rvol_result = bundle.get("relative_volume")
        rvol_val = rvol_result.value if rvol_result else 0.0
        if rvol_val < self._rvol_min:
            return None

        # MACD histogram for momentum direction confirmation
        macd_result = bundle.get("macd")
        if macd_result is not None:
            macd_hist = macd_result.metadata.get("histogram", 0.0)
            if donchian.direction == "long" and macd_hist < 0:
                return None
            if donchian.direction == "short" and macd_hist > 0:
                return None

        # Direction from breakout
        if donchian.direction == "long":
            direction = Direction.LONG
        elif donchian.direction == "short":
            direction = Direction.SHORT
        else:
            return None

        # ATR for wider exit geometry
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

        # Confidence: squeeze tightness + RVOL
        confidence = 0.6
        confidence += min((self._rvol_min - 1.0) * 0.1, 0.1)
        confidence += min((rvol_val - self._rvol_min) * 0.05, 0.1)
        # Bollinger bandwidth squeeze confirmation
        bb = bundle.get("bollinger")
        if bb is not None:
            bandwidth = bb.metadata.get("bandwidth", 0.0)
            if bandwidth < 0.01:  # tight Bollinger = extra conviction
                confidence += 0.1
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
                "flavor": "squeeze",
                "width_percentile": width_pct,
                "rvol": rvol_val,
                "macd_hist": macd_result.metadata.get("histogram", 0.0) if macd_result else 0.0,
                "width": meta.get("width", 0.0),
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
            width_pct=round(width_pct, 1),
            rvol=round(rvol_val, 1),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def reset(self) -> None:
        self._signals_today = 0
        self._squeeze_active = False
        self._squeeze_bars = 0
