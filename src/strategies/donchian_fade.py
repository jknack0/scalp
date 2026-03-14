"""Donchian Flavor 2: Mean reversion fade from channel extremes.

Fades Donchian band touches in RANGE_BOUND regimes. Enters LONG at
lower band touch (oversold RSI), SHORT at upper band touch (overbought RSI).
Tight targets (VWAP/midline reversion) and defensive stops.
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

logger = get_logger("donchian_fade")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class DonchianFadeStrategy:
    """Donchian mean reversion fade — duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "donchian_fade")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 4)

        exit_cfg = config.get("exit", {})
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 30)

        fade_cfg = config.get("fade", {})
        self._target_atr: float = fade_cfg.get("target_atr", 1.5)
        self._stop_atr: float = fade_cfg.get("stop_atr", 1.25)
        self._rsi_oversold: float = fade_cfg.get("rsi_oversold", 30.0)
        self._rsi_overbought: float = fade_cfg.get("rsi_overbought", 70.0)
        self._use_vwap_target: bool = fade_cfg.get("use_vwap_target", True)

        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> DonchianFadeStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)

        if self._signals_today >= self._max_signals_per_day:
            return None

        # Universal filters: session_time, regime_v2, adx ceiling
        filter_result = self._filter_engine.evaluate(bundle)
        if not filter_result.passes:
            return None

        # Donchian band touch detection
        donchian = bundle.get("donchian_channel")
        if donchian is None:
            return None
        meta = donchian.metadata

        upper_touch = meta.get("upper_touch", False)
        lower_touch = meta.get("lower_touch", False)
        if not upper_touch and not lower_touch:
            return None

        # RSI confirmation (direction-dependent)
        rsi = bundle.get("rsi_momentum")
        rsi_val = rsi.value if rsi else 50.0

        if lower_touch and rsi_val < self._rsi_oversold:
            direction = Direction.LONG
        elif upper_touch and rsi_val > self._rsi_overbought:
            direction = Direction.SHORT
        else:
            return None

        # ATR for exit geometry
        atr_result = bundle.get("atr")
        atr_raw = atr_result.metadata.get("atr_raw", 0.0) if atr_result else 0.0
        if atr_raw <= 0:
            return None

        entry_price = bar.close
        mid = meta.get("mid", entry_price)

        # Target: VWAP reversion or ATR-based
        vwap_result = bundle.get("vwap_session")
        vwap = vwap_result.metadata.get("vwap", 0.0) if vwap_result else 0.0

        if direction == Direction.LONG:
            # Target toward midline/VWAP, stop below the band
            if self._use_vwap_target and vwap > entry_price:
                target = vwap
            else:
                target = entry_price + self._target_atr * atr_raw
            stop = entry_price - self._stop_atr * atr_raw
        else:
            if self._use_vwap_target and vwap < entry_price:
                target = vwap
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

        # Confidence: RSI extremity + stochastic confirmation
        stoch = bundle.get("stochastic")
        stoch_k = stoch.metadata.get("k", 50.0) if stoch else 50.0

        confidence = 0.5
        if direction == Direction.LONG:
            confidence += min((self._rsi_oversold - rsi_val) * 0.01, 0.15)
            if stoch_k < 20.0:
                confidence += 0.1
        else:
            confidence += min((rsi_val - self._rsi_overbought) * 0.01, 0.15)
            if stoch_k > 80.0:
                confidence += 0.1

        # VWAP deviation bonus
        if vwap_result:
            dev_sd = abs(vwap_result.metadata.get("deviation_sd", 0.0))
            if dev_sd >= 1.5:
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
                "flavor": "fade",
                "touch": "lower" if lower_touch else "upper",
                "rsi": rsi_val,
                "stoch_k": stoch_k,
                "vwap": vwap,
                "mid": mid,
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
            touch="lower" if lower_touch else "upper",
            rsi=round(rsi_val, 1),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def reset(self) -> None:
        self._signals_today = 0
