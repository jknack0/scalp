"""Strategy 1: VWAP Band Reversion with regime gating.

Mean reversion to VWAP when price deviates >=2 SD during range-bound sessions.
All entry gates are declarative filters in the YAML config, evaluated by
FilterEngine before the strategy runs.

The strategy only handles:
1. Direction determination (long/short based on VWAP deviation)
2. Exit geometry computation (target=VWAP, stop=ATR/bar extreme, time stop)

Exit logic is handled by ExitEngine when configured (exits: section in YAML).
Legacy ExitBuilder is used as fallback for Signal geometry (target/stop prices).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from src.core.events import BarEvent
from src.core.logging import get_logger
from src.exits.exit_builder import ExitBuilder, ExitContext
from src.exits.exit_engine import ExitEngine
from src.filters.filter_engine import FilterEngine
from src.models.hmm_regime import RegimeState
from src.models.regime_detector_v2 import RegimeLabel
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle
from src.strategies.base import Direction, Signal

from zoneinfo import ZoneInfo

logger = get_logger("vwap_band_reversion")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class VWAPBandReversionStrategy:
    """VWAP band reversion strategy — standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "vwap_band_reversion")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 3)

        exit_cfg = config.get("exit", {})
        self._exit_builder = ExitBuilder.from_yaml(exit_cfg)
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 30)

        # Legacy early_exit removed — ExitEngine handles all exit conditions

        # Build ExitEngine from declarative exits (new system)
        self.exit_engine = ExitEngine.from_list(config.get("exits"))

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # Map RegimeLabel -> RegimeState for Signal.regime_state field
        self._label_to_state = {
            RegimeLabel.RANGING: RegimeState.RANGE_BOUND,
            RegimeLabel.TRENDING: RegimeState.TRENDING,
            RegimeLabel.HIGH_VOL: RegimeState.VOLATILE,
        }

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> VWAPBandReversionStrategy:
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

        # Need VWAP session data for direction + exit geometry
        vwap_result = bundle.get("vwap_session")
        if vwap_result is None:
            logger.debug("blocked_no_vwap", time=now.strftime("%H:%M"))
            return None
        meta = vwap_result.metadata

        vwap = meta.get("vwap", 0.0)
        sd = meta.get("sd", 0.0)
        deviation_sd = meta.get("deviation_sd", 0.0)
        slope = meta.get("slope", 0.0)
        session_age = meta.get("session_age_bars", 0)

        if vwap == 0.0 or sd == 0.0:
            logger.debug("blocked_vwap_zero", time=now.strftime("%H:%M"))
            return None

        # Direction from VWAP deviation
        direction = Direction.LONG if deviation_sd < 0 else Direction.SHORT

        # Update regime state from V2 signal
        regime_result = bundle.get("regime_v2")
        if regime_result is not None and regime_result.metadata.get("regime_value") is not None:
            label = RegimeLabel(regime_result.metadata["regime_value"])
            self._current_regime = self._label_to_state.get(label, RegimeState.RANGE_BOUND)

        # Log that filters passed — show key signal values
        adx_result = bundle.get("adx")
        rvol_result = bundle.get("relative_volume")
        adx_val = adx_result.value if adx_result else 0.0
        rvol_val = rvol_result.value if rvol_result else 0.0

        # Regime metadata for logging
        regime_name = regime_result.metadata.get("regime", "unknown") if regime_result else "unknown"
        regime_conf = regime_result.metadata.get("confidence", 0.0) if regime_result else 0.0
        regime_size = regime_result.metadata.get("position_size", "unknown") if regime_result else "unknown"

        logger.info("filters_passed", time=now.strftime("%H:%M"),
                    close=bar.close, direction=direction.value,
                    deviation_sd=round(deviation_sd, 2),
                    vwap=round(vwap, 2), sd=round(sd, 2),
                    slope=round(slope, 4), session_age=session_age,
                    adx=round(adx_val, 1), rvol=round(rvol_val, 1),
                    regime=regime_name, regime_conf=round(regime_conf, 3),
                    regime_size=regime_size)

        # Entry at current price
        entry_price = bar.close

        # ATR for stop computation
        atr_raw = 0.0
        atr_result = bundle.get("atr")
        if atr_result is not None:
            atr_raw = atr_result.metadata.get("atr_raw", 0.0)

        # Compute exit via ExitBuilder
        ctx = ExitContext(
            entry_price=entry_price,
            direction=direction.value,
            atr=atr_raw,
            vwap=vwap,
            vwap_sd=sd,
        )
        geo = self._exit_builder.compute(ctx)

        # Also compute bar-low/high stop as alternative (whichever is wider)
        if direction == Direction.LONG:
            bar_stop = bar.low - TICK_SIZE
            stop = min(geo.stop_price, bar_stop)
        else:
            bar_stop = bar.high + TICK_SIZE
            stop = max(geo.stop_price, bar_stop)

        target = geo.target_price

        expiry = now + timedelta(minutes=self._time_stop_minutes)

        # Confidence based on deviation magnitude
        confidence = min(0.6 + abs(deviation_sd - 2.0) * 0.1, 0.9)

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
                "deviation_sd": deviation_sd,
                "vwap": vwap,
                "sd": sd,
                "slope": slope,
                "atr": atr_raw,
                "adx": adx_val,
                "rvol": rvol_val,
                "regime": regime_name,
                "regime_confidence": regime_conf,
                "regime_position_size": regime_size,
            },
        )

        valid, reason = signal.validate_geometry()
        if not valid:
            logger.info("blocked_geometry", time=now.strftime("%H:%M"),
                        direction=direction.value, entry=entry_price,
                        target=round(target, 2), stop=round(stop, 2),
                        reason=reason)
            return None

        self._signals_today += 1
        logger.info(
            "signal_generated",
            component=self.strategy_id,
            direction=direction.value,
            entry=entry_price,
            target=round(target, 2),
            stop=round(stop, 2),
            deviation_sd=round(deviation_sd, 2),
            adx=round(adx_val, 1),
            rvol=round(rvol_val, 1),
            atr=round(atr_raw, 2),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def reset(self) -> None:
        self._signals_today = 0
