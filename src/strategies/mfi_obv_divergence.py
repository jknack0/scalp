"""Strategy 13: MFI + OBV Accumulation/Distribution Divergence.

Detects divergence between MFI (price-volume oscillator) and OBV slope:
- LONG: MFI oversold (< 20) BUT OBV slope positive (accumulation).
  Price falling but money flowing in — reversal setup.
- SHORT: MFI overbought (> 80) BUT OBV slope negative (distribution).
  Price rising but money flowing out — reversal setup.

Filters out strong trends (ADX >= 30) to avoid catching falling knives.
All other entry gates (session time, relative volume, MFI passes) are
declarative filters in the YAML config, evaluated by FilterEngine.
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

logger = get_logger("mfi_obv_divergence")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class MFIOBVDivergenceStrategy:
    """MFI + OBV divergence strategy — standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "mfi_obv_divergence")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 5)

        # ADX threshold — avoid strong trends
        sig_cfgs = config.get("signal_configs", {})
        adx_cfg = sig_cfgs.get("adx", {})
        self._adx_max: float = adx_cfg.get("threshold", 30.0)

        # MFI thresholds
        mfi_cfg = sig_cfgs.get("mfi", {})
        self._mfi_oversold: float = mfi_cfg.get("oversold", 20.0)
        self._mfi_overbought: float = mfi_cfg.get("overbought", 80.0)

        exit_cfg = config.get("exit", {})
        self._exit_builder = ExitBuilder.from_yaml(exit_cfg)
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 15)

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> MFIOBVDivergenceStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)

        if self._signals_today >= self._max_signals_per_day:
            logger.debug("blocked_daily_limit", time=now.strftime("%H:%M"),
                         signals_today=self._signals_today)
            return None

        # Run all declarative filters (session_time, mfi passes, relative_volume)
        filter_result = self._filter_engine.evaluate(bundle)
        if not filter_result.passes:
            logger.debug("blocked_filters", time=now.strftime("%H:%M"),
                         close=bar.close,
                         reasons=filter_result.block_reasons[:3])
            return None

        # Need MFI for divergence detection
        mfi_result = bundle.get("mfi")
        if mfi_result is None:
            logger.debug("blocked_no_mfi", time=now.strftime("%H:%M"))
            return None

        mfi_value = mfi_result.value

        # Need OBV for divergence detection
        obv_result = bundle.get("obv")
        if obv_result is None:
            logger.debug("blocked_no_obv", time=now.strftime("%H:%M"))
            return None

        obv_slope = obv_result.value

        # ADX check — avoid strong trends
        adx_result = bundle.get("adx")
        adx_val = adx_result.value if adx_result else 0.0
        if adx_result is not None and adx_val >= self._adx_max:
            logger.debug("blocked_adx_too_high", time=now.strftime("%H:%M"),
                         adx=round(adx_val, 1), threshold=self._adx_max)
            return None

        # Core divergence detection:
        # LONG: MFI oversold (< 20) AND OBV slope > 0 (accumulation)
        # SHORT: MFI overbought (> 80) AND OBV slope < 0 (distribution)
        if mfi_value < self._mfi_oversold and obv_slope > 0:
            direction = Direction.LONG
        elif mfi_value > self._mfi_overbought and obv_slope < 0:
            direction = Direction.SHORT
        else:
            # No divergence — MFI and OBV agree, or MFI is neutral
            logger.debug("blocked_no_divergence", time=now.strftime("%H:%M"),
                         mfi=round(mfi_value, 1), obv_slope=round(obv_slope, 2))
            return None

        # ATR for exit geometry
        atr_raw = 0.0
        atr_result = bundle.get("atr")
        if atr_result is not None:
            atr_raw = atr_result.metadata.get("atr_raw", 0.0)

        # Relative volume for metadata
        rvol_result = bundle.get("relative_volume")
        rvol_val = rvol_result.value if rvol_result else 0.0

        logger.info("divergence_detected", time=now.strftime("%H:%M"),
                     close=bar.close, direction=direction.value,
                     mfi=round(mfi_value, 1), obv_slope=round(obv_slope, 2),
                     adx=round(adx_val, 1), rvol=round(rvol_val, 1),
                     atr=round(atr_raw, 2))

        # Entry at current close
        entry_price = bar.close

        # Compute exit via ExitBuilder
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

        # Confidence based on divergence strength
        # Stronger MFI extreme + steeper OBV slope = higher confidence
        if direction == Direction.LONG:
            mfi_strength = max(0.0, (self._mfi_oversold - mfi_value) / self._mfi_oversold)
        else:
            mfi_strength = max(0.0, (mfi_value - self._mfi_overbought) / (100.0 - self._mfi_overbought))
        confidence = min(0.5 + mfi_strength * 0.3 + min(abs(obv_slope) / 1000.0, 0.2), 0.9)

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
                "mfi": mfi_value,
                "obv_slope": obv_slope,
                "adx": adx_val,
                "atr": atr_raw,
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
            mfi=round(mfi_value, 1),
            obv_slope=round(obv_slope, 2),
            adx=round(adx_val, 1),
            atr=round(atr_raw, 2),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def reset(self) -> None:
        self._signals_today = 0
