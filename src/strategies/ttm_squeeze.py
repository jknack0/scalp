"""Strategy 4: TTM Squeeze Breakout.

Identifies volatility compression (Bollinger Bands inside Keltner Channels).
When squeeze releases after sufficient bars, enters in the breakout direction
confirmed by RSI momentum.

All entry gates are declarative filters in the YAML config, evaluated by
FilterEngine before the strategy runs.

The strategy handles:
1. Squeeze state tracking (was_squeezing, squeeze_bar_count)
2. Release detection (squeeze -> no squeeze with min bar count)
3. Direction determination (close vs Keltner bands + RSI)
4. Exit geometry computation (ATR-based target/stop, time stop)
5. Early exit: Keltner re-entry within 3 bars, adverse momentum
"""

from __future__ import annotations

from datetime import datetime, timedelta
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

logger = get_logger("ttm_squeeze")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class TTMSqueezeStrategy:
    """TTM Squeeze breakout strategy — standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "ttm_squeeze")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 3)

        # Squeeze detection config
        squeeze_cfg = config.get("squeeze", {})
        self._min_squeeze_bars: int = squeeze_cfg.get("min_squeeze_bars", 6)

        # RSI thresholds for directional confirmation
        sig_cfgs = config.get("signal_configs", {})
        rsi_cfg = sig_cfgs.get("rsi_momentum", {})
        self._rsi_long_min: float = rsi_cfg.get("long_min", 55.0)
        self._rsi_short_max: float = rsi_cfg.get("short_max", 45.0)

        exit_cfg = config.get("exit", {})
        self._exit_builder = ExitBuilder.from_yaml(exit_cfg)
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 30)

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND
        self._was_squeezing = False
        self._squeeze_bar_count = 0

    @classmethod
    def from_yaml(cls, path: str) -> TTMSqueezeStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)

        # Get Bollinger and Keltner signals for squeeze detection
        boll_result = bundle.get("bollinger")
        kelt_result = bundle.get("keltner_channel")

        if boll_result is None or kelt_result is None:
            logger.debug("blocked_missing_signals", time=now.strftime("%H:%M"),
                         has_bollinger=boll_result is not None,
                         has_keltner=kelt_result is not None)
            return None

        boll_meta = boll_result.metadata
        kelt_meta = kelt_result.metadata

        boll_upper = boll_meta.get("upper", 0.0)
        boll_lower = boll_meta.get("lower", 0.0)
        kelt_upper = kelt_meta.get("upper", 0.0)
        kelt_lower = kelt_meta.get("lower", 0.0)

        # Check current squeeze state: BB inside KC
        squeezing = (boll_upper < kelt_upper) and (boll_lower > kelt_lower)

        if squeezing:
            self._squeeze_bar_count += 1
            self._was_squeezing = True
            logger.debug("squeezing", time=now.strftime("%H:%M"),
                         squeeze_bars=self._squeeze_bar_count,
                         close=bar.close)
            return None

        # Not squeezing — check for release
        if not self._was_squeezing or self._squeeze_bar_count < self._min_squeeze_bars:
            # Reset state and skip — either never squeezed or too short
            self._was_squeezing = False
            self._squeeze_bar_count = 0
            return None

        # Squeeze release detected! Reset squeeze state
        squeeze_bars = self._squeeze_bar_count
        self._was_squeezing = False
        self._squeeze_bar_count = 0

        logger.info("squeeze_release", time=now.strftime("%H:%M"),
                     squeeze_bars=squeeze_bars, close=bar.close,
                     kelt_upper=round(kelt_upper, 2),
                     kelt_lower=round(kelt_lower, 2))

        # Check daily limit
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

        # Direction: close must be outside Keltner bands
        rsi_result = bundle.get("rsi_momentum")
        rsi_val = rsi_result.value if rsi_result else 50.0

        if bar.close > kelt_upper and rsi_val > self._rsi_long_min:
            direction = Direction.LONG
        elif bar.close < kelt_lower and rsi_val < self._rsi_short_max:
            direction = Direction.SHORT
        else:
            logger.info("blocked_no_direction", time=now.strftime("%H:%M"),
                        close=bar.close, kelt_upper=round(kelt_upper, 2),
                        kelt_lower=round(kelt_lower, 2), rsi=round(rsi_val, 1),
                        reason="close between KC bands or RSI not confirming")
            return None

        # Get signal values for logging
        rvol_result = bundle.get("relative_volume")
        rvol_val = rvol_result.value if rvol_result else 0.0

        logger.info("filters_passed", time=now.strftime("%H:%M"),
                     close=bar.close, direction=direction.value,
                     squeeze_bars=squeeze_bars,
                     kelt_upper=round(kelt_upper, 2),
                     kelt_lower=round(kelt_lower, 2),
                     rsi=round(rsi_val, 1), rvol=round(rvol_val, 1))

        # Entry at current price
        entry_price = bar.close

        # ATR for exit computation
        atr_raw = 0.0
        atr_result = bundle.get("atr")
        if atr_result is not None:
            atr_raw = atr_result.metadata.get("atr_raw", 0.0)

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

        # Confidence based on squeeze duration
        confidence = min(0.5 + squeeze_bars * 0.03, 0.9)

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
                "squeeze_bars": squeeze_bars,
                "kelt_upper": kelt_upper,
                "kelt_lower": kelt_lower,
                "kelt_mid": kelt_meta.get("mid", 0.0),
                "boll_upper": boll_upper,
                "boll_lower": boll_lower,
                "rsi": rsi_val,
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
            squeeze_bars=squeeze_bars,
            rsi=round(rsi_val, 1),
            rvol=round(rvol_val, 1),
            atr=round(atr_raw, 2),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def reset(self) -> None:
        self._signals_today = 0
        self._was_squeezing = False
        self._squeeze_bar_count = 0
