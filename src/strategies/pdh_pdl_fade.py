"""Strategy 11: Prior Day Level Fade.

Fades price at prior day high (PDH) and prior day low (PDL) with rejection
candle confirmation and RSI extremes. Targets partial reversion toward
prior day close (PDC).

All entry gates are declarative filters in the YAML config, evaluated by
FilterEngine before the strategy runs.

The strategy handles:
1. Direction from prior_day_levels signal (short near PDH, long near PDL)
2. Rejection candle confirmation (wick ratio >= 0.6)
3. RSI directional check (long needs RSI < 40, short needs RSI > 60)
4. pd_range minimum check (>= 5.0 points)
5. Dynamic target: 25% of pd_range toward PDC
6. Fixed stop: 10 ticks (2.50 points) beyond the level
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

logger = get_logger("pdh_pdl_fade")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25
STOP_TICKS = 10  # 10 ticks = 2.50 points
WICK_RATIO_MIN = 0.6
MIN_PD_RANGE = 5.0
TARGET_REVERSION_PCT = 0.25


def _tick_align(price: float) -> float:
    """Round price to nearest MES tick (0.25)."""
    return round(price / TICK_SIZE) * TICK_SIZE


class PDHPDLFadeStrategy:
    """Prior day level fade strategy -- standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "pdh_pdl_fade")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 3)

        # RSI thresholds for directional check
        sig_cfgs = config.get("signal_configs", {})
        rsi_cfg = sig_cfgs.get("rsi_momentum", {})
        self._rsi_long_max: float = rsi_cfg.get("long_threshold", 40.0)
        self._rsi_short_min: float = rsi_cfg.get("short_threshold", 60.0)

        exit_cfg = config.get("exit", {})
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 20)

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> PDHPDLFadeStrategy:
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

        # Need prior_day_levels for direction + exit geometry
        pdl_result = bundle.get("prior_day_levels")
        if pdl_result is None:
            logger.debug("blocked_no_prior_day_levels", time=now.strftime("%H:%M"))
            return None

        meta = pdl_result.metadata
        pdh = meta.get("pdh", 0.0)
        pdl = meta.get("pdl", 0.0)
        pdc = meta.get("pdc", 0.0)
        pd_range = meta.get("pd_range", 0.0)
        near_pdh = meta.get("near_pdh", False)
        near_pdl = meta.get("near_pdl", False)

        if pdh == 0.0 or pdl == 0.0:
            logger.debug("blocked_no_pdh_pdl", time=now.strftime("%H:%M"))
            return None

        # Determine direction from prior_day_levels signal
        if near_pdh:
            direction = Direction.SHORT
        elif near_pdl:
            direction = Direction.LONG
        else:
            return None

        # pd_range minimum check
        if pd_range < MIN_PD_RANGE:
            logger.debug("blocked_pd_range_too_small", time=now.strftime("%H:%M"),
                         pd_range=round(pd_range, 2))
            return None

        # Rejection candle confirmation
        bar_range = bar.high - bar.low
        if bar_range < TICK_SIZE:
            logger.debug("blocked_no_bar_range", time=now.strftime("%H:%M"))
            return None

        if direction == Direction.LONG:
            # LONG near PDL: need lower wick rejection
            lower_wick = min(bar.open, bar.close) - bar.low
            wick_ratio = lower_wick / bar_range
            if wick_ratio < WICK_RATIO_MIN or bar.close <= bar.open:
                logger.debug("blocked_no_rejection_candle",
                             time=now.strftime("%H:%M"),
                             direction="LONG",
                             wick_ratio=round(wick_ratio, 2),
                             close_gt_open=(bar.close > bar.open))
                return None
        else:
            # SHORT near PDH: need upper wick rejection
            upper_wick = bar.high - max(bar.open, bar.close)
            wick_ratio = upper_wick / bar_range
            if wick_ratio < WICK_RATIO_MIN or bar.close >= bar.open:
                logger.debug("blocked_no_rejection_candle",
                             time=now.strftime("%H:%M"),
                             direction="SHORT",
                             wick_ratio=round(wick_ratio, 2),
                             close_lt_open=(bar.close < bar.open))
                return None

        # RSI directional check
        rsi_result = bundle.get("rsi_momentum")
        rsi_val = rsi_result.value if rsi_result else 50.0

        if direction == Direction.LONG and rsi_val >= self._rsi_long_max:
            logger.info("blocked_rsi_directional",
                        time=now.strftime("%H:%M"),
                        direction="LONG", rsi=round(rsi_val, 1),
                        threshold=self._rsi_long_max,
                        reason=f"need RSI < {self._rsi_long_max}")
            return None
        if direction == Direction.SHORT and rsi_val <= self._rsi_short_min:
            logger.info("blocked_rsi_directional",
                        time=now.strftime("%H:%M"),
                        direction="SHORT", rsi=round(rsi_val, 1),
                        threshold=self._rsi_short_min,
                        reason=f"need RSI > {self._rsi_short_min}")
            return None

        # Log that all gates passed
        atr_result = bundle.get("atr")
        atr_raw = atr_result.metadata.get("atr_raw", 0.0) if atr_result else 0.0
        rvol_result = bundle.get("relative_volume")
        rvol_val = rvol_result.value if rvol_result else 0.0

        logger.info("filters_passed", time=now.strftime("%H:%M"),
                    close=bar.close, direction=direction.value,
                    pdh=round(pdh, 2), pdl=round(pdl, 2),
                    pdc=round(pdc, 2), pd_range=round(pd_range, 2),
                    rsi=round(rsi_val, 1), atr=round(atr_raw, 2),
                    rvol=round(rvol_val, 1))

        # Entry at current close
        entry_price = bar.close

        # Dynamic target: 25% of pd_range toward PDC
        if direction == Direction.LONG:
            target_raw = entry_price + (pdc - entry_price) * TARGET_REVERSION_PCT
        else:
            target_raw = entry_price - (entry_price - pdc) * TARGET_REVERSION_PCT
        target = _tick_align(target_raw)

        # Fixed stop: 10 ticks (2.50 points) beyond the level
        stop_distance = STOP_TICKS * TICK_SIZE  # 2.50 points
        if direction == Direction.LONG:
            stop = pdl - stop_distance
        else:
            stop = pdh + stop_distance

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

        # Confidence based on wick strength and proximity
        confidence = min(0.5 + wick_ratio * 0.3 + (1.0 - pdl_result.value / 2.0) * 0.1, 0.9)

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
                "pdh": pdh,
                "pdl": pdl,
                "pdc": pdc,
                "pd_range": pd_range,
                "near_pdh": near_pdh,
                "near_pdl": near_pdl,
                "wick_ratio": round(wick_ratio, 3),
                "rsi": rsi_val,
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
            pdh=round(pdh, 2),
            pdl=round(pdl, 2),
            pdc=round(pdc, 2),
            wick_ratio=round(wick_ratio, 3),
            rsi=round(rsi_val, 1),
            atr=round(atr_raw, 2),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def reset(self) -> None:
        self._signals_today = 0
