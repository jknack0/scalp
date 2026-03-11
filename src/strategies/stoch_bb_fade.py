"""Strategy 7: Stochastic + Bollinger Band Fade.

Fades extreme moves when stochastic crosses over in an extreme zone
AND price is outside Bollinger Bands, during low-ADX (range-bound) sessions.

The strategy only handles:
1. Direction from stochastic crossover (long=oversold cross, short=overbought cross)
2. Bollinger band cross-validation (long needs close <= lower, short needs close >= upper)
3. Exit geometry via ExitBuilder (sd_band target/stop)

All entry gates (session time, ADX, VWAP slope, relative volume, stochastic passes)
are declarative filters in the YAML config, evaluated by FilterEngine.
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

logger = get_logger("stoch_bb_fade")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class StochBBFadeStrategy:
    """Stochastic + Bollinger Band fade strategy — standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "stoch_bb_fade")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 4)

        exit_cfg = config.get("exit", {})
        self._exit_builder = ExitBuilder.from_yaml(exit_cfg)
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 20)

        # Parse early exit conditions from YAML
        self._early_exits = exit_cfg.get("early_exit", [])

        # ADX threshold for early exit
        self._adx_exit_threshold: float = 30.0
        for cond in self._early_exits:
            if cond.get("type") == "adx_breakout":
                self._adx_exit_threshold = cond.get("threshold", 30.0)

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> StochBBFadeStrategy:
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

        # Get stochastic result for direction
        stoch_result = bundle.get("stochastic")
        if stoch_result is None:
            logger.debug("blocked_no_stochastic", time=now.strftime("%H:%M"))
            return None

        # Direction from stochastic crossover
        if stoch_result.direction == "long":
            direction = Direction.LONG
        elif stoch_result.direction == "short":
            direction = Direction.SHORT
        else:
            logger.debug("blocked_stoch_no_direction", time=now.strftime("%H:%M"))
            return None

        # Bollinger band cross-validation
        bb_result = bundle.get("bollinger")
        if bb_result is None:
            logger.debug("blocked_no_bollinger", time=now.strftime("%H:%M"))
            return None

        bb_meta = bb_result.metadata
        bb_lower = bb_meta.get("lower", 0.0)
        bb_upper = bb_meta.get("upper", 0.0)

        if direction == Direction.LONG and bar.close > bb_lower:
            logger.debug("blocked_bb_mismatch", time=now.strftime("%H:%M"),
                         direction="LONG", close=bar.close,
                         lower=round(bb_lower, 2),
                         reason="close > lower BB")
            return None
        if direction == Direction.SHORT and bar.close < bb_upper:
            logger.debug("blocked_bb_mismatch", time=now.strftime("%H:%M"),
                         direction="SHORT", close=bar.close,
                         upper=round(bb_upper, 2),
                         reason="close < upper BB")
            return None

        # Need VWAP session data for sd_band exit geometry
        vwap_result = bundle.get("vwap_session")
        if vwap_result is None:
            logger.debug("blocked_no_vwap", time=now.strftime("%H:%M"))
            return None
        vwap_meta = vwap_result.metadata
        vwap = vwap_meta.get("vwap", 0.0)
        sd = vwap_meta.get("sd", 0.0)

        if vwap == 0.0 or sd == 0.0:
            logger.debug("blocked_vwap_zero", time=now.strftime("%H:%M"))
            return None

        # Entry at current close
        entry_price = bar.close

        # ATR for early exit checks
        atr_raw = 0.0
        atr_result = bundle.get("atr")
        if atr_result is not None:
            atr_raw = atr_result.metadata.get("atr_raw", 0.0)

        # Compute exit via ExitBuilder (sd_band target + sd_band stop)
        ctx = ExitContext(
            entry_price=entry_price,
            direction=direction.value,
            atr=atr_raw,
            vwap=vwap,
            vwap_sd=sd,
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

        # Confidence from stochastic extremity
        stoch_k = stoch_result.metadata.get("k", 50.0)
        if direction == Direction.LONG:
            confidence = min(0.6 + (20.0 - stoch_k) * 0.02, 0.9)
        else:
            confidence = min(0.6 + (stoch_k - 80.0) * 0.02, 0.9)

        # Log signal context
        adx_result = bundle.get("adx")
        rvol_result = bundle.get("relative_volume")
        adx_val = adx_result.value if adx_result else 0.0
        rvol_val = rvol_result.value if rvol_result else 0.0
        stoch_d = stoch_result.metadata.get("d", 0.0)

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
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "bb_lower": bb_lower,
                "bb_upper": bb_upper,
                "vwap": vwap,
                "sd": sd,
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
            stoch_k=round(stoch_k, 1),
            stoch_d=round(stoch_d, 1),
            bb_lower=round(bb_lower, 2),
            bb_upper=round(bb_upper, 2),
            adx=round(adx_val, 1),
            rvol=round(rvol_val, 1),
            atr=round(atr_raw, 2),
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

        if exit_type == "adverse_momentum":
            return self._check_adverse_momentum_exit(cond, bar, bundle, bars_in_trade, direction, fill_price)

        if exit_type == "adx_breakout":
            return self._check_adx_breakout_exit(cond, bundle)

        return None

    def _check_adverse_momentum_exit(
        self,
        cond: dict,
        bar: BarEvent,
        bundle: SignalBundle,
        bars_in_trade: int,
        direction: Direction,
        fill_price: float,
    ) -> str | None:
        """Exit if unrealized loss exceeds ATR multiple within first N bars."""
        max_bars = cond.get("bars", 2)
        atr_mult = cond.get("atr_multiple", 1.0)

        if bars_in_trade > max_bars:
            return None

        atr_result = bundle.get("atr")
        if atr_result is None:
            return None
        atr_raw = atr_result.metadata.get("atr_raw", 0.0)
        if atr_raw <= 0:
            return None

        if direction == Direction.LONG:
            unrealized = bar.close - fill_price
        else:
            unrealized = fill_price - bar.close

        threshold = -atr_mult * atr_raw
        if unrealized < threshold:
            logger.info("early_exit_adverse_momentum",
                        direction=direction.value,
                        bars_in_trade=bars_in_trade,
                        unrealized=round(unrealized, 2),
                        threshold=round(threshold, 2))
            return "early:adverse_momentum"
        return None

    def _check_adx_breakout_exit(
        self, cond: dict, bundle: SignalBundle
    ) -> str | None:
        """Exit if ADX crosses above threshold (trend developing, fade thesis broken)."""
        threshold = cond.get("threshold", 30.0)
        adx_result = bundle.get("adx")
        if adx_result is None:
            return None
        adx = adx_result.value
        if adx > threshold:
            logger.info("early_exit_adx_breakout", adx=round(adx, 1), threshold=threshold)
            return "early:adx_breakout"
        return None

    def reset(self) -> None:
        self._signals_today = 0
