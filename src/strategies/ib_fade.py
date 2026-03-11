"""Strategy 14: Initial Balance Fade.

Fades touches of the IB high/low during range-bound sessions. Enters on
rejection candle confirmation (wick >= 50% of bar range + RSI extreme) and
targets the IB midpoint.

All entry gates are declarative filters in the YAML config, evaluated by
FilterEngine before the strategy runs.

The strategy only handles:
1. Direction from initial_balance signal (long near IB low, short near IB high)
2. Rejection candle confirmation (wick + RSI directional check)
3. IB range validation (3-20 points)
4. Exit geometry (target=IB mid, stop=10 ticks beyond IB extreme, 30m time stop)
5. Early exits (IB break, ADX trending)
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

logger = get_logger("ib_fade")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25
STOP_TICKS = 10  # 10 ticks = 2.50 points beyond IB extreme


class IBFadeStrategy:
    """IB Fade strategy — standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "ib_fade")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 3)

        # RSI thresholds for rejection confirmation
        sig_cfgs = config.get("signal_configs", {})
        rsi_cfg = sig_cfgs.get("rsi_momentum", {})
        self._rsi_long_max: float = rsi_cfg.get("long_threshold", 35.0)
        self._rsi_short_min: float = rsi_cfg.get("short_threshold", 65.0)

        # IB range bounds
        ib_cfg = sig_cfgs.get("initial_balance", {})
        self._ib_range_min: float = ib_cfg.get("range_min", 3.0)
        self._ib_range_max: float = ib_cfg.get("range_max", 20.0)

        exit_cfg = config.get("exit", {})
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 30)
        self._stop_ticks: int = exit_cfg.get("stop_ticks", STOP_TICKS)

        # Parse early exit conditions from YAML
        self._early_exits = exit_cfg.get("early_exit", [])

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> IBFadeStrategy:
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

        # Need initial_balance data for direction + IB levels
        ib_result = bundle.get("initial_balance")
        if ib_result is None:
            logger.debug("blocked_no_ib", time=now.strftime("%H:%M"))
            return None
        meta = ib_result.metadata

        ib_high = meta.get("ib_high", 0.0)
        ib_low = meta.get("ib_low", 0.0)
        ib_mid = meta.get("ib_mid", 0.0)
        ib_range = meta.get("ib_range", 0.0)
        near_ib_high = meta.get("near_ib_high", False)
        near_ib_low = meta.get("near_ib_low", False)

        # Validate IB range
        if ib_range < self._ib_range_min or ib_range > self._ib_range_max:
            logger.debug("blocked_ib_range", time=now.strftime("%H:%M"),
                         ib_range=round(ib_range, 2),
                         min=self._ib_range_min, max=self._ib_range_max)
            return None

        # Direction from initial_balance signal
        if near_ib_low:
            direction = Direction.LONG
        elif near_ib_high:
            direction = Direction.SHORT
        else:
            return None

        # Get RSI for rejection confirmation
        rsi_result = bundle.get("rsi_momentum")
        rsi_val = rsi_result.value if rsi_result else 50.0

        # Rejection candle confirmation
        bar_range = bar.high - bar.low
        if bar_range < 1e-10:
            logger.debug("blocked_zero_range_bar", time=now.strftime("%H:%M"))
            return None

        lower_wick = min(bar.open, bar.close) - bar.low
        upper_wick = bar.high - max(bar.open, bar.close)

        if direction == Direction.LONG:
            # Need RSI < threshold AND lower wick >= 50% of bar range
            if rsi_val >= self._rsi_long_max:
                logger.debug("blocked_rsi_long", time=now.strftime("%H:%M"),
                             rsi=round(rsi_val, 1), threshold=self._rsi_long_max)
                return None
            if lower_wick < 0.5 * bar_range:
                logger.debug("blocked_wick_long", time=now.strftime("%H:%M"),
                             lower_wick=round(lower_wick, 2),
                             bar_range=round(bar_range, 2))
                return None
        else:  # SHORT
            # Need RSI > threshold AND upper wick >= 50% of bar range
            if rsi_val <= self._rsi_short_min:
                logger.debug("blocked_rsi_short", time=now.strftime("%H:%M"),
                             rsi=round(rsi_val, 1), threshold=self._rsi_short_min)
                return None
            if upper_wick < 0.5 * bar_range:
                logger.debug("blocked_wick_short", time=now.strftime("%H:%M"),
                             upper_wick=round(upper_wick, 2),
                             bar_range=round(bar_range, 2))
                return None

        # Entry at current close
        entry_price = bar.close

        # Target = IB midpoint
        target = ib_mid

        # Stop = fixed ticks beyond IB extreme
        stop_offset = self._stop_ticks * TICK_SIZE  # 10 * 0.25 = 2.50
        if direction == Direction.LONG:
            stop = ib_low - stop_offset
        else:
            stop = ib_high + stop_offset

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

        # Confidence: closer to IB extreme = higher confidence
        ib_result_value = ib_result.value  # min distance to IB extreme
        proximity_pct = max(0.0, 1.0 - ib_result_value / 2.0)
        confidence = min(0.5 + proximity_pct * 0.3, 0.9)

        # Gather signal values for metadata
        adx_result = bundle.get("adx")
        rvol_result = bundle.get("relative_volume")
        atr_result = bundle.get("atr")
        adx_val = adx_result.value if adx_result else 0.0
        rvol_val = rvol_result.value if rvol_result else 0.0
        atr_val = atr_result.metadata.get("atr_raw", 0.0) if atr_result else 0.0

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
                "ib_high": ib_high,
                "ib_low": ib_low,
                "ib_mid": ib_mid,
                "ib_range": ib_range,
                "rsi": rsi_val,
                "adx": adx_val,
                "rvol": rvol_val,
                "atr": atr_val,
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
            ib_high=round(ib_high, 2),
            ib_low=round(ib_low, 2),
            ib_range=round(ib_range, 2),
            rsi=round(rsi_val, 1),
            adx=round(adx_val, 1),
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
            reason = self._eval_early_exit(cond, bar, bundle, direction)
            if reason is not None:
                return reason
        return None

    def _eval_early_exit(
        self,
        cond: dict,
        bar: BarEvent,
        bundle: SignalBundle,
        direction: Direction,
    ) -> str | None:
        """Evaluate a single early exit condition."""
        exit_type = cond.get("type", "")

        if exit_type == "ib_break":
            return self._check_ib_break_exit(cond, bar, bundle, direction)

        if exit_type == "adx_trending":
            return self._check_adx_trending_exit(cond, bundle)

        return None

    def _check_ib_break_exit(
        self,
        cond: dict,
        bar: BarEvent,
        bundle: SignalBundle,
        direction: Direction,
    ) -> str | None:
        """Exit if price breaks beyond IB range (thesis invalidated).

        Shorts exit if close > IB_high + buffer.
        Longs exit if close < IB_low - buffer.
        """
        buffer = cond.get("buffer_points", 1.0)
        ib_result = bundle.get("initial_balance")
        if ib_result is None:
            return None
        meta = ib_result.metadata
        ib_high = meta.get("ib_high", 0.0)
        ib_low = meta.get("ib_low", 0.0)

        if direction == Direction.SHORT and bar.close > ib_high + buffer:
            logger.info("early_exit_ib_break", direction="SHORT",
                        close=bar.close, ib_high=ib_high, buffer=buffer)
            return "early:ib_break"
        if direction == Direction.LONG and bar.close < ib_low - buffer:
            logger.info("early_exit_ib_break", direction="LONG",
                        close=bar.close, ib_low=ib_low, buffer=buffer)
            return "early:ib_break"
        return None

    def _check_adx_trending_exit(
        self,
        cond: dict,
        bundle: SignalBundle,
    ) -> str | None:
        """Exit if ADX rises above threshold (trend developing, fade thesis fails)."""
        threshold = cond.get("threshold", 28.0)
        adx_result = bundle.get("adx")
        if adx_result is None:
            return None
        if adx_result.value > threshold:
            logger.info("early_exit_adx_trending",
                        adx=round(adx_result.value, 1), threshold=threshold)
            return "early:adx_trending"
        return None

    def reset(self) -> None:
        self._signals_today = 0
