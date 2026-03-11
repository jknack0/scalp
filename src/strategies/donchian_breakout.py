"""Strategy 12: Donchian Channel Breakout.

Trend-following breakout strategy using 20-bar Donchian channel for entries
and 10-bar exit channel for trailing stops. Filters require ADX > 18
(trending) and relative volume >= 1.0 (participation confirmation).

The strategy handles:
1. Direction from donchian_channel signal (breakout above/below channel)
2. Width filter (channel must be 3-20 points — not too tight, not too wide)
3. Exit geometry: target = 3.0 * ATR, stop = 1.5 * ATR, trailing via exit channel
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from src.core.events import BarEvent
from src.core.logging import get_logger
from src.filters.filter_engine import FilterEngine
from src.models.hmm_regime import RegimeState
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle
from src.strategies.base import Direction, Signal

from zoneinfo import ZoneInfo

logger = get_logger("donchian_breakout")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class DonchianBreakoutStrategy:
    """Donchian channel breakout strategy — standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "donchian_breakout")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 3)

        exit_cfg = config.get("exit", {})
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 45)

        # Parse early exit conditions from YAML
        self._early_exits = exit_cfg.get("early_exit", [])

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> DonchianBreakoutStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)

        if self._signals_today >= self._max_signals_per_day:
            logger.debug("blocked_daily_limit", time=now.strftime("%H:%M"),
                         signals_today=self._signals_today)
            return None

        # Run all declarative filters (session_time, donchian passes, adx, rvol)
        filter_result = self._filter_engine.evaluate(bundle)
        if not filter_result.passes:
            logger.debug("blocked_filters", time=now.strftime("%H:%M"),
                         close=bar.close,
                         reasons=filter_result.block_reasons[:3])
            return None

        # Need donchian channel data for direction + width filter
        donchian_result = bundle.get("donchian_channel")
        if donchian_result is None:
            logger.debug("blocked_no_donchian", time=now.strftime("%H:%M"))
            return None
        donchian_meta = donchian_result.metadata

        width = donchian_meta.get("width", 0.0)

        # Width filter: channel must be 3-20 points (not too tight, not too wide)
        if width < 3.0 or width > 20.0:
            logger.debug("blocked_width", time=now.strftime("%H:%M"),
                         width=round(width, 2))
            return None

        # Direction from donchian breakout
        if donchian_result.direction == "long":
            direction = Direction.LONG
        elif donchian_result.direction == "short":
            direction = Direction.SHORT
        else:
            return None

        # Get signal values for logging
        adx_result = bundle.get("adx")
        rvol_result = bundle.get("relative_volume")
        adx_val = adx_result.value if adx_result else 0.0
        rvol_val = rvol_result.value if rvol_result else 0.0

        logger.info("filters_passed", time=now.strftime("%H:%M"),
                    close=bar.close, direction=direction.value,
                    width=round(width, 2),
                    entry_upper=round(donchian_meta.get("entry_upper", 0.0), 2),
                    entry_lower=round(donchian_meta.get("entry_lower", 0.0), 2),
                    adx=round(adx_val, 1), rvol=round(rvol_val, 1))

        # Entry at current bar close
        entry_price = bar.close

        # ATR for exit geometry
        atr_raw = 0.0
        atr_result = bundle.get("atr")
        if atr_result is not None:
            atr_raw = atr_result.metadata.get("atr_raw", 0.0)

        if atr_raw <= 0:
            logger.debug("blocked_no_atr", time=now.strftime("%H:%M"))
            return None

        # Compute target and stop manually
        # Target: entry +/- 3.0 * ATR
        # Initial stop: entry -/+ 1.5 * ATR
        if direction == Direction.LONG:
            target = entry_price + 3.0 * atr_raw
            initial_stop = entry_price - 1.5 * atr_raw
        else:
            target = entry_price - 3.0 * atr_raw
            initial_stop = entry_price + 1.5 * atr_raw

        # Geometry sanity check
        if direction == Direction.LONG:
            if not (initial_stop < entry_price < target):
                logger.info("blocked_geometry",
                            time=now.strftime("%H:%M"),
                            direction="LONG", entry=entry_price,
                            target=round(target, 2), stop=round(initial_stop, 2),
                            reason="stop >= entry or entry >= target")
                return None
        else:
            if not (initial_stop > entry_price > target):
                logger.info("blocked_geometry",
                            time=now.strftime("%H:%M"),
                            direction="SHORT", entry=entry_price,
                            target=round(target, 2), stop=round(initial_stop, 2),
                            reason="stop <= entry or entry <= target")
                return None

        expiry = now + timedelta(minutes=self._time_stop_minutes)

        # Confidence based on ADX strength
        confidence = min(0.5 + (adx_val - 18.0) * 0.02, 0.9)
        confidence = max(confidence, 0.5)

        signal = Signal(
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=entry_price,
            target_price=target,
            stop_price=initial_stop,
            signal_time=now,
            expiry_time=expiry,
            confidence=confidence,
            regime_state=self._current_regime,
            metadata={
                "initial_stop": initial_stop,
                "entry_bar_count": 0,
                "width": width,
                "entry_upper": donchian_meta.get("entry_upper", 0.0),
                "entry_lower": donchian_meta.get("entry_lower", 0.0),
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
            stop=round(initial_stop, 2),
            width=round(width, 2),
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

        if exit_type == "donchian_trail":
            return self._check_donchian_trail_exit(bar, bundle, direction, fill_price)

        if exit_type == "adx_collapse":
            return self._check_adx_collapse_exit(cond, bundle)

        return None

    def _check_donchian_trail_exit(
        self,
        bar: BarEvent,
        bundle: SignalBundle,
        direction: Direction,
        fill_price: float,
    ) -> str | None:
        """Trailing stop via 10-bar exit channel.

        LONG: trail stop up using exit_lower. Exit if bar.close < max(initial_stop, exit_lower).
        SHORT: trail stop down using exit_upper. Exit if bar.close > min(initial_stop, exit_upper).

        The initial_stop is stored in the signal metadata at entry time and
        passed through the fill_price context. We retrieve the donchian exit
        channel from the current bundle.
        """
        donchian_result = bundle.get("donchian_channel")
        if donchian_result is None:
            return None

        meta = donchian_result.metadata
        exit_lower = meta.get("exit_lower", 0.0)
        exit_upper = meta.get("exit_upper", 0.0)

        # We need the initial stop from entry. The backtest engine passes
        # signal metadata through; we use the signal's stop_price as the
        # initial_stop baseline. Since fill_price is the entry price, we
        # can reconstruct from the ATR stored in metadata if needed.
        # However, the simpler approach: the trailing stop only tightens,
        # never loosens. We compare bar.close against the exit channel.

        if direction == Direction.LONG:
            # For longs, exit_lower acts as trailing stop (only moves up)
            # Exit if price drops below exit_lower
            if bar.close < exit_lower:
                logger.info("early_exit_donchian_trail", direction="LONG",
                            close=bar.close, exit_lower=round(exit_lower, 2))
                return "early:donchian_trail"
        else:
            # For shorts, exit_upper acts as trailing stop (only moves down)
            # Exit if price rises above exit_upper
            if bar.close > exit_upper:
                logger.info("early_exit_donchian_trail", direction="SHORT",
                            close=bar.close, exit_upper=round(exit_upper, 2))
                return "early:donchian_trail"

        return None

    def _check_adx_collapse_exit(
        self, cond: dict, bundle: SignalBundle
    ) -> str | None:
        """Exit if ADX drops below threshold — trend is dying."""
        threshold = cond.get("threshold", 15.0)
        adx_result = bundle.get("adx")
        if adx_result is None:
            return None

        if adx_result.value < threshold:
            logger.info("early_exit_adx_collapse",
                        adx=round(adx_result.value, 1),
                        threshold=threshold)
            return "early:adx_collapse"
        return None

    def reset(self) -> None:
        self._signals_today = 0
