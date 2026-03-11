"""Strategy 6: POC Magnet + Value Area Bounce.

Fades price as it approaches prior session VAH/VAL, targeting a move
back towards POC. This is the OPPOSITE of value_area_reversion (which
trades re-entry INTO VA after price opens outside). This strategy
trades the BOUNCE off VA boundaries.

Entry: price near VAH (short) or VAL (long) with RSI confirmation.
Target: halfway from entry to POC (dynamic).
Stop: ATR multiple (2.0x).
Early exit: price closes beyond VA boundary by >3 pts, or adverse momentum.
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

logger = get_logger("poc_va_bounce")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class POCVABounceStrategy:
    """POC Magnet + Value Area Bounce — standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "poc_va_bounce")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 4)

        # RSI thresholds for directional check
        sig_cfgs = config.get("signal_configs", {})
        rsi_cfg = sig_cfgs.get("rsi_momentum", {})
        self._rsi_long_max: float = rsi_cfg.get("long_threshold", 35.0)
        self._rsi_short_min: float = rsi_cfg.get("short_threshold", 65.0)

        exit_cfg = config.get("exit", {})
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 25)
        self._stop_atr_mult: float = exit_cfg.get("stop", {}).get("multiplier", 2.0)

        # Parse early exit conditions from YAML
        self._early_exits = exit_cfg.get("early_exit", [])

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # VA-specific config
        va_cfg = config.get("va_bounce", {})
        self._min_va_width: float = va_cfg.get("min_va_width", 5.0)
        self._max_distance_to_boundary: float = va_cfg.get("max_distance_to_boundary", 1.0)
        self._poc_target_fraction: float = va_cfg.get("poc_target_fraction", 0.5)
        self._beyond_va_exit_pts: float = va_cfg.get("beyond_va_exit_pts", 3.0)

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> POCVABounceStrategy:
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

        # Need value_area signal for VA levels and direction
        va_result = bundle.get("value_area")
        if va_result is None:
            logger.debug("blocked_no_value_area", time=now.strftime("%H:%M"))
            return None

        va_meta = va_result.metadata
        vah = va_meta.get("vah", 0.0)
        val = va_meta.get("val", 0.0)
        poc = va_meta.get("poc", 0.0)
        va_width = va_meta.get("va_width", 0.0)
        price_position = va_meta.get("price_position", "inside")
        distance_to_vah = va_meta.get("distance_to_vah", 0.0)
        distance_to_val = va_meta.get("distance_to_val", 0.0)

        if vah == 0.0 or val == 0.0 or poc == 0.0:
            logger.debug("blocked_va_zero", time=now.strftime("%H:%M"))
            return None

        # Reject narrow VA (no room to trade)
        if va_width < self._min_va_width:
            logger.debug("blocked_va_narrow", time=now.strftime("%H:%M"),
                         va_width=round(va_width, 2), min_width=self._min_va_width)
            return None

        # Direction from value_area signal
        va_direction_str = va_result.direction
        if va_direction_str == "long":
            direction = Direction.LONG
        elif va_direction_str == "short":
            direction = Direction.SHORT
        else:
            return None

        # RSI directional check
        rsi_result = bundle.get("rsi_momentum")
        if rsi_result is not None:
            rsi = rsi_result.value
            if direction == Direction.LONG and rsi > self._rsi_long_max:
                logger.debug("blocked_rsi_directional",
                             time=now.strftime("%H:%M"),
                             direction="LONG", rsi=round(rsi, 1),
                             threshold=self._rsi_long_max)
                return None
            if direction == Direction.SHORT and rsi < self._rsi_short_min:
                logger.debug("blocked_rsi_directional",
                             time=now.strftime("%H:%M"),
                             direction="SHORT", rsi=round(rsi, 1),
                             threshold=self._rsi_short_min)
                return None

        # Price position check — must be near the boundary
        max_dist = self._max_distance_to_boundary
        if direction == Direction.LONG:
            # Near VAL: price_position == "below" OR close to VAL
            if not (price_position == "below" or abs(distance_to_val) <= max_dist):
                logger.debug("blocked_price_position", time=now.strftime("%H:%M"),
                             direction="LONG", price_position=price_position,
                             distance_to_val=round(distance_to_val, 2))
                return None
        else:
            # Near VAH: price_position == "above" OR close to VAH
            if not (price_position == "above" or abs(distance_to_vah) <= max_dist):
                logger.debug("blocked_price_position", time=now.strftime("%H:%M"),
                             direction="SHORT", price_position=price_position,
                             distance_to_vah=round(distance_to_vah, 2))
                return None

        # Log that all gates passed
        rsi_val = rsi_result.value if rsi_result else 0.0
        adx_result = bundle.get("adx")
        adx_val = adx_result.value if adx_result else 0.0
        rvol_result = bundle.get("relative_volume")
        rvol_val = rvol_result.value if rvol_result else 0.0

        logger.info("filters_passed", time=now.strftime("%H:%M"),
                    close=bar.close, direction=direction.value,
                    vah=round(vah, 2), val=round(val, 2), poc=round(poc, 2),
                    va_width=round(va_width, 2), price_position=price_position,
                    rsi=round(rsi_val, 1), adx=round(adx_val, 1),
                    rvol=round(rvol_val, 1))

        # Entry at current price
        entry_price = bar.close

        # ATR for stop computation
        atr_raw = 0.0
        atr_result = bundle.get("atr")
        if atr_result is not None:
            atr_raw = atr_result.metadata.get("atr_raw", 0.0)

        if atr_raw <= 0:
            logger.debug("blocked_atr_zero", time=now.strftime("%H:%M"))
            return None

        # Dynamic target: halfway from entry to POC
        if direction == Direction.LONG:
            target_price = entry_price + (poc - entry_price) * self._poc_target_fraction
            stop_price = entry_price - self._stop_atr_mult * atr_raw
        else:
            target_price = entry_price - (entry_price - poc) * self._poc_target_fraction
            stop_price = entry_price + self._stop_atr_mult * atr_raw

        # Geometry sanity check
        if direction == Direction.LONG:
            if not (stop_price < entry_price < target_price):
                logger.info("blocked_geometry",
                            time=now.strftime("%H:%M"),
                            direction="LONG", entry=entry_price,
                            target=round(target_price, 2),
                            stop=round(stop_price, 2),
                            poc=round(poc, 2))
                return None
        else:
            if not (stop_price > entry_price > target_price):
                logger.info("blocked_geometry",
                            time=now.strftime("%H:%M"),
                            direction="SHORT", entry=entry_price,
                            target=round(target_price, 2),
                            stop=round(stop_price, 2),
                            poc=round(poc, 2))
                return None

        expiry = now + timedelta(minutes=self._time_stop_minutes)

        # Confidence based on proximity to boundary (closer = higher)
        nearest_dist = va_result.value
        confidence = min(0.7 + max(0.0, max_dist - nearest_dist) * 0.1, 0.9)

        signal = Signal(
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            signal_time=now,
            expiry_time=expiry,
            confidence=confidence,
            regime_state=self._current_regime,
            metadata={
                "vah": vah,
                "val": val,
                "poc": poc,
                "va_width": va_width,
                "price_position": price_position,
                "distance_to_vah": distance_to_vah,
                "distance_to_val": distance_to_val,
                "atr": atr_raw,
                "rsi": rsi_val,
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
            target=round(target_price, 2),
            stop=round(stop_price, 2),
            poc=round(poc, 2),
            vah=round(vah, 2),
            val=round(val, 2),
            atr=round(atr_raw, 2),
            rsi=round(rsi_val, 1),
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

        if exit_type == "beyond_va":
            return self._check_beyond_va_exit(cond, bar, bundle, direction)

        if exit_type == "adverse_momentum":
            return self._check_adverse_momentum_exit(cond, bar, bundle, bars_in_trade, direction, fill_price)

        return None

    def _check_beyond_va_exit(
        self, cond: dict, bar: BarEvent, bundle: SignalBundle, direction: Direction
    ) -> str | None:
        """Exit if price closes beyond VA boundary by > threshold pts against position.

        LONG (entered near VAL): exit if price drops below VAL by > threshold
        SHORT (entered near VAH): exit if price rises above VAH by > threshold
        """
        threshold = cond.get("threshold_pts", self._beyond_va_exit_pts)

        va_result = bundle.get("value_area")
        if va_result is None:
            return None

        va_meta = va_result.metadata
        vah = va_meta.get("vah", 0.0)
        val = va_meta.get("val", 0.0)

        if vah == 0.0 or val == 0.0:
            return None

        price = bar.close

        if direction == Direction.LONG and price < val - threshold:
            logger.info("early_exit_beyond_va", direction="LONG",
                        price=price, val=round(val, 2), threshold=threshold)
            return "early:beyond_va"
        if direction == Direction.SHORT and price > vah + threshold:
            logger.info("early_exit_beyond_va", direction="SHORT",
                        price=price, vah=round(vah, 2), threshold=threshold)
            return "early:beyond_va"
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

    def reset(self) -> None:
        self._signals_today = 0
