"""Strategy 1: VWAP Band Reversion with regime gating.

Mean reversion to VWAP when price deviates >=2 SD during range-bound sessions.
All entry gates are declarative filters in the YAML config, evaluated by
FilterEngine before the strategy runs.

The strategy only handles:
1. Direction determination (long/short based on VWAP deviation)
2. RSI directional check (long needs oversold, short needs overbought)
3. Exit geometry computation (target=VWAP, stop=ATR/bar extreme, time stop)
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

logger = get_logger("vwap_band_reversion")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class VWAPBandReversionStrategy:
    """VWAP band reversion strategy — standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "vwap_band_reversion")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 3)

        # RSI thresholds for directional check (read from signal_configs)
        sig_cfgs = config.get("signal_configs", {})
        rsi_cfg = sig_cfgs.get("rsi_momentum", {})
        self._rsi_long_max: float = rsi_cfg.get("long_threshold", 20.0)
        self._rsi_short_min: float = rsi_cfg.get("short_threshold", 80.0)

        exit_cfg = config.get("exit", {})
        self._exit_builder = ExitBuilder.from_yaml(exit_cfg)
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 30)

        # Parse early exit conditions from YAML
        self._early_exits = exit_cfg.get("early_exit", [])

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.LOW_VOL_RANGE

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

        # Log that filters passed — show all signal values
        rsi_result = bundle.get("rsi_momentum")
        adx_result = bundle.get("adx")
        rvol_result = bundle.get("relative_volume")
        rsi_val = rsi_result.value if rsi_result else 0.0
        adx_val = adx_result.value if adx_result else 0.0
        rvol_val = rvol_result.value if rvol_result else 0.0

        logger.info("filters_passed", time=now.strftime("%H:%M"),
                    close=bar.close, direction=direction.value,
                    deviation_sd=round(deviation_sd, 2),
                    vwap=round(vwap, 2), sd=round(sd, 2),
                    slope=round(slope, 4), session_age=session_age,
                    rsi=round(rsi_val, 1), adx=round(adx_val, 1),
                    rvol=round(rvol_val, 1))

        # RSI extreme check (directional — can't be a simple filter)
        if rsi_result is not None:
            rsi = rsi_result.value
            if direction == Direction.LONG and rsi > self._rsi_long_max:
                logger.info("blocked_rsi_directional",
                            time=now.strftime("%H:%M"),
                            direction="LONG", rsi=round(rsi, 1),
                            threshold=self._rsi_long_max,
                            reason=f"need RSI < {self._rsi_long_max}")
                return None
            if direction == Direction.SHORT and rsi < self._rsi_short_min:
                logger.info("blocked_rsi_directional",
                            time=now.strftime("%H:%M"),
                            direction="SHORT", rsi=round(rsi, 1),
                            threshold=self._rsi_short_min,
                            reason=f"need RSI > {self._rsi_short_min}")
                return None

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
            target=round(target, 2),
            stop=round(stop, 2),
            deviation_sd=round(deviation_sd, 2),
            rsi=round(rsi_val, 1),
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
        Returns an exit reason string (e.g. "early:vwap_slope") or None.
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

        if exit_type == "vwap_slope":
            return self._check_vwap_slope_exit(cond, bundle, direction)

        if exit_type == "adverse_momentum":
            return self._check_adverse_momentum_exit(cond, bar, bundle, bars_in_trade, direction, fill_price)

        if exit_type == "rsi_failure":
            return self._check_rsi_failure_exit(cond, bundle, bars_in_trade, direction)

        return None

    def _check_vwap_slope_exit(
        self, cond: dict, bundle: SignalBundle, direction: Direction
    ) -> str | None:
        """Exit if VWAP slope is moving against position direction.

        Long trades need flat/rising VWAP; short trades need flat/falling.
        If slope magnitude exceeds threshold AND direction is adverse, exit.
        """
        threshold = cond.get("threshold", 0.3)
        vwap_result = bundle.get("vwap_session")
        if vwap_result is None:
            return None
        slope = vwap_result.metadata.get("slope", 0.0)

        # Long: adverse slope is negative (VWAP falling away from entry)
        # Short: adverse slope is positive (VWAP rising away from entry)
        if direction == Direction.LONG and slope < -threshold:
            logger.info("early_exit_vwap_slope", direction="LONG", slope=round(slope, 4), threshold=threshold)
            return "early:vwap_slope"
        if direction == Direction.SHORT and slope > threshold:
            logger.info("early_exit_vwap_slope", direction="SHORT", slope=round(slope, 4), threshold=threshold)
            return "early:vwap_slope"
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
        """Exit if unrealized loss exceeds ATR multiple within first N bars.

        Catches trades that go immediately wrong — the faster you cut, the less damage.
        """
        max_bars = cond.get("bars", 2)
        atr_mult = cond.get("atr_multiple", 1.0)

        if bars_in_trade > max_bars:
            return None  # Only applies to early bars

        atr_result = bundle.get("atr")
        if atr_result is None:
            return None
        atr_raw = atr_result.metadata.get("atr_raw", 0.0)
        if atr_raw <= 0:
            return None

        # Unrealized P&L (using bar close as mark)
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

    def _check_rsi_failure_exit(
        self, cond: dict, bundle: SignalBundle, bars_in_trade: int, direction: Direction
    ) -> str | None:
        """Exit if RSI hasn't recovered after N bars in trade.

        If we entered long on oversold, RSI should be recovering. If it's
        still deeply oversold after N bars, thesis is failing.
        """
        min_bars = cond.get("bars", 3)
        if bars_in_trade < min_bars:
            return None  # Too early to judge

        rsi_result = bundle.get("rsi_momentum")
        if rsi_result is None:
            return None
        rsi = rsi_result.value

        if direction == Direction.LONG:
            long_min = cond.get("long_min", 25.0)
            if rsi < long_min:
                logger.info("early_exit_rsi_failure", direction="LONG",
                            rsi=round(rsi, 1), threshold=long_min,
                            bars_in_trade=bars_in_trade)
                return "early:rsi_failure"
        else:
            short_max = cond.get("short_max", 75.0)
            if rsi > short_max:
                logger.info("early_exit_rsi_failure", direction="SHORT",
                            rsi=round(rsi, 1), threshold=short_max,
                            bars_in_trade=bars_in_trade)
                return "early:rsi_failure"
        return None

    def reset(self) -> None:
        self._signals_today = 0
