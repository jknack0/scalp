"""Strategy 10: Volatility Regime Switcher (HMM-Gated).

Meta-strategy that adapts based on ADX regime:
- Mode A (ADX < 22): Mean reversion to VWAP on extreme deviation + RSI.
- Mode B (ADX >= 22): Momentum via EMA crossover aligned with recent bars.

All entry gates are declarative filters in the YAML config, evaluated by
FilterEngine before the strategy runs. Exit geometry is computed manually
in on_bar because target/stop types differ per mode.

The strategy stores "mode" in signal metadata so check_early_exit knows
which exit rules to apply.
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

logger = get_logger("regime_switcher")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class RegimeSwitcherStrategy:
    """Volatility Regime Switcher — standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "regime_switcher")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 5)

        # Regime thresholds
        regime_cfg = config.get("regime", {})
        self._adx_threshold: float = regime_cfg.get("adx_threshold", 22.0)
        self._momentum_confirm_bars: int = regime_cfg.get("momentum_confirm_bars", 3)

        # Exit configs per mode
        exit_cfg = config.get("exit", {})

        mode_a = exit_cfg.get("mode_a", {})
        self._mode_a_time_stop: int = mode_a.get("time_stop_minutes", 15)
        self._mode_a_stop_atr_mult: float = mode_a.get("stop_atr_multiple", 2.0)
        self._mode_a_confidence: float = mode_a.get("confidence", 0.7)
        self._mode_a_deviation_threshold: float = mode_a.get("vwap_deviation_threshold", 2.0)
        self._mode_a_rsi_long_max: float = mode_a.get("rsi_long_max", 25.0)
        self._mode_a_rsi_short_min: float = mode_a.get("rsi_short_min", 75.0)

        mode_b = exit_cfg.get("mode_b", {})
        self._mode_b_time_stop: int = mode_b.get("time_stop_minutes", 25)
        self._mode_b_target_atr_mult: float = mode_b.get("target_atr_multiple", 2.0)
        self._mode_b_stop_atr_mult: float = mode_b.get("stop_atr_multiple", 1.0)
        self._mode_b_confidence: float = mode_b.get("confidence", 0.55)

        # Early exit thresholds
        early_cfg = exit_cfg.get("early_exit", {})
        self._ee_vwap_slope_threshold: float = early_cfg.get("vwap_slope_threshold", 0.4)
        self._ee_adx_trend_threshold: float = early_cfg.get("adx_trend_threshold", 25.0)
        self._ee_adx_range_threshold: float = early_cfg.get("adx_range_threshold", 18.0)
        self._ee_adverse_bars: int = early_cfg.get("adverse_momentum_bars", 2)
        self._ee_adverse_atr_mult: float = early_cfg.get("adverse_momentum_atr_multiple", 1.5)

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND
        self._recent_closes: list[float] = []

    @classmethod
    def from_yaml(cls, path: str) -> RegimeSwitcherStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)

        # Track recent closes for momentum confirmation
        self._recent_closes.append(bar.close)
        if len(self._recent_closes) > self._momentum_confirm_bars + 1:
            self._recent_closes = self._recent_closes[-(self._momentum_confirm_bars + 1):]

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

        # Get ADX to determine mode
        adx_result = bundle.get("adx")
        if adx_result is None:
            logger.debug("blocked_no_adx", time=now.strftime("%H:%M"))
            return None
        adx_val = adx_result.value

        # Get ATR for exit geometry
        atr_result = bundle.get("atr")
        atr_raw = 0.0
        if atr_result is not None:
            atr_raw = atr_result.metadata.get("atr_raw", 0.0)
        if atr_raw <= 0:
            logger.debug("blocked_no_atr", time=now.strftime("%H:%M"))
            return None

        # Route to appropriate mode
        if adx_val < self._adx_threshold:
            return self._mode_a_reversion(bar, bundle, now, adx_val, atr_raw)
        else:
            return self._mode_b_momentum(bar, bundle, now, adx_val, atr_raw)

    def _mode_a_reversion(
        self,
        bar: BarEvent,
        bundle: SignalBundle,
        now: datetime,
        adx_val: float,
        atr_raw: float,
    ) -> Signal | None:
        """Mode A: Mean reversion when ADX < threshold.

        Requires VWAP deviation >= 2.0 SD AND RSI(5) extreme.
        Target = VWAP, stop = entry +/- 2.0*ATR.
        """
        # Need VWAP session data
        vwap_result = bundle.get("vwap_session")
        if vwap_result is None:
            logger.debug("blocked_no_vwap", time=now.strftime("%H:%M"), mode="reversion")
            return None
        meta = vwap_result.metadata

        vwap = meta.get("vwap", 0.0)
        sd = meta.get("sd", 0.0)
        deviation_sd = meta.get("deviation_sd", 0.0)
        slope = meta.get("slope", 0.0)

        if vwap == 0.0 or sd == 0.0:
            logger.debug("blocked_vwap_zero", time=now.strftime("%H:%M"), mode="reversion")
            return None

        # Check deviation threshold
        if abs(deviation_sd) < self._mode_a_deviation_threshold:
            logger.debug("blocked_deviation", time=now.strftime("%H:%M"),
                         deviation_sd=round(deviation_sd, 2),
                         threshold=self._mode_a_deviation_threshold)
            return None

        # Direction from VWAP deviation
        direction = Direction.LONG if deviation_sd < 0 else Direction.SHORT

        # RSI extreme check
        rsi_result = bundle.get("rsi_momentum")
        if rsi_result is None:
            logger.debug("blocked_no_rsi", time=now.strftime("%H:%M"), mode="reversion")
            return None
        rsi_val = rsi_result.value

        if direction == Direction.LONG and rsi_val > self._mode_a_rsi_long_max:
            logger.info("blocked_rsi_directional", time=now.strftime("%H:%M"),
                        mode="reversion", direction="LONG",
                        rsi=round(rsi_val, 1), threshold=self._mode_a_rsi_long_max)
            return None
        if direction == Direction.SHORT and rsi_val < self._mode_a_rsi_short_min:
            logger.info("blocked_rsi_directional", time=now.strftime("%H:%M"),
                        mode="reversion", direction="SHORT",
                        rsi=round(rsi_val, 1), threshold=self._mode_a_rsi_short_min)
            return None

        # Entry at current close
        entry_price = bar.close

        # Target = VWAP
        target = vwap

        # Stop = entry -/+ 2.0 * ATR
        stop_distance = self._mode_a_stop_atr_mult * atr_raw
        if direction == Direction.LONG:
            stop = entry_price - stop_distance
        else:
            stop = entry_price + stop_distance

        # Geometry sanity check
        if direction == Direction.LONG:
            if not (stop < entry_price < target):
                logger.info("blocked_geometry", time=now.strftime("%H:%M"),
                            mode="reversion", direction="LONG",
                            entry=entry_price, target=round(target, 2),
                            stop=round(stop, 2))
                return None
        else:
            if not (stop > entry_price > target):
                logger.info("blocked_geometry", time=now.strftime("%H:%M"),
                            mode="reversion", direction="SHORT",
                            entry=entry_price, target=round(target, 2),
                            stop=round(stop, 2))
                return None

        expiry = now + timedelta(minutes=self._mode_a_time_stop)

        rvol_result = bundle.get("relative_volume")
        rvol_val = rvol_result.value if rvol_result else 0.0

        signal = Signal(
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=entry_price,
            target_price=target,
            stop_price=stop,
            signal_time=now,
            expiry_time=expiry,
            confidence=self._mode_a_confidence,
            regime_state=self._current_regime,
            metadata={
                "mode": "reversion",
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
            mode="reversion",
            direction=direction.value,
            entry=entry_price,
            target=round(target, 2),
            stop=round(stop, 2),
            deviation_sd=round(deviation_sd, 2),
            rsi=round(rsi_val, 1),
            adx=round(adx_val, 1),
            rvol=round(rvol_val, 1),
            atr=round(atr_raw, 2),
            confidence=self._mode_a_confidence,
            signal_id=signal.id,
        )
        return signal

    def _mode_b_momentum(
        self,
        bar: BarEvent,
        bundle: SignalBundle,
        now: datetime,
        adx_val: float,
        atr_raw: float,
    ) -> Signal | None:
        """Mode B: Momentum when ADX >= threshold.

        Requires EMA crossover AND direction aligns with last N bars.
        Target = entry +/- 2.0*ATR, stop = entry -/+ 1.0*ATR.
        """
        # Need EMA crossover signal
        ema_result = bundle.get("ema_crossover")
        if ema_result is None:
            logger.debug("blocked_no_ema", time=now.strftime("%H:%M"), mode="momentum")
            return None

        # Must have an active crossover
        if not ema_result.passes:
            logger.debug("blocked_no_crossover", time=now.strftime("%H:%M"),
                         mode="momentum", ema_value=round(ema_result.value, 4))
            return None

        # Direction from EMA crossover
        if ema_result.direction == "long":
            direction = Direction.LONG
        elif ema_result.direction == "short":
            direction = Direction.SHORT
        else:
            logger.debug("blocked_ema_no_direction", time=now.strftime("%H:%M"),
                         mode="momentum")
            return None

        # Confirm direction aligns with last N bars
        if not self._confirm_momentum(direction):
            logger.debug("blocked_momentum_confirm", time=now.strftime("%H:%M"),
                         mode="momentum", direction=direction.value,
                         recent_closes=self._recent_closes[-4:])
            return None

        # Entry at current close
        entry_price = bar.close

        # Target = entry +/- 2.0 * ATR
        target_distance = self._mode_b_target_atr_mult * atr_raw
        stop_distance = self._mode_b_stop_atr_mult * atr_raw

        if direction == Direction.LONG:
            target = entry_price + target_distance
            stop = entry_price - stop_distance
        else:
            target = entry_price - target_distance
            stop = entry_price + stop_distance

        # Geometry sanity check
        if direction == Direction.LONG:
            if not (stop < entry_price < target):
                logger.info("blocked_geometry", time=now.strftime("%H:%M"),
                            mode="momentum", direction="LONG",
                            entry=entry_price, target=round(target, 2),
                            stop=round(stop, 2))
                return None
        else:
            if not (stop > entry_price > target):
                logger.info("blocked_geometry", time=now.strftime("%H:%M"),
                            mode="momentum", direction="SHORT",
                            entry=entry_price, target=round(target, 2),
                            stop=round(stop, 2))
                return None

        expiry = now + timedelta(minutes=self._mode_b_time_stop)

        rsi_result = bundle.get("rsi_momentum")
        rsi_val = rsi_result.value if rsi_result else 0.0
        rvol_result = bundle.get("relative_volume")
        rvol_val = rvol_result.value if rvol_result else 0.0

        signal = Signal(
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=entry_price,
            target_price=target,
            stop_price=stop,
            signal_time=now,
            expiry_time=expiry,
            confidence=self._mode_b_confidence,
            regime_state=self._current_regime,
            metadata={
                "mode": "momentum",
                "ema_spread": ema_result.value,
                "ema_crossed": ema_result.metadata.get("crossed", "none"),
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
            mode="momentum",
            direction=direction.value,
            entry=entry_price,
            target=round(target, 2),
            stop=round(stop, 2),
            ema_spread=round(ema_result.value, 4),
            rsi=round(rsi_val, 1),
            adx=round(adx_val, 1),
            rvol=round(rvol_val, 1),
            atr=round(atr_raw, 2),
            confidence=self._mode_b_confidence,
            signal_id=signal.id,
        )
        return signal

    def _confirm_momentum(self, direction: Direction) -> bool:
        """Check that the last N bars align with the crossover direction.

        For LONG: each of the last N closes must be higher than the previous.
        For SHORT: each must be lower.
        """
        n = self._momentum_confirm_bars
        if len(self._recent_closes) < n + 1:
            return False

        recent = self._recent_closes[-(n + 1):]
        if direction == Direction.LONG:
            return all(recent[i + 1] > recent[i] for i in range(n))
        else:
            return all(recent[i + 1] < recent[i] for i in range(n))

    def check_early_exit(
        self,
        bar: BarEvent,
        bundle: SignalBundle,
        bars_in_trade: int,
        direction: Direction,
        fill_price: float,
        signal_metadata: dict | None = None,
    ) -> str | None:
        """Check if any early exit condition fires (OR logic).

        Routes to mode-specific exit checks based on signal metadata.
        """
        meta = signal_metadata or {}
        mode = meta.get("mode", "reversion")

        if mode == "reversion":
            return self._check_reversion_exits(bar, bundle, direction)
        else:
            return self._check_momentum_exits(bar, bundle, bars_in_trade, direction, fill_price)

    def _check_reversion_exits(
        self,
        bar: BarEvent,
        bundle: SignalBundle,
        direction: Direction,
    ) -> str | None:
        """Mode A early exits:
        1. VWAP slope > 0.4 adverse.
        2. ADX crosses above 25.
        """
        # Check VWAP slope adverse
        vwap_result = bundle.get("vwap_session")
        if vwap_result is not None:
            slope = vwap_result.metadata.get("slope", 0.0)
            # Long: adverse slope is negative (VWAP falling)
            # Short: adverse slope is positive (VWAP rising)
            if direction == Direction.LONG and slope < -self._ee_vwap_slope_threshold:
                logger.info("early_exit_vwap_slope", mode="reversion",
                            direction="LONG", slope=round(slope, 4),
                            threshold=self._ee_vwap_slope_threshold)
                return "early:vwap_slope"
            if direction == Direction.SHORT and slope > self._ee_vwap_slope_threshold:
                logger.info("early_exit_vwap_slope", mode="reversion",
                            direction="SHORT", slope=round(slope, 4),
                            threshold=self._ee_vwap_slope_threshold)
                return "early:vwap_slope"

        # Check ADX crossing above trend threshold (regime changing)
        adx_result = bundle.get("adx")
        if adx_result is not None:
            adx_val = adx_result.value
            if adx_val > self._ee_adx_trend_threshold:
                logger.info("early_exit_adx_trend", mode="reversion",
                            adx=round(adx_val, 1),
                            threshold=self._ee_adx_trend_threshold)
                return "early:adx_trend"

        return None

    def _check_momentum_exits(
        self,
        bar: BarEvent,
        bundle: SignalBundle,
        bars_in_trade: int,
        direction: Direction,
        fill_price: float,
    ) -> str | None:
        """Mode B early exits:
        1. ADX drops below 18 (trend fading).
        2. Adverse momentum: 2 bars with 1.5x ATR move against.
        """
        # Check ADX dropping below range threshold
        adx_result = bundle.get("adx")
        if adx_result is not None:
            adx_val = adx_result.value
            if adx_val < self._ee_adx_range_threshold:
                logger.info("early_exit_adx_range", mode="momentum",
                            adx=round(adx_val, 1),
                            threshold=self._ee_adx_range_threshold)
                return "early:adx_range"

        # Check adverse momentum
        if bars_in_trade <= self._ee_adverse_bars:
            atr_result = bundle.get("atr")
            if atr_result is not None:
                atr_raw = atr_result.metadata.get("atr_raw", 0.0)
                if atr_raw > 0:
                    if direction == Direction.LONG:
                        unrealized = bar.close - fill_price
                    else:
                        unrealized = fill_price - bar.close

                    threshold = -self._ee_adverse_atr_mult * atr_raw
                    if unrealized < threshold:
                        logger.info("early_exit_adverse_momentum", mode="momentum",
                                    direction=direction.value,
                                    bars_in_trade=bars_in_trade,
                                    unrealized=round(unrealized, 2),
                                    threshold=round(threshold, 2))
                        return "early:adverse_momentum"

        return None

    def reset(self) -> None:
        self._signals_today = 0
        self._recent_closes.clear()
