"""VWAP Reversion strategy with regime-switching modes.

Switches between REVERSION (fade extremes back to VWAP when flat) and
PULLBACK (buy/sell pullbacks to VWAP in trend direction). Uses VWAPCalculator
slope to determine mode, with first-kiss detection for confidence boosts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.models.hmm_regime import RegimeState
from src.strategies.base import (
    Direction,
    Signal,
    StrategyBase,
    StrategyConfig,
    _ET,
)

if TYPE_CHECKING:
    from src.features.feature_hub import FeatureHub
    from src.models.hmm_regime import HMMRegimeClassifier

logger = get_logger("vwap_strategy")

# Avoid circular import
from src.core.events import BarEvent, TickEvent  # noqa: E402

# Session start as ET time
_SESSION_START = time(9, 30)


class VWAPMode(Enum):
    """VWAP strategy operating modes."""

    REVERSION = "REVERSION"
    PULLBACK = "PULLBACK"
    NEUTRAL = "NEUTRAL"


@dataclass
class VWAPConfig(StrategyConfig):
    """VWAP Reversion-specific configuration, extends StrategyConfig."""

    strategy_id: str = "vwap_reversion"
    entry_sd_reversion: float = 2.0
    stop_sd: float = 2.0
    flat_slope_threshold: float = 0.003
    trending_slope_threshold: float = 0.008
    first_kiss_lookback_bars: int = 6
    min_session_age_minutes: int = 15
    require_reversal_candle: bool = False
    pullback_entry_sd: float = 0.5
    mode_cooldown_bars: int = 5
    max_signals_per_day: int = 4
    require_hmm_states: list[RegimeState] = field(default_factory=list)
    expiry_minutes: int = 60
    reversion_hmm_states: list[RegimeState] = field(
        default_factory=lambda: [RegimeState.LOW_VOL_RANGE, RegimeState.MEAN_REVERSION]
    )
    pullback_hmm_states: list[RegimeState] = field(
        default_factory=lambda: [RegimeState.HIGH_VOL_UP, RegimeState.HIGH_VOL_DOWN]
    )


class VWAPStrategy(StrategyBase):
    """VWAP Reversion strategy with mode switching.

    Modes:
    - REVERSION: flat VWAP, fade extremes back to VWAP
    - PULLBACK: trending VWAP, buy/sell pullbacks in trend direction
    - NEUTRAL: ambiguous slope, no trade
    """

    def __init__(
        self,
        config: VWAPConfig,
        feature_hub: FeatureHub,
        hmm_classifier: HMMRegimeClassifier | None = None,
    ) -> None:
        super().__init__(config, feature_hub, hmm_classifier)
        self._vwap_config = config
        self._mode: VWAPMode = VWAPMode.NEUTRAL
        self._prev_mode: VWAPMode = VWAPMode.NEUTRAL
        self._bars_since_mode_change: int = 0
        self._prev_bar_close: float = 0.0

    def _bar_et_datetime(self, bar: BarEvent) -> datetime:
        """Convert bar timestamp_ns to ET datetime."""
        return datetime.fromtimestamp(
            bar.timestamp_ns / 1_000_000_000, tz=_ET
        )

    def _update_mode(self) -> None:
        """Update operating mode based on VWAP slope."""
        slope = self.feature_hub.vwap.slope_20bar
        abs_slope = abs(slope)

        if abs_slope < self._vwap_config.flat_slope_threshold:
            new_mode = VWAPMode.REVERSION
        elif abs_slope > self._vwap_config.trending_slope_threshold:
            new_mode = VWAPMode.PULLBACK
        else:
            new_mode = VWAPMode.NEUTRAL

        if new_mode != self._mode:
            self._prev_mode = self._mode
            self._bars_since_mode_change = 0
            logger.info(
                "vwap_mode_change",
                from_mode=self._mode.value,
                to_mode=new_mode.value,
                slope=slope,
            )
            self._mode = new_mode

    def _session_age_minutes(self, bar: BarEvent) -> float:
        """Minutes since 9:30 ET for the given bar."""
        bar_dt = self._bar_et_datetime(bar)
        session_open = bar_dt.replace(
            hour=_SESSION_START.hour,
            minute=_SESSION_START.minute,
            second=0,
            microsecond=0,
        )
        return (bar_dt - session_open).total_seconds() / 60.0

    def _is_reversal_candle(self, bar: BarEvent, direction: Direction) -> bool:
        """Check if bar closes back toward VWAP (reversal confirmation)."""
        if self._prev_bar_close == 0.0:
            return False
        if direction == Direction.LONG:
            return bar.close > self._prev_bar_close
        else:
            return bar.close < self._prev_bar_close

    def on_tick(self, tick: TickEvent) -> None:
        """No-op -- VWAP strategy is bar-driven."""

    def on_bar(self, bar: BarEvent) -> None:
        """Process a completed bar: update features, mode, then check for signal."""
        self._base_on_bar(bar)
        self._update_mode()
        self._bars_since_mode_change += 1
        self.generate_signal(bar)
        self._prev_bar_close = bar.close

    def generate_signal(self, bar: BarEvent | None = None) -> Signal | None:
        """Attempt to generate a trading signal based on current mode."""
        if bar is None:
            return None

        bar_dt = self._bar_et_datetime(bar)

        # Gate: session + daily signal count
        if not self.can_generate_signal(bar_dt):
            return None

        # Gate: mode cooldown
        if self._bars_since_mode_change <= self._vwap_config.mode_cooldown_bars:
            return None

        # Gate: minimum session age
        if self._session_age_minutes(bar) < self._vwap_config.min_session_age_minutes:
            return None

        if self._mode == VWAPMode.REVERSION:
            return self._generate_reversion_signal(bar, bar_dt)
        elif self._mode == VWAPMode.PULLBACK:
            return self._generate_pullback_signal(bar, bar_dt)
        return None

    def _generate_reversion_signal(
        self, bar: BarEvent, bar_dt: datetime
    ) -> Signal | None:
        """Generate a reversion signal: fade extremes back to VWAP."""
        vwap_calc = self.feature_hub.vwap
        dev_sd = vwap_calc.deviation_sd
        vwap = vwap_calc.vwap
        sd = vwap_calc._sd

        # Must be at or beyond entry SD threshold
        if abs(dev_sd) < self._vwap_config.entry_sd_reversion:
            return None

        # Direction: below VWAP → LONG, above → SHORT
        direction = Direction.LONG if dev_sd < 0 else Direction.SHORT

        # Reversal candle check
        if self._vwap_config.require_reversal_candle:
            if not self._is_reversal_candle(bar, direction):
                return None

        # HMM gate (mode-specific)
        if self._vwap_config.reversion_hmm_states:
            if self._current_regime not in self._vwap_config.reversion_hmm_states:
                return None

        # Compute geometry
        entry = bar.close
        target = vwap
        if direction == Direction.LONG:
            stop = vwap - self._vwap_config.stop_sd * sd
        else:
            stop = vwap + self._vwap_config.stop_sd * sd

        # Confidence
        hmm_best_prob = (
            float(max(self._regime_probs))
            if max(self._regime_probs) > 0
            else 0.8
        )
        confidence = min(1.0, abs(dev_sd) / 3.0) * hmm_best_prob

        # First-kiss boost
        first_kiss = vwap_calc.first_kiss_detected(
            bar.close,
            self._vwap_config.first_kiss_lookback_bars,
            self._vwap_config.entry_sd_reversion,
        )
        if first_kiss:
            confidence = min(1.0, confidence + 0.15)
            # Tighten stop to 3.0 SD
            if direction == Direction.LONG:
                stop = vwap - 3.0 * sd
            else:
                stop = vwap + 3.0 * sd

        slope = vwap_calc.slope_20bar
        metadata = {
            "mode": self._mode.value,
            "deviation_sd": dev_sd,
            "vwap": vwap,
            "slope": slope,
            "hmm_state": self._current_regime.name,
            "first_kiss": first_kiss,
            "session_age_minutes": self._session_age_minutes(bar),
        }

        return self._make_signal(
            direction=direction,
            entry=entry,
            target=target,
            stop=stop,
            confidence=confidence,
            expiry_seconds=self._vwap_config.expiry_minutes * 60,
            metadata=metadata,
            now=bar_dt,
        )

    def _generate_pullback_signal(
        self, bar: BarEvent, bar_dt: datetime
    ) -> Signal | None:
        """Generate a pullback signal: trade pullbacks to VWAP in trend direction."""
        vwap_calc = self.feature_hub.vwap
        dev_sd = vwap_calc.deviation_sd
        vwap = vwap_calc.vwap
        sd = vwap_calc._sd
        slope = vwap_calc.slope_20bar

        # Must be near VWAP
        if abs(dev_sd) > self._vwap_config.pullback_entry_sd:
            return None

        # Direction from slope
        direction = Direction.LONG if slope > 0 else Direction.SHORT

        # Continuation candle
        if self._prev_bar_close == 0.0:
            return None
        if direction == Direction.LONG and bar.close <= self._prev_bar_close:
            return None
        if direction == Direction.SHORT and bar.close >= self._prev_bar_close:
            return None

        # HMM gate (mode-specific)
        if self._vwap_config.pullback_hmm_states:
            if self._current_regime not in self._vwap_config.pullback_hmm_states:
                return None

        # Compute geometry
        entry = bar.close
        distance = abs(entry - vwap)
        if direction == Direction.LONG:
            target = entry + 2 * distance
            stop = vwap_calc.band_lower_1sd
        else:
            target = entry - 2 * distance
            stop = vwap_calc.band_upper_1sd

        # Confidence (pullback is lower conviction)
        hmm_best_prob = (
            float(max(self._regime_probs))
            if max(self._regime_probs) > 0
            else 0.8
        )
        confidence = 0.7 * hmm_best_prob

        metadata = {
            "mode": self._mode.value,
            "deviation_sd": dev_sd,
            "vwap": vwap,
            "slope": slope,
            "hmm_state": self._current_regime.name,
            "session_age_minutes": self._session_age_minutes(bar),
        }

        return self._make_signal(
            direction=direction,
            entry=entry,
            target=target,
            stop=stop,
            confidence=confidence,
            expiry_seconds=self._vwap_config.expiry_minutes * 60,
            metadata=metadata,
            now=bar_dt,
        )

    def reset(self) -> None:
        """Reset daily state."""
        super().reset()
        self._mode = VWAPMode.NEUTRAL
        self._prev_mode = VWAPMode.NEUTRAL
        self._bars_since_mode_change = 0
        self._prev_bar_close = 0.0
