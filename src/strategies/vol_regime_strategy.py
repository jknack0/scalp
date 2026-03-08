"""Volatility Regime Switcher strategy.

Adapts trading behavior to volatility regime using ATR semi-variance features.
High-vol: momentum pullbacks with wide targets. Low-vol: fade VWAP extensions
with tight targets. Uses HMM cross-validation per mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.models.hmm_regime import RegimeState
from src.strategies.base import (
    TICK_SIZE,
    Direction,
    Signal,
    StrategyBase,
    StrategyConfig,
    _ET,
)

if TYPE_CHECKING:
    from src.features.feature_hub import FeatureHub
    from src.filters.vpin_monitor import VPINMonitor
    from src.models.hmm_regime import HMMRegimeClassifier

logger = get_logger("vol_regime_strategy")

# Avoid circular import
from src.core.events import BarEvent, TickEvent  # noqa: E402


class VolRegime(Enum):
    """Volatility regime classification."""

    HIGH_VOL = "HIGH_VOL"
    LOW_VOL = "LOW_VOL"
    TRANSITIONING = "TRANSITIONING"


@dataclass
class VolRegimeConfig(StrategyConfig):
    """Vol Regime Switcher configuration, extends StrategyConfig."""

    strategy_id: str = "vol_regime_switcher"
    high_vol_target_ticks: int = 8
    high_vol_stop_ticks: int = 2
    low_vol_target_ticks: int = 2
    low_vol_stop_ticks: int = 30
    low_vol_entry_sd: float = 1.5
    pullback_bars: int = 3
    transition_cooldown_bars: int = 2
    max_signals_per_day: int = 4
    require_hmm_states: list[RegimeState] = field(default_factory=list)
    high_vol_hmm_states: list[RegimeState] = field(
        default_factory=lambda: [
            RegimeState.HIGH_VOL_UP,
            RegimeState.HIGH_VOL_DOWN,
            RegimeState.BREAKOUT,
        ]
    )
    low_vol_hmm_states: list[RegimeState] = field(
        default_factory=lambda: [
            RegimeState.LOW_VOL_RANGE,
            RegimeState.MEAN_REVERSION,
        ]
    )
    use_vpin_regime: bool = False
    expiry_minutes: int = 30


class VolRegimeStrategy(StrategyBase):
    """Volatility Regime Switcher strategy.

    High-vol mode: trade momentum pullbacks with wide targets (8T) and tight
    stops (2T). Low-vol mode: fade VWAP extensions with tight targets (2T)
    and wide stops (30T). TRANSITIONING: no trade.
    """

    def __init__(
        self,
        config: VolRegimeConfig,
        feature_hub: FeatureHub,
        hmm_classifier: HMMRegimeClassifier | None = None,
        vpin_monitor: VPINMonitor | None = None,
    ) -> None:
        super().__init__(config, feature_hub, hmm_classifier)
        self._vol_config = config
        self._vpin_monitor = vpin_monitor
        self._regime: VolRegime = VolRegime.TRANSITIONING
        self._prev_regime: VolRegime = VolRegime.TRANSITIONING
        self._bars_since_transition: int = 0
        self._pullback_count: int = 0
        self._prev_close: float = 0.0

    def _bar_et_datetime(self, bar: BarEvent) -> datetime:
        """Convert bar timestamp_ns to ET datetime."""
        return datetime.fromtimestamp(
            bar.timestamp_ns / 1_000_000_000, tz=_ET
        )

    def on_tick(self, tick: TickEvent) -> None:
        """No-op -- vol regime strategy is bar-driven."""

    def on_bar(self, bar: BarEvent) -> None:
        """Process a completed bar: update features, regime, pullback, signal."""
        self._base_on_bar(bar)
        self._update_regime()
        self._bars_since_transition += 1
        self._update_pullback(bar)
        self.generate_signal(bar)
        self._prev_close = bar.close

    def _update_regime(self) -> None:
        """Classify volatility regime from VPIN or ATR, cross-validate with HMM."""
        if self._vol_config.use_vpin_regime and self._vpin_monitor is not None:
            new_regime = self._regime_from_vpin()
        else:
            new_regime = self._regime_from_atr()

        # HMM cross-validation per mode
        if new_regime == VolRegime.HIGH_VOL and self._vol_config.high_vol_hmm_states:
            if self._current_regime not in self._vol_config.high_vol_hmm_states:
                new_regime = VolRegime.TRANSITIONING

        if new_regime == VolRegime.LOW_VOL and self._vol_config.low_vol_hmm_states:
            if self._current_regime not in self._vol_config.low_vol_hmm_states:
                new_regime = VolRegime.TRANSITIONING

        if new_regime != self._regime:
            self._prev_regime = self._regime
            self._bars_since_transition = 0
            self._pullback_count = 0
            logger.info(
                "vol_regime_change",
                from_regime=self._regime.value,
                to_regime=new_regime.value,
            )
            self._regime = new_regime

    def _regime_from_vpin(self) -> VolRegime:
        """Classify regime using VPIN monitor."""
        regime, vpin = self._vpin_monitor.get_regime()
        if regime == "trending":
            return VolRegime.HIGH_VOL
        elif regime == "mean_reversion":
            return VolRegime.LOW_VOL
        return VolRegime.TRANSITIONING

    def _regime_from_atr(self) -> VolRegime:
        """Classify regime using ATR percentile (legacy)."""
        atr_regime = self.feature_hub.atr.vol_regime
        if atr_regime == "HIGH":
            return VolRegime.HIGH_VOL
        elif atr_regime == "LOW":
            return VolRegime.LOW_VOL
        return VolRegime.TRANSITIONING

    def _update_pullback(self, bar: BarEvent) -> None:
        """Track consecutive bars against dominant direction (high-vol mode)."""
        if self._regime != VolRegime.HIGH_VOL:
            return
        if self._prev_close == 0.0:
            return

        dominant = self.feature_hub.atr.dominant_direction

        if dominant == "UP" and bar.close < self._prev_close:
            self._pullback_count += 1
        elif dominant == "DOWN" and bar.close > self._prev_close:
            self._pullback_count += 1
        else:
            self._pullback_count = 0

    def generate_signal(self, bar: BarEvent | None = None) -> Signal | None:
        """Attempt to generate a trading signal based on current regime."""
        if bar is None:
            return None

        bar_dt = self._bar_et_datetime(bar)

        # Gate: session + daily signal count (base HMM gate is empty by default)
        if not self.can_generate_signal(bar_dt):
            return None

        # Gate: transition cooldown
        if self._bars_since_transition <= self._vol_config.transition_cooldown_bars:
            return None

        # Gate: no trading in TRANSITIONING
        if self._regime == VolRegime.TRANSITIONING:
            return None

        if self._regime == VolRegime.HIGH_VOL:
            return self._generate_high_vol_signal(bar, bar_dt)
        if self._regime == VolRegime.LOW_VOL:
            return self._generate_low_vol_signal(bar, bar_dt)

        return None

    def _generate_high_vol_signal(
        self, bar: BarEvent, bar_dt: datetime
    ) -> Signal | None:
        """Generate momentum pullback signal in high-vol regime."""
        # Gate: enough consecutive pullback bars
        if self._pullback_count < self._vol_config.pullback_bars:
            return None

        dominant = self.feature_hub.atr.dominant_direction
        if dominant not in ("UP", "DOWN"):
            return None

        direction = Direction.LONG if dominant == "UP" else Direction.SHORT

        entry = bar.close
        if direction == Direction.LONG:
            target = entry + self._vol_config.high_vol_target_ticks * TICK_SIZE
            stop = entry - self._vol_config.high_vol_stop_ticks * TICK_SIZE
        else:
            target = entry - self._vol_config.high_vol_target_ticks * TICK_SIZE
            stop = entry + self._vol_config.high_vol_stop_ticks * TICK_SIZE

        # Confidence: normalized semi-variance asymmetry
        sv_up = self.feature_hub.atr.semi_variance_up
        sv_down = self.feature_hub.atr.semi_variance_down
        max_sv = max(sv_up, sv_down)
        if max_sv > 0:
            confidence = min(1.0, abs(sv_up - sv_down) / max_sv)
        else:
            confidence = 0.0

        # Reset pullback counter to prevent double-firing
        self._pullback_count = 0

        metadata = {
            "regime": self._regime.value,
            "dominant_direction": dominant,
            "semi_var_up": sv_up,
            "semi_var_down": sv_down,
            "pullback_bars": self._vol_config.pullback_bars,
            "atr_ticks": self.feature_hub.atr.atr_ticks,
            "hmm_state": self._current_regime.name,
        }

        return self._make_signal(
            direction=direction,
            entry=entry,
            target=target,
            stop=stop,
            confidence=confidence,
            expiry_seconds=self._vol_config.expiry_minutes * 60,
            metadata=metadata,
            now=bar_dt,
        )

    def _generate_low_vol_signal(
        self, bar: BarEvent, bar_dt: datetime
    ) -> Signal | None:
        """Generate VWAP fade signal in low-vol regime."""
        dev_sd = self.feature_hub.vwap.deviation_sd

        # Gate: price must be at least low_vol_entry_sd from VWAP
        if abs(dev_sd) < self._vol_config.low_vol_entry_sd:
            return None

        # Direction: below VWAP → LONG, above → SHORT
        direction = Direction.LONG if dev_sd < 0 else Direction.SHORT

        entry = bar.close
        if direction == Direction.LONG:
            target = entry + self._vol_config.low_vol_target_ticks * TICK_SIZE
            stop = entry - self._vol_config.low_vol_stop_ticks * TICK_SIZE
        else:
            target = entry - self._vol_config.low_vol_target_ticks * TICK_SIZE
            stop = entry + self._vol_config.low_vol_stop_ticks * TICK_SIZE

        # Confidence: deviation magnitude / 3.0, capped at 1.0
        confidence = min(1.0, abs(dev_sd) / 3.0)

        metadata = {
            "regime": self._regime.value,
            "deviation_sd": dev_sd,
            "vwap": self.feature_hub.vwap.vwap,
            "atr_ticks": self.feature_hub.atr.atr_ticks,
            "hmm_state": self._current_regime.name,
        }

        return self._make_signal(
            direction=direction,
            entry=entry,
            target=target,
            stop=stop,
            confidence=confidence,
            expiry_seconds=self._vol_config.expiry_minutes * 60,
            metadata=metadata,
            now=bar_dt,
        )

    def reset(self) -> None:
        """Reset daily state."""
        super().reset()
        self._regime = VolRegime.TRANSITIONING
        self._prev_regime = VolRegime.TRANSITIONING
        self._bars_since_transition = 0
        self._pullback_count = 0
        self._prev_close = 0.0
