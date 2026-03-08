"""Opening Range Breakout (ORB) strategy.

Tracks the 9:30–9:45 ET high/low range, then triggers on bar closes
beyond that range with volume, VWAP, HMM, and time filters. One-shot
per day by default.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
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
    _parse_time,
)

if TYPE_CHECKING:
    from src.features.feature_hub import FeatureHub
    from src.models.hmm_regime import HMMRegimeClassifier

logger = get_logger("orb")

# Avoid circular import
from src.core.events import BarEvent, TickEvent  # noqa: E402


class ORBState(Enum):
    """ORB state machine phases."""

    WAITING = "WAITING"
    COLLECTING_RANGE = "COLLECTING_RANGE"
    WATCHING_BREAKOUT = "WATCHING_BREAKOUT"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    INACTIVE = "INACTIVE"


@dataclass
class ORBConfig(StrategyConfig):
    """ORB-specific configuration, extends StrategyConfig."""

    strategy_id: str = "orb"
    orb_start: str = "09:30"
    orb_end: str = "09:45"
    breakout_bar_type: str = "5m"
    target_multiplier: float = 0.5
    volume_multiplier: float = 1.5
    max_signal_time: str = "11:00"
    require_vwap_alignment: bool = True
    max_signals_per_day: int = 1
    require_hmm_states: list[RegimeState] = field(
        default_factory=lambda: [
            RegimeState.HIGH_VOL_UP,
            RegimeState.HIGH_VOL_DOWN,
            RegimeState.BREAKOUT,
        ]
    )
    expiry_minutes: int = 90
    slippage_ticks: int = 1


class ORBStrategy(StrategyBase):
    """Opening Range Breakout strategy.

    State machine: WAITING → COLLECTING_RANGE → WATCHING_BREAKOUT →
    SIGNAL_GENERATED or INACTIVE.
    """

    def __init__(
        self,
        config: ORBConfig,
        feature_hub: FeatureHub,
        hmm_classifier: HMMRegimeClassifier | None = None,
    ) -> None:
        super().__init__(config, feature_hub, hmm_classifier)
        self._orb_config = config
        self._state = ORBState.WAITING
        self._range_high: float = -math.inf
        self._range_low: float = math.inf
        self._range_volume: int = 0
        self._range_bar_count: int = 0
        self._volume_history: deque[int] = deque(maxlen=100)
        # 5-minute bar aggregator for breakout detection (time-based)
        self._agg_open: float = 0.0
        self._agg_high: float = -math.inf
        self._agg_low: float = math.inf
        self._agg_close: float = 0.0
        self._agg_volume: int = 0
        self._agg_start_ns: int = 0  # timestamp of first bar in window

    def _bar_et_time(self, bar: BarEvent) -> time:
        """Convert bar timestamp_ns to ET time object."""
        dt_utc = datetime.fromtimestamp(
            bar.timestamp_ns / 1_000_000_000, tz=_ET
        )
        return dt_utc.time()

    def _bar_et_datetime(self, bar: BarEvent) -> datetime:
        """Convert bar timestamp_ns to ET datetime."""
        return datetime.fromtimestamp(
            bar.timestamp_ns / 1_000_000_000, tz=_ET
        )

    def on_tick(self, tick: TickEvent) -> None:
        """No-op — ORB is bar-driven."""

    def on_bar(self, bar: BarEvent) -> None:
        """Process a completed bar through the ORB state machine."""
        self._base_on_bar(bar)
        bar_time = self._bar_et_time(bar)

        orb_start = _parse_time(self._orb_config.orb_start)
        orb_end = _parse_time(self._orb_config.orb_end)
        max_time = _parse_time(self._orb_config.max_signal_time)

        # State transitions
        if self._state == ORBState.WAITING:
            if bar_time >= orb_start:
                self._state = ORBState.COLLECTING_RANGE
                # Process this bar as part of the range
                self._update_range(bar)

        elif self._state == ORBState.COLLECTING_RANGE:
            if bar_time >= orb_end:
                # Freeze range, transition to watching
                self._state = ORBState.WATCHING_BREAKOUT
                logger.info(
                    "orb_range_frozen",
                    range_high=self._range_high,
                    range_low=self._range_low,
                    range_width=self._range_high - self._range_low,
                    range_volume=self._range_volume,
                    bar_count=self._range_bar_count,
                )
                # This bar is post-range, track its volume
                self._volume_history.append(bar.volume)
            else:
                self._update_range(bar)

        elif self._state == ORBState.WATCHING_BREAKOUT:
            if bar_time >= max_time:
                self._state = ORBState.INACTIVE
                return
            self._volume_history.append(bar.volume)
            # Native 5m bars: check breakout directly
            if bar.bar_type == self._orb_config.breakout_bar_type:
                signal = self.generate_signal(bar)
                if signal is not None:
                    self._state = ORBState.SIGNAL_GENERATED
            else:
                # Aggregate bars into 5-minute windows for breakout check
                self._aggregate_bar(bar)
                elapsed_ns = bar.timestamp_ns - self._agg_start_ns
                if elapsed_ns >= 300_000_000_000:  # 5 minutes in nanoseconds
                    agg_bar = BarEvent(
                        symbol=bar.symbol,
                        open=self._agg_open,
                        high=self._agg_high,
                        low=self._agg_low,
                        close=self._agg_close,
                        volume=self._agg_volume,
                        bar_type=self._orb_config.breakout_bar_type,
                        timestamp_ns=bar.timestamp_ns,
                    )
                    self._reset_agg()
                    signal = self.generate_signal(agg_bar)
                    if signal is not None:
                        self._state = ORBState.SIGNAL_GENERATED

    def _update_range(self, bar: BarEvent) -> None:
        """Update range high/low/volume during collection phase."""
        self._range_high = max(self._range_high, bar.high)
        self._range_low = min(self._range_low, bar.low)
        self._range_volume += bar.volume
        self._range_bar_count += 1

    def _aggregate_bar(self, bar: BarEvent) -> None:
        """Accumulate a bar into the 5-minute aggregator."""
        if self._agg_start_ns == 0:
            self._agg_open = bar.open
            self._agg_start_ns = bar.timestamp_ns
        self._agg_high = max(self._agg_high, bar.high)
        self._agg_low = min(self._agg_low, bar.low)
        self._agg_close = bar.close
        self._agg_volume += bar.volume

    def _reset_agg(self) -> None:
        """Reset the 5-minute bar aggregator."""
        self._agg_open = 0.0
        self._agg_high = -math.inf
        self._agg_low = math.inf
        self._agg_close = 0.0
        self._agg_volume = 0
        self._agg_start_ns = 0

    def generate_signal(self, bar: BarEvent | None = None) -> Signal | None:
        """Check for breakout and generate signal if all filters pass."""
        if bar is None:
            return None

        bar_dt = self._bar_et_datetime(bar)

        if not self.can_generate_signal(bar_dt):
            return None
        if self._state != ORBState.WATCHING_BREAKOUT:
            return None

        orb_width = self._range_high - self._range_low

        # Check breakout direction based on close
        if bar.close > self._range_high:
            direction = Direction.LONG
        elif bar.close < self._range_low:
            direction = Direction.SHORT
        else:
            return None

        # Volume filter
        avg_volume = self._avg_volume()
        if avg_volume > 0:
            volume_ratio = bar.volume / avg_volume
            if bar.volume < self._orb_config.volume_multiplier * avg_volume:
                return None
        else:
            volume_ratio = 1.0

        # VWAP alignment filter
        if self._orb_config.require_vwap_alignment and self._last_snapshot is not None:
            vwap = self._last_snapshot.vwap
            if direction == Direction.LONG and vwap >= bar.close:
                return None
            if direction == Direction.SHORT and vwap <= bar.close:
                return None

        # Compute signal geometry
        slippage = self._orb_config.slippage_ticks * TICK_SIZE
        if direction == Direction.LONG:
            entry = bar.close + slippage
            target = entry + orb_width * self._orb_config.target_multiplier
            stop = self._range_low
        else:
            entry = bar.close - slippage
            target = entry - orb_width * self._orb_config.target_multiplier
            stop = self._range_high

        # Confidence
        hmm_best_prob = float(max(self._regime_probs)) if max(self._regime_probs) > 0 else 0.8
        confidence = min(1.0, volume_ratio / self._orb_config.volume_multiplier) * hmm_best_prob

        # Compute time since open
        orb_start = _parse_time(self._orb_config.orb_start)
        open_dt = bar_dt.replace(hour=orb_start.hour, minute=orb_start.minute, second=0)
        time_since_open = (bar_dt - open_dt).total_seconds() / 60.0

        # Metadata
        vwap_at_signal = self._last_snapshot.vwap if self._last_snapshot else 0.0
        metadata = {
            "orb_high": self._range_high,
            "orb_low": self._range_low,
            "orb_width_ticks": orb_width / TICK_SIZE,
            "breakout_volume": bar.volume,
            "avg_volume": avg_volume,
            "volume_ratio": volume_ratio,
            "vwap_at_signal": vwap_at_signal,
            "hmm_state": self._current_regime.name,
            "time_since_open_minutes": time_since_open,
        }

        expiry_seconds = self._orb_config.expiry_minutes * 60

        return self._make_signal(
            direction=direction,
            entry=entry,
            target=target,
            stop=stop,
            confidence=confidence,
            expiry_seconds=expiry_seconds,
            metadata=metadata,
            now=bar_dt,
        )

    def _avg_volume(self) -> float:
        """Average volume from rolling history."""
        if not self._volume_history:
            return 0.0
        return sum(self._volume_history) / len(self._volume_history)

    def reset(self) -> None:
        """Reset daily state. Volume history is preserved across sessions."""
        super().reset()
        self._state = ORBState.WAITING
        self._range_high = -math.inf
        self._range_low = math.inf
        self._range_volume = 0
        self._range_bar_count = 0
        self._reset_agg()
