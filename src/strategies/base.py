"""Strategy base class and shared interfaces.

Defines the Signal dataclass, Direction enum, StrategyConfig,
and StrategyBase ABC that concrete strategies can inherit from.

Standalone strategies (ORB, VWAP, NoiseBreakout) do NOT inherit from
StrategyBase — they duck-type the on_bar/reset interface instead.
StrategyBase is kept as a lightweight ABC for any future strategies
that want session gating and signal construction helpers.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from src.core.logging import get_logger
from src.models.hmm_regime import RegimeState

logger = get_logger("strategy")

# MES contract constants
TICK_SIZE = 0.25
TICK_VALUE = 1.25  # $5 × 0.25
POINT_VALUE = 5.0


class Direction(Enum):
    """Trade direction — separate from SignalEvent's BUY/SELL strings."""

    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class Signal:
    """Rich strategy signal with entry/target/stop geometry.

    Distinct from SignalEvent (the lighter bus message). Conversion to
    SignalEvent happens in the orchestrator.
    """

    strategy_id: str
    direction: Direction
    entry_price: float
    target_price: float
    stop_price: float
    signal_time: datetime
    expiry_time: datetime
    confidence: float
    regime_state: RegimeState
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def validate_geometry(self) -> tuple[bool, str]:
        """Check that stop/entry/target are in correct order for direction.

        Returns (is_valid, reason).  Reason is empty string when valid.
        """
        if self.direction == Direction.LONG:
            if not (self.stop_price < self.entry_price < self.target_price):
                return False, "LONG: need stop < entry < target"
        else:
            if not (self.stop_price > self.entry_price > self.target_price):
                return False, "SHORT: need stop > entry > target"
        return True, ""

    @property
    def risk_reward_ratio(self) -> float:
        """Reward / risk in price distance. Returns 0 if risk is zero."""
        reward = abs(self.target_price - self.entry_price)
        risk = abs(self.entry_price - self.stop_price)
        if risk == 0:
            return 0.0
        return reward / risk

    @property
    def ticks_to_target(self) -> float:
        """Distance to target in ticks."""
        return abs(self.target_price - self.entry_price) / TICK_SIZE

    @property
    def ticks_to_stop(self) -> float:
        """Distance to stop in ticks."""
        return abs(self.entry_price - self.stop_price) / TICK_SIZE


@dataclass
class StrategyConfig:
    """Mutable strategy configuration — may be loaded from YAML."""

    strategy_id: str = "default"
    symbol: str = "MESM6"
    max_signals_per_day: int = 5
    session_start: str = "09:30"
    session_end: str = "16:00"
    excluded_windows: list[tuple[str, str]] = field(default_factory=list)
    require_hmm_states: list[RegimeState] = field(default_factory=list)
    min_confidence: float = 0.6


# US Eastern timezone (DST-aware)
from zoneinfo import ZoneInfo

_ET = ZoneInfo("US/Eastern")


def _parse_time(s: str) -> time:
    """Parse HH:MM string to time object."""
    parts = s.split(":")
    return time(int(parts[0]), int(parts[1]))


class StrategyBase(ABC):
    """Abstract base class for trading strategies.

    Provides session gating, signal construction, and daily state management.
    Standalone strategies (ORB, VWAP, NoiseBreakout) do NOT use this — they
    duck-type on_bar/reset directly. This ABC is kept for any strategy that
    wants the built-in session/HMM/signal helpers.
    """

    def __init__(
        self,
        config: StrategyConfig,
    ) -> None:
        self.config = config

        self._current_regime: RegimeState = RegimeState.RANGE_BOUND
        self._signals_today: int = 0
        self._signals_generated: list[Signal] = []
        self._bars_processed: int = 0
        self._bar_window: list[BarEvent] = []

    # ── Abstract methods ─────────────────────────────────────────────

    @abstractmethod
    def on_tick(self, tick: TickEvent) -> None:
        """Process a real-time tick."""
        ...

    @abstractmethod
    def on_bar(self, bar: BarEvent) -> Signal | None:
        """Process a completed bar.

        Returns a Signal if one was generated, None otherwise.
        """
        ...

    @abstractmethod
    def generate_signal(self) -> Signal | None:
        """Attempt to generate a trading signal."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset daily state. Must call super().reset()."""
        ...

    # ── Concrete methods ─────────────────────────────────────────────

    def _base_on_bar(self, bar: BarEvent) -> None:
        """Update bar counter and window."""
        self._bars_processed += 1

        # Maintain bar window for filter evaluation (last 500 bars)
        self._bar_window.append(bar)
        if len(self._bar_window) > 500:
            self._bar_window = self._bar_window[-500:]

    def is_active_session(self, now: datetime | None = None) -> bool:
        """Check if current time is within the trading session window."""
        if now is None:
            now = datetime.now(_ET)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=_ET)

        current_time = now.time()
        session_start = _parse_time(self.config.session_start)
        session_end = _parse_time(self.config.session_end)

        if not (session_start <= current_time <= session_end):
            return False

        for start_str, end_str in self.config.excluded_windows:
            ex_start = _parse_time(start_str)
            ex_end = _parse_time(end_str)
            if ex_start <= current_time <= ex_end:
                return False

        return True

    def is_valid_hmm_state(self) -> bool:
        """Check if current regime is in the allowed set."""
        if not self.config.require_hmm_states:
            return True
        return self._current_regime in self.config.require_hmm_states

    def can_generate_signal(self, now: datetime | None = None) -> bool:
        """Combined gate: session + HMM state + daily signal count."""
        if not self.is_active_session(now):
            return False
        if not self.is_valid_hmm_state():
            return False
        if self._signals_today >= self.config.max_signals_per_day:
            return False
        return True

    def _make_signal(
        self,
        direction: Direction,
        entry: float,
        target: float,
        stop: float,
        confidence: float,
        expiry_seconds: int = 60,
        metadata: dict | None = None,
        now: datetime | None = None,
    ) -> Signal | None:
        """Construct a Signal, enforce min_confidence, increment counter."""
        if confidence < self.config.min_confidence:
            return None

        if now is None:
            now = datetime.now(_ET)

        signal = Signal(
            strategy_id=self.config.strategy_id,
            direction=direction,
            entry_price=entry,
            target_price=target,
            stop_price=stop,
            signal_time=now,
            expiry_time=now + timedelta(seconds=expiry_seconds),
            confidence=confidence,
            regime_state=self._current_regime,
            metadata=metadata or {},
        )
        self._signals_today += 1
        self._signals_generated.append(signal)
        self._log_signal(signal)
        return signal

    def _log_signal(self, signal: Signal) -> None:
        """Structured log of a generated signal."""
        logger.info(
            "signal_generated",
            strategy_id=signal.strategy_id,
            direction=signal.direction.value,
            entry=signal.entry_price,
            target=signal.target_price,
            stop=signal.stop_price,
            confidence=signal.confidence,
            regime=signal.regime_state.name,
            rr=round(signal.risk_reward_ratio, 2),
            signal_id=signal.id,
        )

    def get_daily_metrics(self) -> dict:
        """Return daily strategy performance metrics."""
        return {
            "strategy_id": self.config.strategy_id,
            "signals_today": self._signals_today,
            "bars_processed": self._bars_processed,
            "current_regime": self._current_regime.name,
        }

    def reset(self) -> None:
        """Reset daily state."""
        self._signals_today = 0
        self._signals_generated.clear()
        self._bars_processed = 0
        self._bar_window.clear()


# Avoid circular imports — import event types at module level for type hints
from src.core.events import BarEvent, TickEvent  # noqa: E402
