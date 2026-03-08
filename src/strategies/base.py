"""Strategy base class and shared interfaces.

Defines the Signal dataclass, Direction enum, StrategyConfig, HMMFeatureBuffer,
and StrategyBase ABC that all concrete strategies inherit from.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from src.core.logging import get_logger
from src.models.hmm_regime import RegimeState

if TYPE_CHECKING:
    from src.features.feature_hub import FeatureHub, FeatureVector
    from src.models.hmm_regime import HMMRegimeClassifier

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


class HMMFeatureBuffer:
    """Bridges FeatureHub snapshots to HMM's (W, 6) input matrix.

    Accumulates FeatureVectors and close prices, then builds the
    6-feature matrix the HMM expects: atr_ticks, vwap_dev_sd, cvd_slope,
    poc_distance_ticks, realized_vol, return_20bar — with rolling z-score
    normalization.
    """

    def __init__(self, maxlen: int = 300, zscore_window: int = 250) -> None:
        self._features: deque[FeatureVector] = deque(maxlen=maxlen)
        self._closes: deque[float] = deque(maxlen=maxlen + 21)
        self._maxlen = maxlen
        self._zscore_window = zscore_window

    def update(self, fvec: FeatureVector, close: float) -> None:
        """Append a new bar's feature snapshot and close price."""
        self._features.append(fvec)
        self._closes.append(close)

    def is_ready(self) -> bool:
        """True when enough bars for a 50-row matrix with rolling features."""
        return len(self._features) >= 50 and len(self._closes) >= 70

    def build_matrix(self) -> np.ndarray | None:
        """Build (50, 6) z-score normalized feature matrix for HMM.

        Returns None if not enough data.
        """
        if not self.is_ready():
            return None

        n = len(self._features)
        closes = np.array(self._closes, dtype=np.float64)

        # Extract 4 features from FeatureVectors
        atr_ticks = np.array([f.atr_ticks for f in self._features], dtype=np.float64)
        vwap_dev_sd = np.array([f.vwap_dev_sd for f in self._features], dtype=np.float64)
        cvd_slope = np.array([f.cvd_slope for f in self._features], dtype=np.float64)
        poc_dist = np.array(
            [f.poc_distance_ticks for f in self._features], dtype=np.float64
        )

        # Compute realized_vol: 20-bar rolling std of log returns
        # Use the last n closes (aligned with features)
        aligned_closes = closes[-n:]
        log_rets = np.zeros(n, dtype=np.float64)
        log_rets[1:] = np.log(
            np.maximum(aligned_closes[1:], 1e-10) / np.maximum(aligned_closes[:-1], 1e-10)
        )

        realized_vol = np.full(n, np.nan, dtype=np.float64)
        for i in range(19, n):
            realized_vol[i] = np.std(log_rets[i - 19 : i + 1], ddof=1)

        # Compute return_20bar: 20-bar log return
        return_20bar = np.full(n, np.nan, dtype=np.float64)
        for i in range(20, n):
            return_20bar[i] = np.log(
                max(aligned_closes[i], 1e-10) / max(aligned_closes[i - 20], 1e-10)
            )

        # Stack raw (n, 6)
        raw = np.column_stack(
            [atr_ticks, vwap_dev_sd, cvd_slope, poc_dist, realized_vol, return_20bar]
        )

        # Rolling z-score normalization
        normed = self._rolling_zscore(raw)

        # Return last 50 valid rows
        last_50 = normed[-50:]

        # Check for NaN/inf
        if np.any(np.isnan(last_50)) or np.any(np.isinf(last_50)):
            return None

        return last_50

    def _rolling_zscore(self, raw: np.ndarray) -> np.ndarray:
        """Apply rolling z-score normalization per column.

        Uses cumulative sums for O(n) computation instead of O(n*window).
        Falls back to simple expanding window for NaN-heavy columns.
        """
        n, k = raw.shape
        normed = np.full_like(raw, np.nan)
        window = self._zscore_window

        for col in range(k):
            series = raw[:, col]
            nan_mask = np.isnan(series)

            # If too many NaNs, use the slow path for this column
            if nan_mask.sum() > n * 0.3:
                for i in range(1, n):
                    if nan_mask[i]:
                        continue
                    w = min(i + 1, window)
                    chunk = series[i - w + 1 : i + 1]
                    valid = chunk[~np.isnan(chunk)]
                    if len(valid) < 2:
                        continue
                    mean = np.mean(valid)
                    std = np.std(valid, ddof=1)
                    normed[i, col] = 0.0 if std < 1e-10 else (series[i] - mean) / std
            else:
                # Fast path: fill NaNs with 0 and use cumsum
                filled = np.where(nan_mask, 0.0, series)
                valid_count = np.cumsum(~nan_mask).astype(np.float64)
                cumsum = np.cumsum(filled)
                cumsum2 = np.cumsum(filled ** 2)

                for i in range(1, n):
                    if nan_mask[i]:
                        continue
                    w = min(i + 1, window)
                    j = i - w  # start index - 1
                    cnt = valid_count[i] - (valid_count[j] if j >= 0 else 0)
                    if cnt < 2:
                        continue
                    s = cumsum[i] - (cumsum[j] if j >= 0 else 0)
                    s2 = cumsum2[i] - (cumsum2[j] if j >= 0 else 0)
                    mean = s / cnt
                    var = (s2 / cnt) - mean ** 2
                    var = var * cnt / (cnt - 1)  # Bessel correction
                    std = np.sqrt(max(var, 0.0))
                    normed[i, col] = 0.0 if std < 1e-10 else (series[i] - mean) / std

        return normed

    def clear(self) -> None:
        """Clear all buffered data."""
        self._features.clear()
        self._closes.clear()


# US Eastern timezone (DST-aware)
from zoneinfo import ZoneInfo

_ET = ZoneInfo("US/Eastern")


def _parse_time(s: str) -> time:
    """Parse HH:MM string to time object."""
    parts = s.split(":")
    return time(int(parts[0]), int(parts[1]))


class StrategyBase(ABC):
    """Abstract base class for all trading strategies.

    Concrete strategies implement on_tick, on_bar, generate_signal, and reset.
    The base class provides session gating, HMM regime tracking, signal
    construction, and daily metrics.

    Strategies are pure state machines — no async, no EventBus ownership.
    Called externally by the orchestrator for easy testing/backtesting.
    """

    def __init__(
        self,
        config: StrategyConfig,
        feature_hub: FeatureHub,
        hmm_classifier: HMMRegimeClassifier | None = None,
    ) -> None:
        self.config = config
        self.feature_hub = feature_hub
        self.hmm_classifier = hmm_classifier

        self._hmm_buffer = HMMFeatureBuffer()
        self._current_regime: RegimeState = RegimeState.LOW_VOL_RANGE
        self._regime_probs: np.ndarray = np.zeros(len(RegimeState), dtype=np.float64)
        self._hmm_update_interval: int = 10  # Only run HMM every N bars
        self._hmm_bar_counter: int = 0
        self._signals_today: int = 0
        self._signals_generated: list[Signal] = []
        self._bars_processed: int = 0
        self._last_snapshot: FeatureVector | None = None

    # ── Abstract methods (concrete strategies implement) ─────────────

    @abstractmethod
    def on_tick(self, tick: TickEvent) -> None:
        """Process a real-time tick. Must be implemented by subclass."""
        ...

    @abstractmethod
    def on_bar(self, bar: BarEvent) -> None:
        """Process a completed bar. Must call _base_on_bar(bar) first."""
        ...

    @abstractmethod
    def generate_signal(self) -> Signal | None:
        """Attempt to generate a trading signal. Returns None if no signal."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset daily state. Must call super().reset()."""
        ...

    # ── Concrete methods (base provides) ─────────────────────────────

    def _base_on_bar(self, bar: BarEvent) -> None:
        """Update FeatureHub, take snapshot, feed HMM buffer, refresh regime.

        Concrete strategies must call this at the start of their on_bar().
        """
        self.feature_hub.on_bar(
            timestamp_ns=bar.timestamp_ns,
            open_=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )
        self._bars_processed += 1

        snapshot = self.feature_hub.snapshot()
        self._last_snapshot = snapshot

        # Feed HMM buffer
        self._hmm_buffer.update(snapshot, bar.close)

        # Refresh regime if HMM is available and buffer is ready
        # Only run every N bars — regime doesn't change per-minute
        if self.hmm_classifier is not None:
            self._hmm_bar_counter += 1
            if self._hmm_bar_counter >= self._hmm_update_interval:
                self._hmm_bar_counter = 0
                matrix = self._hmm_buffer.build_matrix()
                if matrix is not None:
                    state, probs = self.hmm_classifier.predict_proba(matrix)
                    self._current_regime = state
                    self._regime_probs = probs

    def is_active_session(self, now: datetime | None = None) -> bool:
        """Check if current time is within the trading session window.

        Args:
            now: Override current time (for testing). If None, uses UTC now
                 converted to ET.
        """
        if now is None:
            now = datetime.now(_ET)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=_ET)

        current_time = now.time()
        session_start = _parse_time(self.config.session_start)
        session_end = _parse_time(self.config.session_end)

        if not (session_start <= current_time <= session_end):
            return False

        # Check excluded windows
        for start_str, end_str in self.config.excluded_windows:
            ex_start = _parse_time(start_str)
            ex_end = _parse_time(end_str)
            if ex_start <= current_time <= ex_end:
                return False

        return True

    def is_valid_hmm_state(self) -> bool:
        """Check if current regime is in the allowed set.

        Returns True if require_hmm_states is empty (all states allowed).
        """
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
        """Construct a Signal, enforce min_confidence, increment counter.

        Returns None if confidence is below min_confidence threshold.
        """
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
            "hmm_buffer_size": len(self._hmm_buffer._features),
        }

    def reset(self) -> None:
        """Reset daily state. HMM buffer is NOT reset (preserves warmup)."""
        self._signals_today = 0
        self._signals_generated.clear()
        self._bars_processed = 0
        self._last_snapshot = None


# Avoid circular imports — import event types at module level for type hints
from src.core.events import BarEvent, TickEvent  # noqa: E402
