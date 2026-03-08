"""CVD Divergence + POC strategy.

Detects when price and cumulative volume delta disagree at swing points
(exhaustion signal), filtered by proximity to prior session POC.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
    from src.models.hmm_regime import HMMRegimeClassifier

logger = get_logger("cvd_divergence")

# Avoid circular import
from src.core.events import BarEvent, TickEvent  # noqa: E402


# ── Data structures ─────────────────────────────────────────────────


class SwingType(Enum):
    HIGH = "HIGH"
    LOW = "LOW"


@dataclass(frozen=True)
class Swing:
    """A confirmed swing point with price and CVD snapshot."""

    bar_index: int
    price: float
    cvd_value: float
    timestamp_ns: int
    swing_type: SwingType


@dataclass(frozen=True)
class Divergence:
    """Detected divergence between price and CVD at swing points."""

    divergence_type: Direction  # SHORT for bearish, LONG for bullish
    magnitude: float  # 0.0–1.0
    current_swing: Swing
    prev_swing: Swing


# ── Swing detector ──────────────────────────────────────────────────


class SwingDetector:
    """Detects local swing highs/lows using a lookback window.

    Buffers 2*lookback + 1 bars. When the center bar's close is higher
    (or lower) than all surrounding closes, it's a swing high (or low).
    Swings are confirmed lookback bars late.
    """

    def __init__(self, lookback: int = 5, max_swings: int = 5) -> None:
        self._lookback = lookback
        self._window_size = 2 * lookback + 1
        self._max_swings = max_swings

        # Circular buffer of (bar_index, close, cvd_value, timestamp_ns)
        self._buffer: deque[tuple[int, float, float, int]] = deque(
            maxlen=self._window_size
        )

        self._recent_highs: deque[Swing] = deque(maxlen=max_swings)
        self._recent_lows: deque[Swing] = deque(maxlen=max_swings)

    def on_bar(
        self, bar_index: int, close: float, cvd_value: float, timestamp_ns: int
    ) -> list[Swing]:
        """Process a bar and return any newly detected swings (0 or 1)."""
        self._buffer.append((bar_index, close, cvd_value, timestamp_ns))

        if len(self._buffer) < self._window_size:
            return []

        # Check center bar
        center = self._lookback
        center_close = self._buffer[center][1]

        swings: list[Swing] = []

        # Swing high: center close > all surrounding closes
        is_high = all(
            center_close > self._buffer[i][1]
            for i in range(self._window_size)
            if i != center
        )
        if is_high:
            swing = Swing(
                bar_index=self._buffer[center][0],
                price=center_close,
                cvd_value=self._buffer[center][2],
                timestamp_ns=self._buffer[center][3],
                swing_type=SwingType.HIGH,
            )
            self._recent_highs.append(swing)
            swings.append(swing)

        # Swing low: center close < all surrounding closes
        is_low = all(
            center_close < self._buffer[i][1]
            for i in range(self._window_size)
            if i != center
        )
        if is_low:
            swing = Swing(
                bar_index=self._buffer[center][0],
                price=center_close,
                cvd_value=self._buffer[center][2],
                timestamp_ns=self._buffer[center][3],
                swing_type=SwingType.LOW,
            )
            self._recent_lows.append(swing)
            swings.append(swing)

        return swings

    @property
    def recent_highs(self) -> list[Swing]:
        return list(self._recent_highs)

    @property
    def recent_lows(self) -> list[Swing]:
        return list(self._recent_lows)


# ── Divergence detector ────────────────────────────────────────────


class DivergenceDetector:
    """Checks for price/CVD divergence between consecutive swing points."""

    def __init__(self, threshold_pct: float = 0.15) -> None:
        self._threshold_pct = threshold_pct

    def check_bearish(self, swing_highs: list[Swing]) -> Divergence | None:
        """Bearish divergence: price higher high + CVD lower high."""
        if len(swing_highs) < 2:
            return None

        prev = swing_highs[-2]
        curr = swing_highs[-1]

        # Price higher high
        if curr.price <= prev.price:
            return None

        # CVD lower high
        if curr.cvd_value >= prev.cvd_value:
            return None

        magnitude = self._magnitude(prev.cvd_value, curr.cvd_value)
        if magnitude < self._threshold_pct:
            return None

        return Divergence(
            divergence_type=Direction.SHORT,
            magnitude=min(magnitude, 1.0),
            current_swing=curr,
            prev_swing=prev,
        )

    def check_bullish(self, swing_lows: list[Swing]) -> Divergence | None:
        """Bullish divergence: price lower low + CVD higher low."""
        if len(swing_lows) < 2:
            return None

        prev = swing_lows[-2]
        curr = swing_lows[-1]

        # Price lower low
        if curr.price >= prev.price:
            return None

        # CVD higher low
        if curr.cvd_value <= prev.cvd_value:
            return None

        magnitude = self._magnitude(prev.cvd_value, curr.cvd_value)
        if magnitude < self._threshold_pct:
            return None

        return Divergence(
            divergence_type=Direction.LONG,
            magnitude=min(magnitude, 1.0),
            current_swing=curr,
            prev_swing=prev,
        )

    @staticmethod
    def _magnitude(prev_cvd: float, curr_cvd: float) -> float:
        """CVD divergence magnitude: |prev - curr| / |prev|."""
        if abs(prev_cvd) < 1e-10:
            return 0.0
        return abs(prev_cvd - curr_cvd) / abs(prev_cvd)


# ── Configuration ───────────────────────────────────────────────────


@dataclass
class CVDDivergenceConfig(StrategyConfig):
    """CVD Divergence strategy configuration."""

    strategy_id: str = "cvd_divergence"
    swing_lookback_bars: int = 5
    divergence_threshold_pct: float = 0.15
    poc_proximity_ticks: int = 6
    target_ticks: int = 14  # 3.5 points
    stop_buffer_ticks: int = 2
    max_hold_bars: int = 4
    max_signals_per_day: int = 3
    require_hmm_states: list[RegimeState] = field(
        default_factory=lambda: [
            RegimeState.HIGH_VOL_UP,
            RegimeState.HIGH_VOL_DOWN,
            RegimeState.BREAKOUT,
        ]
    )
    expiry_minutes: int = 30


# ── Strategy ────────────────────────────────────────────────────────


class CVDDivergenceStrategy(StrategyBase):
    """CVD Divergence strategy: fade exhaustion at swing points near POC.

    Detects when price makes a new swing high/low but CVD doesn't confirm,
    indicating order flow exhaustion. Filtered by proximity to prior session
    POC for confluence.
    """

    def __init__(
        self,
        config: CVDDivergenceConfig,
        feature_hub: FeatureHub,
        hmm_classifier: HMMRegimeClassifier | None = None,
    ) -> None:
        super().__init__(config, feature_hub, hmm_classifier)
        self._cvd_config = config
        self._swing_detector = SwingDetector(
            lookback=config.swing_lookback_bars,
        )
        self._divergence_detector = DivergenceDetector(
            threshold_pct=config.divergence_threshold_pct,
        )
        self._bar_index: int = 0
        self._last_divergence_bar: int = -1

    def on_tick(self, tick: TickEvent) -> None:
        """No-op — CVD divergence is bar-driven."""

    def on_bar(self, bar: BarEvent) -> None:
        """Process a completed bar: update features, detect swings, check divergence."""
        self._base_on_bar(bar)
        self._bar_index += 1

        cvd_value = self.feature_hub.cvd.cvd
        new_swings = self._swing_detector.on_bar(
            self._bar_index, bar.close, cvd_value, bar.timestamp_ns
        )

        if new_swings:
            sig = self.generate_signal(bar)
            # Signal is stored internally via _make_signal

    def generate_signal(self, bar: BarEvent | None = None) -> Signal | None:
        """Check for divergence at recent swings and generate signal if valid."""
        if bar is None:
            return None

        bar_dt = datetime.fromtimestamp(
            bar.timestamp_ns / 1_000_000_000, tz=_ET
        )

        if not self.can_generate_signal(bar_dt):
            return None

        # Check bearish divergence first, then bullish
        divergence = self._divergence_detector.check_bearish(
            self._swing_detector.recent_highs
        )
        if divergence is None:
            divergence = self._divergence_detector.check_bullish(
                self._swing_detector.recent_lows
            )

        if divergence is None:
            return None

        # Dedup: don't fire twice from the same swing
        if divergence.current_swing.bar_index == self._last_divergence_bar:
            return None

        # POC proximity filter
        poc_dist = self.feature_hub.volume_profile.poc_distance_ticks
        if poc_dist > self._cvd_config.poc_proximity_ticks:
            return None

        # Entry geometry
        entry = bar.close
        cfg = self._cvd_config

        if divergence.divergence_type == Direction.SHORT:
            target = entry - cfg.target_ticks * TICK_SIZE
            stop = divergence.current_swing.price + cfg.stop_buffer_ticks * TICK_SIZE
        else:
            target = entry + cfg.target_ticks * TICK_SIZE
            stop = divergence.current_swing.price - cfg.stop_buffer_ticks * TICK_SIZE

        # Confidence: divergence magnitude * POC proximity factor
        max_dist = float(cfg.poc_proximity_ticks)
        if max_dist > 0:
            poc_factor = 1.0 - 0.5 * (poc_dist / max_dist)
        else:
            poc_factor = 1.0
        confidence = divergence.magnitude * poc_factor

        # Metadata
        vp = self.feature_hub.volume_profile
        metadata = {
            "divergence_type": divergence.divergence_type.value,
            "magnitude": round(divergence.magnitude, 4),
            "current_swing_price": divergence.current_swing.price,
            "prev_swing_price": divergence.prev_swing.price,
            "current_swing_cvd": divergence.current_swing.cvd_value,
            "prev_swing_cvd": divergence.prev_swing.cvd_value,
            "poc_distance_ticks": round(poc_dist, 2),
            "prior_poc": vp.prior_poc,
            "prior_vah": vp.prior_vah,
            "prior_val": vp.prior_val,
            "hmm_state": self._current_regime.name,
        }

        # Expiry: max_hold_bars * 5 minutes (5-minute bars assumed)
        expiry_seconds = cfg.max_hold_bars * 5 * 60

        self._last_divergence_bar = divergence.current_swing.bar_index

        return self._make_signal(
            direction=divergence.divergence_type,
            entry=entry,
            target=target,
            stop=stop,
            confidence=confidence,
            expiry_seconds=expiry_seconds,
            metadata=metadata,
            now=bar_dt,
        )

    def reset(self) -> None:
        """Reset daily state and re-create internal detectors."""
        super().reset()
        self._swing_detector = SwingDetector(
            lookback=self._cvd_config.swing_lookback_bars,
        )
        self._divergence_detector = DivergenceDetector(
            threshold_pct=self._cvd_config.divergence_threshold_pct,
        )
        self._bar_index = 0
        self._last_divergence_bar = -1
