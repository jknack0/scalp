"""L2 order book strategy — iceberg detection + absorption trading.

Combines two L2 signals into a single strategy:

1. ICEBERG: Hidden order detected (level persists, absorbs flow, refreshes)
   → Trade WITH the iceberg (front-run institutional flow)

2. ABSORPTION: Level repeatedly tested but holds (no iceberg required)
   → Fade into the level (it's real S/R)
   → If level breaks → stop out

Iceberg signals get a confidence boost since the hidden order is stronger
evidence of institutional intent.

Entry: Signal on bid side → LONG, signal on ask side → SHORT
Stop:  Beyond the signal level (if it breaks, edge is gone)
Target: Fixed tick distance from entry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from typing import Literal

from src.filters import L2Snapshot
from src.filters.iceberg_absorption import (
    AbsorptionDetector,
    AbsorptionSignal,
    IcebergConfig,
    IcebergDetector,
    IcebergSignal,
    TradeEvent,
)
from src.models.hmm_regime import RegimeState
from src.strategies.base import TICK_SIZE, Direction, Signal


@dataclass
class L2StrategyConfig:
    """Configuration for the combined L2 strategy."""

    strategy_id: str = "l2_book"

    # --- Iceberg detection ---
    min_appearances: int = 10
    consumed_threshold: int = 50
    min_iceberg_confidence: float = 0.3
    top_n_levels: int = 3

    # --- Absorption detection ---
    test_threshold: int = 5  # min tests to trigger absorption
    window_seconds: float = 30.0  # absorption window
    min_absorbed_size: int = 30  # min contracts absorbed

    # --- Signal weighting ---
    # Iceberg signals get this confidence boost on top of detector confidence
    iceberg_boost: float = 0.2
    # Absorption-only signals get a flat confidence
    absorption_base_confidence: float = 0.4

    # --- Trade geometry (ticks: 1 tick = 0.25 pts = $1.25) ---
    target_ticks: int = 8
    stop_ticks: int = 8  # beyond signal level

    # --- Position management ---
    max_signals_per_day: int = 5
    expiry_seconds: int = 300  # 5 min max hold

    # --- L2 sampling rate for detectors (ms) ---
    l2_sample_ms: int = 100

    # --- Session (ET) ---
    session_start: str = "09:35"  # skip first 5min of open
    session_end: str = "15:50"  # stop 10min before close


@dataclass(frozen=True)
class L2Signal:
    """Internal signal before conversion to trade Signal."""

    timestamp: datetime
    price: float  # the S/R level
    side: Literal["bid", "ask"]
    confidence: float
    signal_type: Literal["iceberg", "absorption", "iceberg+absorption"]
    estimated_size: int  # total size absorbed/hidden
    metadata: dict = field(default_factory=dict)


@dataclass
class L2Strategy:
    """Combined iceberg + absorption L2 strategy.

    Not a StrategyBase subclass — operates on L2 snapshots + trades,
    not bar events.
    """

    config: L2StrategyConfig = field(default_factory=L2StrategyConfig)

    def __post_init__(self) -> None:
        iceberg_cfg = IcebergConfig(
            min_appearances=self.config.min_appearances,
            consumed_threshold=self.config.consumed_threshold,
            top_n_levels=self.config.top_n_levels,
            test_threshold=self.config.test_threshold,
            window_seconds=self.config.window_seconds,
            persist=False,
        )
        self._iceberg_det = IcebergDetector(config=iceberg_cfg)
        self._absorption_det = AbsorptionDetector(config=iceberg_cfg)
        self._last_l2_ns: int = 0
        self._signals_today: int = 0
        self._current_date: datetime | None = None
        self._session_start = self._parse_time(self.config.session_start)
        self._session_end = self._parse_time(self.config.session_end)
        # Track recently signaled levels to avoid duplicates
        self._signaled_levels: set[tuple[float, str]] = set()
        # Last absorption signals (for engine to check level breaks)
        self._last_absorption_signals: list[AbsorptionSignal] = []

    def on_l2(self, snapshot: L2Snapshot) -> Signal | None:
        """Process an L2 snapshot. Returns Signal if entry triggered."""
        # Daily reset
        if (
            self._current_date is None
            or snapshot.timestamp.date() != self._current_date.date()
        ):
            self.reset()
            self._current_date = snapshot.timestamp

        if not self._is_session(snapshot.timestamp):
            return None

        if self._signals_today >= self.config.max_signals_per_day:
            return None

        # Rate-limit L2 feeds to detectors
        ts_ns = int(snapshot.timestamp.timestamp() * 1e9)
        sample_ns = self.config.l2_sample_ms * 1_000_000
        if ts_ns - self._last_l2_ns < sample_ns:
            return None
        self._last_l2_ns = ts_ns

        # Feed both detectors
        iceberg_signals = self._iceberg_det.push_l2(snapshot)
        absorption_signals = self._absorption_det.push_l2(snapshot)
        self._last_absorption_signals = absorption_signals

        # Build unified signal list
        best = self._pick_best_signal(
            iceberg_signals, absorption_signals, snapshot.timestamp
        )
        if best is None:
            return None

        # Don't re-signal the same level
        level_key = (best.price, best.side)
        if level_key in self._signaled_levels:
            return None
        self._signaled_levels.add(level_key)

        return self._build_trade_signal(best, snapshot)

    def on_trade(self, trade: TradeEvent) -> None:
        """Feed a trade to detectors (tracks consumption at levels)."""
        self._iceberg_det.push_trade(trade)
        self._absorption_det.push_trade(trade)

    def get_absorption_signals(self) -> list[AbsorptionSignal]:
        """Return absorption signals from the last on_l2() call.

        Used by the replay engine to detect when a position's level breaks.
        Does NOT re-push data to the detector.
        """
        return self._last_absorption_signals

    def _pick_best_signal(
        self,
        icebergs: list[IcebergSignal],
        absorptions: list[AbsorptionSignal],
        ts: datetime,
    ) -> L2Signal | None:
        """Merge iceberg + absorption signals, pick the best one."""
        candidates: list[L2Signal] = []

        # Index absorption signals by (price, side) for cross-referencing
        absorbing_levels: dict[tuple[float, str], AbsorptionSignal] = {}
        for a in absorptions:
            if a.status == "absorbing":
                absorbing_levels[(a.price, a.side)] = a

        # Process icebergs
        for ice in icebergs:
            if ice.confidence < self.config.min_iceberg_confidence:
                continue

            # Check if this level is also absorbing
            key = (ice.price, ice.side)
            if key in absorbing_levels:
                ab = absorbing_levels.pop(key)  # consume it
                confidence = min(
                    ice.confidence + self.config.iceberg_boost, 1.0
                )
                candidates.append(
                    L2Signal(
                        timestamp=ts,
                        price=ice.price,
                        side=ice.side,
                        confidence=confidence,
                        signal_type="iceberg+absorption",
                        estimated_size=ice.estimated_total_size
                        + ab.absorbed_size,
                        metadata={
                            "iceberg_confidence": ice.confidence,
                            "absorption_tests": ab.test_count,
                            "absorption_absorbed": ab.absorbed_size,
                        },
                    )
                )
            else:
                candidates.append(
                    L2Signal(
                        timestamp=ts,
                        price=ice.price,
                        side=ice.side,
                        confidence=min(
                            ice.confidence + self.config.iceberg_boost, 1.0
                        ),
                        signal_type="iceberg",
                        estimated_size=ice.estimated_total_size,
                    )
                )

        # Remaining absorption signals (not paired with iceberg)
        for key, ab in absorbing_levels.items():
            if ab.absorbed_size < self.config.min_absorbed_size:
                continue
            candidates.append(
                L2Signal(
                    timestamp=ts,
                    price=ab.price,
                    side=ab.side,
                    confidence=self.config.absorption_base_confidence,
                    signal_type="absorption",
                    estimated_size=ab.absorbed_size,
                    metadata={
                        "test_count": ab.test_count,
                    },
                )
            )

        if not candidates:
            return None

        # Pick highest confidence
        return max(candidates, key=lambda s: s.confidence)

    def _build_trade_signal(
        self, sig: L2Signal, snapshot: L2Snapshot
    ) -> Signal | None:
        """Convert an L2Signal into a trade Signal."""
        if not snapshot.bids or not snapshot.asks:
            return None

        best_bid = snapshot.bids[0][0]
        best_ask = snapshot.asks[0][0]
        tick = TICK_SIZE

        if sig.side == "bid":
            # S/R on bid side (support) → LONG
            direction = Direction.LONG
            entry = best_ask  # buy at ask
            stop = sig.price - (self.config.stop_ticks * tick)
            target = entry + (self.config.target_ticks * tick)
        else:
            # S/R on ask side (resistance) → SHORT
            direction = Direction.SHORT
            entry = best_bid  # sell at bid
            stop = sig.price + (self.config.stop_ticks * tick)
            target = entry - (self.config.target_ticks * tick)

        self._signals_today += 1

        return Signal(
            strategy_id=self.config.strategy_id,
            direction=direction,
            entry_price=entry,
            target_price=target,
            stop_price=stop,
            signal_time=sig.timestamp,
            expiry_time=sig.timestamp
            + timedelta(seconds=self.config.expiry_seconds),
            confidence=sig.confidence,
            regime_state=RegimeState.LOW_VOL_RANGE,
            metadata={
                "signal_type": sig.signal_type,
                "level_price": sig.price,
                "level_side": sig.side,
                "estimated_size": sig.estimated_size,
                **sig.metadata,
            },
        )

    def _is_session(self, ts: datetime) -> bool:
        """Check RTH session bounds (expects ET timestamps)."""
        t = ts.time()
        return self._session_start <= t <= self._session_end

    def reset(self) -> None:
        """Reset daily state. Detectors keep their level tracking."""
        self._signals_today = 0
        self._signaled_levels.clear()

    @staticmethod
    def _parse_time(s: str) -> dt_time:
        parts = s.split(":")
        return dt_time(int(parts[0]), int(parts[1]))
