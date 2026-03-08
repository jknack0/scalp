"""VPIN (Volume-Synchronized Probability of Informed Trading) regime filter.

Accumulates trades into fixed-volume buckets, classifies each trade as
buyer- or seller-initiated via tick rule, and computes a rolling VPIN
metric. Classifies the market regime as "trending", "mean_reversion",
or "undefined" based on configurable thresholds.

Gating logic:
- VWAP strategy blocked in "trending" regime
- ORB and CVD strategies blocked in "mean_reversion" regime
- VolRegime strategy: no VPIN gate (handles its own regime)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl


Regime = Literal["mean_reversion", "trending", "undefined"]

# Strategies that should be blocked in each regime
BLOCKED_IN_TRENDING: frozenset[str] = frozenset({"vwap_reversion"})
BLOCKED_IN_MEAN_REVERSION: frozenset[str] = frozenset({"orb", "cvd_divergence"})


@dataclass(frozen=True)
class VPINConfig:
    """Tuning knobs for the VPIN monitor."""

    bucket_size: int = 100
    n_buckets: int = 50
    trending_threshold: float = 0.55
    mean_reversion_threshold: float = 0.38
    persist_every_n: int = 10
    parquet_dir: str = "data/vpin_states"


@dataclass(frozen=True)
class VPINState:
    """Snapshot of VPIN state after a bucket completes."""

    timestamp: datetime
    vpin: float
    regime: Regime
    bucket_count: int
    last_bucket_imbalance: float


@dataclass
class VPINMonitor:
    """Rolling VPIN calculator with regime classification and Parquet persistence.

    Feed trades via on_tick() for live L1 data, or on_bar_approx() for
    bar-level backtesting. Buckets complete when accumulated volume reaches
    bucket_size, at which point VPIN is recalculated.
    """

    config: VPINConfig = field(default_factory=VPINConfig)
    persist: bool = False

    def __post_init__(self) -> None:
        self._buy_vol: float = 0.0
        self._sell_vol: float = 0.0
        self._bucket_imbalances: deque[float] = deque(maxlen=self.config.n_buckets)
        self._bucket_count: int = 0
        self._prev_price: float | None = None
        self._latest_state: VPINState | None = None
        self._pending: list[VPINState] = []
        self._flush_count: int = 0
        self._last_timestamp: datetime | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_tick(self, price: float, size: int, bid: float, ask: float, timestamp: datetime | None = None) -> None:
        """Classify a trade and accumulate into the current volume bucket.

        Uses the same classification as CVDCalculator:
        - price >= ask -> buy
        - price <= bid -> sell
        - mid-spread -> tick rule
        """
        if timestamp is not None:
            self._last_timestamp = timestamp

        # Classify aggressor
        if price >= ask:
            self._buy_vol += size
        elif price <= bid:
            self._sell_vol += size
        else:
            # Tick rule fallback
            if self._prev_price is not None and price > self._prev_price:
                self._buy_vol += size
            elif self._prev_price is not None and price < self._prev_price:
                self._sell_vol += size
            # else: unchanged price, split evenly
            else:
                self._buy_vol += size / 2
                self._sell_vol += size / 2

        self._prev_price = price

        # Check if bucket is full
        total = self._buy_vol + self._sell_vol
        while total >= self.config.bucket_size:
            overflow = total - self.config.bucket_size
            # Scale buy/sell to fit exactly one bucket
            if total > 0:
                buy_ratio = self._buy_vol / total
            else:
                buy_ratio = 0.5
            bucket_buy = self.config.bucket_size * buy_ratio
            bucket_sell = self.config.bucket_size * (1 - buy_ratio)
            self._complete_bucket(bucket_buy, bucket_sell)

            # Carry forward overflow with same ratio
            self._buy_vol = overflow * buy_ratio
            self._sell_vol = overflow * (1 - buy_ratio)
            total = self._buy_vol + self._sell_vol

    def on_bar_approx(self, open_: float, close: float, volume: int, timestamp: datetime | None = None,
                       high: float | None = None, low: float | None = None) -> None:
        """Approximate VPIN from bar data (backtesting fallback).

        Uses the Bulk Volume Classification (BVC) method: buy_pct is
        proportional to (close - low) / (high - low) when H/L range is
        available. Falls back to binary classification if only open/close.
        """
        if timestamp is not None:
            self._last_timestamp = timestamp

        if volume <= 0:
            return

        # BVC: proportional split based on bar position within range
        if high is not None and low is not None and high > low:
            buy_pct = (close - low) / (high - low)
        elif close > open_:
            buy_pct = 1.0
        elif close < open_:
            buy_pct = 0.0
        else:
            buy_pct = 0.5

        self._buy_vol += volume * buy_pct
        self._sell_vol += volume * (1 - buy_pct)

        # Check if bucket is full
        total = self._buy_vol + self._sell_vol
        while total >= self.config.bucket_size:
            overflow = total - self.config.bucket_size
            if total > 0:
                buy_ratio = self._buy_vol / total
            else:
                buy_ratio = 0.5
            bucket_buy = self.config.bucket_size * buy_ratio
            bucket_sell = self.config.bucket_size * (1 - buy_ratio)
            self._complete_bucket(bucket_buy, bucket_sell)

            self._buy_vol = overflow * buy_ratio
            self._sell_vol = overflow * (1 - buy_ratio)
            total = self._buy_vol + self._sell_vol

    def get_regime(self) -> tuple[Regime, float]:
        """Return the current regime classification and VPIN value.

        Returns:
            (regime, vpin) tuple. regime is "undefined" if not enough buckets.
        """
        if self._latest_state is None:
            return "undefined", 0.0
        return self._latest_state.regime, self._latest_state.vpin

    def should_block(self, strategy_id: str) -> tuple[bool, str]:
        """Check whether current VPIN regime should block a given strategy.

        Returns:
            (blocked, reason) -- reason is empty string when not blocked.
        """
        regime, vpin = self.get_regime()

        if regime == "undefined":
            return False, ""

        if regime == "trending" and strategy_id in BLOCKED_IN_TRENDING:
            return True, f"VPIN regime=trending (vpin={vpin:.3f}), {strategy_id} blocked"

        if regime == "mean_reversion" and strategy_id in BLOCKED_IN_MEAN_REVERSION:
            return True, f"VPIN regime=mean_reversion (vpin={vpin:.3f}), {strategy_id} blocked"

        return False, ""

    @property
    def bucket_count(self) -> int:
        """Total number of completed volume buckets."""
        return self._bucket_count

    @property
    def latest_state(self) -> VPINState | None:
        """Most recently computed state, or None if no buckets completed."""
        return self._latest_state

    @property
    def vpin(self) -> float:
        """Current VPIN value, or 0.0 if no buckets."""
        return self._latest_state.vpin if self._latest_state else 0.0

    def reset(self) -> None:
        """Reset all state for a new session."""
        self._buy_vol = 0.0
        self._sell_vol = 0.0
        self._bucket_imbalances.clear()
        self._bucket_count = 0
        self._prev_price = None
        self._latest_state = None
        self._pending.clear()
        self._flush_count = 0
        self._last_timestamp = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _complete_bucket(self, buy_vol: float, sell_vol: float) -> None:
        """Finalize a volume bucket and recalculate VPIN."""
        imbalance = abs(buy_vol - sell_vol) / self.config.bucket_size
        self._bucket_imbalances.append(imbalance)
        self._bucket_count += 1

        # VPIN = rolling mean of bucket imbalances
        vpin = sum(self._bucket_imbalances) / len(self._bucket_imbalances)

        # Classify regime
        if len(self._bucket_imbalances) < 5:
            regime: Regime = "undefined"
        elif vpin > self.config.trending_threshold:
            regime = "trending"
        elif vpin < self.config.mean_reversion_threshold:
            regime = "mean_reversion"
        else:
            regime = "undefined"

        from datetime import datetime, timezone
        ts = self._last_timestamp or datetime.now(timezone.utc)

        state = VPINState(
            timestamp=ts,
            vpin=vpin,
            regime=regime,
            bucket_count=self._bucket_count,
            last_bucket_imbalance=imbalance,
        )
        self._latest_state = state

        # Batch for periodic Parquet flush
        if self.persist:
            self._pending.append(state)
            self._flush_count += 1
            if self._flush_count % self.config.persist_every_n == 0:
                self._flush_parquet()

    def _flush_parquet(self) -> None:
        """Append pending states to a date-partitioned Parquet file."""
        if not self._pending:
            return

        rows = [
            {
                "timestamp": s.timestamp,
                "vpin": s.vpin,
                "regime": s.regime,
                "bucket_count": s.bucket_count,
                "last_bucket_imbalance": s.last_bucket_imbalance,
            }
            for s in self._pending
        ]
        df = pl.DataFrame(rows)

        date_str = self._pending[-1].timestamp.strftime("%Y-%m-%d")
        out_dir = Path(self.config.parquet_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"vpin_{date_str}.parquet"

        if out_path.exists():
            existing = pl.read_parquet(out_path)
            df = pl.concat([existing, df])

        df.write_parquet(out_path, compression="zstd")
        self._pending.clear()
