"""L2 order book features: absorption, spoofing, depth, weighted mid.

Consumes L2 snapshots (10 depth levels per side) and computes microstructure
signals. Designed for both live streaming and bar-aggregated backtesting.

Features:
- Weighted Mid Price: size-weighted mid vs raw mid (skew detection)
- Depth Deterioration: current depth vs rolling average (thinning = vol expansion)
- Absorption Detection: passive order absorbing aggressive flow without moving
- Spoof Score: levels appearing/disappearing without getting traded
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class L2Level:
    """Single price level in the order book."""
    price: float
    size: int


@dataclass(frozen=True, slots=True)
class L2Snapshot:
    """Full L2 order book snapshot (up to 10 levels per side)."""
    timestamp_ns: int
    bids: tuple[L2Level, ...]  # best bid first (descending price)
    asks: tuple[L2Level, ...]  # best ask first (ascending price)


@dataclass(frozen=True, slots=True)
class L2Signals:
    """Computed L2 order book signals."""
    # Mid price signals
    raw_mid: float
    weighted_mid: float
    mid_skew: float  # weighted_mid - raw_mid (positive = bid pressure)

    # Depth signals
    total_bid_depth: int
    total_ask_depth: int
    bid_depth_ratio: float  # current / rolling avg (< 1 = thinning)
    ask_depth_ratio: float

    # Depth imbalance across all levels
    depth_imbalance: float  # (bid - ask) / (bid + ask) over top N levels

    # Mid price drift (slope of weighted mid over recent snapshots)
    mid_drift: float

    # Absorption
    bid_absorption: bool  # passive bid absorbing sells
    ask_absorption: bool  # passive ask absorbing buys
    absorption_level: float  # price level where absorption detected (0 if none)
    absorption_size_consumed: int  # total aggressive volume absorbed

    # Spoofing
    spoof_score: int  # count of levels that appeared/disappeared without trading


# ---------------------------------------------------------------------------
# Weighted Mid Price
# ---------------------------------------------------------------------------

def weighted_mid(snap: L2Snapshot) -> tuple[float, float, float]:
    """Compute raw mid, weighted mid, and skew.

    Weighted mid uses top-of-book sizes as weights:
        wmid = (ask_size * bid_price + bid_size * ask_price) / (bid_size + ask_size)

    When bid_size > ask_size, wmid shifts toward ask (price is pulled up).

    Returns:
        (raw_mid, weighted_mid, skew)
    """
    if not snap.bids or not snap.asks:
        mid = snap.bids[0].price if snap.bids else (snap.asks[0].price if snap.asks else 0.0)
        return mid, mid, 0.0

    bid = snap.bids[0]
    ask = snap.asks[0]
    raw = (bid.price + ask.price) / 2.0

    total_size = bid.size + ask.size
    if total_size == 0:
        return raw, raw, 0.0

    wmid = (ask.size * bid.price + bid.size * ask.price) / total_size
    return raw, wmid, wmid - raw


# ---------------------------------------------------------------------------
# Depth Deterioration
# ---------------------------------------------------------------------------

class DepthTracker:
    """Tracks rolling average depth and computes deterioration ratios."""

    def __init__(self, window: int = 30, n_levels: int = 5) -> None:
        self._window = window
        self._n_levels = n_levels
        self._bid_depths: deque[int] = deque(maxlen=window)
        self._ask_depths: deque[int] = deque(maxlen=window)

    def update(self, snap: L2Snapshot) -> tuple[int, int, float, float]:
        """Update with new snapshot, return (bid_depth, ask_depth, bid_ratio, ask_ratio)."""
        bid_depth = sum(lv.size for lv in snap.bids[:self._n_levels])
        ask_depth = sum(lv.size for lv in snap.asks[:self._n_levels])

        self._bid_depths.append(bid_depth)
        self._ask_depths.append(ask_depth)

        if len(self._bid_depths) < 2:
            return bid_depth, ask_depth, 1.0, 1.0

        avg_bid = np.mean(self._bid_depths)
        avg_ask = np.mean(self._ask_depths)

        bid_ratio = bid_depth / avg_bid if avg_bid > 0 else 1.0
        ask_ratio = ask_depth / avg_ask if avg_ask > 0 else 1.0

        return bid_depth, ask_depth, float(bid_ratio), float(ask_ratio)


# ---------------------------------------------------------------------------
# Absorption Detection
# ---------------------------------------------------------------------------

@dataclass
class _LevelTrack:
    """Tracks a single price level across consecutive snapshots."""
    price: float
    initial_size: int
    current_size: int
    consecutive_hits: int = 0  # snapshots where size decreased but level held
    total_consumed: int = 0


class AbsorptionDetector:
    """Detects passive order absorption at price levels.

    Absorption = a price level stays in the top N for many consecutive
    snapshots while its size is being consumed (decreasing) but the level
    doesn't move. This indicates a large hidden/iceberg order.
    """

    def __init__(
        self,
        n_levels: int = 3,
        min_consecutive: int = 10,
        min_consumption_pct: float = 0.3,
    ) -> None:
        self._n_levels = n_levels
        self._min_consecutive = min_consecutive
        self._min_consumption_pct = min_consumption_pct
        self._bid_tracks: dict[float, _LevelTrack] = {}
        self._ask_tracks: dict[float, _LevelTrack] = {}

    def update(self, snap: L2Snapshot) -> tuple[bool, bool, float, int]:
        """Check for absorption on both sides.

        Returns:
            (bid_absorption, ask_absorption, absorption_price, consumed_size)
        """
        bid_abs, bid_price, bid_consumed = self._check_side(
            [lv for lv in snap.bids[:self._n_levels]],
            self._bid_tracks,
        )
        ask_abs, ask_price, ask_consumed = self._check_side(
            [lv for lv in snap.asks[:self._n_levels]],
            self._ask_tracks,
        )

        # Return the stronger signal
        if bid_abs and ask_abs:
            if bid_consumed >= ask_consumed:
                return True, False, bid_price, bid_consumed
            return False, True, ask_price, ask_consumed
        if bid_abs:
            return True, False, bid_price, bid_consumed
        if ask_abs:
            return False, True, ask_price, ask_consumed
        return False, False, 0.0, 0

    def _check_side(
        self,
        levels: list[L2Level],
        tracks: dict[float, _LevelTrack],
    ) -> tuple[bool, float, int]:
        """Check one side for absorption. Returns (detected, price, consumed)."""
        current_prices = {lv.price for lv in levels}

        # Remove tracks for levels that disappeared
        gone = [p for p in tracks if p not in current_prices]
        for p in gone:
            del tracks[p]

        detected = False
        best_price = 0.0
        best_consumed = 0

        for lv in levels:
            if lv.price in tracks:
                track = tracks[lv.price]
                if lv.size < track.current_size:
                    # Size decreased but level held — absorption
                    consumed = track.current_size - lv.size
                    track.total_consumed += consumed
                    track.consecutive_hits += 1
                    track.current_size = lv.size
                elif lv.size > track.current_size:
                    # Size increased — replenishment (iceberg reload)
                    track.consecutive_hits += 1
                    track.total_consumed += track.current_size  # old size consumed
                    track.current_size = lv.size
                else:
                    # No change — still holding
                    track.consecutive_hits += 1

                # Check if absorption threshold met
                if (
                    track.consecutive_hits >= self._min_consecutive
                    and track.total_consumed > track.initial_size * self._min_consumption_pct
                    and track.total_consumed > best_consumed
                ):
                    detected = True
                    best_price = lv.price
                    best_consumed = track.total_consumed
            else:
                # New level appearing — start tracking
                tracks[lv.price] = _LevelTrack(
                    price=lv.price,
                    initial_size=lv.size,
                    current_size=lv.size,
                )

        return detected, best_price, best_consumed


# ---------------------------------------------------------------------------
# Spoof Detection
# ---------------------------------------------------------------------------

class SpoofDetector:
    """Detects potential spoofing: levels that appear and vanish without trading.

    A spoof is a level that:
    1. Appeared in the top N levels
    2. Disappeared within `lookback` snapshots
    3. Was not at the best bid/ask (so it wasn't traded through)
    """

    def __init__(self, n_levels: int = 5, lookback: int = 5) -> None:
        self._n_levels = n_levels
        self._lookback = lookback
        self._recent_bids: deque[set[float]] = deque(maxlen=lookback)
        self._recent_asks: deque[set[float]] = deque(maxlen=lookback)
        self._recent_best_bid: deque[float] = deque(maxlen=lookback)
        self._recent_best_ask: deque[float] = deque(maxlen=lookback)

    def update(self, snap: L2Snapshot) -> int:
        """Count spoof-like events in the recent window.

        Returns:
            Number of levels that appeared and disappeared without being
            at the best price (i.e., not likely traded).
        """
        bid_prices = {lv.price for lv in snap.bids[:self._n_levels]}
        ask_prices = {lv.price for lv in snap.asks[:self._n_levels]}
        best_bid = snap.bids[0].price if snap.bids else 0.0
        best_ask = snap.asks[0].price if snap.asks else 0.0

        self._recent_bids.append(bid_prices)
        self._recent_asks.append(ask_prices)
        self._recent_best_bid.append(best_bid)
        self._recent_best_ask.append(best_ask)

        if len(self._recent_bids) < 3:
            return 0

        spoof_count = 0
        current_bids = self._recent_bids[-1]
        current_asks = self._recent_asks[-1]

        # Check bids: levels present in middle snapshots but gone now
        for i in range(len(self._recent_bids) - 2):
            past_bids = self._recent_bids[i]
            appeared = past_bids - (self._recent_bids[max(0, i - 1)] if i > 0 else set())
            for price in appeared:
                if price not in current_bids and price != self._recent_best_bid[i]:
                    spoof_count += 1

        # Check asks
        for i in range(len(self._recent_asks) - 2):
            past_asks = self._recent_asks[i]
            appeared = past_asks - (self._recent_asks[max(0, i - 1)] if i > 0 else set())
            for price in appeared:
                if price not in current_asks and price != self._recent_best_ask[i]:
                    spoof_count += 1

        return spoof_count


# ---------------------------------------------------------------------------
# Mid Price Drift (linear regression slope)
# ---------------------------------------------------------------------------

class MidDriftTracker:
    """Tracks weighted mid price drift via linear regression slope."""

    def __init__(self, window: int = 20) -> None:
        self._window = window
        self._mids: deque[float] = deque(maxlen=window)

    def update(self, weighted_mid: float) -> float:
        """Add new weighted mid, return slope (points per snapshot)."""
        self._mids.append(weighted_mid)
        if len(self._mids) < 3:
            return 0.0

        y = np.array(self._mids)
        x = np.arange(len(y), dtype=np.float64)
        # Fast slope: cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2))
        return slope


# ---------------------------------------------------------------------------
# Composite L2 Feature Calculator
# ---------------------------------------------------------------------------

class L2BookFeatures:
    """Composite L2 feature calculator.

    Maintains all sub-calculators and produces an L2Signals snapshot
    from each L2Snapshot update.

    Usage:
        l2 = L2BookFeatures()
        for snap in l2_snapshots:
            signals = l2.on_snapshot(snap)
    """

    def __init__(
        self,
        depth_window: int = 30,
        depth_levels: int = 5,
        absorption_levels: int = 3,
        absorption_min_consecutive: int = 10,
        spoof_levels: int = 5,
        spoof_lookback: int = 5,
        drift_window: int = 20,
    ) -> None:
        self._depth = DepthTracker(window=depth_window, n_levels=depth_levels)
        self._absorption = AbsorptionDetector(
            n_levels=absorption_levels,
            min_consecutive=absorption_min_consecutive,
        )
        self._spoof = SpoofDetector(n_levels=spoof_levels, lookback=spoof_lookback)
        self._drift = MidDriftTracker(window=drift_window)
        self._depth_levels = depth_levels
        self._last_signals: L2Signals | None = None

    @property
    def snapshot(self) -> L2Signals | None:
        """Most recent computed signals."""
        return self._last_signals

    def on_snapshot(self, snap: L2Snapshot) -> L2Signals:
        """Process an L2 snapshot and compute all signals."""
        # Mid prices
        raw, wmid, skew = weighted_mid(snap)

        # Depth
        bid_depth, ask_depth, bid_ratio, ask_ratio = self._depth.update(snap)
        total = bid_depth + ask_depth
        depth_imb = (bid_depth - ask_depth) / total if total > 0 else 0.0

        # Drift
        drift = self._drift.update(wmid)

        # Absorption
        bid_abs, ask_abs, abs_price, abs_consumed = self._absorption.update(snap)

        # Spoofing
        spoof = self._spoof.update(snap)

        signals = L2Signals(
            raw_mid=raw,
            weighted_mid=wmid,
            mid_skew=skew,
            total_bid_depth=bid_depth,
            total_ask_depth=ask_depth,
            bid_depth_ratio=bid_ratio,
            ask_depth_ratio=ask_ratio,
            depth_imbalance=depth_imb,
            mid_drift=drift,
            bid_absorption=bid_abs,
            ask_absorption=ask_abs,
            absorption_level=abs_price,
            absorption_size_consumed=abs_consumed,
            spoof_score=spoof,
        )
        self._last_signals = signals
        return signals
