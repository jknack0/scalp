"""Order Book Imbalance (OBI) feature calculator.

Computes rolling bid/ask size imbalance from L1-enriched bars.
Each bar must have avg_bid_size and avg_ask_size columns (aggregated
from L1 TBBO tick data).

Key features:
- raw_imbalance: (bid - ask) / (bid + ask), range [-1, +1]
- smoothed_imbalance: EMA of raw_imbalance
- imbalance_zscore: rolling z-score of raw_imbalance
- aggressive_ratio: aggressive_buy_vol / (aggressive_buy_vol + aggressive_sell_vol)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class OBISnapshot:
    """Point-in-time order book imbalance features."""

    raw_imbalance: float = 0.0       # (bid - ask) / (bid + ask), [-1, +1]
    smoothed_imbalance: float = 0.0  # EMA of raw_imbalance
    imbalance_zscore: float = 0.0    # rolling z-score
    aggressive_ratio: float = 0.5    # buy_vol / (buy_vol + sell_vol), [0, 1]
    bid_size: float = 0.0
    ask_size: float = 0.0


class OrderBookImbalance:
    """Tracks order book imbalance from L1-enriched bar data.

    Call on_bar() with bid/ask sizes and aggressive volumes each bar.
    Then read the snapshot for current feature values.
    """

    def __init__(
        self,
        ema_span: int = 20,
        zscore_window: int = 50,
    ) -> None:
        self._ema_span = ema_span
        self._ema_alpha = 2.0 / (ema_span + 1)
        self._zscore_window = zscore_window

        self._raw_history: deque[float] = deque(maxlen=zscore_window)
        self._smoothed: float = 0.0
        self._initialized: bool = False

        # Aggressive volume tracking
        self._agg_buy_total: float = 0.0
        self._agg_sell_total: float = 0.0
        self._agg_buy_window: deque[float] = deque(maxlen=zscore_window)
        self._agg_sell_window: deque[float] = deque(maxlen=zscore_window)

        self._snapshot = OBISnapshot()

    @property
    def snapshot(self) -> OBISnapshot:
        return self._snapshot

    @property
    def raw_imbalance(self) -> float:
        return self._snapshot.raw_imbalance

    @property
    def smoothed_imbalance(self) -> float:
        return self._snapshot.smoothed_imbalance

    @property
    def imbalance_zscore(self) -> float:
        return self._snapshot.imbalance_zscore

    @property
    def aggressive_ratio(self) -> float:
        return self._snapshot.aggressive_ratio

    def on_bar(
        self,
        avg_bid_size: float,
        avg_ask_size: float,
        aggressive_buy_vol: float = 0.0,
        aggressive_sell_vol: float = 0.0,
    ) -> OBISnapshot:
        """Update with new bar's order book data.

        Args:
            avg_bid_size: Average bid size at BBO during this bar.
            avg_ask_size: Average ask size at BBO during this bar.
            aggressive_buy_vol: Total volume from aggressive buys (lifted asks).
            aggressive_sell_vol: Total volume from aggressive sells (hit bids).

        Returns:
            Updated OBISnapshot.
        """
        # Raw imbalance
        total = avg_bid_size + avg_ask_size
        raw = (avg_bid_size - avg_ask_size) / total if total > 0 else 0.0
        self._raw_history.append(raw)

        # EMA smoothing
        if not self._initialized:
            self._smoothed = raw
            self._initialized = True
        else:
            self._smoothed = self._ema_alpha * raw + (1 - self._ema_alpha) * self._smoothed

        # Z-score
        if len(self._raw_history) >= 10:
            arr = np.array(self._raw_history)
            mean = arr.mean()
            std = arr.std(ddof=1)
            zscore = (raw - mean) / std if std > 1e-10 else 0.0
        else:
            zscore = 0.0

        # Aggressive volume ratio
        self._agg_buy_window.append(aggressive_buy_vol)
        self._agg_sell_window.append(aggressive_sell_vol)
        total_buy = sum(self._agg_buy_window)
        total_sell = sum(self._agg_sell_window)
        total_agg = total_buy + total_sell
        agg_ratio = total_buy / total_agg if total_agg > 0 else 0.5

        self._snapshot = OBISnapshot(
            raw_imbalance=raw,
            smoothed_imbalance=self._smoothed,
            imbalance_zscore=zscore,
            aggressive_ratio=agg_ratio,
            bid_size=avg_bid_size,
            ask_size=avg_ask_size,
        )
        return self._snapshot

    def reset(self) -> None:
        """Reset all state (new session)."""
        self._raw_history.clear()
        self._smoothed = 0.0
        self._initialized = False
        self._agg_buy_window.clear()
        self._agg_sell_window.clear()
        self._snapshot = OBISnapshot()
