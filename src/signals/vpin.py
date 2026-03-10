from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from collections import deque
from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class VPINConfig:
    bucket_size: int = 100
    n_buckets: int = 50
    trending_threshold: float = 0.55
    mean_reversion_threshold: float = 0.38


@SignalRegistry.register
class VPINSignal(SignalBase):
    name = "vpin"

    def __init__(self, config: VPINConfig | None = None) -> None:
        self.config = config or VPINConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        if not bars:
            return SignalResult(
                value=0.0,
                passes=False,
                direction="none",
                metadata={"vpin": 0.0, "bucket_count": 0, "regime": "undefined"},
            )

        bucket_size = self.config.bucket_size
        n_buckets = self.config.n_buckets

        # Fill volume buckets using BVC classification
        bucket_imbalances: deque[float] = deque(maxlen=n_buckets)
        bucket_buy = 0.0
        bucket_sell = 0.0
        bucket_remaining = float(bucket_size)

        for bar in bars:
            vol = float(bar.volume)
            if vol <= 0:
                continue

            # BVC bar approximation
            if bar.high != bar.low:
                buy_pct = (bar.close - bar.low) / (bar.high - bar.low)
            else:
                buy_pct = 0.5

            bar_buy = vol * buy_pct
            bar_sell = vol - bar_buy

            # Distribute this bar's volume across buckets
            remaining_buy = bar_buy
            remaining_sell = bar_sell
            remaining_vol = vol

            while remaining_vol > 1e-9:
                fill = min(remaining_vol, bucket_remaining)
                # Proportion of this bar's remaining volume
                if remaining_vol > 0:
                    proportion = fill / remaining_vol
                else:
                    break

                bucket_buy += remaining_buy * proportion
                bucket_sell += remaining_sell * proportion
                remaining_buy -= remaining_buy * proportion
                remaining_sell -= remaining_sell * proportion
                remaining_vol -= fill
                bucket_remaining -= fill

                if bucket_remaining < 1e-9:
                    # Bucket is full
                    imbalance = abs(bucket_buy - bucket_sell) / bucket_size
                    bucket_imbalances.append(imbalance)
                    bucket_buy = 0.0
                    bucket_sell = 0.0
                    bucket_remaining = float(bucket_size)

        bucket_count = len(bucket_imbalances)

        if bucket_count < n_buckets:
            return SignalResult(
                value=0.0,
                passes=False,
                direction="none",
                metadata={
                    "vpin": 0.0,
                    "bucket_count": bucket_count,
                    "regime": "undefined",
                },
            )

        # VPIN = mean of last n_buckets imbalances
        vpin = float(np.mean(list(bucket_imbalances)))

        # Regime classification
        if vpin > self.config.trending_threshold:
            regime = "trending"
        elif vpin < self.config.mean_reversion_threshold:
            regime = "mean_reversion"
        else:
            regime = "undefined"

        return SignalResult(
            value=vpin,
            passes=True,
            direction="none",
            metadata={
                "vpin": vpin,
                "bucket_count": bucket_count,
                "regime": regime,
            },
        )
