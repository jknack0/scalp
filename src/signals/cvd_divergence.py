from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


def _ols_slope(y: np.ndarray) -> float:
    """Compute OLS slope of y against integer indices."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = (n - 1) / 2.0
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    return float(num / den) if den > 0 else 0.0


@dataclass(frozen=True)
class CVDDivergenceConfig:
    lookback_bars: int = 20
    z_threshold: float = 1.5


@SignalRegistry.register
class CVDDivergenceSignal(SignalBase):
    name = "cvd_divergence"

    def __init__(self, config: CVDDivergenceConfig | None = None) -> None:
        self.config = config or CVDDivergenceConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        lookback = self.config.lookback_bars

        if len(bars) < lookback:
            return SignalResult(
                value=0.0,
                passes=False,
                direction="none",
                metadata={"cvd_slope": 0.0, "price_slope": 0.0, "z_score": 0.0},
            )

        n = len(bars)

        # Compute per-bar delta using L1 fields when available, else bar approx
        deltas = np.empty(n, dtype=np.float64)
        closes = np.empty(n, dtype=np.float64)

        for i, bar in enumerate(bars):
            closes[i] = bar.close

            if bar.aggressive_buy_vol > 0 or bar.aggressive_sell_vol > 0:
                # L1 data available
                buy_vol = bar.aggressive_buy_vol
                sell_vol = bar.aggressive_sell_vol
            else:
                # BVC bar approximation
                vol = float(bar.volume)
                if bar.high != bar.low:
                    buy_pct = (bar.close - bar.low) / (bar.high - bar.low)
                else:
                    buy_pct = 0.5
                buy_vol = vol * buy_pct
                sell_vol = vol - buy_vol

            deltas[i] = buy_vol - sell_vol

        # Cumulative volume delta
        cvd = np.cumsum(deltas)

        # Compute CVD slope over last lookback_bars
        cvd_window = cvd[-lookback:]
        price_window = closes[-lookback:]

        cvd_slope = _ols_slope(cvd_window)
        price_slope = _ols_slope(price_window)

        # Rolling CVD slopes for z-score: compute slope for each sub-window
        # ending at positions from (lookback-1) to (n-1)
        num_slopes = min(n - lookback + 1, lookback)
        if num_slopes < 2:
            z_score = 0.0
        else:
            slopes = np.empty(num_slopes, dtype=np.float64)
            for j in range(num_slopes):
                end_idx = n - num_slopes + 1 + j
                start_idx = end_idx - lookback
                slopes[j] = _ols_slope(cvd[start_idx:end_idx])

            slope_std = float(np.std(slopes, ddof=1))
            z_score = float(cvd_slope / slope_std) if slope_std > 0 else 0.0

        # Divergence detection: opposite signs
        divergence = (price_slope > 0 and cvd_slope < 0) or (
            price_slope < 0 and cvd_slope > 0
        )

        passes = divergence and abs(z_score) >= self.config.z_threshold

        # Direction
        if divergence:
            if price_slope > 0 and cvd_slope < 0:
                direction = "short"  # bearish divergence
            else:
                direction = "long"  # bullish divergence
        else:
            direction = "none"

        return SignalResult(
            value=z_score,
            passes=passes,
            direction=direction,
            metadata={
                "cvd_slope": cvd_slope,
                "price_slope": price_slope,
                "z_score": z_score,
                "divergence": divergence,
            },
        )
