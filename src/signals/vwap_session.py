"""Stateless session VWAP signal — replaces _VWAPAccumulator in strategies.

Computes session VWAP, standard deviation, deviation in SDs, slope, and
first-kiss detection from a bar window.  All state is derived from the bar
list on each call — no incremental accumulation needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

import numpy as np

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry

from zoneinfo import ZoneInfo

_ET = ZoneInfo("US/Eastern")
_SESSION_START = time(9, 30)

MES_TICK_SIZE = 0.25


@dataclass(frozen=True)
class VWAPSessionConfig:
    slope_window: int = 20
    first_kiss_lookback_bars: int = 6
    first_kiss_sd_threshold: float = 1.5
    flat_slope_threshold: float = 0.003
    trending_slope_threshold: float = 0.008


@SignalRegistry.register
class VWAPSessionSignal(SignalBase):
    """Session VWAP with SD bands, slope, mode, and first-kiss detection.

    Returns:
        value: deviation_sd (signed, negative = below VWAP)
        passes: True (informational — always passes)
        direction: "long" if below VWAP, "short" if above, "none" if near
        metadata: vwap, sd, slope, deviation_sd, mode, first_kiss,
                  session_age_bars, atr (if enough bars)
    """

    name = "vwap_session"

    def __init__(self, config: VWAPSessionConfig | None = None) -> None:
        self.config = config or VWAPSessionConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        if len(bars) < 2:
            return SignalResult(
                value=0.0,
                passes=True,
                direction="none",
                metadata=self._empty_metadata(),
            )

        # Find session bars (current trading day only)
        session_bars = self._extract_session_bars(bars)
        if len(session_bars) < 2:
            return SignalResult(
                value=0.0,
                passes=True,
                direction="none",
                metadata=self._empty_metadata(),
            )

        # Compute VWAP and SD from session bars
        typical_prices = np.array(
            [(b.high + b.low + b.close) / 3.0 for b in session_bars],
            dtype=np.float64,
        )
        volumes = np.array(
            [b.volume for b in session_bars], dtype=np.float64,
        )

        # Filter zero-volume bars
        mask = volumes > 0
        if not np.any(mask):
            return SignalResult(
                value=0.0,
                passes=True,
                direction="none",
                metadata=self._empty_metadata(),
            )

        tp = typical_prices[mask]
        vol = volumes[mask]

        sum_pv = float(np.sum(tp * vol))
        sum_vol = float(np.sum(vol))
        sum_pv2 = float(np.sum(vol * tp * tp))

        vwap = sum_pv / sum_vol
        var = max(0.0, sum_pv2 / sum_vol - vwap * vwap)
        sd = float(np.sqrt(var))

        last_price = session_bars[-1].close

        # Deviation in SDs
        deviation_sd = 0.0
        if sd > 0:
            deviation_sd = (last_price - vwap) / sd

        # Slope from VWAP history (last slope_window bars)
        slope = self._compute_slope(session_bars, vwap_func=self._running_vwap)

        # Mode: REVERSION (flat), PULLBACK (trending), NEUTRAL
        abs_slope = abs(slope)
        if abs_slope < self.config.flat_slope_threshold:
            mode = "REVERSION"
        elif abs_slope > self.config.trending_slope_threshold:
            mode = "PULLBACK"
        else:
            mode = "NEUTRAL"

        # First-kiss detection
        first_kiss = self._detect_first_kiss(
            session_bars, vwap, sd, last_price
        )

        # Direction
        if sd > 0 and abs(deviation_sd) >= 1.0:
            direction = "long" if deviation_sd < 0 else "short"
        else:
            direction = "none"

        return SignalResult(
            value=deviation_sd,
            passes=True,
            direction=direction,
            metadata={
                "vwap": vwap,
                "sd": sd,
                "slope": slope,
                "deviation_sd": deviation_sd,
                "mode": mode,
                "first_kiss": first_kiss,
                "session_age_bars": len(session_bars),
            },
        )

    def _extract_session_bars(self, bars: list[BarEvent]) -> list[BarEvent]:
        """Extract bars from the most recent trading session."""
        if not bars:
            return []

        # Find the date of the last bar
        last_dt = datetime.fromtimestamp(
            bars[-1].timestamp_ns / 1_000_000_000, tz=_ET
        )
        last_date = last_dt.date()

        # Collect all bars from the same date
        session_bars: list[BarEvent] = []
        for bar in bars:
            bar_dt = datetime.fromtimestamp(
                bar.timestamp_ns / 1_000_000_000, tz=_ET
            )
            if bar_dt.date() == last_date:
                session_bars.append(bar)

        return session_bars

    def _running_vwap(self, bars: list[BarEvent]) -> list[float]:
        """Compute running VWAP for each bar position."""
        vwaps: list[float] = []
        sum_pv = 0.0
        sum_vol = 0.0
        for bar in bars:
            tp = (bar.high + bar.low + bar.close) / 3.0
            vol = bar.volume
            if vol > 0:
                sum_pv += tp * vol
                sum_vol += vol
            vwaps.append(sum_pv / sum_vol if sum_vol > 0 else 0.0)
        return vwaps

    def _compute_slope(
        self,
        session_bars: list[BarEvent],
        vwap_func,
    ) -> float:
        """Linear regression slope of running VWAP over last slope_window bars."""
        vwaps = vwap_func(session_bars)
        window = self.config.slope_window
        tail = vwaps[-window:] if len(vwaps) >= window else vwaps

        n = len(tail)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2.0
        y_mean = sum(tail) / n
        num = 0.0
        den = 0.0
        for i, y in enumerate(tail):
            dx = i - x_mean
            num += dx * (y - y_mean)
            den += dx * dx

        if den == 0:
            return 0.0
        return num / den

    def _detect_first_kiss(
        self,
        session_bars: list[BarEvent],
        vwap: float,
        sd: float,
        current_price: float,
    ) -> bool:
        """True if price was far from VWAP recently and is now near."""
        if sd == 0 or len(session_bars) < 2:
            return False

        current_dev = abs(current_price - vwap) / sd
        if current_dev > 0.5:
            return False

        lookback = self.config.first_kiss_lookback_bars
        recent_bars = session_bars[-(lookback + 1):-1] if len(session_bars) > lookback else session_bars[:-1]

        threshold = self.config.first_kiss_sd_threshold
        for bar in recent_bars:
            bar_dev = abs(bar.close - vwap) / sd
            if bar_dev >= threshold:
                return True
        return False

    @staticmethod
    def _empty_metadata() -> dict:
        return {
            "vwap": 0.0,
            "sd": 0.0,
            "slope": 0.0,
            "deviation_sd": 0.0,
            "mode": "NEUTRAL",
            "first_kiss": False,
            "session_age_bars": 0,
        }
