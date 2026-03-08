"""Feature composition layer — FeatureVector and FeatureHub.

FeatureHub owns all individual calculators and delegates bar/tick events
to each. snapshot() returns a frozen FeatureVector with all current values.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from src.features.atr import ATRCalculator
from src.features.cvd import CVDCalculator
from src.features.volume_profile import VolumeProfileTracker
from src.features.vwap import VWAPCalculator


@dataclass(frozen=True)
class FeatureVector:
    """Frozen snapshot of all feature values at a point in time."""

    timestamp_ns: int
    # VWAP
    vwap: float
    vwap_dev_sd: float
    vwap_slope: float
    vwap_is_flat: bool
    # ATR
    atr_ticks: float
    vol_regime: str
    semi_var_up: float
    semi_var_down: float
    dominant_dir: str
    # CVD
    cvd: float
    cvd_slope: float
    cvd_zscore: float
    # Volume Profile
    poc_distance_ticks: float
    price_above_poc: bool
    poc_proximity: bool

    def to_array(self) -> np.ndarray:
        """Numeric features as a numpy array (for HMM input).

        Boolean and string fields are encoded:
        - bool → 1.0 / 0.0
        - vol_regime → LOW=0, NORMAL=1, HIGH=2
        - dominant_dir → DOWN=0, NEUTRAL=1, UP=2
        """
        regime_map = {"LOW": 0.0, "NORMAL": 1.0, "HIGH": 2.0}
        dir_map = {"DOWN": 0.0, "NEUTRAL": 1.0, "UP": 2.0}
        return np.array([
            self.vwap,
            self.vwap_dev_sd,
            self.vwap_slope,
            1.0 if self.vwap_is_flat else 0.0,
            self.atr_ticks,
            regime_map.get(self.vol_regime, 1.0),
            self.semi_var_up,
            self.semi_var_down,
            dir_map.get(self.dominant_dir, 1.0),
            self.cvd,
            self.cvd_slope,
            self.cvd_zscore,
            self.poc_distance_ticks,
            1.0 if self.price_above_poc else 0.0,
            1.0 if self.poc_proximity else 0.0,
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        """All feature values as a plain dict (for logging/storage)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class FeatureHubConfig:
    """Configuration for FeatureHub calculators."""

    vwap_flat_threshold: float = 0.001
    atr_period: int = 14
    tick_size: float = 0.25
    point_value: float = 5.0
    cvd_slope_window: int = 20
    cvd_zscore_window: int = 20


class FeatureHub:
    """Composition layer that owns and delegates to all feature calculators.

    Feed bars via on_bar() and ticks via on_tick(). Take a snapshot of all
    current feature values with snapshot().
    """

    def __init__(self, config: FeatureHubConfig | None = None) -> None:
        cfg = config or FeatureHubConfig()

        self.vwap = VWAPCalculator(flat_threshold=cfg.vwap_flat_threshold)
        self.atr = ATRCalculator(
            period=cfg.atr_period,
            tick_size=cfg.tick_size,
            point_value=cfg.point_value,
        )
        self.cvd = CVDCalculator(
            slope_window=cfg.cvd_slope_window,
            zscore_window=cfg.cvd_zscore_window,
        )
        self.volume_profile = VolumeProfileTracker(tick_size=cfg.tick_size)

        self._last_timestamp_ns: int = 0
        self._last_close: float = 0.0
        self._prev_close: float = 0.0

    def on_bar(
        self,
        timestamp_ns: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> None:
        """Feed a new bar to all calculators."""
        self._prev_close = self._last_close
        self._last_timestamp_ns = timestamp_ns
        self._last_close = close

        # Typical price for VWAP
        typical = (high + low + close) / 3.0
        self.vwap.on_bar(typical, volume)
        self.atr.on_bar(high, low, close)
        self.cvd.on_bar_approx(open_, close, volume)
        self.cvd.on_bar_close()
        self.volume_profile.on_bar(close, volume)

    def on_tick(
        self,
        price: float,
        size: int,
        bid: float,
        ask: float,
    ) -> None:
        """Feed a single L1 trade to CVD calculator."""
        self.cvd.on_tick(price, size, bid, ask)

    def snapshot(self) -> FeatureVector:
        """Capture current feature values as a frozen FeatureVector."""
        return FeatureVector(
            timestamp_ns=self._last_timestamp_ns,
            vwap=self.vwap.vwap,
            vwap_dev_sd=self.vwap.deviation_sd,
            vwap_slope=self.vwap.slope_20bar,
            vwap_is_flat=self.vwap.is_flat,
            atr_ticks=self.atr.atr_ticks,
            vol_regime=self.atr.vol_regime,
            semi_var_up=self.atr.semi_variance_up,
            semi_var_down=self.atr.semi_variance_down,
            dominant_dir=self.atr.dominant_direction,
            cvd=self.cvd.cvd,
            cvd_slope=self.cvd.cvd_slope_20bar,
            cvd_zscore=self.cvd.cvd_zscore,
            poc_distance_ticks=self.volume_profile.poc_distance_ticks,
            price_above_poc=self.volume_profile.price_above_poc,
            poc_proximity=self.volume_profile.poc_proximity,
        )

    def reset(self) -> None:
        """Reset all calculators for a new session."""
        self.vwap.reset()
        self.atr.reset()
        self.cvd.reset()
        self.volume_profile.reset()
        self._last_timestamp_ns = 0
        self._last_close = 0.0
        self._prev_close = 0.0
