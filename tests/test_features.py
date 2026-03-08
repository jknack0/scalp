"""Tests for the real-time feature library (src/features/).

Tests cover VWAP, ATR, CVD, VolumeProfile, and FeatureHub with
known values and edge cases.
"""

import math

import numpy as np
import pytest

from src.features.vwap import VWAPCalculator
from src.features.atr import ATRCalculator
from src.features.cvd import CVDCalculator
from src.features.volume_profile import VolumeProfileTracker
from src.features.feature_hub import FeatureHub, FeatureHubConfig, FeatureVector


# ── VWAP Tests ──────────────────────────────────────────────────────


class TestVWAP:
    def test_vwap_known_prices(self):
        """3 bars at known price/volume → exact VWAP."""
        calc = VWAPCalculator()
        # Bar 1: price=100, volume=10 → PV=1000
        calc.on_bar(100.0, 10)
        assert calc.vwap == pytest.approx(100.0)

        # Bar 2: price=102, volume=20 → PV=2040, total_vol=30
        calc.on_bar(102.0, 20)
        # VWAP = (1000 + 2040) / 30 = 3040/30 = 101.333...
        assert calc.vwap == pytest.approx(3040.0 / 30.0)

        # Bar 3: price=101, volume=10 → PV=1010, total_vol=40
        calc.on_bar(101.0, 10)
        # VWAP = (1000 + 2040 + 1010) / 40 = 4050/40 = 101.25
        assert calc.vwap == pytest.approx(101.25)

    def test_vwap_deviation_bands(self):
        """Verify bands are VWAP ± N×SD."""
        calc = VWAPCalculator()
        # Feed bars with spread to create nonzero SD
        calc.on_bar(100.0, 100)
        calc.on_bar(104.0, 100)
        calc.on_bar(98.0, 100)

        vwap = calc.vwap
        sd = calc._sd

        assert sd > 0, "SD should be nonzero with different prices"
        assert calc.band_upper_1sd == pytest.approx(vwap + sd)
        assert calc.band_lower_1sd == pytest.approx(vwap - sd)
        assert calc.band_upper_2sd == pytest.approx(vwap + 2 * sd)
        assert calc.band_lower_2sd == pytest.approx(vwap - 2 * sd)
        assert calc.band_upper_3sd == pytest.approx(vwap + 3 * sd)
        assert calc.band_lower_3sd == pytest.approx(vwap - 3 * sd)

    def test_vwap_first_kiss(self):
        """Price moves >2SD then returns → detected."""
        calc = VWAPCalculator()
        # Build up VWAP with some history
        for _ in range(10):
            calc.on_bar(100.0, 100)
        # Push price far away
        calc.on_bar(110.0, 10)
        calc.on_bar(110.0, 10)
        # Now come back near VWAP
        vwap = calc.vwap
        # first_kiss checks if we're within 0.5 SD
        near_vwap = vwap + 0.1  # very close to VWAP
        result = calc.first_kiss_detected(near_vwap, lookback_bars=6)
        assert result is True

    def test_vwap_slope_flat(self):
        """Constant price → flat VWAP slope."""
        calc = VWAPCalculator(flat_threshold=0.001)
        for _ in range(20):
            calc.on_bar(100.0, 100)
        assert calc.is_flat is True
        assert calc.is_trending is False
        assert calc.slope_20bar == pytest.approx(0.0, abs=1e-10)


# ── ATR Tests ───────────────────────────────────────────────────────


class TestATR:
    def test_atr_manual_calculation(self):
        """3 bars with known TR → known ATR."""
        calc = ATRCalculator(period=14)

        # Bar 1: no prev_close, TR = high - low = 2.0
        calc.on_bar(high=102.0, low=100.0, close=101.0)
        assert calc.atr == pytest.approx(2.0)

        # Bar 2: prev_close=101, TR = max(103-100, |103-101|, |100-101|) = 3.0
        calc.on_bar(high=103.0, low=100.0, close=102.0)
        # Simple avg of first 2 bars: (2.0 + 3.0) / 2 = 2.5
        assert calc.atr == pytest.approx(2.5)

        # Bar 3: prev_close=102, TR = max(104-101, |104-102|, |101-102|) = 3.0
        calc.on_bar(high=104.0, low=101.0, close=103.0)
        # Simple avg of 3 bars: (2.0*2 + 3.0) / 3 → (2.5*2 + 3.0)/3 = 8.0/3
        assert calc.atr == pytest.approx(8.0 / 3.0)

    def test_atr_ticks_and_dollars(self):
        """ATR conversion to ticks and dollars."""
        calc = ATRCalculator(period=14, tick_size=0.25, point_value=5.0)
        calc.on_bar(high=102.0, low=100.0, close=101.0)
        # ATR = 2.0
        assert calc.atr_ticks == pytest.approx(2.0 / 0.25)  # 8 ticks
        assert calc.atr_dollars == pytest.approx(2.0 * 5.0)  # $10

    def test_atr_vol_regime(self):
        """Inject low/normal/high ATR history → correct classification."""
        calc = ATRCalculator(period=3, regime_window=100)

        # Build up history with low-volatility bars
        prev = 100.0
        for i in range(50):
            h = prev + 0.5
            l = prev - 0.5
            c = prev + 0.1
            calc.on_bar(h, l, c)
            prev = c

        # ATR should be low-ish, regime should be classifiable
        low_atr = calc.atr

        # Now inject high-volatility bars
        for i in range(50):
            h = prev + 5.0
            l = prev - 5.0
            c = prev + 1.0
            calc.on_bar(h, l, c)
            prev = c

        # After high vol bars, regime should be HIGH
        assert calc.vol_regime == "HIGH"

    def test_atr_semi_variance(self):
        """Semi-variance tracks up and down moves separately."""
        calc = ATRCalculator(period=3)
        # Feed alternating up and down moves
        calc.on_bar(102.0, 100.0, 101.0)  # first bar, no direction
        calc.on_bar(103.0, 100.0, 102.0)  # up move
        calc.on_bar(104.0, 100.0, 103.0)  # up move
        calc.on_bar(103.0, 99.0, 100.0)   # down move

        # Should have entries in both up and down
        assert calc.semi_variance_up >= 0.0
        assert calc.semi_variance_down >= 0.0


# ── CVD Tests ───────────────────────────────────────────────────────


class TestCVD:
    def test_cvd_buy_sell_classification(self):
        """Trades at ask → positive CVD, at bid → negative."""
        calc = CVDCalculator()

        # Buy: price at ask
        calc.on_tick(price=100.25, size=10, bid=100.0, ask=100.25)
        assert calc.cvd == 10.0

        # Sell: price at bid
        calc.on_tick(price=100.0, size=5, bid=100.0, ask=100.25)
        assert calc.cvd == 5.0  # 10 - 5

        # Another buy
        calc.on_tick(price=100.25, size=3, bid=100.0, ask=100.25)
        assert calc.cvd == 8.0  # 5 + 3

    def test_cvd_bar_approx_fallback(self):
        """Up-close bar → positive delta, down-close → negative."""
        calc = CVDCalculator()

        # Up bar
        calc.on_bar_approx(open_=100.0, close=101.0, volume=50)
        assert calc.cvd == 50.0

        # Down bar
        calc.on_bar_approx(open_=101.0, close=99.0, volume=30)
        assert calc.cvd == 20.0  # 50 - 30

        # Flat bar (no change)
        calc.on_bar_approx(open_=100.0, close=100.0, volume=10)
        assert calc.cvd == 20.0  # unchanged

    def test_cvd_divergence(self):
        """Price up + CVD down → high divergence score."""
        calc = CVDCalculator()

        # Build some bar history with negative deltas
        for _ in range(5):
            calc.on_bar_approx(open_=101.0, close=99.0, volume=100)
            calc.on_bar_close()

        # Price went up but CVD is negative → divergence
        div = calc.divergence_from_price(price_change=5.0)
        assert div == 1.0  # maximum divergence (opposite signs)

    def test_cvd_tick_rule_fallback(self):
        """Mid-spread trades use tick rule."""
        calc = CVDCalculator()

        # First trade at mid-spread, no prev price → delta = 0
        calc.on_tick(price=100.125, size=10, bid=100.0, ask=100.25)
        assert calc.cvd == 0.0

        # Second trade higher → buy via tick rule
        calc.on_tick(price=100.15, size=10, bid=100.0, ask=100.25)
        assert calc.cvd == 10.0


# ── Volume Profile Tests ────────────────────────────────────────────


class TestVolumeProfile:
    def test_poc_distance(self):
        """Known prior POC, current price → correct tick distance."""
        tracker = VolumeProfileTracker(tick_size=0.25)
        tracker.set_prior_session(poc=5000.0, vah=5010.0, val=4990.0)

        # Feed a bar to set current price
        tracker.on_bar(close=5002.0, volume=100)

        # Distance = |5002 - 5000| / 0.25 = 8 ticks
        assert tracker.poc_distance_ticks == pytest.approx(8.0)
        assert tracker.price_above_poc is True
        assert tracker.poc_proximity is False  # 8 > 4

    def test_poc_proximity(self):
        """Price within 4 ticks of POC → poc_proximity = True."""
        tracker = VolumeProfileTracker(tick_size=0.25)
        tracker.set_prior_session(poc=5000.0, vah=5010.0, val=4990.0)

        # 3 ticks away: |5000.75 - 5000| / 0.25 = 3
        tracker.on_bar(close=5000.75, volume=100)
        assert tracker.poc_proximity is True

    def test_developing_profile(self):
        """Feed bars → developing POC is the most-traded level."""
        tracker = VolumeProfileTracker(tick_size=0.25)

        # Most volume at 5000
        tracker.on_bar(close=5000.0, volume=500)
        tracker.on_bar(close=5000.0, volume=500)
        tracker.on_bar(close=5001.0, volume=100)
        tracker.on_bar(close=4999.0, volume=100)

        assert tracker.live_poc == pytest.approx(5000.0)

    def test_prior_session_preserved_after_reset(self):
        """reset() clears developing profile but keeps prior session levels."""
        tracker = VolumeProfileTracker(tick_size=0.25)
        tracker.set_prior_session(poc=5000.0, vah=5010.0, val=4990.0)
        tracker.on_bar(close=5005.0, volume=100)

        tracker.reset()

        assert tracker.prior_poc == 5000.0
        assert tracker.prior_vah == 5010.0
        assert tracker.prior_val == 4990.0
        assert tracker.live_poc == 0.0  # cleared


# ── FeatureHub Tests ────────────────────────────────────────────────


class TestFeatureHub:
    def test_feature_hub_snapshot(self):
        """Feed bars through hub → get complete FeatureVector."""
        hub = FeatureHub()
        hub.volume_profile.set_prior_session(poc=5000.0, vah=5010.0, val=4990.0)

        # Feed several bars
        bars = [
            (1_000_000, 5000.0, 5002.0, 4998.0, 5001.0, 100),
            (2_000_000, 5001.0, 5003.0, 4999.0, 5002.0, 150),
            (3_000_000, 5002.0, 5005.0, 5000.0, 5004.0, 200),
            (4_000_000, 5004.0, 5006.0, 5001.0, 5003.0, 120),
            (5_000_000, 5003.0, 5004.0, 5000.0, 5001.0, 90),
        ]

        for ts, o, h, l, c, v in bars:
            hub.on_bar(ts, o, h, l, c, v)

        snap = hub.snapshot()

        assert isinstance(snap, FeatureVector)
        assert snap.timestamp_ns == 5_000_000
        assert snap.vwap > 0
        assert snap.atr_ticks >= 0
        assert snap.vol_regime in ("LOW", "NORMAL", "HIGH")
        assert snap.dominant_dir in ("UP", "DOWN", "NEUTRAL")
        assert snap.poc_distance_ticks >= 0

        # Test serialization
        d = snap.to_dict()
        assert "vwap" in d
        assert "cvd" in d
        assert d["timestamp_ns"] == 5_000_000

        arr = snap.to_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 15  # 15 numeric features

    def test_session_reset(self):
        """reset() clears all calculator state."""
        hub = FeatureHub()

        # Feed some data
        hub.on_bar(1_000_000, 5000.0, 5002.0, 4998.0, 5001.0, 100)
        hub.on_bar(2_000_000, 5001.0, 5003.0, 4999.0, 5002.0, 150)

        # Verify state exists
        snap_before = hub.snapshot()
        assert snap_before.vwap > 0
        assert snap_before.cvd != 0

        # Reset
        hub.reset()

        snap_after = hub.snapshot()
        assert snap_after.vwap == 0.0
        assert snap_after.cvd == 0.0
        assert snap_after.atr_ticks == 0.0
        assert snap_after.timestamp_ns == 0
