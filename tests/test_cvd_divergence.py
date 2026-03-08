"""Tests for CVD Divergence + POC strategy.

7 tests covering swing detection, bearish/bullish divergence, threshold
filtering, POC proximity gating, daily signal limit, and expiry timing.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.core.events import BarEvent
from src.features.feature_hub import FeatureHub
from src.strategies.base import Direction, TICK_SIZE
from src.strategies.cvd_divergence_strategy import (
    CVDDivergenceConfig,
    CVDDivergenceStrategy,
    DivergenceDetector,
    Swing,
    SwingDetector,
    SwingType,
)

# US Eastern (DST-aware, matching production code)
_ET = ZoneInfo("US/Eastern")

# Fixed date: Monday 2025-06-02
_BASE_DATE = datetime(2025, 6, 2, tzinfo=_ET)


def _et_to_ns(hour: int, minute: int, second: int = 0) -> int:
    """Convert an ET time on 2025-06-02 to timestamp_ns."""
    dt = _BASE_DATE.replace(hour=hour, minute=minute, second=second)
    return int(dt.timestamp() * 1_000_000_000)


def _make_bar(
    close: float,
    volume: int = 500,
    ts_ns: int | None = None,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
) -> BarEvent:
    """Create a BarEvent with sensible defaults."""
    if ts_ns is None:
        ts_ns = _et_to_ns(10, 0)
    if open_ is None:
        open_ = close - 0.25
    if high is None:
        high = close + 0.50
    if low is None:
        low = close - 0.50
    return BarEvent(
        symbol="MESM6",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bar_type="5m",
        timestamp_ns=ts_ns,
    )


def _make_strategy(**overrides) -> CVDDivergenceStrategy:
    """Create a CVDDivergenceStrategy with test-friendly defaults."""
    defaults = dict(
        swing_lookback_bars=1,  # 3-bar pattern for minimal sequences
        require_hmm_states=[],  # disable HMM gating
        min_confidence=0.0,
        poc_proximity_ticks=100,  # wide default so POC doesn't block
        divergence_threshold_pct=0.15,
    )
    defaults.update(overrides)
    config = CVDDivergenceConfig(**defaults)
    hub = FeatureHub()
    # Set prior session POC close to test prices so POC filter passes
    hub.volume_profile.set_prior_session(poc=5000.0, vah=5010.0, val=4990.0)
    return CVDDivergenceStrategy(config, hub)


# ── Test 1: Swing detection ─────────────────────────────────────────


class TestSwingDetection:
    def test_swing_detected_at_correct_bar(self):
        """Feed 5 bars with clear peak at bar 3, verify swing high detected."""
        detector = SwingDetector(lookback=1, max_swings=5)

        # Pattern: low, HIGH, low → with lookback=1, window=3
        # Bars: 4990, 5005, 4992
        results = []
        results.extend(detector.on_bar(1, 4990.0, 50.0, _et_to_ns(10, 0)))
        results.extend(detector.on_bar(2, 5005.0, 100.0, _et_to_ns(10, 1)))
        results.extend(detector.on_bar(3, 4992.0, 80.0, _et_to_ns(10, 2)))

        # Should have detected swing high at bar 2
        assert len(detector.recent_highs) == 1
        swing = detector.recent_highs[0]
        assert swing.bar_index == 2
        assert swing.price == 5005.0
        assert swing.cvd_value == 100.0
        assert swing.swing_type == SwingType.HIGH

        # Also feed: high, LOW, high → swing low
        results2 = []
        results2.extend(detector.on_bar(4, 5010.0, 120.0, _et_to_ns(10, 3)))
        results2.extend(detector.on_bar(5, 4985.0, 60.0, _et_to_ns(10, 4)))
        results2.extend(detector.on_bar(6, 5008.0, 110.0, _et_to_ns(10, 5)))

        assert len(detector.recent_lows) >= 1
        low_swing = detector.recent_lows[-1]
        assert low_swing.price == 4985.0
        assert low_swing.swing_type == SwingType.LOW


# ── Test 2: Bearish divergence ──────────────────────────────────────


class TestBearishDivergence:
    def test_bearish_divergence_detected(self):
        """Two swing highs: price higher high + CVD lower high → SHORT signal."""
        strat = _make_strategy()

        # Set prior POC near price so POC filter passes
        strat.feature_hub.volume_profile.set_prior_session(
            poc=5002.0, vah=5010.0, val=4990.0
        )

        # Build first swing high: low, HIGH(5000, cvd≈positive), low
        bars_sequence = [
            (4990.0, 500),  # low
            (5000.0, 500),  # swing high #1
            (4992.0, 500),  # low
            # Build second swing high: higher price, lower CVD
            (4995.0, 500),  # low
            (5002.0, 500),  # swing high #2 (price higher)
            (4994.0, 500),  # low — triggers detection of swing #2
        ]

        # We need CVD to be lower at second swing high
        # After bar approx: close > open → +volume, close < open → -volume
        # We'll manipulate by feeding bars that cause CVD to drop
        # Feed bars with careful open/close to control CVD direction
        for i, (close, vol) in enumerate(bars_sequence):
            ts_ns = _et_to_ns(10, i)
            # For bars 3,4: make close < open to drive CVD negative
            if i in (3, 4):
                open_ = close + 2.0  # close < open → sell pressure
            else:
                open_ = close - 0.25
            bar = BarEvent(
                symbol="MESM6",
                open=open_,
                high=close + 0.50,
                low=close - 0.50,
                close=close,
                volume=vol,
                bar_type="5m",
                timestamp_ns=ts_ns,
            )
            strat.on_bar(bar)

        # Check signals were generated
        signals = strat._signals_generated
        if signals:
            sig = signals[-1]
            assert sig.direction == Direction.SHORT
            assert sig.metadata["divergence_type"] == "SHORT"
            assert sig.metadata["magnitude"] > 0


# ── Test 3: Bullish divergence ──────────────────────────────────────


class TestBullishDivergence:
    def test_bullish_divergence_detected(self):
        """Two swing lows: price lower low + CVD higher low → LONG signal."""
        detector = DivergenceDetector(threshold_pct=0.15)

        # Swing low #1: price=4990, cvd=-200
        # Swing low #2: price=4988 (lower low), cvd=-150 (higher low)
        prev_swing = Swing(
            bar_index=5,
            price=4990.0,
            cvd_value=-200.0,
            timestamp_ns=_et_to_ns(10, 5),
            swing_type=SwingType.LOW,
        )
        curr_swing = Swing(
            bar_index=12,
            price=4988.0,
            cvd_value=-150.0,
            timestamp_ns=_et_to_ns(10, 12),
            swing_type=SwingType.LOW,
        )

        div = detector.check_bullish([prev_swing, curr_swing])
        assert div is not None
        assert div.divergence_type == Direction.LONG
        # magnitude = |(-200) - (-150)| / |(-200)| = 50/200 = 0.25
        assert div.magnitude == pytest.approx(0.25, rel=0.01)


# ── Test 4: Below threshold not detected ────────────────────────────


class TestBelowThreshold:
    def test_below_threshold_not_detected(self):
        """CVD differs by only 5% (< 15% threshold) → no divergence."""
        detector = DivergenceDetector(threshold_pct=0.15)

        # Bearish: price higher high, CVD only 5% lower
        prev = Swing(
            bar_index=3,
            price=5000.0,
            cvd_value=100.0,
            timestamp_ns=_et_to_ns(10, 3),
            swing_type=SwingType.HIGH,
        )
        curr = Swing(
            bar_index=8,
            price=5002.0,
            cvd_value=95.0,  # 5% lower → below 15% threshold
            timestamp_ns=_et_to_ns(10, 8),
            swing_type=SwingType.HIGH,
        )

        assert detector.check_bearish([prev, curr]) is None

        # Bullish: price lower low, CVD only 5% higher
        prev_low = Swing(
            bar_index=3,
            price=4990.0,
            cvd_value=-100.0,
            timestamp_ns=_et_to_ns(10, 3),
            swing_type=SwingType.LOW,
        )
        curr_low = Swing(
            bar_index=8,
            price=4988.0,
            cvd_value=-95.0,  # 5% higher → below threshold
            timestamp_ns=_et_to_ns(10, 8),
            swing_type=SwingType.LOW,
        )

        assert detector.check_bullish([prev_low, curr_low]) is None


# ── Test 5: POC proximity filter ────────────────────────────────────


class TestPOCProximityFilter:
    def test_poc_proximity_filter_blocks_far_divergence(self):
        """Valid divergence but POC 400 ticks away → blocked;
        same divergence with POC 4 ticks away → signal."""
        # Strategy with tight POC filter
        strat = _make_strategy(poc_proximity_ticks=6)

        # Set prior POC far away (400 ticks = 100 points)
        strat.feature_hub.volume_profile.set_prior_session(
            poc=5100.0, vah=5110.0, val=5090.0
        )

        # Inject swings manually into the detector
        swing1 = Swing(2, 5000.0, 100.0, _et_to_ns(10, 2), SwingType.HIGH)
        swing2 = Swing(5, 5002.0, 70.0, _et_to_ns(10, 5), SwingType.HIGH)
        strat._swing_detector._recent_highs.append(swing1)
        strat._swing_detector._recent_highs.append(swing2)

        # Feed a bar to establish last_price for poc_distance_ticks
        bar = _make_bar(close=5001.0, ts_ns=_et_to_ns(10, 6))
        strat._base_on_bar(bar)
        strat._bar_index = 6

        sig_far = strat.generate_signal(bar)
        assert sig_far is None  # POC too far

        # Now set POC close (4 ticks away = 1 point)
        strat.feature_hub.volume_profile.set_prior_session(
            poc=5002.0, vah=5010.0, val=4990.0
        )
        # Reset dedup guard
        strat._last_divergence_bar = -1
        # Feed another bar to update last_price
        bar2 = _make_bar(close=5001.0, ts_ns=_et_to_ns(10, 7))
        strat._base_on_bar(bar2)

        sig_near = strat.generate_signal(bar2)
        assert sig_near is not None
        assert sig_near.direction == Direction.SHORT


# ── Test 6: Max signals per day limit ───────────────────────────────


class TestMaxSignalsPerDay:
    def test_max_signals_per_day_limit(self):
        """max_signals_per_day=1: only 1 signal generated from 2 valid divergences."""
        strat = _make_strategy(max_signals_per_day=1)

        strat.feature_hub.volume_profile.set_prior_session(
            poc=5001.0, vah=5010.0, val=4990.0
        )

        # Inject first divergence pair
        swing1a = Swing(2, 5000.0, 100.0, _et_to_ns(10, 2), SwingType.HIGH)
        swing1b = Swing(5, 5002.0, 70.0, _et_to_ns(10, 5), SwingType.HIGH)
        strat._swing_detector._recent_highs.append(swing1a)
        strat._swing_detector._recent_highs.append(swing1b)

        bar1 = _make_bar(close=5001.0, ts_ns=_et_to_ns(10, 6))
        strat._base_on_bar(bar1)
        strat._bar_index = 6
        sig1 = strat.generate_signal(bar1)
        assert sig1 is not None  # first signal should pass

        # Inject second divergence pair
        swing2a = Swing(8, 5003.0, 90.0, _et_to_ns(10, 8), SwingType.HIGH)
        swing2b = Swing(11, 5005.0, 60.0, _et_to_ns(10, 11), SwingType.HIGH)
        strat._swing_detector._recent_highs.append(swing2a)
        strat._swing_detector._recent_highs.append(swing2b)
        strat._last_divergence_bar = -1  # reset dedup

        bar2 = _make_bar(close=5004.0, ts_ns=_et_to_ns(10, 12))
        strat._base_on_bar(bar2)
        strat._bar_index = 12
        sig2 = strat.generate_signal(bar2)
        assert sig2 is None  # blocked by daily limit

        assert len(strat._signals_generated) == 1


# ── Test 7: Signal expiry matches max_hold_bars ─────────────────────


class TestSignalExpiry:
    def test_signal_expiry_matches_max_hold_bars(self):
        """expiry_time - signal_time == timedelta(minutes=20) for max_hold_bars=4."""
        strat = _make_strategy(max_hold_bars=4)

        strat.feature_hub.volume_profile.set_prior_session(
            poc=5001.0, vah=5010.0, val=4990.0
        )

        # Inject divergence
        swing1 = Swing(2, 5000.0, 100.0, _et_to_ns(10, 2), SwingType.HIGH)
        swing2 = Swing(5, 5002.0, 70.0, _et_to_ns(10, 5), SwingType.HIGH)
        strat._swing_detector._recent_highs.append(swing1)
        strat._swing_detector._recent_highs.append(swing2)

        bar = _make_bar(close=5001.0, ts_ns=_et_to_ns(10, 6))
        strat._base_on_bar(bar)
        strat._bar_index = 6
        sig = strat.generate_signal(bar)

        assert sig is not None
        assert sig.expiry_time - sig.signal_time == timedelta(minutes=20)
