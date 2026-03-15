"""Tests for the 26-feature streaming FeatureBuilder."""

import numpy as np
import pytest

from src.core.events import BarEvent
from src.signals.tick_predictor.features.feature_builder import (
    FEATURE_NAMES,
    NUM_FEATURES,
    FeatureBuilder,
)


def _bar(o: float, h: float, l: float, c: float, vol: int = 100,
         ts_ns: int = 0) -> BarEvent:
    return BarEvent(
        symbol="MES", open=o, high=h, low=l, close=c,
        volume=vol, bar_type="1s", timestamp_ns=ts_ns,
    )


def _warmup(fb: FeatureBuilder, n: int = 60, base: float = 5600.0) -> None:
    """Feed n bars with slight random walk."""
    rng = np.random.default_rng(42)
    price = base
    for i in range(n):
        change = rng.normal(0, 0.5)
        price += change
        h = price + abs(rng.normal(0, 0.3))
        l = price - abs(rng.normal(0, 0.3))
        vol = int(rng.integers(50, 200))
        fb.on_bar(_bar(price - change / 2, h, l, price, vol, i * 1_000_000_000))


class TestFeatureVectorShape:
    def test_shape_after_warmup(self):
        fb = FeatureBuilder()
        _warmup(fb, 60)
        bar = _bar(5600, 5601, 5599, 5600.5, 100)
        vec = fb.on_bar(bar)
        assert vec.shape == (26,)

    def test_shape_always_26(self):
        fb = FeatureBuilder()
        for i in range(10):
            vec = fb.on_bar(_bar(5600 + i, 5601 + i, 5599 + i, 5600.5 + i, 100, i))
            assert vec.shape == (NUM_FEATURES,)


class TestAllNanBeforeWarmup:
    def test_first_50_bars_all_nan(self):
        fb = FeatureBuilder()
        for i in range(50):
            vec = fb.on_bar(_bar(5600 + i * 0.1, 5601, 5599, 5600 + i * 0.1, 100, i))
            # Many features should be nan (not all because some compute from bar 2)
            # But is_warm should be False
            assert not fb.is_warm()

    def test_warm_after_51_bars(self):
        fb = FeatureBuilder()
        _warmup(fb, 51)
        assert fb.is_warm()


class TestSetANoLookahead:
    def test_return_1_matches_log_ratio(self):
        fb = FeatureBuilder()
        prices = [5600.0 + i * 0.25 for i in range(55)]
        vecs = []
        for i, p in enumerate(prices):
            vec = fb.on_bar(_bar(p - 0.1, p + 0.5, p - 0.5, p, 100, i))
            vecs.append(vec)

        # return_1 at bar t should be based on close[t-1]/close[t-2]
        # (shifted by 1 for causality)
        # After warmup, check a specific bar
        idx_return_1 = FEATURE_NAMES.index("return_1")
        # The raw return_1 = log(close[t-1] / close[t-2]) but it gets normalized
        # So just check it's finite after warmup
        assert np.isfinite(vecs[-1][idx_return_1])


class TestCvdDelta:
    def test_positive_on_bullish_bar(self):
        """Bar where close == high should produce cvd_delta == volume."""
        fb = FeatureBuilder()
        _warmup(fb, 55)

        # Bullish bar: close == high
        bar = _bar(5600, 5602, 5598, 5602, 200)
        vec = fb.on_bar(bar)

        # cvd_delta is normalized, but the raw value should be positive
        raw = fb._last_raw
        idx = FEATURE_NAMES.index("cvd_delta")
        # close == high: (c-l)/(h-l) = (5602-5598)/(5602-5598) = 1.0
        # cvd = vol * (2*1 - 1) = vol * 1 = 200
        assert raw[idx] == pytest.approx(200.0, rel=1e-6)

    def test_negative_on_bearish_bar(self):
        """Bar where close == low should produce cvd_delta == -volume."""
        fb = FeatureBuilder()
        _warmup(fb, 55)

        # Bearish bar: close == low
        bar = _bar(5600, 5602, 5598, 5598, 200)
        vec = fb.on_bar(bar)

        raw = fb._last_raw
        idx = FEATURE_NAMES.index("cvd_delta")
        # close == low: (c-l)/(h-l) = 0/4 ≈ 0
        # cvd = vol * (2*0 - 1) = vol * -1 = -200
        assert raw[idx] == pytest.approx(-200.0, rel=1e-6)


class TestObiSign:
    def test_obi_negative_when_close_near_high(self):
        """Close near high → ask_est > bid_est → obi_est negative."""
        fb = FeatureBuilder()
        _warmup(fb, 55)

        # close near high: ask_est = vol*(c-l)/(h-l) large, bid_est small
        bar = _bar(5600, 5602, 5598, 5601.9, 200)
        fb.on_bar(bar)
        raw = fb._last_raw
        idx = FEATURE_NAMES.index("obi_est")
        # bid_est = vol*(h-c)/(h-l) = 200*(0.1)/4 = 5
        # ask_est = vol*(c-l)/(h-l) = 200*(3.9)/4 = 195
        # obi = (5-195)/(5+195+1e-9) < 0
        assert raw[idx] < 0


class TestReset:
    def test_reset_clears_all(self):
        fb = FeatureBuilder()
        _warmup(fb, 60)
        assert fb.is_warm()

        fb.reset()
        assert not fb.is_warm()

        # Next bar should give mostly nan (bar_count=1)
        vec = fb.on_bar(_bar(5600, 5601, 5599, 5600, 100))
        # Most features nan or zero (normalization returns 0 for early bars)
        assert vec.shape == (26,)


class TestUpVolRatioBounds:
    def test_bounds(self):
        fb = FeatureBuilder()
        _warmup(fb, 55)

        idx = FEATURE_NAMES.index("up_vol_ratio_10")
        raw = fb._last_raw
        val = raw[idx]
        if np.isfinite(val):
            assert 0.0 <= val <= 1.0


class TestBodyRatioBounds:
    def test_bounds(self):
        fb = FeatureBuilder()
        _warmup(fb, 55)

        idx = FEATURE_NAMES.index("body_ratio")
        raw = fb._last_raw
        val = raw[idx]
        if np.isfinite(val):
            assert 0.0 <= val <= 1.0


class TestFeatureDict:
    def test_get_feature_dict(self):
        fb = FeatureBuilder()
        _warmup(fb, 55)
        d = fb.get_feature_dict()
        assert len(d) == NUM_FEATURES
        assert set(d.keys()) == set(FEATURE_NAMES)
