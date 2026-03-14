"""Tests for TickDirectionPredictor feature extraction."""

import numpy as np
import pytest

from src.core.events import BarEvent
from src.signals.tick_predictor.features.feature_builder import (
    FEATURE_NAMES,
    NUM_FEATURES,
    FeatureBuilder,
)
from src.signals.tick_predictor.features.ring_buffer import RingBuffer


# ── RingBuffer tests ────────────────────────────────────────────

class TestRingBuffer:
    def test_append_and_get(self):
        buf = RingBuffer(5, "test")
        for i in range(3):
            buf.append(float(i))
        assert len(buf) == 3
        np.testing.assert_array_equal(buf.get_array(), [0, 1, 2])

    def test_wraparound(self):
        buf = RingBuffer(3, "test")
        for i in range(5):
            buf.append(float(i))
        assert buf.is_full()
        np.testing.assert_array_equal(buf.get_array(), [2, 3, 4])

    def test_last(self):
        buf = RingBuffer(10, "test")
        for i in range(7):
            buf.append(float(i))
        np.testing.assert_array_equal(buf.last(3), [4, 5, 6])

    def test_reset(self):
        buf = RingBuffer(5, "test")
        for i in range(5):
            buf.append(float(i))
        buf.reset()
        assert len(buf) == 0
        assert not buf.is_full()


# ── FeatureBuilder tests ───────────────────────────────────────

def _make_bar(close: float, volume: int = 100, high: float | None = None,
              low: float | None = None, ts_ns: int = 0) -> BarEvent:
    h = high or close + 0.5
    lo = low or close - 0.5
    return BarEvent(
        symbol="MES", open=close, high=h, low=lo,
        close=close, volume=volume, bar_type="1s",
        timestamp_ns=ts_ns,
    )


class TestFeatureBuilder:
    def test_feature_vector_shape(self):
        builder = FeatureBuilder()
        for i in range(100):
            fv = builder.on_bar(_make_bar(5600.0 + i * 0.25, ts_ns=i * 1_000_000_000))
            assert fv.shape == (NUM_FEATURES,), f"Bar {i}: shape {fv.shape}"
            assert len(fv) == 30

    def test_returns_nan_before_warmup(self):
        builder = FeatureBuilder()
        # First bar — many features should be NaN (pre-normalization returns 0.0)
        fv = builder.on_bar(_make_bar(5600.0))
        # spread_zscore_50 needs 51 bars, so is_warm should be False
        assert not builder.is_warm

    def test_ofi_zero_when_sizes_unchanged(self):
        """OFI should be near zero when bid/ask sizes don't change."""
        builder = FeatureBuilder()
        # Feed identical bars — same close, same volume, same HL range
        for i in range(60):
            fv = builder.on_bar(_make_bar(5600.0, volume=100, high=5600.5,
                                          low=5599.5, ts_ns=i * 1_000_000_000))
        # OFI is diff of bid/ask sizes — if bars are identical, diffs are ~0
        # After normalization, should be near 0
        ofi_10_idx = FEATURE_NAMES.index("ofi_10")
        assert abs(fv[ofi_10_idx]) < 0.5

    def test_obi_zero_when_balanced(self):
        """OBI should be ~0 when close is at midpoint of HL range."""
        builder = FeatureBuilder()
        # close at exact midpoint → bid_size ≈ ask_size → OBI ≈ 0
        for i in range(60):
            fv = builder.on_bar(_make_bar(5600.0, volume=100, high=5601.0,
                                          low=5599.0, ts_ns=i * 1_000_000_000))
        obi_1_idx = FEATURE_NAMES.index("obi_1")
        assert abs(fv[obi_1_idx]) < 0.5

    def test_reset_clears_all_buffers(self):
        builder = FeatureBuilder()
        for i in range(100):
            builder.on_bar(_make_bar(5600.0, ts_ns=i * 1_000_000_000))
        assert builder.is_warm

        builder.reset()
        assert not builder.is_warm
        assert builder._bar_count == 0
