"""Tests for SignalBundle and SignalEngine."""

import time as _time

from src.core.events import BarEvent
from src.signals.base import SignalResult
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle, SignalEngine


def _bar(close: float = 5000.0, volume: int = 100, ts_offset: int = 0) -> BarEvent:
    """Helper: build a BarEvent with sensible defaults."""
    base_ns = 1_700_000_000_000_000_000 + ts_offset * 1_000_000_000
    return BarEvent(
        symbol="MESM6",
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=volume,
        bar_type="1s",
        timestamp_ns=base_ns,
    )


# ── SignalBundle tests ───────────────────────────────────────────────


class TestSignalBundle:
    def test_empty_bundle(self):
        b = EMPTY_BUNDLE
        assert b.bar_count == 0
        assert b.get("atr") is None
        assert b.value("atr") == 0.0
        assert b.passes("atr") is True  # missing signal => True
        assert b.direction("atr") == "none"
        assert b.metadata("atr") == {}
        assert b.has("atr") is False

    def test_bundle_with_results(self):
        results = {
            "spread": SignalResult(value=1.5, passes=True, direction="none"),
            "atr": SignalResult(value=3.2, passes=True, direction="none", metadata={"atr_raw": 3.2}),
        }
        b = SignalBundle(results=results, bar_count=100)

        assert b.bar_count == 100
        assert b.has("spread")
        assert b.has("atr")
        assert not b.has("vpin")

        assert b.value("spread") == 1.5
        assert b.value("atr") == 3.2
        assert b.value("missing", default=99.0) == 99.0

        assert b.metadata("atr") == {"atr_raw": 3.2}
        assert b.metadata("missing") == {}

    def test_bundle_direction(self):
        results = {
            "vwap_session": SignalResult(
                value=-2.0, passes=True, direction="long",
                metadata={"vwap": 5000.0, "sd": 2.0},
            ),
        }
        b = SignalBundle(results=results, bar_count=50)
        assert b.direction("vwap_session") == "long"
        assert b.direction("missing") == "none"

    def test_bundle_passes(self):
        results = {
            "spread": SignalResult(value=3.0, passes=False, direction="none"),
        }
        b = SignalBundle(results=results, bar_count=10)
        assert b.passes("spread") is False
        assert b.passes("missing") is True  # not found => passes

    def test_bundle_frozen(self):
        b = SignalBundle()
        # Frozen — can't assign new attributes
        try:
            b.bar_count = 99  # type: ignore
            assert False, "Should have raised"
        except AttributeError:
            pass


# ── SignalEngine tests ───────────────────────────────────────────────


class TestSignalEngine:
    def test_empty_bars_returns_empty_bundle(self):
        engine = SignalEngine(["spread"])
        bundle = engine.compute([])
        assert bundle is EMPTY_BUNDLE

    def test_compute_spread_signal(self):
        engine = SignalEngine(["spread"])
        bars = [
            _bar(close=5000.0 + i, volume=100, ts_offset=i)
            for i in range(30)
        ]
        bundle = engine.compute(bars)
        assert bundle.has("spread")
        assert bundle.bar_count == 30
        # Spread signal computes a value (exact value depends on bar data)
        assert isinstance(bundle.value("spread"), float)

    def test_compute_multiple_signals(self):
        engine = SignalEngine(["spread", "atr"])
        bars = [_bar(close=5000.0 + i * 0.25, volume=100, ts_offset=i) for i in range(30)]
        bundle = engine.compute(bars)
        assert bundle.has("spread")
        assert bundle.has("atr")
        assert engine.signal_names == ["spread", "atr"]

    def test_signal_configs_passed(self):
        # atr signal accepts lookback_bars config
        engine = SignalEngine(["atr"], signal_configs={"atr": {"lookback_bars": 10}})
        bars = [_bar(close=5000.0 + i * 0.5, volume=100, ts_offset=i) for i in range(30)]
        bundle = engine.compute(bars)
        assert bundle.has("atr")
