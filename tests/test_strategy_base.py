"""Tests for strategy base class and interfaces (src/strategies/base.py).

Uses a MockStrategy to test the ABC's concrete methods: session gating,
signal construction, HMM state filtering, daily limits, and HMMFeatureBuffer.
"""

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from src.core.events import BarEvent
from src.features.feature_hub import FeatureHub, FeatureHubConfig
from src.models.hmm_regime import RegimeState
from src.strategies.base import (
    TICK_SIZE,
    Direction,
    HMMFeatureBuffer,
    Signal,
    StrategyBase,
    StrategyConfig,
)

# US Eastern (DST-aware, matching production code)
_ET = ZoneInfo("US/Eastern")


class MockStrategy(StrategyBase):
    """Minimal concrete strategy for testing base class behavior."""

    def on_tick(self, tick):
        pass

    def on_bar(self, bar):
        self._base_on_bar(bar)

    def generate_signal(self):
        """Always generates a LONG signal at fixed geometry."""
        return self._make_signal(
            direction=Direction.LONG,
            entry=5000.0,
            target=5002.0,
            stop=4999.0,
            confidence=0.75,
            now=datetime(2025, 6, 2, 10, 30, tzinfo=_ET),
        )

    def reset(self):
        super().reset()


def _make_config(**overrides) -> StrategyConfig:
    defaults = dict(strategy_id="mock", max_signals_per_day=5)
    defaults.update(overrides)
    return StrategyConfig(**defaults)


def _make_bar(close: float = 5000.0, volume: int = 100, ts_ns: int = 0) -> BarEvent:
    return BarEvent(
        symbol="MESM6",
        open=close - 0.25,
        high=close + 0.50,
        low=close - 0.50,
        close=close,
        volume=volume,
        bar_type="1s",
        timestamp_ns=ts_ns,
    )


# ── Test 1: Signal properties ───────────────────────────────────────


class TestSignalProperties:
    def test_signal_properties_correct(self):
        """RR ratio, ticks_to_target, ticks_to_stop with known prices."""
        signal = Signal(
            strategy_id="test",
            direction=Direction.LONG,
            entry_price=5000.0,
            target_price=5002.0,  # 8 ticks
            stop_price=4999.0,  # 4 ticks
            signal_time=datetime(2025, 6, 2, 10, 0, tzinfo=_ET),
            expiry_time=datetime(2025, 6, 2, 10, 1, tzinfo=_ET),
            confidence=0.8,
            regime_state=RegimeState.LOW_VOL_RANGE,
        )
        assert signal.risk_reward_ratio == pytest.approx(2.0)
        assert signal.ticks_to_target == pytest.approx(8.0)
        assert signal.ticks_to_stop == pytest.approx(4.0)


# ── Test 2: Unique IDs ──────────────────────────────────────────────


class TestSignalIds:
    def test_signal_unique_ids(self):
        """Two signals have different uuid4 ids."""
        s1 = Signal(
            strategy_id="test",
            direction=Direction.LONG,
            entry_price=5000.0,
            target_price=5002.0,
            stop_price=4999.0,
            signal_time=datetime.now(_ET),
            expiry_time=datetime.now(_ET),
            confidence=0.7,
            regime_state=RegimeState.BREAKOUT,
        )
        s2 = Signal(
            strategy_id="test",
            direction=Direction.SHORT,
            entry_price=5000.0,
            target_price=4998.0,
            stop_price=5001.0,
            signal_time=datetime.now(_ET),
            expiry_time=datetime.now(_ET),
            confidence=0.7,
            regime_state=RegimeState.BREAKOUT,
        )
        assert s1.id != s2.id


# ── Test 3: Session gating ──────────────────────────────────────────


class TestSessionGating:
    def test_session_gating(self):
        """Active session allows signal, inactive blocks."""
        hub = FeatureHub()
        config = _make_config()
        strat = MockStrategy(config, hub)

        # 10:30 ET — within session
        in_session = datetime(2025, 6, 2, 10, 30, tzinfo=_ET)
        assert strat.is_active_session(in_session) is True

        # 04:30 ET — outside session
        out_session = datetime(2025, 6, 2, 4, 30, tzinfo=_ET)
        assert strat.is_active_session(out_session) is False

        # 16:01 ET — after close
        after_close = datetime(2025, 6, 2, 16, 1, tzinfo=_ET)
        assert strat.is_active_session(after_close) is False


# ── Test 4: Signal count limit ───────────────────────────────────────


class TestSignalCountLimit:
    def test_signal_count_limit(self):
        """Third signal blocked when max_signals_per_day=2."""
        hub = FeatureHub()
        config = _make_config(max_signals_per_day=2)
        strat = MockStrategy(config, hub)

        in_session = datetime(2025, 6, 2, 10, 30, tzinfo=_ET)

        # First two signals should succeed
        assert strat.can_generate_signal(in_session) is True
        s1 = strat.generate_signal()
        assert s1 is not None

        assert strat.can_generate_signal(in_session) is True
        s2 = strat.generate_signal()
        assert s2 is not None

        # Third should be blocked
        assert strat.can_generate_signal(in_session) is False


# ── Test 5: HMM feature buffer ──────────────────────────────────────


class TestHMMFeatureBuffer:
    def test_hmm_buffer_builds_matrix(self):
        """Shape (50, 6), no NaN/inf after warmup."""
        hub = FeatureHub()
        buf = HMMFeatureBuffer(maxlen=300)

        # Feed 100 bars to get enough data
        base_price = 5000.0
        rng = np.random.default_rng(42)

        for i in range(100):
            price = base_price + rng.normal(0, 1.0)
            volume = int(100 + rng.integers(0, 50))
            hub.on_bar(
                timestamp_ns=i * 1_000_000_000,
                open_=price - 0.25,
                high=price + 0.50,
                low=price - 0.50,
                close=price,
                volume=volume,
            )
            snap = hub.snapshot()
            buf.update(snap, price)

        assert buf.is_ready() is True
        matrix = buf.build_matrix()
        assert matrix is not None
        assert matrix.shape == (50, 6)
        assert not np.any(np.isnan(matrix))
        assert not np.any(np.isinf(matrix))


# ── Test 6: Reset clears daily state ────────────────────────────────


class TestReset:
    def test_reset_clears_daily_state(self):
        """signals_today=0 after reset."""
        hub = FeatureHub()
        config = _make_config()
        strat = MockStrategy(config, hub)

        # Generate a signal
        s = strat.generate_signal()
        assert s is not None
        assert strat._signals_today == 1

        # Reset
        strat.reset()
        assert strat._signals_today == 0
        assert len(strat._signals_generated) == 0


# ── Test 7: HMM state gating ────────────────────────────────────────


class TestHMMStateGating:
    def test_hmm_state_gating(self):
        """require_hmm_states filters wrong states."""
        hub = FeatureHub()
        config = _make_config(
            require_hmm_states=[RegimeState.BREAKOUT, RegimeState.HIGH_VOL_UP]
        )
        strat = MockStrategy(config, hub)

        in_session = datetime(2025, 6, 2, 10, 30, tzinfo=_ET)

        # Default regime is LOW_VOL_RANGE — not in allowed list
        assert strat._current_regime == RegimeState.LOW_VOL_RANGE
        assert strat.is_valid_hmm_state() is False
        assert strat.can_generate_signal(in_session) is False

        # Manually set to allowed state
        strat._current_regime = RegimeState.BREAKOUT
        assert strat.is_valid_hmm_state() is True
        assert strat.can_generate_signal(in_session) is True
