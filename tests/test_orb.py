"""Tests for ORB (Opening Range Breakout) strategy.

9 tests covering range collection, breakout detection, volume/time filters,
signal geometry, and one-shot-per-day behavior.
"""

import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.core.events import BarEvent
from src.features.feature_hub import FeatureHub
from src.models.hmm_regime import RegimeState
from src.strategies.base import TICK_SIZE, Direction
from src.strategies.orb_strategy import ORBConfig, ORBState, ORBStrategy

# US Eastern (DST-aware, matching production code)
_ET = ZoneInfo("US/Eastern")

# Fixed date: Monday 2025-06-02
_BASE_DATE = datetime(2025, 6, 2, tzinfo=_ET)


def _et_to_ns(hour: int, minute: int, second: int = 0) -> int:
    """Convert an ET time on 2025-06-02 to timestamp_ns."""
    dt = _BASE_DATE.replace(hour=hour, minute=minute, second=second)
    return int(dt.timestamp() * 1_000_000_000)


def _make_bar_at(
    time_str: str,
    close: float,
    volume: int,
    bar_type: str = "1s",
    high: float | None = None,
    low: float | None = None,
) -> BarEvent:
    """Create a BarEvent at a given ET time on 2025-06-02."""
    parts = time_str.split(":")
    hour, minute = int(parts[0]), int(parts[1])
    second = int(parts[2]) if len(parts) > 2 else 0
    ts_ns = _et_to_ns(hour, minute, second)
    return BarEvent(
        symbol="MESM6",
        open=close - 0.25,
        high=high if high is not None else close + 0.50,
        low=low if low is not None else close - 0.50,
        close=close,
        volume=volume,
        bar_type=bar_type,
        timestamp_ns=ts_ns,
    )


def _make_orb(**config_overrides) -> ORBStrategy:
    """Create an ORBStrategy with FeatureHub and optional config overrides."""
    defaults = dict(
        require_hmm_states=[],  # disable HMM gating for simpler tests
        require_vwap_alignment=False,  # disable VWAP gating by default
        min_confidence=0.0,  # don't block on confidence
    )
    defaults.update(config_overrides)
    config = ORBConfig(**defaults)
    hub = FeatureHub()
    strat = ORBStrategy(config, hub)
    return strat


def _feed_range(strat: ORBStrategy, high: float, low: float, n_bars: int = 10) -> None:
    """Feed n bars during 9:30–9:44 to establish the ORB range.

    Creates bars with controlled high/low to set a known range.
    The first bar sets the high, the second sets the low, rest are mid-range.
    """
    mid = (high + low) / 2.0
    for i in range(n_bars):
        second = i * 5  # spread across 9:30–9:30:45
        ts_ns = _et_to_ns(9, 30, second)
        if i == 0:
            bar = BarEvent(
                symbol="MESM6",
                open=mid,
                high=high,
                low=mid - 0.25,
                close=mid + 0.25,
                volume=100,
                bar_type="1s",
                timestamp_ns=ts_ns,
            )
        elif i == 1:
            bar = BarEvent(
                symbol="MESM6",
                open=mid,
                high=mid + 0.25,
                low=low,
                close=mid - 0.25,
                volume=100,
                bar_type="1s",
                timestamp_ns=ts_ns,
            )
        else:
            bar = BarEvent(
                symbol="MESM6",
                open=mid,
                high=mid + 0.25,
                low=mid - 0.25,
                close=mid,
                volume=100,
                bar_type="1s",
                timestamp_ns=ts_ns,
            )
        strat.on_bar(bar)


# ── Test 1: Range correctly collected ──────────────────────────────


class TestRangeCollection:
    def test_range_correctly_collected(self):
        """Feed bars with known highs/lows during 9:30–9:44, verify range."""
        strat = _make_orb()

        # Feed bars during the opening range window
        _feed_range(strat, high=5010.0, low=5000.0, n_bars=10)

        assert strat._range_high == pytest.approx(5010.0)
        assert strat._range_low == pytest.approx(5000.0)

        # Feed a bar at 9:45 to trigger transition
        bar_945 = _make_bar_at("09:45", close=5005.0, volume=100)
        strat.on_bar(bar_945)

        assert strat._state == ORBState.WATCHING_BREAKOUT


# ── Test 2: No signal on wick break only (close inside range) ──────


class TestWickBreakNoSignal:
    def test_no_signal_on_wick_break_only_close(self):
        """Bar with high > orb_high but close < orb_high → no signal."""
        strat = _make_orb()
        _feed_range(strat, high=5010.0, low=5000.0)

        # Transition to WATCHING_BREAKOUT
        strat.on_bar(_make_bar_at("09:45", close=5005.0, volume=100))
        assert strat._state == ORBState.WATCHING_BREAKOUT

        # Seed volume history
        for i in range(5):
            strat.on_bar(_make_bar_at(f"09:46:{i * 10}", close=5005.0, volume=100))

        # Bar with wick above range but close inside
        wick_bar = BarEvent(
            symbol="MESM6",
            open=5008.0,
            high=5012.0,  # above range_high
            low=5007.0,
            close=5009.0,  # below range_high (5010)
            volume=200,
            bar_type="5m",
            timestamp_ns=_et_to_ns(9, 55),
        )
        strat.on_bar(wick_bar)
        assert strat._state == ORBState.WATCHING_BREAKOUT  # no signal generated


# ── Test 3: Long signal above ORB high ─────────────────────────────


class TestLongSignal:
    def test_long_signal_above_orb_high(self):
        """Bar close > orb_high → LONG signal generated."""
        strat = _make_orb()
        _feed_range(strat, high=5010.0, low=5000.0)

        strat.on_bar(_make_bar_at("09:45", close=5005.0, volume=100))

        # Seed volume history
        for i in range(5):
            strat.on_bar(_make_bar_at(f"09:46:{i * 10}", close=5005.0, volume=100))

        # Breakout bar above range
        breakout = _make_bar_at("09:55", close=5012.0, volume=200, bar_type="5m")
        strat.on_bar(breakout)

        assert strat._state == ORBState.SIGNAL_GENERATED
        assert len(strat._signals_generated) == 1
        assert strat._signals_generated[0].direction == Direction.LONG


# ── Test 4: Short signal below ORB low ─────────────────────────────


class TestShortSignal:
    def test_short_signal_below_orb_low(self):
        """Bar close < orb_low → SHORT signal generated."""
        strat = _make_orb()
        _feed_range(strat, high=5010.0, low=5000.0)

        strat.on_bar(_make_bar_at("09:45", close=5005.0, volume=100))

        # Seed volume history
        for i in range(5):
            strat.on_bar(_make_bar_at(f"09:46:{i * 10}", close=5005.0, volume=100))

        # Breakout bar below range
        breakout = _make_bar_at("09:55", close=4998.0, volume=200, bar_type="5m")
        strat.on_bar(breakout)

        assert strat._state == ORBState.SIGNAL_GENERATED
        assert len(strat._signals_generated) == 1
        assert strat._signals_generated[0].direction == Direction.SHORT


# ── Test 5: Volume filter blocks low volume ────────────────────────


class TestVolumeFilter:
    def test_volume_filter_blocks_low_volume(self):
        """Breakout bar with volume below threshold → no signal."""
        strat = _make_orb(volume_multiplier=1.5)
        _feed_range(strat, high=5010.0, low=5000.0)

        strat.on_bar(_make_bar_at("09:45", close=5005.0, volume=100))

        # Seed volume history with avg=100
        for i in range(5):
            strat.on_bar(_make_bar_at(f"09:46:{i * 10}", close=5005.0, volume=100))

        # Breakout bar with low volume (120 < 1.5 * 100 = 150)
        breakout = _make_bar_at("09:55", close=5012.0, volume=120, bar_type="5m")
        strat.on_bar(breakout)

        assert strat._state == ORBState.WATCHING_BREAKOUT  # still watching, no signal
        assert len(strat._signals_generated) == 0


# ── Test 6: No signal after 11:00 ─────────────────────────────────


class TestMaxSignalTime:
    def test_no_signal_after_11am(self):
        """Breakout bar at 11:05 → state is INACTIVE, no signal."""
        strat = _make_orb()
        _feed_range(strat, high=5010.0, low=5000.0)

        strat.on_bar(_make_bar_at("09:45", close=5005.0, volume=100))

        # Seed some volume bars
        for i in range(5):
            strat.on_bar(_make_bar_at(f"09:46:{i * 10}", close=5005.0, volume=100))

        # Bar at 11:05 — should transition to INACTIVE
        late_bar = _make_bar_at("11:05", close=5012.0, volume=200, bar_type="5m")
        strat.on_bar(late_bar)

        assert strat._state == ORBState.INACTIVE
        assert len(strat._signals_generated) == 0


# ── Test 7: Only one signal per day ────────────────────────────────


class TestOneShotPerDay:
    def test_only_one_signal_per_day(self):
        """After first signal, state is SIGNAL_GENERATED, second breakout produces no signal."""
        strat = _make_orb()
        _feed_range(strat, high=5010.0, low=5000.0)

        strat.on_bar(_make_bar_at("09:45", close=5005.0, volume=100))

        # Seed volume
        for i in range(5):
            strat.on_bar(_make_bar_at(f"09:46:{i * 10}", close=5005.0, volume=100))

        # First breakout
        b1 = _make_bar_at("09:55", close=5012.0, volume=200, bar_type="5m")
        strat.on_bar(b1)
        assert strat._state == ORBState.SIGNAL_GENERATED
        assert len(strat._signals_generated) == 1

        # Second breakout attempt — should not produce a signal
        b2 = _make_bar_at("10:05", close=5015.0, volume=300, bar_type="5m")
        strat.on_bar(b2)
        assert len(strat._signals_generated) == 1  # still just 1


# ── Test 8: Target at correct price ───────────────────────────────


class TestTargetPrice:
    def test_target_at_correct_price(self):
        """Long signal target = entry + orb_width * 0.5."""
        strat = _make_orb(target_multiplier=0.5, slippage_ticks=1)
        _feed_range(strat, high=5010.0, low=5000.0)

        strat.on_bar(_make_bar_at("09:45", close=5005.0, volume=100))

        for i in range(5):
            strat.on_bar(_make_bar_at(f"09:46:{i * 10}", close=5005.0, volume=100))

        breakout = _make_bar_at("09:55", close=5012.0, volume=200, bar_type="5m")
        strat.on_bar(breakout)

        signal = strat._signals_generated[0]
        orb_width = 5010.0 - 5000.0  # 10.0 points
        expected_entry = 5012.0 + 1 * TICK_SIZE  # close + slippage
        expected_target = expected_entry + orb_width * 0.5

        assert signal.entry_price == pytest.approx(expected_entry)
        assert signal.target_price == pytest.approx(expected_target)


# ── Test 9: Stop at opposite range boundary ────────────────────────


class TestStopPrice:
    def test_stop_at_opposite_range_boundary(self):
        """Long stop = orb_low, short stop = orb_high."""
        # Test LONG stop
        strat_long = _make_orb()
        _feed_range(strat_long, high=5010.0, low=5000.0)
        strat_long.on_bar(_make_bar_at("09:45", close=5005.0, volume=100))
        for i in range(5):
            strat_long.on_bar(_make_bar_at(f"09:46:{i * 10}", close=5005.0, volume=100))
        strat_long.on_bar(_make_bar_at("09:55", close=5012.0, volume=200, bar_type="5m"))
        assert strat_long._signals_generated[0].stop_price == pytest.approx(5000.0)

        # Test SHORT stop
        strat_short = _make_orb()
        _feed_range(strat_short, high=5010.0, low=5000.0)
        strat_short.on_bar(_make_bar_at("09:45", close=5005.0, volume=100))
        for i in range(5):
            strat_short.on_bar(_make_bar_at(f"09:46:{i * 10}", close=5005.0, volume=100))
        strat_short.on_bar(_make_bar_at("09:55", close=4998.0, volume=200, bar_type="5m"))
        assert strat_short._signals_generated[0].stop_price == pytest.approx(5010.0)
