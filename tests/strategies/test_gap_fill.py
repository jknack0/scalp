"""Tests for Gap Fill strategy."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.core.events import BarEvent
from src.signals.base import SignalResult
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle
from src.strategies.gap_fill import GapFillState, GapFillStrategy

_ET = ZoneInfo("US/Eastern")


def _make_config(**overrides):
    cfg = {
        "strategy": {"strategy_id": "gap_fill", "max_signals_per_day": 1},
        "gap": {
            "min_gap_pct": 0.10,
            "max_gap_pct": 0.50,
            "ib_minutes": 15,
            "adx_max": 25.0,
            "noon_cutoff": True,
        },
        "exit": {"time_stop_minutes": 150, "stop_ib_pct": 0.75},
    }
    cfg["gap"].update(overrides)
    return cfg


def _make_bar(close=5600.0, high=5601.0, low=5599.0, volume=100,
              hour=10, minute=0, day=2):
    dt = datetime(2025, 6, day, hour, minute, tzinfo=_ET)
    ts_ns = int(dt.timestamp() * 1e9)
    return BarEvent(
        symbol="MESM6", open=close - 0.25, high=high, low=low,
        close=close, volume=volume, bar_type="5m", timestamp_ns=ts_ns,
    )


def _make_adx_bundle(adx=15.0):
    results = {
        "adx": SignalResult(value=adx, passes=False, direction="none",
                           metadata={"adx": adx, "plus_di": 10.0, "minus_di": 8.0}),
    }
    return SignalBundle(results=results, bar_count=50)


class TestGapFill:
    def test_detects_gap_down_goes_long(self):
        """Gap down → fade → LONG signal."""
        strat = GapFillStrategy(_make_config())

        # Day 1: establish prior close at 5600
        bar1 = _make_bar(close=5600.0, hour=15, minute=55, day=2)
        strat.on_bar(bar1, EMPTY_BUNDLE)

        # Day 2: open lower (gap down of ~0.18%)
        open_bar = _make_bar(close=5590.0, high=5591.0, low=5588.0, hour=9, minute=30, day=3)
        strat.on_bar(open_bar, EMPTY_BUNDLE)

        # Bars during IB (before 15 min mark)
        for m in range(35, 44, 5):
            bar = _make_bar(close=5589.0, high=5592.0, low=5587.0,
                           hour=9, minute=m, day=3)
            strat.on_bar(bar, EMPTY_BUNDLE)

        # Bar at 9:45 completes IB and triggers entry check with ADX bundle
        signal_bar = _make_bar(close=5589.0, high=5592.0, low=5587.0,
                               hour=9, minute=45, day=3)
        signal = strat.on_bar(signal_bar, _make_adx_bundle())

        assert signal is not None
        assert signal.direction.value == "LONG"
        assert signal.target_price == 5600.0  # Prior close

    def test_detects_gap_up_goes_short(self):
        """Gap up → fade → SHORT signal."""
        strat = GapFillStrategy(_make_config())

        # Day 1
        bar1 = _make_bar(close=5600.0, hour=15, minute=55, day=2)
        strat.on_bar(bar1, EMPTY_BUNDLE)

        # Day 2: open higher (gap up ~0.18%)
        open_bar = _make_bar(close=5610.0, high=5612.0, low=5608.0, hour=9, minute=30, day=3)
        strat.on_bar(open_bar, EMPTY_BUNDLE)

        for m in range(35, 44, 5):
            bar = _make_bar(close=5611.0, high=5612.0, low=5608.0,
                           hour=9, minute=m, day=3)
            strat.on_bar(bar, EMPTY_BUNDLE)

        signal_bar = _make_bar(close=5611.0, high=5612.0, low=5608.0,
                               hour=9, minute=45, day=3)
        signal = strat.on_bar(signal_bar, _make_adx_bundle())

        assert signal is not None
        assert signal.direction.value == "SHORT"
        assert signal.target_price == 5600.0

    def test_blocks_small_gap(self):
        """Gap too small (< 0.10%) is ignored."""
        strat = GapFillStrategy(_make_config())

        bar1 = _make_bar(close=5600.0, hour=15, minute=55, day=2)
        strat.on_bar(bar1, EMPTY_BUNDLE)

        # Gap of only 2 points = 0.036% — too small
        open_bar = _make_bar(close=5598.0, hour=9, minute=30, day=3)
        strat.on_bar(open_bar, EMPTY_BUNDLE)

        for m in range(35, 50, 5):
            bar = _make_bar(close=5598.0, hour=9, minute=m, day=3)
            strat.on_bar(bar, EMPTY_BUNDLE)

        assert strat._state == GapFillState.DONE

    def test_blocks_large_gap(self):
        """Gap too large (> 0.50%) is ignored."""
        strat = GapFillStrategy(_make_config())

        bar1 = _make_bar(close=5600.0, hour=15, minute=55, day=2)
        strat.on_bar(bar1, EMPTY_BUNDLE)

        # Gap of 30 points = 0.54% — too large
        open_bar = _make_bar(close=5570.0, high=5571.0, low=5568.0, hour=9, minute=30, day=3)
        strat.on_bar(open_bar, EMPTY_BUNDLE)

        for m in range(35, 50, 5):
            bar = _make_bar(close=5570.0, hour=9, minute=m, day=3)
            strat.on_bar(bar, EMPTY_BUNDLE)

        assert strat._state == GapFillState.DONE

    def test_one_trade_per_day(self):
        """Max 1 signal per day."""
        strat = GapFillStrategy(_make_config())
        strat._signals_today = 1
        strat._state = GapFillState.READY
        strat._prior_close = 5600.0
        strat._session_open = 5590.0
        strat._ib_high = 5592.0
        strat._ib_low = 5588.0
        strat._current_date = datetime(2025, 6, 3).date()

        bar = _make_bar(close=5590.0, hour=10, minute=0, day=3)
        signal = strat.on_bar(bar, _make_adx_bundle())
        assert signal is None
        assert strat._state == GapFillState.DONE

    def test_from_yaml(self):
        strat = GapFillStrategy.from_yaml("config/strategies/gap_fill.yaml")
        assert strat.strategy_id == "gap_fill"
        assert strat._min_gap_pct == 0.10
