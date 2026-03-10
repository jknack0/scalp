"""Tests for Value Area Reversion strategy."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.core.events import BarEvent
from src.signals.base import SignalResult
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle
from src.strategies.value_area_reversion import VAState, ValueAreaReversionStrategy

_ET = ZoneInfo("US/Eastern")


def _make_config(**overrides):
    cfg = {
        "strategy": {"strategy_id": "value_area_reversion", "max_signals_per_day": 1},
        "value_area": {
            "confirmation_minutes": 5,  # short for testing
            "price_step": 0.25,
            "adx_max": 20.0,
            "stop_buffer_ticks": 4,
            "min_va_width": 2.0,
        },
        "exit": {"time_stop_minutes": 120},
    }
    cfg["value_area"].update(overrides)
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


class TestValueAreaReversion:
    def _build_prior_session(self, strat, poc=5600.0, half_range=10.0):
        """Feed bars to build a prior session with known VA."""
        # Build bars centered around POC with volume concentrated there
        for i in range(100):
            # Most volume at POC, less at edges
            offset = (i % 20 - 10) * 0.5
            price = poc + offset
            vol = 200 if abs(offset) < 3 else 50  # More vol near center
            total_min = 30 + i * 5
            bar = _make_bar(
                close=price,
                high=price + 0.25,
                low=price - 0.25,
                volume=vol,
                hour=9 + total_min // 60,
                minute=total_min % 60,
                day=2,
            )
            strat.on_bar(bar, EMPTY_BUNDLE)

    def test_computes_value_area(self):
        """Prior session bars produce a valid VA."""
        strat = ValueAreaReversionStrategy(_make_config())
        self._build_prior_session(strat, poc=5600.0)

        # Trigger new day to compute VA
        bar = _make_bar(close=5600.0, hour=9, minute=30, day=3)
        strat.on_bar(bar, EMPTY_BUNDLE)

        assert strat._prior_poc > 0
        assert strat._prior_vah >= strat._prior_poc
        assert strat._prior_val <= strat._prior_poc
        assert strat._prior_vah > strat._prior_val  # VA has width

    def test_open_inside_va_no_signal(self):
        """Opening inside the VA → no setup."""
        strat = ValueAreaReversionStrategy(_make_config())
        self._build_prior_session(strat, poc=5600.0)

        # New day: open inside VA
        bar = _make_bar(close=5600.0, hour=9, minute=30, day=3)
        strat.on_bar(bar, EMPTY_BUNDLE)
        assert strat._state == VAState.DONE

    def test_open_below_val_watching(self):
        """Opening below VAL → watching for re-entry → LONG."""
        strat = ValueAreaReversionStrategy(_make_config())
        self._build_prior_session(strat, poc=5600.0)

        # Force known VA for predictable test
        strat._prior_val = 5595.0
        strat._prior_vah = 5605.0
        strat._prior_poc = 5600.0

        # New day: open below VAL
        bar = _make_bar(close=5590.0, hour=9, minute=30, day=3)
        strat._current_date = None  # Force new day
        strat.on_bar(bar, EMPTY_BUNDLE)
        assert strat._state == VAState.WATCHING

    def test_reentry_triggers_confirmation(self):
        """Price re-entering VA starts confirmation period."""
        strat = ValueAreaReversionStrategy(_make_config())
        strat._prior_val = 5595.0
        strat._prior_vah = 5605.0
        strat._prior_poc = 5600.0
        strat._state = VAState.WATCHING
        strat._reentry_direction = None
        strat._current_date = datetime(2025, 6, 3).date()

        # Set up for LONG (opened below)
        strat._reentry_direction = None
        strat._state = VAState.CHECKING_OPEN

        # Open below VA
        bar1 = _make_bar(close=5590.0, hour=9, minute=30, day=3)
        strat.on_bar(bar1, EMPTY_BUNDLE)
        assert strat._state == VAState.WATCHING

        # Price re-enters VA
        bar2 = _make_bar(close=5596.0, hour=9, minute=45, day=3)
        strat.on_bar(bar2, _make_adx_bundle())
        assert strat._state == VAState.CONFIRMING

    def test_signal_after_confirmation(self):
        """After confirmation period, signal is generated."""
        strat = ValueAreaReversionStrategy(_make_config(confirmation_minutes=5))
        strat._prior_val = 5595.0
        strat._prior_vah = 5605.0
        strat._prior_poc = 5600.0
        strat._state = VAState.CONFIRMING
        strat._reentry_direction = strat._reentry_direction  # Will be set below
        strat._current_date = datetime(2025, 6, 3).date()
        strat._signals_today = 0

        # Simulate re-entry from below (LONG)
        from src.strategies.base import Direction
        strat._reentry_direction = Direction.LONG
        strat._reentry_time = datetime(2025, 6, 3, 9, 45, tzinfo=_ET)

        # Bar 6 minutes after re-entry (past confirmation)
        bar = _make_bar(close=5597.0, hour=9, minute=51, day=3)
        signal = strat.on_bar(bar, _make_adx_bundle())

        assert signal is not None
        assert signal.direction.value == "LONG"
        assert signal.target_price == 5600.0  # POC

    def test_breaks_back_out_resets(self):
        """Price breaking back out of VA during confirmation resets to watching."""
        strat = ValueAreaReversionStrategy(_make_config())
        strat._prior_val = 5595.0
        strat._prior_vah = 5605.0
        strat._prior_poc = 5600.0
        strat._state = VAState.CONFIRMING
        strat._current_date = datetime(2025, 6, 3).date()
        strat._reentry_time = datetime(2025, 6, 3, 9, 45, tzinfo=_ET)

        from src.strategies.base import Direction
        strat._reentry_direction = Direction.LONG

        # Price drops back below VAL
        bar = _make_bar(close=5593.0, hour=9, minute=50, day=3)
        strat.on_bar(bar, _make_adx_bundle())
        assert strat._state == VAState.WATCHING

    def test_adx_blocks_signal(self):
        """High ADX blocks signal generation during confirmation."""
        strat = ValueAreaReversionStrategy(_make_config(confirmation_minutes=5))
        strat._prior_val = 5595.0
        strat._prior_vah = 5605.0
        strat._prior_poc = 5600.0
        strat._state = VAState.CONFIRMING
        strat._current_date = datetime(2025, 6, 3).date()
        strat._signals_today = 0

        from src.strategies.base import Direction
        strat._reentry_direction = Direction.LONG
        strat._reentry_time = datetime(2025, 6, 3, 9, 45, tzinfo=_ET)

        bar = _make_bar(close=5597.0, hour=9, minute=51, day=3)
        signal = strat.on_bar(bar, _make_adx_bundle(adx=25.0))
        assert signal is None

    def test_from_yaml(self):
        strat = ValueAreaReversionStrategy.from_yaml("config/strategies/value_area_reversion.yaml")
        assert strat.strategy_id == "value_area_reversion"
        assert strat._confirmation_minutes == 60

    def test_reset(self):
        strat = ValueAreaReversionStrategy(_make_config())
        strat._signals_today = 1
        strat._state = VAState.DONE
        strat.reset()
        assert strat._signals_today == 0
        assert strat._state == VAState.WAITING
