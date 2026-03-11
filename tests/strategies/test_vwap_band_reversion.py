"""Tests for VWAP Band Reversion strategy."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.core.events import BarEvent
from src.signals.base import SignalResult
from src.signals.signal_bundle import SignalBundle
from src.strategies.vwap_band_reversion import VWAPBandReversionStrategy

_ET = ZoneInfo("US/Eastern")


def _make_config(filter_overrides=None):
    """Build config with declarative filters matching the YAML structure."""
    filters = [
        {"signal": "session_time", "expr": ">= 585"},
        {"signal": "session_time", "expr": "<= 900"},
        {"signal": "vwap_session", "field": "session_age_bars", "expr": ">= 5"},
        {"signal": "vwap_session", "field": "slope", "expr": "abs <= 0.5"},
        {"signal": "vwap_session", "field": "deviation_sd", "expr": "abs >= 2.0"},
        {"signal": "adx", "expr": "< 20.0"},
        {"signal": "relative_volume", "expr": ">= 1.5"},
    ]
    if filter_overrides:
        filters = filter_overrides

    return {
        "strategy": {"strategy_id": "vwap_band_reversion", "max_signals_per_day": 3},
        "filters": filters,
        "exit": {
            "target": {"type": "vwap"},
            "stop": {"type": "atr_multiple", "multiplier": 1.5},
            "time_stop_minutes": 30,
        },
    }


def _make_bar(close=5600.0, high=5601.0, low=5599.0, volume=100, ts_ns=None):
    if ts_ns is None:
        # 10:00 AM ET on a weekday
        dt = datetime(2025, 6, 2, 10, 0, tzinfo=_ET)
        ts_ns = int(dt.timestamp() * 1e9)
    return BarEvent(
        symbol="MESM6", open=close - 0.25, high=high, low=low,
        close=close, volume=volume, bar_type="5m", timestamp_ns=ts_ns,
    )


def _make_bundle(deviation_sd=-2.5, vwap=5600.0, sd=5.0, slope=0.0,
                 session_age=50, rsi=5.0, adx=15.0,
                 atr_raw=3.0, rvol=2.0, session_time=600.0):
    """Build a SignalBundle with all gates passing for a LONG signal."""
    results = {
        "vwap_session": SignalResult(
            value=deviation_sd, passes=True, direction="long",
            metadata={"vwap": vwap, "sd": sd, "slope": slope,
                      "deviation_sd": deviation_sd, "mode": "REVERSION",
                      "first_kiss": False, "session_age_bars": session_age},
        ),
        "rsi_momentum": SignalResult(value=rsi, passes=True, direction="short"),
        "adx": SignalResult(value=adx, passes=False, direction="none",
                           metadata={"adx": adx, "plus_di": 10.0, "minus_di": 8.0}),
        "atr": SignalResult(value=12.0, passes=True, direction="none",
                           metadata={"atr_ticks": 12.0, "atr_raw": atr_raw,
                                     "vol_regime": "NORMAL", "atr_percentile": 50.0}),
        "relative_volume": SignalResult(value=rvol, passes=True, direction="none",
                                       metadata={"rvol": rvol}),
        "session_time": SignalResult(value=session_time, passes=True, direction="none",
                                    metadata={"minutes_since_midnight": session_time}),
    }
    return SignalBundle(results=results, bar_count=100)


class TestVWAPBandReversion:
    def test_generates_long_signal(self):
        """All gates pass for oversold below VWAP -> LONG signal."""
        strat = VWAPBandReversionStrategy(_make_config())
        bar = _make_bar(close=5590.0, low=5589.0)
        bundle = _make_bundle(deviation_sd=-2.5, vwap=5600.0, sd=5.0)

        signal = strat.on_bar(bar, bundle)

        assert signal is not None
        assert signal.direction.value == "LONG"
        assert signal.entry_price == 5590.0
        assert signal.target_price == 5600.0  # VWAP target
        assert signal.stop_price < signal.entry_price

    def test_generates_short_signal(self):
        """All gates pass for overbought above VWAP -> SHORT signal."""
        strat = VWAPBandReversionStrategy(_make_config())
        bar = _make_bar(close=5610.0, high=5611.0)
        bundle = _make_bundle(deviation_sd=2.5, vwap=5600.0, sd=5.0,
                             rsi=95.0)

        signal = strat.on_bar(bar, bundle)

        assert signal is not None
        assert signal.direction.value == "SHORT"
        assert signal.target_price == 5600.0  # VWAP target

    def test_blocks_when_adx_high(self):
        """ADX >= 20 blocks signal (trending market)."""
        strat = VWAPBandReversionStrategy(_make_config())
        bar = _make_bar(close=5590.0)
        bundle = _make_bundle(adx=25.0)

        signal = strat.on_bar(bar, bundle)
        assert signal is None

    def test_blocks_when_deviation_insufficient(self):
        """Price within bands blocks signal."""
        strat = VWAPBandReversionStrategy(_make_config())
        bar = _make_bar(close=5598.0)
        bundle = _make_bundle(deviation_sd=-1.0)

        signal = strat.on_bar(bar, bundle)
        assert signal is None

    def test_blocks_low_volume(self):
        """Low relative volume blocks signal."""
        strat = VWAPBandReversionStrategy(_make_config())
        bar = _make_bar(close=5590.0)
        bundle = _make_bundle(rvol=0.8)

        signal = strat.on_bar(bar, bundle)
        assert signal is None

    def test_blocks_steep_vwap_slope(self):
        """Steep VWAP slope blocks signal (VWAP is trending)."""
        strat = VWAPBandReversionStrategy(_make_config())
        bar = _make_bar(close=5590.0)
        bundle = _make_bundle(slope=1.5)

        signal = strat.on_bar(bar, bundle)
        assert signal is None

    def test_respects_daily_limit(self):
        """Max signals per day enforced."""
        strat = VWAPBandReversionStrategy(_make_config())
        bar = _make_bar(close=5590.0, low=5589.0)
        bundle = _make_bundle()

        # Generate 3 signals
        for _ in range(3):
            sig = strat.on_bar(bar, bundle)
            assert sig is not None

        # 4th should be blocked
        sig = strat.on_bar(bar, bundle)
        assert sig is None

    def test_reset_clears_count(self):
        strat = VWAPBandReversionStrategy(_make_config())
        strat._signals_today = 3
        strat.reset()
        assert strat._signals_today == 0

    def test_from_yaml(self):
        strat = VWAPBandReversionStrategy.from_yaml("config/strategies/vwap_band_reversion.yaml")
        assert strat.strategy_id == "vwap_band_reversion"
        assert not strat._filter_engine.is_empty
