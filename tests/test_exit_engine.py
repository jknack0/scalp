"""Tests for ExitEngine — declarative exit condition evaluation."""

import pytest

from src.core.events import BarEvent
from src.exits.exit_engine import (
    AdverseSignalExit,
    ExitContext,
    ExitEngine,
    ExitResult,
    RegimeExit,
    StaticStop,
    StaticTarget,
    TimeStop,
    TrailingStop,
    VWAPReversionTarget,
    VolatilityExpansionExit,
    build_condition,
)
from src.signals.base import SignalResult
from src.signals.signal_bundle import SignalBundle


def _bar(close=5600.0, high=5605.0, low=5595.0, open_=5598.0, volume=100):
    return BarEvent(
        symbol="MES", open=open_, high=high, low=low, close=close,
        volume=volume, bar_type="5m", timestamp_ns=0,
    )


def _bundle(**signals):
    """Build a SignalBundle from keyword signal results."""
    results = {}
    for name, val in signals.items():
        if isinstance(val, SignalResult):
            results[name] = val
        elif isinstance(val, dict):
            results[name] = SignalResult(
                value=val.get("value", 0.0),
                passes=val.get("passes", True),
                direction=val.get("direction", "none"),
                metadata=val.get("metadata", {}),
            )
        else:
            results[name] = SignalResult(value=float(val), passes=True)
    return SignalBundle(results=results)


def _ctx(direction="LONG", fill_price=5600.0, bars_in_trade=5,
         bar=None, bundle=None, entry_snapshot=None, peak_price=0.0):
    return ExitContext(
        bar=bar or _bar(),
        bundle=bundle or _bundle(),
        direction=direction,
        fill_price=fill_price,
        bars_in_trade=bars_in_trade,
        entry_snapshot=entry_snapshot or {"atr": 5.0},
        peak_price=peak_price,
    )


# ── build_condition / ExitEngine.from_list ──────────────────────────

class TestExitEngineConstruction:
    def test_from_empty_list(self):
        engine = ExitEngine.from_list(None)
        assert engine.is_empty

    def test_from_list_filters_disabled(self):
        configs = [
            {"type": "static_stop", "enabled": True, "atr_multiple": 1.0},
            {"type": "time_stop", "enabled": False, "max_bars": 10},
        ]
        engine = ExitEngine.from_list(configs)
        assert len(engine.conditions) == 1
        assert isinstance(engine.conditions[0], StaticStop)

    def test_enabled_defaults_true(self):
        configs = [{"type": "time_stop", "max_bars": 10}]
        engine = ExitEngine.from_list(configs)
        assert len(engine.conditions) == 1

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown exit condition type"):
            build_condition({"type": "bogus_exit"})

    def test_has_type(self):
        engine = ExitEngine.from_list([
            {"type": "static_stop", "atr_multiple": 1.0},
            {"type": "time_stop", "max_bars": 10},
        ])
        assert engine.has_type("static_stop")
        assert engine.has_type("time_stop")
        assert not engine.has_type("trailing_stop")


# ── StaticTarget ────────────────────────────────────────────────────

class TestStaticTarget:
    def test_long_target_hit(self):
        cond = StaticTarget({"atr_multiple": 1.5})
        # fill=5600, atr=5, target=5607.5
        ctx = _ctx(direction="LONG", fill_price=5600.0,
                    bar=_bar(high=5608.0, close=5607.0))
        assert cond.evaluate(ctx) == "tp:static_target"

    def test_long_target_not_hit(self):
        cond = StaticTarget({"atr_multiple": 1.5})
        ctx = _ctx(direction="LONG", fill_price=5600.0,
                    bar=_bar(high=5605.0, close=5604.0))
        assert cond.evaluate(ctx) is None

    def test_short_target_hit(self):
        cond = StaticTarget({"atr_multiple": 2.0})
        # fill=5600, atr=5, target=5590
        ctx = _ctx(direction="SHORT", fill_price=5600.0,
                    bar=_bar(low=5589.0, close=5590.0))
        assert cond.evaluate(ctx) == "tp:static_target"

    def test_zero_atr_returns_none(self):
        cond = StaticTarget({"atr_multiple": 1.5})
        ctx = _ctx(entry_snapshot={"atr": 0.0})
        assert cond.evaluate(ctx) is None


# ── StaticStop ──────────────────────────────────────────────────────

class TestStaticStop:
    def test_long_stop_hit(self):
        cond = StaticStop({"atr_multiple": 1.0})
        # fill=5600, atr=5, stop=5595
        ctx = _ctx(direction="LONG", fill_price=5600.0,
                    bar=_bar(low=5594.0, close=5595.0))
        assert cond.evaluate(ctx) == "stop:static_stop"

    def test_long_stop_not_hit(self):
        cond = StaticStop({"atr_multiple": 1.0})
        ctx = _ctx(direction="LONG", fill_price=5600.0,
                    bar=_bar(low=5596.0, close=5598.0))
        assert cond.evaluate(ctx) is None

    def test_short_stop_hit(self):
        cond = StaticStop({"atr_multiple": 1.0})
        # fill=5600, atr=5, stop=5605
        ctx = _ctx(direction="SHORT", fill_price=5600.0,
                    bar=_bar(high=5606.0, close=5604.0))
        assert cond.evaluate(ctx) == "stop:static_stop"


# ── TrailingStop ────────────────────────────────────────────────────

class TestTrailingStop:
    def test_not_activated_below_threshold(self):
        cond = TrailingStop({"atr_multiple": 1.0, "activate_after_ticks": 4})
        # Only 2 ticks in profit — should not activate
        ctx = _ctx(direction="LONG", fill_price=5600.0,
                    bar=_bar(high=5601.0, close=5600.5))
        assert cond.evaluate(ctx) is None

    def test_long_trailing_fires(self):
        cond = TrailingStop({"atr_multiple": 1.0, "activate_after_ticks": 4})
        # 8 ticks in profit, peak at 5610, now pulled back to 5604
        ctx = _ctx(direction="LONG", fill_price=5600.0,
                    bar=_bar(high=5604.0, close=5604.0),
                    peak_price=5610.0,
                    entry_snapshot={"atr": 5.0})
        result = cond.evaluate(ctx)
        assert result == "stop:trailing"

    def test_long_trailing_holds(self):
        cond = TrailingStop({"atr_multiple": 1.0, "activate_after_ticks": 4})
        # 8 ticks in profit, peak at 5610, close at 5606 — within trail
        ctx = _ctx(direction="LONG", fill_price=5600.0,
                    bar=_bar(high=5610.0, close=5606.0),
                    peak_price=5610.0,
                    entry_snapshot={"atr": 5.0})
        result = cond.evaluate(ctx)
        assert result is None

    def test_short_trailing_fires(self):
        cond = TrailingStop({"atr_multiple": 1.0, "activate_after_ticks": 4})
        # Short from 5600, best low was 5590, now bounced to 5596
        ctx = _ctx(direction="SHORT", fill_price=5600.0,
                    bar=_bar(low=5596.0, close=5596.0),
                    peak_price=5590.0,
                    entry_snapshot={"atr": 5.0})
        result = cond.evaluate(ctx)
        assert result == "stop:trailing"


# ── TimeStop ────────────────────────────────────────────────────────

class TestTimeStop:
    def test_fires_at_max_bars(self):
        cond = TimeStop({"max_bars": 10})
        ctx = _ctx(bars_in_trade=10)
        assert cond.evaluate(ctx) == "stop:time"

    def test_holds_before_max_bars(self):
        cond = TimeStop({"max_bars": 10})
        ctx = _ctx(bars_in_trade=9)
        assert cond.evaluate(ctx) is None


# ── VWAPReversionTarget ─────────────────────────────────────────────

class TestVWAPReversionTarget:
    def test_fires_when_bar_high_touches_vwap_long(self):
        """LONG: bar.high reaches within target_sd_band of VWAP."""
        cond = VWAPReversionTarget({"target_sd_band": 0.5})
        # VWAP=5610, SD=5. Bar high=5609 → deviation = (5609-5610)/5 = -0.2 → |0.2| < 0.5 ✓
        bundle = _bundle(vwap_session={"value": 0.0, "metadata": {
            "vwap": 5610.0, "sd": 5.0, "deviation_sd": -2.0,
        }})
        bar = _bar(close=5600.0, high=5609.0, low=5595.0)
        ctx = _ctx(direction="LONG", bundle=bundle, bar=bar)
        assert cond.evaluate(ctx) == "tp:reversion_target"

    def test_fires_when_bar_low_touches_vwap_short(self):
        """SHORT: bar.low reaches within target_sd_band of VWAP."""
        cond = VWAPReversionTarget({"target_sd_band": 0.5})
        # VWAP=5590, SD=5. Bar low=5591 → deviation = (5591-5590)/5 = 0.2 → |0.2| < 0.5 ✓
        bundle = _bundle(vwap_session={"value": 0.0, "metadata": {
            "vwap": 5590.0, "sd": 5.0, "deviation_sd": 2.0,
        }})
        bar = _bar(close=5600.0, high=5605.0, low=5591.0)
        ctx = _ctx(direction="SHORT", bundle=bundle, bar=bar)
        assert cond.evaluate(ctx) == "tp:reversion_target"

    def test_holds_when_extended(self):
        """Bar high still far from VWAP → no TP."""
        cond = VWAPReversionTarget({"target_sd_band": 0.5})
        # VWAP=5620, SD=5. Bar high=5605 → deviation = (5605-5620)/5 = -3.0 → |3.0| > 0.5
        bundle = _bundle(vwap_session={"value": 0.0, "metadata": {
            "vwap": 5620.0, "sd": 5.0, "deviation_sd": -4.0,
        }})
        ctx = _ctx(direction="LONG", bundle=bundle)
        assert cond.evaluate(ctx) is None

    def test_holds_when_no_vwap_signal(self):
        cond = VWAPReversionTarget({"target_sd_band": 0.5})
        ctx = _ctx(bundle=_bundle())
        assert cond.evaluate(ctx) is None

    def test_holds_when_vwap_zero(self):
        cond = VWAPReversionTarget({"target_sd_band": 0.5})
        bundle = _bundle(vwap_session={"value": 0.0, "metadata": {
            "vwap": 0.0, "sd": 0.0, "deviation_sd": 0.0,
        }})
        ctx = _ctx(bundle=bundle)
        assert cond.evaluate(ctx) is None


# ── AdverseSignalExit ───────────────────────────────────────────────

class TestAdverseSignalExit:
    def test_long_adverse_slope(self):
        cond = AdverseSignalExit({
            "signal": "vwap_session", "field": "slope",
            "long_threshold": -0.3, "short_threshold": 0.3,
        })
        bundle = _bundle(vwap_session={"value": 0.0, "metadata": {"slope": -0.5}})
        ctx = _ctx(direction="LONG", bundle=bundle)
        assert cond.evaluate(ctx) is not None

    def test_long_ok_slope(self):
        cond = AdverseSignalExit({
            "signal": "vwap_session", "field": "slope",
            "long_threshold": -0.3, "short_threshold": 0.3,
        })
        bundle = _bundle(vwap_session={"value": 0.0, "metadata": {"slope": -0.1}})
        ctx = _ctx(direction="LONG", bundle=bundle)
        assert cond.evaluate(ctx) is None

    def test_short_adverse_slope(self):
        cond = AdverseSignalExit({
            "signal": "vwap_session", "field": "slope",
            "long_threshold": -0.3, "short_threshold": 0.3,
        })
        bundle = _bundle(vwap_session={"value": 0.0, "metadata": {"slope": 0.5}})
        ctx = _ctx(direction="SHORT", bundle=bundle)
        assert cond.evaluate(ctx) is not None

    def test_uses_value_when_no_field(self):
        cond = AdverseSignalExit({
            "signal": "adx", "long_threshold": -10.0, "short_threshold": 10.0,
        })
        bundle = _bundle(adx=SignalResult(value=15.0, passes=True))
        ctx = _ctx(direction="SHORT", bundle=bundle)
        assert cond.evaluate(ctx) is not None


# ── RegimeExit ──────────────────────────────────────────────────────

class TestRegimeExit:
    def test_fires_on_hostile_regime_long(self):
        cond = RegimeExit({
            "hmm_signal": "hmm_regime",
            "hostile_regimes_long": [1],
            "hostile_regimes_short": [0],
            "min_bars_before_active": 2,
        })
        bundle = _bundle(hmm_regime=SignalResult(value=1.0, passes=False))
        ctx = _ctx(direction="LONG", bars_in_trade=3, bundle=bundle)
        assert cond.evaluate(ctx) == "early:regime_flip"

    def test_holds_in_safe_regime(self):
        cond = RegimeExit({
            "hmm_signal": "hmm_regime",
            "hostile_regimes_long": [1],
            "hostile_regimes_short": [0],
            "min_bars_before_active": 2,
        })
        bundle = _bundle(hmm_regime=SignalResult(value=0.0, passes=True))
        ctx = _ctx(direction="LONG", bars_in_trade=3, bundle=bundle)
        assert cond.evaluate(ctx) is None

    def test_skips_early_bars(self):
        cond = RegimeExit({
            "hmm_signal": "hmm_regime",
            "hostile_regimes_long": [1],
            "min_bars_before_active": 5,
        })
        bundle = _bundle(hmm_regime=SignalResult(value=1.0, passes=False))
        ctx = _ctx(direction="LONG", bars_in_trade=3, bundle=bundle)
        assert cond.evaluate(ctx) is None


# ── VolatilityExpansionExit ─────────────────────────────────────────

class TestVolatilityExpansionExit:
    def test_fires_on_vol_expansion(self):
        cond = VolatilityExpansionExit({
            "expansion_multiple": 1.5, "min_bars_before_active": 2,
        })
        bundle = _bundle(atr={"value": 10.0, "metadata": {"atr_raw": 10.0}})
        ctx = _ctx(bars_in_trade=5, entry_snapshot={"atr": 5.0}, bundle=bundle)
        assert cond.evaluate(ctx) == "early:vol_expansion"

    def test_holds_when_vol_normal(self):
        cond = VolatilityExpansionExit({
            "expansion_multiple": 1.5, "min_bars_before_active": 2,
        })
        bundle = _bundle(atr={"value": 6.0, "metadata": {"atr_raw": 6.0}})
        ctx = _ctx(bars_in_trade=5, entry_snapshot={"atr": 5.0}, bundle=bundle)
        assert cond.evaluate(ctx) is None

    def test_skips_early_bars(self):
        cond = VolatilityExpansionExit({
            "expansion_multiple": 1.5, "min_bars_before_active": 3,
        })
        bundle = _bundle(atr={"value": 10.0, "metadata": {"atr_raw": 10.0}})
        ctx = _ctx(bars_in_trade=1, entry_snapshot={"atr": 5.0}, bundle=bundle)
        assert cond.evaluate(ctx) is None


# ── ExitEngine.evaluate (OR logic) ──────────────────────────────────

class TestExitEngineEvaluation:
    def test_or_logic_first_fires(self):
        engine = ExitEngine.from_list([
            {"type": "time_stop", "max_bars": 10},
            {"type": "static_stop", "atr_multiple": 1.0},
        ])
        # Time stop fires first (bars_in_trade=12)
        ctx = _ctx(bars_in_trade=12, fill_price=5600.0,
                    bar=_bar(low=5598.0))
        result = engine.evaluate(ctx)
        assert result.should_exit
        assert result.reason == "stop:time"

    def test_no_conditions_holds(self):
        engine = ExitEngine.from_list([])
        ctx = _ctx()
        result = engine.evaluate(ctx)
        assert not result.should_exit

    def test_all_conditions_hold(self):
        engine = ExitEngine.from_list([
            {"type": "time_stop", "max_bars": 20},
            {"type": "static_stop", "atr_multiple": 3.0},
        ])
        ctx = _ctx(bars_in_trade=5, fill_price=5600.0,
                    bar=_bar(low=5596.0))
        result = engine.evaluate(ctx)
        assert not result.should_exit


# ── get_bracket_prices ──────────────────────────────────────────────

class TestGetBracketPrices:
    def test_long_bracket(self):
        engine = ExitEngine.from_list([
            {"type": "static_target", "atr_multiple": 2.0},
            {"type": "static_stop", "atr_multiple": 1.0},
        ])
        target, stop = engine.get_bracket_prices(
            fill_price=5600.0, direction="LONG",
            entry_snapshot={"atr": 5.0},
        )
        assert target == 5610.0  # 5600 + 5*2
        assert stop == 5595.0    # 5600 - 5*1

    def test_short_bracket(self):
        engine = ExitEngine.from_list([
            {"type": "static_target", "atr_multiple": 2.0},
            {"type": "static_stop", "atr_multiple": 1.0},
        ])
        target, stop = engine.get_bracket_prices(
            fill_price=5600.0, direction="SHORT",
            entry_snapshot={"atr": 5.0},
        )
        assert target == 5590.0  # 5600 - 5*2
        assert stop == 5605.0    # 5600 + 5*1

    def test_no_atr_returns_zeros(self):
        engine = ExitEngine.from_list([
            {"type": "static_target", "atr_multiple": 2.0},
        ])
        target, stop = engine.get_bracket_prices(
            fill_price=5600.0, direction="LONG",
            entry_snapshot={"atr": 0.0},
        )
        assert target == 0.0
        assert stop == 0.0
