"""Tests for Volatility Regime Switcher strategy.

7 tests covering regime classification, pullback signals, low-vol VWAP fade,
transition cooldown, and direction from dominant semi-variance.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.core.events import BarEvent
from src.features.feature_hub import FeatureHub
from src.strategies.base import TICK_SIZE, Direction
from src.strategies.vol_regime_strategy import (
    VolRegime,
    VolRegimeConfig,
    VolRegimeStrategy,
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
    hour: int,
    minute: int,
    second: int,
    close: float,
    volume: int = 500,
    high: float | None = None,
    low: float | None = None,
) -> BarEvent:
    """Create a BarEvent at a given ET time."""
    return BarEvent(
        symbol="MESM6",
        open=close - 0.25,
        high=high if high is not None else close + 0.50,
        low=low if low is not None else close - 0.50,
        close=close,
        volume=volume,
        bar_type="1s",
        timestamp_ns=_et_to_ns(hour, minute, second),
    )


def _make_strategy(**config_overrides) -> VolRegimeStrategy:
    """Create VolRegimeStrategy with FeatureHub and optional config overrides."""
    defaults = dict(
        high_vol_hmm_states=[],  # disable HMM cross-validation
        low_vol_hmm_states=[],
        require_hmm_states=[],
        min_confidence=0.0,
    )
    defaults.update(config_overrides)
    config = VolRegimeConfig(**defaults)
    hub = FeatureHub()
    return VolRegimeStrategy(config, hub)


def _warm_up(
    strat: VolRegimeStrategy,
    n_bars: int,
    base_price: float,
    volume: int = 500,
    start_minute: int = 30,
    high_range: float = 1.0,
) -> None:
    """Feed n bars starting at 9:{start_minute} ET to warm up features.

    Alternates price +-1 around base to build nonzero SD and ATR.
    """
    for i in range(n_bars):
        second = i % 60
        minute = start_minute + i // 60
        sign = 1 if i % 2 == 0 else -1
        price = base_price + sign * 1.0
        bar = BarEvent(
            symbol="MESM6",
            open=price - 0.25,
            high=price + high_range,
            low=price - high_range,
            close=price,
            volume=volume,
            bar_type="1s",
            timestamp_ns=_et_to_ns(9, minute, second),
        )
        strat.on_bar(bar)


# ── Test 1: Regime classified HIGH_VOL ──────────────────────────────


class TestRegimeClassifiedHighVol:
    def test_regime_classified_high_vol(self):
        """Feed bars that push ATR above 75th percentile -> HIGH_VOL."""
        strat = _make_strategy()

        # Build a baseline of low ATR bars
        for i in range(50):
            second = i % 60
            minute = 30 + i // 60
            bar = BarEvent(
                symbol="MESM6",
                open=5000.0,
                high=5000.50,
                low=4999.50,
                close=5000.0,
                volume=500,
                bar_type="1s",
                timestamp_ns=_et_to_ns(9, minute, second),
            )
            strat.on_bar(bar)

        # Now feed bars with much wider range to spike ATR above 75th
        for i in range(30):
            second = i % 60
            minute = 31 + i // 60
            price = 5000.0 + (i % 2) * 10  # big swings
            bar = BarEvent(
                symbol="MESM6",
                open=price - 5.0,
                high=price + 5.0,
                low=price - 5.0,
                close=price,
                volume=500,
                bar_type="1s",
                timestamp_ns=_et_to_ns(10, minute, second),
            )
            strat.on_bar(bar)

        assert strat._regime == VolRegime.HIGH_VOL


# ── Test 2: Regime classified LOW_VOL ───────────────────────────────


class TestRegimeClassifiedLowVol:
    def test_regime_classified_low_vol(self):
        """Feed bars with low ATR percentile -> LOW_VOL."""
        strat = _make_strategy()

        # Build a baseline with moderate-high ATR bars
        for i in range(50):
            second = i % 60
            minute = 30 + i // 60
            price = 5000.0 + (i % 2) * 5.0  # moderate swings
            bar = BarEvent(
                symbol="MESM6",
                open=price - 2.5,
                high=price + 2.5,
                low=price - 2.5,
                close=price,
                volume=500,
                bar_type="1s",
                timestamp_ns=_et_to_ns(9, minute, second),
            )
            strat.on_bar(bar)

        # Now feed tight-range bars to push ATR below 25th percentile
        for i in range(30):
            second = i % 60
            minute = 31 + i // 60
            bar = BarEvent(
                symbol="MESM6",
                open=5000.0,
                high=5000.10,
                low=4999.90,
                close=5000.0,
                volume=500,
                bar_type="1s",
                timestamp_ns=_et_to_ns(10, minute, second),
            )
            strat.on_bar(bar)

        assert strat._regime == VolRegime.LOW_VOL


# ── Test 3: TRANSITIONING blocks signals ────────────────────────────


class TestTransitioningBlocksSignals:
    def test_transitioning_blocks_signals(self):
        """When regime is TRANSITIONING, no signal even with valid conditions."""
        strat = _make_strategy(transition_cooldown_bars=0)
        _warm_up(strat, 25, 5000.0)

        # Force TRANSITIONING
        strat._regime = VolRegime.TRANSITIONING
        strat._bars_since_transition = 100  # past cooldown

        # Feed bar with conditions that would trigger both modes
        bar = _make_bar(10, 5, 0, 4990.0)
        strat._base_on_bar(bar)
        sig = strat.generate_signal(bar)
        assert sig is None


# ── Test 4: High-vol signal requires pullback ───────────────────────


class TestHighVolSignalRequiresPullback:
    def test_high_vol_signal_requires_pullback(self):
        """In HIGH_VOL, no signal until pullback_bars consecutive bars against
        dominant direction, then signal fires."""
        strat = _make_strategy(
            pullback_bars=3,
            transition_cooldown_bars=0,
        )
        _warm_up(strat, 25, 5000.0)

        # Force HIGH_VOL with UP dominant direction
        strat._regime = VolRegime.HIGH_VOL
        strat._bars_since_transition = 100
        strat._pullback_count = 0

        # Inject semi-variance to make dominant direction UP
        # Need values with high variance for up and low variance for down
        strat.feature_hub.atr._up_trs.clear()
        strat.feature_hub.atr._down_trs.clear()
        for i in range(20):
            strat.feature_hub.atr._up_trs.append(1.0 + i * 0.5)  # high spread
        for _ in range(20):
            strat.feature_hub.atr._down_trs.append(1.0)  # zero spread
        assert strat.feature_hub.atr.dominant_direction == "UP"

        # Feed 2 pullback bars (close < prev_close) — not enough for signal
        strat._prev_close = 5002.0
        bar1 = _make_bar(10, 10, 0, 5001.0)
        strat._base_on_bar(bar1)
        strat._regime = VolRegime.HIGH_VOL  # re-force after _update_regime
        strat._bars_since_transition = 100
        sig1 = strat.generate_signal(bar1)
        strat._pullback_count = 1  # manually track since we're not calling on_bar
        strat._prev_close = bar1.close

        bar2 = _make_bar(10, 10, 1, 5000.0)
        strat._base_on_bar(bar2)
        strat._regime = VolRegime.HIGH_VOL
        strat._bars_since_transition = 100
        sig2 = strat.generate_signal(bar2)
        strat._pullback_count = 2
        strat._prev_close = bar2.close

        assert sig1 is None
        assert sig2 is None

        # 3rd pullback bar — should trigger signal
        bar3 = _make_bar(10, 10, 2, 4999.0)
        strat._base_on_bar(bar3)
        strat._regime = VolRegime.HIGH_VOL
        strat._bars_since_transition = 100
        strat._pullback_count = 3
        # Re-inject semi-variance (on_bar may have added to deques)
        strat.feature_hub.atr._up_trs.clear()
        strat.feature_hub.atr._down_trs.clear()
        for i in range(20):
            strat.feature_hub.atr._up_trs.append(1.0 + i * 0.5)
        for _ in range(20):
            strat.feature_hub.atr._down_trs.append(1.0)
        sig3 = strat.generate_signal(bar3)

        assert sig3 is not None
        assert sig3.direction == Direction.LONG  # dominant is UP


# ── Test 5: Low-vol signal at VWAP extension ────────────────────────


class TestLowVolSignalAtVwapExtension:
    def test_low_vol_signal_at_vwap_extension(self):
        """In LOW_VOL, signal when price reaches 1.5 SD from VWAP; no signal near VWAP."""
        strat = _make_strategy(
            low_vol_entry_sd=1.5,
            transition_cooldown_bars=0,
        )
        _warm_up(strat, 25, 5000.0)

        # Force LOW_VOL
        strat._regime = VolRegime.LOW_VOL
        strat._bars_since_transition = 100

        vwap_val = strat.feature_hub.vwap.vwap
        sd = strat.feature_hub.vwap._sd
        assert sd > 0, "SD should be non-zero after warmup"

        # Bar near VWAP → no signal
        near_bar = _make_bar(10, 5, 0, vwap_val + 0.1 * sd)
        strat._base_on_bar(near_bar)
        strat._regime = VolRegime.LOW_VOL
        strat._bars_since_transition = 100
        sig_near = strat.generate_signal(near_bar)
        assert sig_near is None

        # Bar at 2.0 SD above VWAP → SHORT signal
        far_price = vwap_val + 2.0 * sd
        far_bar = _make_bar(10, 6, 0, far_price)
        strat._base_on_bar(far_bar)
        strat._regime = VolRegime.LOW_VOL
        strat._bars_since_transition = 100
        sig_far = strat.generate_signal(far_bar)
        assert sig_far is not None
        assert sig_far.direction == Direction.SHORT


# ── Test 6: Transition cooldown ─────────────────────────────────────


class TestTransitionCooldown:
    def test_transition_cooldown(self):
        """After regime change, no signal for transition_cooldown_bars, then resume."""
        strat = _make_strategy(
            transition_cooldown_bars=2,
            low_vol_entry_sd=0.1,  # very low threshold so signal would fire
        )
        _warm_up(strat, 25, 5000.0)

        # Force LOW_VOL with fresh transition (bars_since = 0)
        strat._regime = VolRegime.LOW_VOL
        strat._bars_since_transition = 0

        vwap_val = strat.feature_hub.vwap.vwap
        sd = strat.feature_hub.vwap._sd

        # Bar 1 after transition: cooldown active (1 <= 2)
        strat._bars_since_transition = 1
        bar1 = _make_bar(10, 10, 0, vwap_val + 2.0 * max(sd, 1.0))
        strat._base_on_bar(bar1)
        strat._regime = VolRegime.LOW_VOL
        strat._bars_since_transition = 1
        sig1 = strat.generate_signal(bar1)
        assert sig1 is None

        # Bar 2: still in cooldown (2 <= 2)
        strat._bars_since_transition = 2
        bar2 = _make_bar(10, 10, 1, vwap_val + 2.0 * max(sd, 1.0))
        strat._base_on_bar(bar2)
        strat._regime = VolRegime.LOW_VOL
        strat._bars_since_transition = 2
        sig2 = strat.generate_signal(bar2)
        assert sig2 is None

        # Bar 3: cooldown over (3 > 2)
        strat._bars_since_transition = 3
        bar3 = _make_bar(10, 10, 2, vwap_val + 2.0 * max(sd, 1.0))
        strat._base_on_bar(bar3)
        strat._regime = VolRegime.LOW_VOL
        strat._bars_since_transition = 3
        sig3 = strat.generate_signal(bar3)
        assert sig3 is not None


# ── Test 7: Direction follows dominant semi-variance ────────────────


class TestDirectionFollowsDominant:
    def test_direction_follows_dominant_semi_variance(self):
        """In HIGH_VOL, direction matches dominant_direction: UP→LONG, DOWN→SHORT."""
        # ── UP → LONG ──
        strat_up = _make_strategy(pullback_bars=1, transition_cooldown_bars=0)
        _warm_up(strat_up, 25, 5000.0)

        strat_up._regime = VolRegime.HIGH_VOL
        strat_up._bars_since_transition = 100

        # Force dominant UP (need high variance for up, zero for down)
        strat_up.feature_hub.atr._up_trs.clear()
        strat_up.feature_hub.atr._down_trs.clear()
        for i in range(20):
            strat_up.feature_hub.atr._up_trs.append(1.0 + i * 0.5)
        for _ in range(20):
            strat_up.feature_hub.atr._down_trs.append(1.0)
        assert strat_up.feature_hub.atr.dominant_direction == "UP"

        strat_up._prev_close = 5002.0
        strat_up._pullback_count = 1  # meets pullback_bars=1

        bar_up = _make_bar(10, 15, 0, 5001.0)
        strat_up._base_on_bar(bar_up)
        strat_up._regime = VolRegime.HIGH_VOL
        strat_up._bars_since_transition = 100
        # Re-inject after on_bar may modify deques
        strat_up.feature_hub.atr._up_trs.clear()
        strat_up.feature_hub.atr._down_trs.clear()
        for i in range(20):
            strat_up.feature_hub.atr._up_trs.append(1.0 + i * 0.5)
        for _ in range(20):
            strat_up.feature_hub.atr._down_trs.append(1.0)
        sig_up = strat_up.generate_signal(bar_up)
        assert sig_up is not None
        assert sig_up.direction == Direction.LONG

        # ── DOWN → SHORT ──
        strat_down = _make_strategy(pullback_bars=1, transition_cooldown_bars=0)
        _warm_up(strat_down, 25, 5000.0)

        strat_down._regime = VolRegime.HIGH_VOL
        strat_down._bars_since_transition = 100

        # Force dominant DOWN (need high variance for down, zero for up)
        strat_down.feature_hub.atr._up_trs.clear()
        strat_down.feature_hub.atr._down_trs.clear()
        for _ in range(20):
            strat_down.feature_hub.atr._up_trs.append(1.0)
        for i in range(20):
            strat_down.feature_hub.atr._down_trs.append(1.0 + i * 0.5)
        assert strat_down.feature_hub.atr.dominant_direction == "DOWN"

        strat_down._prev_close = 4998.0
        strat_down._pullback_count = 1

        bar_down = _make_bar(10, 15, 0, 4999.0)
        strat_down._base_on_bar(bar_down)
        strat_down._regime = VolRegime.HIGH_VOL
        strat_down._bars_since_transition = 100
        # Re-inject after on_bar
        strat_down.feature_hub.atr._up_trs.clear()
        strat_down.feature_hub.atr._down_trs.clear()
        for _ in range(20):
            strat_down.feature_hub.atr._up_trs.append(1.0)
        for i in range(20):
            strat_down.feature_hub.atr._down_trs.append(1.0 + i * 0.5)
        sig_down = strat_down.generate_signal(bar_down)
        assert sig_down is not None
        assert sig_down.direction == Direction.SHORT
