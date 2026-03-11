"""Tests for the bar-replay backtest engine.

Uses a MockStrategy that generates signals on demand — no Parquet data needed.
Tests cover: fill mechanics, slippage, commission, equity tracking, and
no-lookahead-bias guarantee.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from src.analysis.commission_model import CostModel
from src.backtesting.engine import SimulatedOMS, TICK_SIZE, TICK_VALUE, POINT_VALUE
from src.backtesting.metrics import MetricsCalculator, Trade
from src.backtesting.slippage import VolatilitySlippageModel
from src.core.events import BarEvent
from src.strategies.base import Direction, Signal, StrategyBase, StrategyConfig

from zoneinfo import ZoneInfo

_ET = ZoneInfo("US/Eastern")


# ── Helpers ───────────────────────────────────────────────────────────


def _make_bar(
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 100,
    ts_ns: int = 0,
) -> BarEvent:
    return BarEvent(
        symbol="MESM6",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bar_type="1s",
        timestamp_ns=ts_ns,
    )


def _make_signal(
    direction: Direction = Direction.LONG,
    entry: float = 5000.0,
    target: float = 5002.0,
    stop: float = 4998.0,
    expiry_seconds: int = 300,
    strategy_id: str = "mock",
) -> Signal:
    from src.models.hmm_regime import RegimeState

    now = datetime(2025, 6, 2, 10, 30, tzinfo=_ET)
    return Signal(
        strategy_id=strategy_id,
        direction=direction,
        entry_price=entry,
        target_price=target,
        stop_price=stop,
        signal_time=now,
        expiry_time=now + timedelta(seconds=expiry_seconds),
        confidence=0.8,
        regime_state=RegimeState.RANGE_BOUND,
    )


def _make_oms(
    calm_ticks: float = 1.0,
    active_ticks: float = 2.0,
    event_ticks: float = 3.0,
    atr_75th: float = 2.0,
    commission_per_side: float = 0.35,
) -> SimulatedOMS:
    cost = CostModel(
        broker_name="Test",
        broker_commission_per_side=commission_per_side,
    )
    slippage = VolatilitySlippageModel(
        atr_75th_percentile=atr_75th,
        calm_ticks=calm_ticks,
        active_ticks=active_ticks,
        event_ticks=event_ticks,
    )
    return SimulatedOMS(commission_model=cost, slippage_model=slippage)


def _bar_time(minute_offset: int = 0) -> datetime:
    return datetime(2025, 6, 2, 10, 30, tzinfo=_ET) + timedelta(minutes=minute_offset)


def _bar_date() -> date:
    return date(2025, 6, 2)


# ── Test 1: Long fills and targets ────────────────────────────────────


class TestLongFillsAndTargets:
    def test_long_fills_and_targets(self):
        """3-bar sequence: signal bar 0, entry fill bar 1, target hit bar 2.

        LONG signal: entry=5000, target=5002, stop=4998.
        Bar 1 low touches 5000 → fill at 5000 (limit).
        Bar 2 high reaches 5002 → target exit at 5002.
        gross_pnl = (5002 - 5000) * 5.0 = 10.0
        """
        oms = _make_oms()
        signal = _make_signal(
            direction=Direction.LONG,
            entry=5000.0,
            target=5002.0,
            stop=4998.0,
        )

        # Bar 0: signal generated
        oms.on_signal(signal, bar_index=0)
        bar0 = _make_bar(5001.0, 5001.5, 5000.5, 5001.0)
        trades = oms.on_bar(bar0, 0, _bar_time(0), _bar_date(), 1.0)
        assert len(trades) == 0  # No fill on signal bar

        # Bar 1: price touches entry → fill
        bar1 = _make_bar(5001.0, 5001.5, 4999.5, 5000.5)
        trades = oms.on_bar(bar1, 1, _bar_time(1), _bar_date(), 1.0)
        assert len(trades) == 0  # Filled but no exit yet
        assert oms.open_position_count == 1

        # Bar 2: price reaches target → exit
        bar2 = _make_bar(5001.0, 5002.5, 5000.5, 5002.0)
        trades = oms.on_bar(bar2, 2, _bar_time(2), _bar_date(), 1.0)
        assert len(trades) == 1

        trade = trades[0]
        assert trade.direction == Direction.LONG
        assert trade.exit_reason == "target"
        assert trade.entry_price == 5000.0
        assert trade.exit_price == 5002.0
        assert trade.gross_pnl == pytest.approx(10.0)  # (5002-5000)*5
        assert trade.net_pnl > 0  # gross - commission > 0
        assert trade.entry_slippage_ticks == 0.0  # Limit entry
        assert trade.exit_slippage_ticks == 0.0  # Limit target exit


# ── Test 2: Stop loss fills with slippage ──────────────────────────────


class TestStopLossWithSlippage:
    def test_stop_loss_fills_with_slippage(self):
        """LONG stop hit: fill price is stop - slippage * tick_size.

        LONG: entry=5000, stop=4998. Calm slippage=1 tick.
        Stop bar low <= 4998 → exit at 4998 - 1*0.25 = 4997.75.
        """
        oms = _make_oms(calm_ticks=1.0)
        signal = _make_signal(
            direction=Direction.LONG,
            entry=5000.0,
            target=5004.0,
            stop=4998.0,
        )

        # Signal + entry fill
        oms.on_signal(signal, bar_index=0)
        bar1 = _make_bar(5001.0, 5001.5, 4999.5, 5000.5)
        oms.on_bar(bar1, 1, _bar_time(1), _bar_date(), 1.0)
        assert oms.open_position_count == 1

        # Stop hit
        bar2 = _make_bar(4999.0, 4999.5, 4997.5, 4998.0)
        trades = oms.on_bar(bar2, 2, _bar_time(2), _bar_date(), 1.0)
        assert len(trades) == 1

        trade = trades[0]
        assert trade.exit_reason == "stop"
        # Adverse slippage: stop at 4998, slipped to 4997.75
        assert trade.exit_price == pytest.approx(4998.0 - 1.0 * TICK_SIZE)
        assert trade.exit_slippage_ticks == pytest.approx(1.0)

        # SHORT stop: slippage goes the other way
        oms2 = _make_oms(calm_ticks=1.0)
        short_signal = _make_signal(
            direction=Direction.SHORT,
            entry=5000.0,
            target=4996.0,
            stop=5002.0,
        )
        oms2.on_signal(short_signal, bar_index=0)
        bar_fill = _make_bar(4999.0, 5000.5, 4998.5, 4999.5)
        oms2.on_bar(bar_fill, 1, _bar_time(1), _bar_date(), 1.0)

        bar_stop = _make_bar(5001.0, 5002.5, 5000.5, 5002.0)
        trades = oms2.on_bar(bar_stop, 2, _bar_time(2), _bar_date(), 1.0)
        assert len(trades) == 1
        assert trades[0].exit_price == pytest.approx(5002.0 + 1.0 * TICK_SIZE)


# ── Test 3: Commission deducted ──────────────────────────────────────


class TestCommissionDeducted:
    def test_commission_deducted(self):
        """Known commission $0.35/side = $0.70 round-trip.

        net_pnl = gross_pnl - commission - slippage_cost
        """
        oms = _make_oms(commission_per_side=0.35, calm_ticks=0.0)
        signal = _make_signal(
            direction=Direction.LONG,
            entry=5000.0,
            target=5002.0,
            stop=4998.0,
        )

        oms.on_signal(signal, bar_index=0)
        # Entry fill
        bar1 = _make_bar(5001.0, 5001.0, 4999.5, 5000.5)
        oms.on_bar(bar1, 1, _bar_time(1), _bar_date(), 0.5)

        # Target hit
        bar2 = _make_bar(5001.0, 5002.5, 5000.5, 5002.0)
        trades = oms.on_bar(bar2, 2, _bar_time(2), _bar_date(), 0.5)
        assert len(trades) == 1

        trade = trades[0]
        assert trade.commission == pytest.approx(0.70)
        gross = (5002.0 - 5000.0) * POINT_VALUE  # 10.0
        assert trade.gross_pnl == pytest.approx(gross)
        # No slippage (calm_ticks=0, target is limit)
        assert trade.slippage_cost == pytest.approx(0.0)
        assert trade.net_pnl == pytest.approx(gross - 0.70)


# ── Test 4: Volatility-conditioned slippage ──────────────────────────


class TestVolatilityConditionedSlippage:
    def test_volatility_conditioned_slippage(self):
        """Active (high ATR) → 2 ticks; event day → 3 ticks."""
        oms_active = _make_oms(
            calm_ticks=1.0, active_ticks=2.0, event_ticks=3.0, atr_75th=2.0
        )
        signal = _make_signal(
            direction=Direction.LONG,
            entry=5000.0,
            target=5004.0,
            stop=4998.0,
        )

        # Fill the entry
        oms_active.on_signal(signal, bar_index=0)
        bar1 = _make_bar(5001.0, 5001.0, 4999.5, 5000.5)
        oms_active.on_bar(bar1, 1, _bar_time(1), _bar_date(), 1.0)

        # High ATR = 3.0 >= threshold 2.0 → active → 2 ticks slippage
        bar_stop = _make_bar(4999.0, 4999.0, 4997.5, 4998.0)
        trades = oms_active.on_bar(bar_stop, 2, _bar_time(2), _bar_date(), 3.0)
        assert len(trades) == 1
        assert trades[0].exit_slippage_ticks == pytest.approx(2.0)
        assert trades[0].exit_price == pytest.approx(4998.0 - 2.0 * TICK_SIZE)

        # Event day: inject the date as event day
        oms_event = _make_oms(
            calm_ticks=1.0, active_ticks=2.0, event_ticks=3.0, atr_75th=2.0
        )
        oms_event._slippage_model.add_event_date(_bar_date())

        signal2 = _make_signal(
            direction=Direction.LONG,
            entry=5000.0,
            target=5004.0,
            stop=4998.0,
        )
        oms_event.on_signal(signal2, bar_index=0)
        bar1 = _make_bar(5001.0, 5001.0, 4999.5, 5000.5)
        oms_event.on_bar(bar1, 1, _bar_time(1), _bar_date(), 1.0)

        bar_stop2 = _make_bar(4999.0, 4999.0, 4997.5, 4998.0)
        trades = oms_event.on_bar(bar_stop2, 2, _bar_time(2), _bar_date(), 1.0)
        assert len(trades) == 1
        assert trades[0].exit_slippage_ticks == pytest.approx(3.0)
        assert trades[0].exit_price == pytest.approx(4998.0 - 3.0 * TICK_SIZE)


# ── Test 5: Equity curve tracks trades ────────────────────────────────


class TestEquityCurveTracksTrades:
    def test_equity_curve_tracks_trades(self):
        """Equity starts at initial_capital, adjusts with each trade's net_pnl."""
        from src.models.hmm_regime import RegimeState

        initial_capital = 10_000.0

        # Simulate 2 trades: one win, one loss
        now = datetime(2025, 6, 2, 10, 30, tzinfo=_ET)
        trade_win = Trade(
            trade_id="t1",
            strategy_id="mock",
            direction=Direction.LONG,
            entry_price=5000.0,
            exit_price=5002.0,
            entry_time=now,
            exit_time=now + timedelta(minutes=5),
            size=1,
            gross_pnl=10.0,
            slippage_cost=0.0,
            commission=0.70,
            net_pnl=9.30,
            exit_reason="target",
            bars_held=300,
            entry_slippage_ticks=0.0,
            exit_slippage_ticks=0.0,
        )
        trade_loss = Trade(
            trade_id="t2",
            strategy_id="mock",
            direction=Direction.LONG,
            entry_price=5000.0,
            exit_price=4998.0,
            entry_time=now + timedelta(minutes=10),
            exit_time=now + timedelta(minutes=15),
            size=1,
            gross_pnl=-10.0,
            slippage_cost=1.25,
            commission=0.70,
            net_pnl=-11.95,
            exit_reason="stop",
            bars_held=300,
            entry_slippage_ticks=0.0,
            exit_slippage_ticks=1.0,
        )

        metrics, eq_curve, daily_pnl = MetricsCalculator.from_trades(
            [trade_win, trade_loss], initial_capital
        )

        # Equity curve: initial → after win → after loss
        eq_values = eq_curve["equity"].to_list()
        assert eq_values[0] == pytest.approx(initial_capital)
        assert eq_values[1] == pytest.approx(initial_capital + 9.30)
        assert eq_values[2] == pytest.approx(initial_capital + 9.30 - 11.95)

        # Metrics
        assert metrics.total_trades == 2
        assert metrics.win_rate == pytest.approx(0.5)
        assert metrics.net_pnl == pytest.approx(9.30 - 11.95)


# ── Test 6: No lookahead bias ─────────────────────────────────────────


class TestNoLookaheadBias:
    def test_no_lookahead_bias(self):
        """Signal on bar N with entry_price: fill must NOT occur on bar N itself.

        Even if bar N's range includes the entry price, the order is only
        eligible for fill on bar N+1 or later.
        """
        oms = _make_oms()
        signal = _make_signal(
            direction=Direction.LONG,
            entry=5000.0,
            target=5002.0,
            stop=4998.0,
        )

        # Signal on bar 5
        oms.on_signal(signal, bar_index=5)

        # Bar 5: range includes entry (low=4999.5 <= 5000.0) — must NOT fill
        bar5 = _make_bar(5001.0, 5001.5, 4999.5, 5000.5)
        trades = oms.on_bar(bar5, 5, _bar_time(0), _bar_date(), 1.0)
        assert len(trades) == 0
        assert oms.open_position_count == 0
        assert oms.pending_entry_count == 1  # Still pending

        # Bar 6: same range — now it CAN fill
        bar6 = _make_bar(5001.0, 5001.5, 4999.5, 5000.5)
        trades = oms.on_bar(bar6, 6, _bar_time(1), _bar_date(), 1.0)
        assert len(trades) == 0  # Filled but not exited
        assert oms.open_position_count == 1
        assert oms.pending_entry_count == 0
