"""Tests for the risk manager pre-trade checks."""

import time

import pytest

from src.core.events import FillEvent, SignalEvent
from src.risk.risk_manager import RiskManager


def _make_signal(direction: str = "BUY", **kwargs) -> SignalEvent:
    defaults = dict(
        strategy_id="orb", direction=direction,
        strength=0.8, reason="test", timestamp_ns=time.time_ns(),
    )
    defaults.update(kwargs)
    return SignalEvent(**defaults)


def _make_fill(direction: str = "BUY", price: float = 5500.0, **kwargs) -> FillEvent:
    defaults = dict(
        order_id="o1", symbol="MESM6", direction=direction,
        fill_price=price, fill_size=1, commission=0.70,
        timestamp_ns=time.time_ns(),
    )
    defaults.update(kwargs)
    return FillEvent(**defaults)


def test_approved_order_passes_all_checks(risk_manager: RiskManager):
    """Signal passes when all limits are within bounds."""
    result = risk_manager.check_order(_make_signal(), current_position=0, session_valid=True)
    assert result.approved is True
    assert result.reason == "all checks passed"


def test_daily_loss_limit_triggers_halt(risk_manager: RiskManager):
    """Recording fills that breach daily loss limit triggers halt."""
    # Open a long position
    risk_manager.record_fill(_make_fill("BUY", price=5500.0))
    # Close at a big loss (> $100 max daily loss)
    risk_manager.record_fill(_make_fill("SELL", price=5400.0))
    assert risk_manager.is_halted is True


def test_halted_manager_rejects_all_orders(risk_manager: RiskManager):
    """Once halted, all check_order calls return approved=False."""
    risk_manager.halt("test halt")
    result = risk_manager.check_order(_make_signal(), current_position=0, session_valid=True)
    assert result.approved is False
    assert "halted" in result.reason


def test_max_position_blocks_order(risk_manager: RiskManager):
    """Signal rejected when current position is at max."""
    # Already at max long position
    result = risk_manager.check_order(_make_signal("BUY"), current_position=1, session_valid=True)
    assert result.approved is False
    assert "position limit" in result.reason


def test_max_position_allows_reduce(risk_manager: RiskManager):
    """Signal to reduce/close position is allowed even at max."""
    result = risk_manager.check_order(_make_signal("SELL"), current_position=1, session_valid=True)
    assert result.approved is True


def test_signal_count_limit(risk_manager: RiskManager):
    """Signal rejected after max_signals_per_day is reached."""
    # Max is 5 for test fixture
    for _ in range(5):
        result = risk_manager.check_order(_make_signal(), current_position=0, session_valid=True)
        assert result.approved is True

    # 6th should be rejected
    result = risk_manager.check_order(_make_signal(), current_position=0, session_valid=True)
    assert result.approved is False
    assert "signal limit" in result.reason


def test_session_invalid_blocks_order(risk_manager: RiskManager):
    """Signal rejected when session_valid=False."""
    result = risk_manager.check_order(_make_signal(), current_position=0, session_valid=False)
    assert result.approved is False
    assert "outside RTH" in result.reason


def test_halt_resets_on_new_session(risk_manager: RiskManager):
    """reset_daily() clears halt flag and counters."""
    risk_manager.halt("test")
    assert risk_manager.is_halted is True

    risk_manager.reset_daily()
    assert risk_manager.is_halted is False
    assert risk_manager.daily_pnl == 0.0


def test_record_fill_updates_position(risk_manager: RiskManager):
    """record_fill correctly updates current_position tracking."""
    assert risk_manager.current_position == 0

    risk_manager.record_fill(_make_fill("BUY"))
    assert risk_manager.current_position == 1

    risk_manager.record_fill(_make_fill("SELL"))
    assert risk_manager.current_position == 0


def test_record_fill_tracks_pnl(risk_manager: RiskManager):
    """record_fill computes realized P&L on position close."""
    risk_manager.record_fill(_make_fill("BUY", price=5500.0))
    risk_manager.record_fill(_make_fill("SELL", price=5510.0))
    # P&L = (5510 - 5500) * 1 - commission(0.70) - commission(0.70 on open)
    # Open fill deducts commission: -0.70
    # Close fill: pnl = 10.0 - 0.70 = 9.30
    # Total: -0.70 + 9.30 = 8.60
    assert abs(risk_manager.daily_pnl - 8.60) < 0.01
