"""Shared pytest fixtures for the test suite."""

import pytest

from src.core.events import EventBus
from src.risk.risk_manager import RiskManager


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus(maxsize=100)


@pytest.fixture
def risk_manager() -> RiskManager:
    return RiskManager(
        max_daily_loss_usd=100.0,
        max_position_contracts=1,
        max_signals_per_day=5,
    )
