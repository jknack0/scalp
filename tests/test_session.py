"""Tests for session management."""

from datetime import datetime, time, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src.core.session import SessionManager


ET = ZoneInfo("America/New_York")


def _make_session() -> SessionManager:
    return SessionManager(session_start=time(9, 30), session_end=time(16, 0))


def _mock_now(session: SessionManager, dt: datetime):
    """Patch session.now() to return a fixed datetime."""
    return patch.object(type(session), "now", return_value=dt)


def test_is_rth_during_trading_hours():
    """is_rth() returns True during 9:30-16:00 ET on a weekday."""
    session = _make_session()
    # Monday at 10:00 AM ET
    monday_10am = datetime(2026, 3, 2, 10, 0, tzinfo=ET)
    with _mock_now(session, monday_10am):
        assert session.is_rth() is True


def test_is_rth_at_open():
    """is_rth() returns True at exactly 9:30."""
    session = _make_session()
    monday_930 = datetime(2026, 3, 2, 9, 30, tzinfo=ET)
    with _mock_now(session, monday_930):
        assert session.is_rth() is True


def test_is_rth_before_open():
    """is_rth() returns False before 9:30 ET."""
    session = _make_session()
    monday_9am = datetime(2026, 3, 2, 9, 0, tzinfo=ET)
    with _mock_now(session, monday_9am):
        assert session.is_rth() is False


def test_is_rth_at_close():
    """is_rth() returns False at exactly 16:00 (close is exclusive)."""
    session = _make_session()
    monday_4pm = datetime(2026, 3, 2, 16, 0, tzinfo=ET)
    with _mock_now(session, monday_4pm):
        assert session.is_rth() is False


def test_is_rth_weekend():
    """is_rth() returns False on weekends."""
    session = _make_session()
    saturday = datetime(2026, 3, 7, 12, 0, tzinfo=ET)
    with _mock_now(session, saturday):
        assert session.is_rth() is False


def test_time_to_open_during_rth():
    """time_to_open() returns zero when in session."""
    session = _make_session()
    monday_10am = datetime(2026, 3, 2, 10, 0, tzinfo=ET)
    with _mock_now(session, monday_10am):
        assert session.time_to_open() == timedelta(0)


def test_time_to_open_before_session():
    """time_to_open() returns time until 9:30 on same day."""
    session = _make_session()
    monday_8am = datetime(2026, 3, 2, 8, 0, tzinfo=ET)
    with _mock_now(session, monday_8am):
        tto = session.time_to_open()
        assert tto == timedelta(hours=1, minutes=30)


def test_seconds_in_session():
    """seconds_in_session() returns elapsed seconds since open."""
    session = _make_session()
    monday_10am = datetime(2026, 3, 2, 10, 0, tzinfo=ET)
    with _mock_now(session, monday_10am):
        assert session.seconds_in_session() == 1800.0  # 30 minutes


def test_seconds_in_session_outside_rth():
    """seconds_in_session() returns 0 outside RTH."""
    session = _make_session()
    monday_8am = datetime(2026, 3, 2, 8, 0, tzinfo=ET)
    with _mock_now(session, monday_8am):
        assert session.seconds_in_session() == 0.0
