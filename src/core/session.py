"""Trading session management for MES futures."""

import asyncio
from datetime import datetime, time, timedelta
from typing import Any, Callable, Coroutine
from zoneinfo import ZoneInfo

from src.core.logging import get_logger

logger = get_logger("session")

Callback = Callable[..., Coroutine[Any, Any, None]]


class SessionManager:
    """Manages RTH session boundaries and triggers callbacks.

    MES regular trading hours: 9:30 AM - 4:00 PM Eastern.
    This bot trades RTH only for Phase 1.
    """

    def __init__(
        self,
        session_start: time = time(9, 30),
        session_end: time = time(16, 0),
        timezone: str = "America/New_York",
    ) -> None:
        self._session_start = session_start
        self._session_end = session_end
        self._tz = ZoneInfo(timezone)
        self._open_callbacks: list[Callback] = []
        self._close_callbacks: list[Callback] = []
        self._running = False

    def now(self) -> datetime:
        """Current time in session timezone."""
        return datetime.now(self._tz)

    def is_rth(self) -> bool:
        """Is current time within regular trading hours?

        Returns False on weekends (Sat/Sun).
        """
        current = self.now()
        if current.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        return self._session_start <= current.time() < self._session_end

    def time_to_open(self) -> timedelta:
        """Time until next session open. Returns zero if currently in session."""
        if self.is_rth():
            return timedelta(0)

        now = self.now()
        today_open = now.replace(
            hour=self._session_start.hour,
            minute=self._session_start.minute,
            second=0,
            microsecond=0,
        )

        if now < today_open and now.weekday() < 5:
            # Before open today (weekday)
            return today_open - now

        # After close or weekend — find next weekday
        next_open = today_open + timedelta(days=1)
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
        return next_open - now

    def seconds_in_session(self) -> float:
        """Seconds elapsed since session open. Returns 0.0 if outside RTH."""
        if not self.is_rth():
            return 0.0

        now = self.now()
        session_open = now.replace(
            hour=self._session_start.hour,
            minute=self._session_start.minute,
            second=0,
            microsecond=0,
        )
        return (now - session_open).total_seconds()

    def on_session_open(self, callback: Callback) -> None:
        """Register async callback to fire at session open."""
        self._open_callbacks.append(callback)

    def on_session_close(self, callback: Callback) -> None:
        """Register async callback to fire at session close."""
        self._close_callbacks.append(callback)

    async def _fire_callbacks(self, callbacks: list[Callback], label: str) -> None:
        """Fire all callbacks, logging errors without crashing."""
        for cb in callbacks:
            try:
                await cb()
            except Exception:
                logger.exception("session_callback_error", label=label, callback=cb.__name__)

    async def run(self) -> None:
        """Background task that fires session open/close callbacks.

        Checks every 30 seconds. Fires callbacks once per boundary crossing.
        """
        self._running = True
        was_rth = self.is_rth()
        logger.info("session_manager_started", is_rth=was_rth)

        while self._running:
            await asyncio.sleep(30)
            is_rth_now = self.is_rth()

            if is_rth_now and not was_rth:
                logger.info("session_open")
                await self._fire_callbacks(self._open_callbacks, "session_open")

            if not is_rth_now and was_rth:
                logger.info("session_close")
                await self._fire_callbacks(self._close_callbacks, "session_close")

            was_rth = is_rth_now

    def stop(self) -> None:
        """Signal the run loop to exit."""
        self._running = False
