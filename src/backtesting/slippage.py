"""Volatility-conditioned slippage model for backtest fill simulation.

Slippage varies by market conditions:
- Calm markets (low ATR): 1 tick
- Active markets (high ATR): 2 ticks
- Event days (FOMC/NFP/CPI): 3 ticks

Only applied to market-order exits (stops, expiry, session close).
Limit orders (entries, targets) fill at specified price — no slippage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.analysis.intraday_profile import EventDay, get_event_days


@dataclass(frozen=True)
class SlippageResult:
    """Slippage estimate for a single fill."""

    ticks: float
    reason: str  # "calm", "active", "event_day"


class VolatilitySlippageModel:
    """ATR + event-day conditioned slippage model.

    Tiers:
        event_day  → event_ticks (default 3)
        ATR >= 75th → active_ticks (default 2)
        otherwise  → calm_ticks (default 1)
    """

    def __init__(
        self,
        atr_75th_percentile: float = 2.0,
        calm_ticks: float = 1.0,
        active_ticks: float = 2.0,
        event_ticks: float = 3.0,
    ) -> None:
        self.atr_75th_percentile = atr_75th_percentile
        self.calm_ticks = calm_ticks
        self.active_ticks = active_ticks
        self.event_ticks = event_ticks
        self._event_dates: set[date] = set()

    def load_event_days(self, years: list[int]) -> None:
        """Load FOMC/NFP/CPI dates from intraday_profile for given years."""
        for year in years:
            for event_day in get_event_days(year):
                self._event_dates.add(event_day.date)

    def add_event_date(self, d: date) -> None:
        """Manually inject an event date."""
        self._event_dates.add(d)

    def compute_slippage(
        self, bar_date: date, current_atr_ticks: float
    ) -> SlippageResult:
        """Compute slippage ticks based on date and current ATR.

        Args:
            bar_date: Date of the current bar.
            current_atr_ticks: Current ATR in tick units.

        Returns:
            SlippageResult with tick count and reason.
        """
        if bar_date in self._event_dates:
            return SlippageResult(ticks=self.event_ticks, reason="event_day")

        if current_atr_ticks >= self.atr_75th_percentile:
            return SlippageResult(ticks=self.active_ticks, reason="active")

        return SlippageResult(ticks=self.calm_ticks, reason="calm")
