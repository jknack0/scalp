"""Pre-trade risk manager for MES scalping."""

import time as time_mod
from dataclasses import dataclass

from src.core.events import FillEvent, SignalEvent
from src.core.logging import get_logger

logger = get_logger("risk")


@dataclass
class RiskCheckResult:
    """Result of a pre-trade risk check."""

    approved: bool
    reason: str


class RiskManager:
    """Validates signals against risk limits before order submission.

    Checks (in order):
    1. Is trading halted? (daily loss limit hit)
    2. Is session valid? (within RTH)
    3. Position limit check (max contracts)
    4. Daily signal count limit
    """

    def __init__(
        self,
        max_daily_loss_usd: float = 150.0,
        max_position_contracts: int = 1,
        max_signals_per_day: int = 10,
        tick_value: float = 1.25,
    ) -> None:
        self._max_daily_loss = max_daily_loss_usd
        self._max_position = max_position_contracts
        self._max_signals = max_signals_per_day
        self._tick_value = tick_value

        self._daily_pnl: float = 0.0
        self._current_position: int = 0  # +1 long, -1 short, 0 flat
        self._signal_count: int = 0
        self._is_halted: bool = False
        self._halt_reason: str = ""
        self._avg_entry_price: float = 0.0

    def check_order(
        self,
        signal: SignalEvent,
        current_position: int,
        session_valid: bool,
    ) -> RiskCheckResult:
        """Validate a signal against all risk checks."""
        if self._is_halted:
            return RiskCheckResult(False, f"halted: {self._halt_reason}")

        if not session_valid:
            return RiskCheckResult(False, "outside RTH session")

        # Position limit: block if adding to position at max
        would_increase = (
            (signal.direction == "BUY" and current_position >= self._max_position)
            or (signal.direction == "SELL" and current_position <= -self._max_position)
        )
        if would_increase:
            return RiskCheckResult(
                False, f"position limit ({current_position}/{self._max_position})"
            )

        if self._signal_count >= self._max_signals:
            return RiskCheckResult(
                False, f"signal limit ({self._signal_count}/{self._max_signals})"
            )

        self._signal_count += 1
        logger.info(
            "order_approved",
            direction=signal.direction,
            strategy=signal.strategy_id,
            signal_count=self._signal_count,
        )
        return RiskCheckResult(True, "all checks passed")

    def record_fill(self, fill: FillEvent) -> None:
        """Update position and P&L tracking after a fill."""
        prev_position = self._current_position
        size = fill.fill_size if fill.direction == "BUY" else -fill.fill_size

        # Compute realized P&L if closing/reducing a position
        if prev_position != 0 and (
            (prev_position > 0 and fill.direction == "SELL")
            or (prev_position < 0 and fill.direction == "BUY")
        ):
            closed_qty = min(abs(prev_position), fill.fill_size)
            if prev_position > 0:
                pnl = (fill.fill_price - self._avg_entry_price) * closed_qty
            else:
                pnl = (self._avg_entry_price - fill.fill_price) * closed_qty
            pnl -= fill.commission
            self._daily_pnl += pnl
            logger.info(
                "fill_recorded",
                direction=fill.direction,
                price=fill.fill_price,
                realized_pnl=round(pnl, 2),
                daily_pnl=round(self._daily_pnl, 2),
            )
        else:
            # Opening or adding — just deduct commission
            self._daily_pnl -= fill.commission
            logger.info(
                "fill_recorded",
                direction=fill.direction,
                price=fill.fill_price,
                commission=fill.commission,
            )

        # Update position
        self._current_position = prev_position + size

        # Update average entry price
        if self._current_position == 0:
            self._avg_entry_price = 0.0
        elif prev_position == 0 or (prev_position > 0) != (self._current_position > 0):
            # New position or flipped side
            self._avg_entry_price = fill.fill_price
        else:
            # Adding to existing position — weighted average
            total = abs(prev_position) + fill.fill_size
            self._avg_entry_price = (
                self._avg_entry_price * abs(prev_position)
                + fill.fill_price * fill.fill_size
            ) / total

        # Check daily loss limit
        if self._daily_pnl <= -self._max_daily_loss:
            self.halt(f"daily loss limit hit (${self._daily_pnl:.2f})")

    def reset_daily(self) -> None:
        """Reset daily counters. Called at session open."""
        self._daily_pnl = 0.0
        self._signal_count = 0
        self._is_halted = False
        self._halt_reason = ""
        logger.info("daily_reset")

    def halt(self, reason: str) -> None:
        """Halt all trading for the rest of the session."""
        self._is_halted = True
        self._halt_reason = reason
        logger.warning("trading_halted", reason=reason, daily_pnl=self._daily_pnl)

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def is_halted(self) -> bool:
        return self._is_halted

    @property
    def current_position(self) -> int:
        return self._current_position
