"""Commission math model for MES futures scalping."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CostModel:
    """Cost model for a single broker/plan combination.

    All per-side values. Tick value fixed at MES = $1.25.
    """

    broker_name: str
    broker_commission_per_side: float
    exchange_fee: float = 0.0        # set to 0 if already included in all-in rate
    nfa_fee: float = 0.0             # set to 0 if already included in all-in rate
    avg_slippage_ticks: float = 1.0
    tick_value: float = 1.25         # MES: 0.25 points × $5/point

    @property
    def commission_per_side(self) -> float:
        return self.broker_commission_per_side + self.exchange_fee + self.nfa_fee

    def round_trip_commission(self) -> float:
        return self.commission_per_side * 2

    def slippage_cost(self) -> float:
        return self.avg_slippage_ticks * self.tick_value * 2  # both sides

    def round_trip_cost(self) -> float:
        return self.round_trip_commission() + self.slippage_cost()

    def gross_win(self, target_ticks: int) -> float:
        return target_ticks * self.tick_value

    def net_win(self, target_ticks: int) -> float:
        return self.gross_win(target_ticks) - self.round_trip_cost()

    def gross_loss(self, stop_ticks: int) -> float:
        return stop_ticks * self.tick_value

    def net_loss(self, stop_ticks: int) -> float:
        return self.gross_loss(stop_ticks) + self.round_trip_cost()

    def breakeven_win_rate(self, target_ticks: int, stop_ticks: int) -> float:
        nw = self.net_win(target_ticks)
        nl = self.net_loss(stop_ticks)
        if nw + nl == 0:
            return 1.0
        if nw <= 0:
            return 1.0  # impossible to profit
        return nl / (nw + nl)

    def profit_expectancy(
        self, target_ticks: int, stop_ticks: int, win_rate: float
    ) -> float:
        return win_rate * self.net_win(target_ticks) - (1 - win_rate) * self.net_loss(
            stop_ticks
        )

    def annual_commission_cost(
        self, trades_per_day: int = 5, trading_days: int = 250
    ) -> float:
        return self.round_trip_commission() * trades_per_day * trading_days

    def min_viable_target(
        self, stop_ticks: int = 8, max_breakeven_wr: float = 0.50
    ) -> int:
        """Smallest target where breakeven WR < max_breakeven_wr."""
        for t in range(1, 101):
            if self.breakeven_win_rate(t, stop_ticks) < max_breakeven_wr:
                return t
        return 100  # unreachable in practice


@dataclass
class BrokerComparison:
    """Compare multiple broker cost models side by side."""

    models: list[CostModel] = field(default_factory=list)

    def add(self, model: CostModel) -> None:
        self.models.append(model)

    def breakeven_matrix(
        self,
        targets: list[int] | None = None,
        stops: list[int] | None = None,
    ) -> dict[str, list[list[float]]]:
        """Breakeven win rate matrix for each broker.

        Returns {broker_name: [[rate for each stop] for each target]}.
        """
        if targets is None:
            targets = [4, 6, 8, 10, 12, 16]
        if stops is None:
            stops = [2, 3, 4, 5, 6, 8]

        result: dict[str, list[list[float]]] = {}
        for m in self.models:
            matrix = []
            for t in targets:
                row = [m.breakeven_win_rate(t, s) for s in stops]
                matrix.append(row)
            result[m.broker_name] = matrix
        return result

    def annual_cost_table(
        self, trades_per_day_options: list[int] | None = None
    ) -> dict[str, list[float]]:
        """Annual commission cost for each broker at different trade frequencies.

        Returns {broker_name: [cost at each frequency]}.
        """
        if trades_per_day_options is None:
            trades_per_day_options = [5, 10, 15, 20]

        result: dict[str, list[float]] = {}
        for m in self.models:
            result[m.broker_name] = [
                m.annual_commission_cost(tpd) for tpd in trades_per_day_options
            ]
        return result


# ── Pre-built broker configurations ──────────────────────────────────────────

def tradovate_free() -> CostModel:
    """Tradovate Free plan — $0.35/side all-in."""
    return CostModel(
        broker_name="Tradovate Free",
        broker_commission_per_side=0.35,
    )


def tradovate_lifetime() -> CostModel:
    """Tradovate Lifetime plan — $0.09/side all-in."""
    return CostModel(
        broker_name="Tradovate Lifetime",
        broker_commission_per_side=0.09,
    )


def edgeclear_rithmic() -> CostModel:
    """EdgeClear + Rithmic — estimated $0.32/side all-in.

    Broker: $0.10, exchange: ~$0.20, NFA: $0.02.
    """
    return CostModel(
        broker_name="EdgeClear + Rithmic",
        broker_commission_per_side=0.10,
        exchange_fee=0.20,
        nfa_fee=0.02,
    )


def ninjatrader_free() -> CostModel:
    """NinjaTrader Free plan — $0.39/side all-in."""
    return CostModel(
        broker_name="NinjaTrader Free",
        broker_commission_per_side=0.39,
    )


def all_brokers() -> list[CostModel]:
    """All broker configs for comparison."""
    return [
        tradovate_free(),
        tradovate_lifetime(),
        edgeclear_rithmic(),
        ninjatrader_free(),
    ]
