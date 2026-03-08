"""Backtest metrics computation — Trade, BacktestMetrics, BacktestResult.

Calculates Sharpe, Sortino, max drawdown, profit factor, win rate, and
builds equity curve / daily P&L DataFrames from a list of Trade objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import polars as pl

from src.strategies.base import Direction


@dataclass(frozen=True)
class Trade:
    """Single completed round-trip trade."""

    trade_id: str
    strategy_id: str
    direction: Direction
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    size: int
    gross_pnl: float
    slippage_cost: float
    commission: float
    net_pnl: float
    exit_reason: str  # "target", "stop", "expiry", "session_close"
    bars_held: int
    entry_slippage_ticks: float
    exit_slippage_ticks: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestMetrics:
    """Aggregate performance statistics."""

    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: float
    gross_pnl: float
    net_pnl: float
    total_commission: float
    total_slippage: float
    avg_slippage_ticks: float
    avg_bars_held: float
    best_trade: float
    worst_trade: float
    avg_daily_pnl: float
    trading_days: int


@dataclass
class BacktestResult:
    """Complete backtest output."""

    trades: list[Trade]
    equity_curve: pl.DataFrame  # columns: timestamp, equity
    daily_pnl: pl.DataFrame  # columns: date, pnl, cumulative_pnl
    metrics: BacktestMetrics
    config_summary: dict


class MetricsCalculator:
    """Computes backtest metrics from a list of trades."""

    @staticmethod
    def from_trades(
        trades: list[Trade], initial_capital: float
    ) -> tuple[BacktestMetrics, pl.DataFrame, pl.DataFrame]:
        """Build metrics, equity curve, and daily P&L from trades.

        Returns:
            (BacktestMetrics, equity_curve DataFrame, daily_pnl DataFrame)
        """
        if not trades:
            empty_metrics = BacktestMetrics(
                total_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown_pct=0.0,
                max_drawdown_duration_days=0.0,
                gross_pnl=0.0,
                net_pnl=0.0,
                total_commission=0.0,
                total_slippage=0.0,
                avg_slippage_ticks=0.0,
                avg_bars_held=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                avg_daily_pnl=0.0,
                trading_days=0,
            )
            eq = pl.DataFrame(
                {"timestamp": [trades[0].exit_time if trades else datetime.min], "equity": [initial_capital]}
            ) if trades else pl.DataFrame({"timestamp": pl.Series([], dtype=pl.Datetime), "equity": pl.Series([], dtype=pl.Float64)})
            dp = pl.DataFrame({"date": pl.Series([], dtype=pl.Date), "pnl": pl.Series([], dtype=pl.Float64), "cumulative_pnl": pl.Series([], dtype=pl.Float64)})
            return empty_metrics, eq, dp

        # Build equity curve
        equity = initial_capital
        eq_timestamps = [trades[0].entry_time]
        eq_values = [float(initial_capital)]
        for t in trades:
            equity += t.net_pnl
            eq_timestamps.append(t.exit_time)
            eq_values.append(equity)

        equity_curve = pl.DataFrame({
            "timestamp": eq_timestamps,
            "equity": eq_values,
        })

        # Build daily P&L
        daily_map: dict[date, float] = {}
        for t in trades:
            d = t.exit_time.date() if isinstance(t.exit_time, datetime) else t.exit_time
            daily_map[d] = daily_map.get(d, 0.0) + t.net_pnl

        sorted_dates = sorted(daily_map.keys())
        daily_pnls = [daily_map[d] for d in sorted_dates]
        cum_pnl = np.cumsum(daily_pnls).tolist()

        daily_pnl_df = pl.DataFrame({
            "date": sorted_dates,
            "pnl": daily_pnls,
            "cumulative_pnl": cum_pnl,
        })

        # Compute metrics
        net_pnls = [t.net_pnl for t in trades]
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p <= 0]

        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        pf = MetricsCalculator.profit_factor(trades)

        daily_returns = np.array(daily_pnls) / initial_capital if daily_pnls else np.array([])
        sharpe = MetricsCalculator.sharpe(daily_returns)
        sortino = MetricsCalculator.sortino(daily_returns)

        dd_pct, dd_dur = MetricsCalculator.max_drawdown(eq_values)

        gross_pnl = sum(t.gross_pnl for t in trades)
        net_pnl = sum(t.net_pnl for t in trades)
        total_commission = sum(t.commission for t in trades)
        total_slippage = sum(t.slippage_cost for t in trades)
        avg_slip = float(np.mean([t.entry_slippage_ticks + t.exit_slippage_ticks for t in trades]))
        avg_bars = float(np.mean([t.bars_held for t in trades]))

        metrics = BacktestMetrics(
            total_trades=total_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=pf,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=dd_pct,
            max_drawdown_duration_days=dd_dur,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_commission=total_commission,
            total_slippage=total_slippage,
            avg_slippage_ticks=avg_slip,
            avg_bars_held=avg_bars,
            best_trade=max(net_pnls),
            worst_trade=min(net_pnls),
            avg_daily_pnl=float(np.mean(daily_pnls)) if daily_pnls else 0.0,
            trading_days=len(sorted_dates),
        )

        return metrics, equity_curve, daily_pnl_df

    @staticmethod
    def sharpe(
        daily_returns: np.ndarray,
        risk_free: float = 0.0,
        periods: int = 252,
    ) -> float:
        """Annualized Sharpe ratio from daily returns."""
        if len(daily_returns) < 2:
            return 0.0
        excess = daily_returns - risk_free / periods
        std = float(np.std(excess, ddof=1))
        if std < 1e-10:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(periods))

    @staticmethod
    def sortino(
        daily_returns: np.ndarray,
        target: float = 0.0,
        periods: int = 252,
    ) -> float:
        """Annualized Sortino ratio from daily returns."""
        if len(daily_returns) < 2:
            return 0.0
        excess = daily_returns - target
        downside = daily_returns[daily_returns < target] - target
        if len(downside) < 1:
            return 0.0
        downside_std = float(np.sqrt(np.mean(downside**2)))
        if downside_std < 1e-10:
            return 0.0
        return float(np.mean(excess) / downside_std * np.sqrt(periods))

    @staticmethod
    def max_drawdown(equity_series: list[float]) -> tuple[float, float]:
        """Compute max drawdown percentage and duration in days.

        Args:
            equity_series: List of equity values over time.

        Returns:
            (max_drawdown_pct, duration_days) — drawdown as a positive
            percentage (e.g. 5.0 means 5%).
        """
        if len(equity_series) < 2:
            return 0.0, 0.0

        arr = np.array(equity_series, dtype=np.float64)
        peak = np.maximum.accumulate(arr)
        drawdowns = (peak - arr) / np.where(peak > 0, peak, 1.0)

        max_dd_pct = float(np.max(drawdowns)) * 100.0

        # Duration: longest streak where equity < peak
        max_dur = 0
        current_dur = 0
        for i in range(len(arr)):
            if arr[i] < peak[i]:
                current_dur += 1
                max_dur = max(max_dur, current_dur)
            else:
                current_dur = 0

        return max_dd_pct, float(max_dur)

    @staticmethod
    def profit_factor(trades: list[Trade]) -> float:
        """Gross wins / gross losses. Returns 0 if no losses."""
        gross_wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_losses = abs(sum(t.net_pnl for t in trades if t.net_pnl <= 0))
        if gross_losses < 1e-10:
            return float("inf") if gross_wins > 0 else 0.0
        return gross_wins / gross_losses
