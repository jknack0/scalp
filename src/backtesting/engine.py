"""Bar-replay backtest engine.

Feeds historical 1s bars to strategies, simulates order fills against
bar OHLC, and computes performance metrics. Synchronous — no asyncio needed.

Signal-to-fill pipeline:
1. strategy.on_bar(bar_event) — may append to _signals_generated
2. Engine detects new signals via len(_signals_generated) diff
3. New signals → oms.on_signal() creates PendingOrder
4. Entry: next bar+ where price reaches entry_price (limit, no slippage)
5. Exit: target (limit, no slippage), stop (market, slippage applied)
6. Ambiguity: if both stop and target in same bar → stop hit first
7. Expiry/session close → market exit with slippage
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

import polars as pl

from src.analysis.commission_model import CostModel, tradovate_free
from src.backtesting.metrics import (
    BacktestResult,
    MetricsCalculator,
    Trade,
)
from src.backtesting.slippage import SlippageResult, VolatilitySlippageModel
from src.core.events import BarEvent
from src.core.logging import get_logger
from src.filters.spread_monitor import SpreadConfig, SpreadMonitor, SpreadSnapshot
from src.filters.vpin_monitor import VPINConfig, VPINMonitor
from src.strategies.base import Direction, Signal, StrategyBase

logger = get_logger("backtest")


def _run_coro(coro):
    """Run an async coroutine synchronously (for backtest context)."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Nested event loop — create a new one in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# MES constants
TICK_SIZE = 0.25
TICK_VALUE = 1.25
POINT_VALUE = 5.0

# US Eastern timezone (DST-aware)
from zoneinfo import ZoneInfo

_ET = ZoneInfo("US/Eastern")


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    strategies: list[StrategyBase]
    start_date: date
    end_date: date
    initial_capital: float = 10_000.0
    commission_model: CostModel = field(default_factory=tradovate_free)
    slippage_model: VolatilitySlippageModel = field(
        default_factory=VolatilitySlippageModel
    )
    bar_type: str = "1s"
    resample_freq: str | None = None  # e.g. "5s", "15s", "1m" — None keeps raw 1s
    symbol: str = "MESM6"
    session_start: time = field(default_factory=lambda: time(9, 30))
    session_end: time = field(default_factory=lambda: time(16, 0))
    max_position: int = 1
    parquet_dir: str = "data/parquet"
    l1_parquet_dir: str | None = None  # Set to "data/l1" to use L1 tick data for OBI enrichment
    l1_bar_seconds: int = 5  # Bar aggregation interval when using L1 data
    use_rth_bars: bool = False  # True = pre-built RTH bars, skip resample + session filter
    rth_parquet_dir: str = "data/parquet_5s_rth"
    use_hmm: bool = False
    hmm_model_path: str = "models/hmm/v1"
    spread_monitor: SpreadMonitor | None = None
    vpin_monitor: VPINMonitor | None = None


@dataclass
class PendingOrder:
    """Tracks an order from signal through fill to close."""

    order_id: str
    signal: Signal
    strategy_id: str
    direction: Direction
    entry_price: float
    target_price: float
    stop_price: float
    entry_bar_index: int  # bar index when signal was created
    expiry_time: datetime
    status: str = "pending_entry"  # "pending_entry", "open", "closed"
    fill_price: float = 0.0
    fill_bar_index: int = 0
    fill_time: datetime | None = None
    exit_price: float = 0.0
    exit_time: datetime | None = None
    exit_bar_index: int = 0
    exit_reason: str = ""
    entry_slippage_ticks: float = 0.0
    exit_slippage_ticks: float = 0.0


class SimulatedOMS:
    """Simulated order management system for backtesting.

    Handles signal → pending order → entry fill → exit fill lifecycle.
    """

    def __init__(
        self,
        commission_model: CostModel,
        slippage_model: VolatilitySlippageModel,
        max_position: int = 1,
        tick_size: float = TICK_SIZE,
        tick_value: float = TICK_VALUE,
        point_value: float = POINT_VALUE,
    ) -> None:
        self._commission_model = commission_model
        self._slippage_model = slippage_model
        self._max_position = max_position
        self._tick_size = tick_size
        self._tick_value = tick_value
        self._point_value = point_value
        self._orders: list[PendingOrder] = []

    def on_signal(self, signal: Signal, bar_index: int) -> str:
        """Create a PendingOrder from a strategy signal.

        Returns:
            order_id of the created order.
        """
        order_id = str(uuid.uuid4())[:8]
        order = PendingOrder(
            order_id=order_id,
            signal=signal,
            strategy_id=signal.strategy_id,
            direction=signal.direction,
            entry_price=signal.entry_price,
            target_price=signal.target_price,
            stop_price=signal.stop_price,
            entry_bar_index=bar_index,
            expiry_time=signal.expiry_time,
        )
        self._orders.append(order)
        return order_id

    def on_bar(
        self,
        bar: BarEvent,
        bar_index: int,
        bar_time: datetime,
        bar_date: date,
        current_atr_ticks: float,
    ) -> list[Trade]:
        """Process a bar: check for entry fills on pending, exit fills on open.

        Returns:
            List of completed trades (may be empty).
        """
        completed: list[Trade] = []

        for order in self._orders:
            if order.status == "closed":
                continue

            # --- Pending entry orders ---
            if order.status == "pending_entry":
                # Check expiry first
                if bar_time >= order.expiry_time:
                    order.status = "closed"
                    continue

                # Don't fill on the same bar the signal was generated
                if bar_index <= order.entry_bar_index:
                    continue

                # Check if too many open positions
                if self.open_position_count >= self._max_position:
                    continue

                # Limit entry: fill at entry_price if bar touches it
                filled = False
                if order.direction == Direction.LONG:
                    if bar.low <= order.entry_price:
                        filled = True
                elif order.direction == Direction.SHORT:
                    if bar.high >= order.entry_price:
                        filled = True

                if filled:
                    order.status = "open"
                    order.fill_price = order.entry_price  # Limit fill, no slippage
                    order.fill_bar_index = bar_index
                    order.fill_time = bar_time
                    order.entry_slippage_ticks = 0.0

            # --- Open positions: check exits ---
            if order.status == "open":
                trade = self._check_exit(
                    order, bar, bar_index, bar_time, bar_date, current_atr_ticks
                )
                if trade is not None:
                    completed.append(trade)

        # Clean up closed orders
        self._orders = [o for o in self._orders if o.status != "closed"]
        return completed

    def _check_exit(
        self,
        order: PendingOrder,
        bar: BarEvent,
        bar_index: int,
        bar_time: datetime,
        bar_date: date,
        current_atr_ticks: float,
    ) -> Trade | None:
        """Check if an open position should be exited on this bar.

        Ambiguity rule: if both stop and target are within the bar's range,
        assume stop hit first (conservative).
        """
        stop_hit = False
        target_hit = False

        if order.direction == Direction.LONG:
            stop_hit = bar.low <= order.stop_price
            target_hit = bar.high >= order.target_price
        else:  # SHORT
            stop_hit = bar.high >= order.stop_price
            target_hit = bar.low <= order.target_price

        # Ambiguity: both in same bar → stop first (conservative)
        if stop_hit and target_hit:
            return self._fill_exit(
                order, bar_index, bar_time, bar_date, current_atr_ticks,
                reason="stop", use_slippage=True,
            )

        if stop_hit:
            return self._fill_exit(
                order, bar_index, bar_time, bar_date, current_atr_ticks,
                reason="stop", use_slippage=True,
            )

        if target_hit:
            return self._fill_exit(
                order, bar_index, bar_time, bar_date, current_atr_ticks,
                reason="target", use_slippage=False,
            )

        # Check expiry on open positions
        if bar_time >= order.expiry_time:
            return self._fill_exit(
                order, bar_index, bar_time, bar_date, current_atr_ticks,
                reason="expiry", use_slippage=True, market_price=bar.close,
            )

        return None

    def _fill_exit(
        self,
        order: PendingOrder,
        bar_index: int,
        bar_time: datetime,
        bar_date: date,
        current_atr_ticks: float,
        reason: str,
        use_slippage: bool,
        market_price: float | None = None,
    ) -> Trade:
        """Close a position and produce a Trade."""
        slip_result = SlippageResult(ticks=0.0, reason="calm")
        if use_slippage:
            slip_result = self._slippage_model.compute_slippage(
                bar_date, current_atr_ticks
            )

        # Determine exit price
        if reason == "target":
            exit_price = order.target_price
        elif reason == "stop":
            # Adverse slippage on stop
            if order.direction == Direction.LONG:
                exit_price = order.stop_price - slip_result.ticks * self._tick_size
            else:
                exit_price = order.stop_price + slip_result.ticks * self._tick_size
        else:
            # expiry or session_close: market exit at close with adverse slippage
            assert market_price is not None
            if order.direction == Direction.LONG:
                exit_price = market_price - slip_result.ticks * self._tick_size
            else:
                exit_price = market_price + slip_result.ticks * self._tick_size

        # P&L
        if order.direction == Direction.LONG:
            gross_pnl = (exit_price - order.fill_price) * self._point_value
        else:
            gross_pnl = (order.fill_price - exit_price) * self._point_value

        commission = self._commission_model.round_trip_commission()
        exit_slippage_ticks = slip_result.ticks if use_slippage else 0.0
        slippage_cost = (
            (order.entry_slippage_ticks + exit_slippage_ticks) * self._tick_value
        )
        net_pnl = gross_pnl - commission - slippage_cost

        order.status = "closed"
        order.exit_price = exit_price
        order.exit_time = bar_time
        order.exit_bar_index = bar_index
        order.exit_reason = reason
        order.exit_slippage_ticks = exit_slippage_ticks

        return Trade(
            trade_id=order.order_id,
            strategy_id=order.strategy_id,
            direction=order.direction,
            entry_price=order.fill_price,
            exit_price=exit_price,
            entry_time=order.fill_time,
            exit_time=bar_time,
            size=1,
            gross_pnl=gross_pnl,
            slippage_cost=slippage_cost,
            commission=commission,
            net_pnl=net_pnl,
            exit_reason=reason,
            bars_held=bar_index - order.fill_bar_index,
            entry_slippage_ticks=order.entry_slippage_ticks,
            exit_slippage_ticks=exit_slippage_ticks,
            metadata={"signal_id": order.signal.id},
        )

    def close_all(
        self,
        close_price: float,
        close_time: datetime,
        bar_index: int,
        bar_date: date,
        current_atr_ticks: float,
        reason: str = "session_close",
    ) -> list[Trade]:
        """Force-close all open positions at market price."""
        completed: list[Trade] = []
        for order in self._orders:
            if order.status == "open":
                trade = self._fill_exit(
                    order, bar_index, close_time, bar_date, current_atr_ticks,
                    reason=reason, use_slippage=True, market_price=close_price,
                )
                completed.append(trade)
            elif order.status == "pending_entry":
                order.status = "closed"
        self._orders = [o for o in self._orders if o.status != "closed"]
        return completed

    @property
    def open_position_count(self) -> int:
        """Number of currently open positions."""
        return sum(1 for o in self._orders if o.status == "open")

    @property
    def pending_entry_count(self) -> int:
        """Number of pending entry orders."""
        return sum(1 for o in self._orders if o.status == "pending_entry")


class BacktestEngine:
    """Bar-replay backtest engine.

    Loads historical bars from Parquet, feeds them to strategies,
    simulates fills via SimulatedOMS, and returns BacktestResult.
    """

    def run(self, config: BacktestConfig) -> BacktestResult:
        """Execute a backtest. Synchronous entry point.

        Args:
            config: BacktestConfig with strategies, date range, etc.

        Returns:
            BacktestResult with trades, equity curve, daily P&L, metrics.
        """
        bars_df = self._load_bars(config)
        if bars_df.is_empty():
            logger.warning("no_bars_loaded", start=str(config.start_date), end=str(config.end_date))
            metrics, eq, dp = MetricsCalculator.from_trades([], config.initial_capital)
            return BacktestResult(
                trades=[], equity_curve=eq, daily_pnl=dp,
                metrics=metrics, config_summary=self._config_summary(config),
            )

        # ── Vectorized preprocessing (replaces per-row Python work) ──
        # Convert naive-UTC timestamps to US/Eastern and compute epoch ns
        bars_df = bars_df.with_columns(
            pl.col("timestamp")
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("US/Eastern")
                .alias("_et_ts"),
            (pl.col("timestamp").cast(pl.Int64) * (1 if bars_df["timestamp"].dtype == pl.Datetime("ns") else 1000)).alias("_timestamp_ns"),
        )

        # Pre-filter to session hours (~72% of bars are outside RTH)
        # Skip if using pre-built RTH bars (already filtered)
        if not config.use_rth_bars:
            bars_df = bars_df.filter(
                (pl.col("_et_ts").dt.time() >= config.session_start)
                & (pl.col("_et_ts").dt.time() < config.session_end)
            )

        if bars_df.is_empty():
            metrics, eq, dp = MetricsCalculator.from_trades([], config.initial_capital)
            return BacktestResult(
                trades=[], equity_curve=eq, daily_pnl=dp,
                metrics=metrics, config_summary=self._config_summary(config),
            )

        # Extract date/time columns for day-boundary and session-close detection
        bars_df = bars_df.with_columns(
            pl.col("_et_ts").dt.date().alias("_bar_date"),
            pl.col("_et_ts").dt.time().alias("_bar_time"),
        )

        oms = SimulatedOMS(
            commission_model=config.commission_model,
            slippage_model=config.slippage_model,
            max_position=config.max_position,
        )

        all_trades: list[Trade] = []
        prev_date: date | None = None
        signal_counts = {s.config.strategy_id: 0 for s in config.strategies}
        session_close_time = _sub_time(config.session_end, seconds=1)

        n_bars = len(bars_df)
        logger.info("backtest_start", bars=n_bars, strategies=len(config.strategies))

        last_bar_time: datetime | None = None
        last_bar_date: date | None = None
        last_close: float = 0.0

        for bar_index, row in enumerate(bars_df.iter_rows(named=True)):
            bar_date = row["_bar_date"]
            bar_time = row["_et_ts"]

            # Session boundaries
            if prev_date is None or bar_date != prev_date:
                if prev_date is not None:
                    trades = oms.close_all(
                        close_price=row["close"],
                        close_time=bar_time,
                        bar_index=bar_index,
                        bar_date=bar_date,
                        current_atr_ticks=self._get_atr(config.strategies),
                        reason="session_close",
                    )
                    all_trades.extend(trades)

                for strat in config.strategies:
                    strat.reset()
                signal_counts = {s.config.strategy_id: 0 for s in config.strategies}

            prev_date = bar_date
            last_bar_time = bar_time
            last_bar_date = bar_date
            last_close = row["close"]

            # Build bar event (timestamp_ns precomputed — no per-row conversion)
            bar_event = BarEvent(
                symbol=config.symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                bar_type="5s" if config.use_rth_bars else (config.resample_freq or config.bar_type),
                timestamp_ns=row["_timestamp_ns"],
                avg_bid_size=float(row.get("avg_bid_size", 0.0) or 0.0),
                avg_ask_size=float(row.get("avg_ask_size", 0.0) or 0.0),
                aggressive_buy_vol=float(row.get("aggressive_buy_vol", 0.0) or 0.0),
                aggressive_sell_vol=float(row.get("aggressive_sell_vol", 0.0) or 0.0),
            )

            # Feed filters with bar data
            if config.spread_monitor and bar_event.avg_bid_size > 0 and bar_event.avg_ask_size > 0:
                snap = SpreadSnapshot(
                    timestamp=bar_time.replace(tzinfo=None) if bar_time.tzinfo else bar_time,
                    bid=bar_event.avg_bid_size,
                    ask=bar_event.avg_ask_size,
                )
                config.spread_monitor.push_sync(snap)

            if config.vpin_monitor and bar_event.volume > 0:
                config.vpin_monitor.on_bar_approx(
                    open_=bar_event.open,
                    close=bar_event.close,
                    volume=bar_event.volume,
                    timestamp=bar_time.replace(tzinfo=None) if bar_time.tzinfo else bar_time,
                    high=bar_event.high,
                    low=bar_event.low,
                )

            # Feed bar to each strategy and detect new signals
            for strat in config.strategies:
                prev_count = len(strat._signals_generated)

                strat.on_bar(bar_event)

                new_count = len(strat._signals_generated)
                if new_count > prev_count:
                    for sig in strat._signals_generated[prev_count:]:
                        if oms.open_position_count < config.max_position:
                            # Spread filter gate
                            if config.spread_monitor:
                                spread_ok, _ = config.spread_monitor.is_spread_normal()
                                if not spread_ok:
                                    continue
                            # VPIN regime gate
                            if config.vpin_monitor:
                                vpin_blocked, _ = config.vpin_monitor.should_block(sig.strategy_id)
                                if vpin_blocked:
                                    continue
                            oms.on_signal(sig, bar_index)

            # Get current ATR for slippage model
            current_atr = self._get_atr(config.strategies)

            # Process fills
            trades = oms.on_bar(
                bar_event, bar_index, bar_time, bar_date, current_atr
            )
            all_trades.extend(trades)

            # Session close: close all at end of session
            if row["_bar_time"] >= session_close_time:
                trades = oms.close_all(
                    close_price=bar_event.close,
                    close_time=bar_time,
                    bar_index=bar_index,
                    bar_date=bar_date,
                    current_atr_ticks=current_atr,
                    reason="session_close",
                )
                all_trades.extend(trades)

        # Final close of any remaining positions
        if n_bars > 0 and last_bar_time is not None:
            trades = oms.close_all(
                close_price=float(last_close),
                close_time=last_bar_time,
                bar_index=n_bars - 1,
                bar_date=last_bar_date,
                current_atr_ticks=self._get_atr(config.strategies),
                reason="session_close",
            )
            all_trades.extend(trades)

        # Compute metrics
        metrics, equity_curve, daily_pnl = MetricsCalculator.from_trades(
            all_trades, config.initial_capital
        )

        logger.info(
            "backtest_complete",
            trades=len(all_trades),
            net_pnl=round(metrics.net_pnl, 2),
            sharpe=round(metrics.sharpe_ratio, 3),
        )

        return BacktestResult(
            trades=all_trades,
            equity_curve=equity_curve,
            daily_pnl=daily_pnl,
            metrics=metrics,
            config_summary=self._config_summary(config),
        )

    def _load_bars(self, config: BacktestConfig) -> pl.DataFrame:
        """Load bars from Parquet files for the date range."""
        import os

        # If L1 data is configured, aggregate ticks into enriched bars
        if config.l1_parquet_dir:
            return self._load_l1_bars(config)

        parquet_dir = config.rth_parquet_dir if config.use_rth_bars else config.parquet_dir

        frames = []
        start_year = config.start_date.year
        end_year = config.end_date.year

        for year in range(start_year, end_year + 1):
            path = os.path.join(parquet_dir, f"year={year}", "data.parquet")
            if os.path.exists(path):
                df = pl.read_parquet(path)
                frames.append(df)

        if not frames:
            return pl.DataFrame()

        df = pl.concat(frames)

        # Filter by date range using naive datetime (matches Parquet Datetime(us) schema)
        start_dt = datetime.combine(config.start_date, time(0, 0))
        end_dt = datetime.combine(config.end_date + timedelta(days=1), time(0, 0))
        df = df.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_dt)
        )
        df = df.sort("timestamp")

        # Resample to coarser bars if requested and not using pre-built RTH bars
        if config.resample_freq and not config.use_rth_bars:
            from src.data.bars import resample_bars
            df = resample_bars(df, config.resample_freq)

        return df

    def _load_l1_bars(self, config: BacktestConfig) -> pl.DataFrame:
        """Load L1 tick data and aggregate into bars with OBI enrichment.

        L1 data has: timestamp, price, size, side, bid_price, ask_price, bid_size, ask_size.
        Aggregates into bars with: open, high, low, close, volume, avg_bid_size, avg_ask_size,
        aggressive_buy_vol, aggressive_sell_vol.
        """
        import os

        frames = []
        for year in range(config.start_date.year, config.end_date.year + 1):
            path = os.path.join(config.l1_parquet_dir, f"year={year}", "data.parquet")
            if os.path.exists(path):
                frames.append(pl.read_parquet(path))

        if not frames:
            logger.warning("no_l1_data", dir=config.l1_parquet_dir)
            return pl.DataFrame()

        df = pl.concat(frames)

        # Strip timezone if present (L1 uses ns precision)
        if df["timestamp"].dtype == pl.Datetime("ns", "UTC") or df["timestamp"].dtype == pl.Datetime("us", "UTC"):
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

        start_dt = datetime.combine(config.start_date, time(0, 0))
        end_dt = datetime.combine(config.end_date + timedelta(days=1), time(0, 0))
        df = df.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_dt)
        )

        if df.is_empty():
            return pl.DataFrame()

        # Classify aggressive trades via Lee-Ready: compare trade price to mid
        df = df.with_columns(
            ((pl.col("bid_price") + pl.col("ask_price")) / 2.0).alias("_mid"),
        )
        df = df.with_columns(
            pl.when(pl.col("price") > pl.col("_mid"))
            .then(pl.col("size"))
            .otherwise(pl.lit(0))
            .alias("_agg_buy"),
            pl.when(pl.col("price") < pl.col("_mid"))
            .then(pl.col("size"))
            .otherwise(pl.lit(0))
            .alias("_agg_sell"),
        )

        # Group into time bars
        interval = f"{config.l1_bar_seconds}s"
        bars = df.group_by_dynamic("timestamp", every=interval).agg(
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
            pl.col("bid_size").mean().alias("avg_bid_size"),
            pl.col("ask_size").mean().alias("avg_ask_size"),
            pl.col("_agg_buy").sum().alias("aggressive_buy_vol"),
            pl.col("_agg_sell").sum().alias("aggressive_sell_vol"),
        )

        # Drop empty bars (no trades)
        bars = bars.filter(pl.col("volume") > 0)
        bars = bars.sort("timestamp")

        logger.info("l1_bars_built", ticks=len(df), bars=len(bars), interval=interval)
        return bars

    def _is_new_session(self, prev_date: date | None, current_date: date) -> bool:
        """Detect session boundary (new trading day)."""
        if prev_date is None:
            return True
        return current_date != prev_date

    def _bar_to_event(self, row: dict, symbol: str, bar_type: str) -> BarEvent:
        """Convert a Parquet row dict to a BarEvent."""
        return BarEvent(
            symbol=symbol,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
            bar_type=bar_type,
            timestamp_ns=int(
                row["timestamp"].replace(tzinfo=timezone.utc).timestamp()
                * 1_000_000_000
            ),
        )

    def _row_to_datetime(self, row: dict) -> datetime:
        """Convert a row's timestamp (naive UTC datetime) to ET datetime."""
        return row["timestamp"].replace(tzinfo=timezone.utc).astimezone(_ET)

    def _get_atr(self, strategies: list[StrategyBase]) -> float:
        """Read ATR from the first strategy's feature hub."""
        if strategies and strategies[0]._last_snapshot is not None:
            return strategies[0]._last_snapshot.atr_ticks
        return 1.0  # Default if no snapshot yet

    def _config_summary(self, config: BacktestConfig) -> dict:
        """Build a summary dict of the backtest configuration."""
        return {
            "start_date": str(config.start_date),
            "end_date": str(config.end_date),
            "initial_capital": config.initial_capital,
            "symbol": config.symbol,
            "bar_type": config.bar_type,
            "max_position": config.max_position,
            "commission_per_side": config.commission_model.commission_per_side,
            "strategies": [s.config.strategy_id for s in config.strategies],
        }


def _sub_time(t: time, seconds: int = 1) -> time:
    """Subtract seconds from a time object."""
    dt = datetime.combine(date.today(), t) - timedelta(seconds=seconds)
    return dt.time()
