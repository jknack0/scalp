"""Bar-replay backtest engine.

Feeds historical 1s bars to strategies, simulates order fills against
bar OHLC, and computes performance metrics. Synchronous — no asyncio needed.

Signal-to-fill pipeline:
1. strategy.on_bar(bar_event) → Signal | None (strategies own their filters)
2. Non-None signals → oms.on_signal() creates PendingOrder
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
from src.filters.filter_engine import FilterEngine
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle, SignalEngine
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


def _get_strategy_id(strategy) -> str:
    """Extract strategy_id from either StrategyBase or standalone strategy."""
    if hasattr(strategy, "config") and hasattr(strategy.config, "strategy_id"):
        return strategy.config.strategy_id
    if hasattr(strategy, "strategy_id"):
        return strategy.strategy_id
    return type(strategy).__name__


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

    strategies: list  # StrategyBase or standalone strategies (duck-typed)
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
    dollar_threshold: float | None = None  # Set to build dollar bars (e.g. 50_000 for MES VWAP)
    l1_parquet_dir: str | None = None  # Set to "data/l1" to use L1 tick data for OBI enrichment
    l1_bar_seconds: int = 5  # Bar aggregation interval when using L1 data
    use_rth_bars: bool = False  # True = pre-built RTH bars, skip resample + session filter
    rth_parquet_dir: str = "data/parquet_5s_rth"
    use_hmm: bool = False
    hmm_model_path: str = "models/hmm/v1"
    prebuilt_bars: pl.DataFrame | None = None  # Skip data loading if bars already aggregated
    # Signal/Filter engine (declarative pipeline)
    signal_engine: SignalEngine | None = None
    filter_engine: FilterEngine | None = None
    prebuilt_bundles: list | None = None  # Pre-computed SignalBundles indexed by bar position
    enriched_signal_names: list[str] | None = None  # Signal names for inline bundle construction from sig_* columns
    # Multi-timeframe filter support: map seq -> prebuilt SignalBundle list
    # seq=1 uses signal_engine/prebuilt_bundles above.
    # Higher seqs supply their own pre-computed bundles (e.g. 15m signals for seq=2).
    filter_stage_bundles: dict[int, list] | None = None  # {2: [bundle_0, bundle_1, ...]}


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
        early_exit_fn=None,
    ) -> list[Trade]:
        """Process a bar: check for entry fills on pending, exit fills on open.

        Args:
            early_exit_fn: Optional callback (order, bar, bar_index) -> str | None.
                If it returns a non-None string, that's the early exit reason.

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
                    order, bar, bar_index, bar_time, bar_date, current_atr_ticks,
                    early_exit_fn=early_exit_fn,
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
        early_exit_fn=None,
    ) -> Trade | None:
        """Check if an open position should be exited on this bar.

        Priority: stop > target > early_exit > expiry.
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

        # Early exit conditions (OR logic — any condition triggers market exit)
        if early_exit_fn is not None:
            early_reason = early_exit_fn(order, bar, bar_index)
            if early_reason is not None:
                return self._fill_exit(
                    order, bar_index, bar_time, bar_date, current_atr_ticks,
                    reason=early_reason, use_slippage=True, market_price=bar.close,
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
        # Auto-detect L1 data need: if signal engine requires L1 fields
        # and l1_parquet_dir isn't set, check if data/l1 exists
        if (
            config.signal_engine is not None
            and config.signal_engine.requires_l1
            and config.l1_parquet_dir is None
            and config.prebuilt_bars is None
        ):
            import os
            default_l1 = "data/l1"
            if os.path.isdir(default_l1):
                logger.info("auto_l1_detected", reason="signal_engine requires L1 fields", dir=default_l1)
                config.l1_parquet_dir = default_l1

        bars_df = config.prebuilt_bars if config.prebuilt_bars is not None else self._load_bars(config)
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

        # Auto-compute multi-timeframe stage bundles if filter rules specify bar freqs
        if (
            config.filter_engine is not None
            and config.filter_engine.bar_freqs
            and config.filter_stage_bundles is None
        ):
            config.filter_stage_bundles = self._precompute_stage_bundles(
                config, bars_df
            )

        oms = SimulatedOMS(
            commission_model=config.commission_model,
            slippage_model=config.slippage_model,
            max_position=config.max_position,
        )

        # Bar window for signal computation
        signal_bar_window: list[BarEvent] = []

        all_trades: list[Trade] = []
        prev_date: date | None = None
        signal_counts = {_get_strategy_id(s): 0 for s in config.strategies}
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
                signal_counts = {_get_strategy_id(s): 0 for s in config.strategies}

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
                avg_bid_price=float(row.get("avg_bid_price", 0.0) or 0.0),
                avg_ask_price=float(row.get("avg_ask_price", 0.0) or 0.0),
                aggressive_buy_vol=float(row.get("aggressive_buy_vol", 0.0) or 0.0),
                aggressive_sell_vol=float(row.get("aggressive_sell_vol", 0.0) or 0.0),
            )

            # Compute signals and evaluate filters
            bundle = EMPTY_BUNDLE
            if config.prebuilt_bundles is not None:
                bundle = config.prebuilt_bundles[bar_index]

                # Evaluate filters — if they fail, skip strategy dispatch
                if config.filter_engine is not None:
                    if not self._evaluate_filters(config, bundle, bar_index):
                        bundle = EMPTY_BUNDLE
            elif config.enriched_signal_names is not None:
                # Build bundle inline from pre-computed sig_* columns in the row
                from src.signals.bundle_from_columns import bundle_from_row
                bundle = bundle_from_row(row, config.enriched_signal_names)

                if config.filter_engine is not None:
                    if not self._evaluate_filters(config, bundle, bar_index):
                        bundle = EMPTY_BUNDLE
            elif config.signal_engine is not None:
                signal_bar_window.append(bar_event)
                if len(signal_bar_window) > 500:
                    signal_bar_window = signal_bar_window[-500:]
                bundle = config.signal_engine.compute(signal_bar_window)

                # Evaluate filters — if they fail, skip strategy dispatch
                if config.filter_engine is not None:
                    if not self._evaluate_filters(config, bundle, bar_index):
                        bundle = EMPTY_BUNDLE

            # Feed bar to each strategy
            for strat in config.strategies:
                try:
                    signal = strat.on_bar(bar_event, bundle)
                except TypeError:
                    signal = strat.on_bar(bar_event)
                if signal is not None and oms.open_position_count < config.max_position:
                    oms.on_signal(signal, bar_index)

            # Get current ATR for slippage model
            current_atr = self._get_atr(config.strategies)

            # Build early exit callback from strategies that support it
            early_exit_fn = self._build_early_exit_fn(
                config.strategies, bar_event, bundle
            )

            # Process fills
            trades = oms.on_bar(
                bar_event, bar_index, bar_time, bar_date, current_atr,
                early_exit_fn=early_exit_fn,
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

    def precompute_bundles(
        self, config: BacktestConfig
    ) -> list[SignalBundle]:
        """Pre-compute SignalBundles for all bars in the dataset.

        Runs the signal engine once over the full bar series so that
        parameter sweeps can reuse bundles without recomputation.
        """
        bars_df = config.prebuilt_bars if config.prebuilt_bars is not None else self._load_bars(config)
        if bars_df.is_empty():
            return []

        # Same preprocessing as run()
        bars_df = bars_df.with_columns(
            pl.col("timestamp")
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("US/Eastern")
                .alias("_et_ts"),
            (pl.col("timestamp").cast(pl.Int64) * (1 if bars_df["timestamp"].dtype == pl.Datetime("ns") else 1000)).alias("_timestamp_ns"),
        )

        if not config.use_rth_bars:
            bars_df = bars_df.filter(
                (pl.col("_et_ts").dt.time() >= config.session_start)
                & (pl.col("_et_ts").dt.time() < config.session_end)
            )

        if bars_df.is_empty() or config.signal_engine is None:
            return [EMPTY_BUNDLE] * len(bars_df)

        bar_type = "5s" if config.use_rth_bars else (config.resample_freq or config.bar_type)
        signal_bar_window: list[BarEvent] = []
        bundles: list[SignalBundle] = []

        for row in bars_df.iter_rows(named=True):
            bar_event = BarEvent(
                symbol=config.symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                bar_type=bar_type,
                timestamp_ns=row["_timestamp_ns"],
                avg_bid_size=float(row.get("avg_bid_size", 0.0) or 0.0),
                avg_ask_size=float(row.get("avg_ask_size", 0.0) or 0.0),
                avg_bid_price=float(row.get("avg_bid_price", 0.0) or 0.0),
                avg_ask_price=float(row.get("avg_ask_price", 0.0) or 0.0),
                aggressive_buy_vol=float(row.get("aggressive_buy_vol", 0.0) or 0.0),
                aggressive_sell_vol=float(row.get("aggressive_sell_vol", 0.0) or 0.0),
            )
            signal_bar_window.append(bar_event)
            if len(signal_bar_window) > 500:
                signal_bar_window = signal_bar_window[-500:]
            bundles.append(config.signal_engine.compute(signal_bar_window))

        return bundles

    def _load_bars(self, config: BacktestConfig) -> pl.DataFrame:
        """Load bars from Parquet files for the date range.

        Checks the persistent bar cache first. If a cached file exists for
        the bar configuration, loads from there instead of rebuilding.
        """
        import os
        from src.data.bar_cache import BarCache

        # If L1 data is configured, aggregate ticks into enriched bars
        if config.l1_parquet_dir:
            return self._load_l1_bars(config)

        # Determine cache name based on bar type
        if config.dollar_threshold is not None:
            cache_name = BarCache.bar_name(dollar_threshold=config.dollar_threshold)
        else:
            cache_name = BarCache.bar_name(freq=config.resample_freq or config.bar_type)
        cached = BarCache.load(cache_name)
        if cached is not None:
            logger.info("bar_cache_hit", name=cache_name, rows=len(cached))
            # Filter to date range
            start_dt = datetime.combine(config.start_date, time(0, 0))
            end_dt = datetime.combine(config.end_date + timedelta(days=1), time(0, 0))
            cached = cached.filter(
                (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_dt)
            )
            return cached.sort("timestamp")

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

        # Build dollar bars or resample to coarser time bars
        if config.dollar_threshold is not None:
            from src.data.bars import build_dollar_bars
            df = build_dollar_bars(df, config.dollar_threshold)
        elif config.resample_freq and not config.use_rth_bars:
            from src.data.bars import resample_bars
            df = resample_bars(df, config.resample_freq)

        # Save full dataset to cache (before date filtering)
        if len(df) > 0:
            BarCache.save(cache_name, df)

        # Filter by date range using naive datetime (matches Parquet Datetime(us) schema)
        start_dt = datetime.combine(config.start_date, time(0, 0))
        end_dt = datetime.combine(config.end_date + timedelta(days=1), time(0, 0))
        df = df.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_dt)
        )
        df = df.sort("timestamp")

        return df

    def _load_l1_bars(self, config: BacktestConfig) -> pl.DataFrame:
        """Load L1 tick data and aggregate into bars with OBI enrichment.

        L1 data has: timestamp, price, size, side, bid_price, ask_price, bid_size, ask_size.
        Aggregates into bars with: open, high, low, close, volume, avg_bid_size, avg_ask_size,
        aggressive_buy_vol, aggressive_sell_vol.

        Checks the persistent bar cache first.
        """
        import os
        from src.data.bar_cache import BarCache

        # Check bar cache
        cache_name = BarCache.bar_name(source="l1", l1_seconds=config.l1_bar_seconds)
        cached = BarCache.load(cache_name)
        if cached is not None:
            logger.info("bar_cache_hit", name=cache_name, rows=len(cached))
            start_dt = datetime.combine(config.start_date, time(0, 0))
            end_dt = datetime.combine(config.end_date + timedelta(days=1), time(0, 0))
            cached = cached.filter(
                (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_dt)
            )
            return cached.sort("timestamp")

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
            pl.col("bid_price").mean().alias("avg_bid_price"),
            pl.col("ask_price").mean().alias("avg_ask_price"),
            pl.col("_agg_buy").sum().alias("aggressive_buy_vol"),
            pl.col("_agg_sell").sum().alias("aggressive_sell_vol"),
        )

        # Drop empty bars (no trades)
        bars = bars.filter(pl.col("volume") > 0)
        bars = bars.sort("timestamp")

        # Save full dataset to cache (before date filtering)
        if len(bars) > 0:
            BarCache.save(cache_name, bars)

        # Filter to date range
        start_dt = datetime.combine(config.start_date, time(0, 0))
        end_dt = datetime.combine(config.end_date + timedelta(days=1), time(0, 0))
        bars = bars.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_dt)
        )

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

    def _precompute_stage_bundles(
        self,
        config: BacktestConfig,
        primary_bars_df: pl.DataFrame,
    ) -> dict[int, list[SignalBundle]]:
        """Pre-compute SignalBundles for multi-timeframe filter sequences.

        For each seq with a `bar` freq in the filter engine, resamples the
        source bars to that freq, computes signals, and builds a time-aligned
        bundle list indexed by primary bar position.

        Returns:
            dict mapping seq -> list of SignalBundles aligned to primary bars.
        """
        from src.data.bars import resample_bars

        fe = config.filter_engine
        bar_freqs = fe.bar_freqs  # {seq: "5m", seq2: "15m", ...}
        signal_engine = config.signal_engine

        if not bar_freqs or signal_engine is None:
            return {}

        # Primary bar timestamps for alignment
        primary_timestamps = primary_bars_df["timestamp"].to_list()
        n_primary = len(primary_timestamps)

        stage_bundles: dict[int, list[SignalBundle]] = {}

        # Group seqs by bar freq to avoid duplicate resampling
        freq_to_seqs: dict[str, list[int]] = {}
        for seq, freq in bar_freqs.items():
            freq_to_seqs.setdefault(freq, []).append(seq)

        bar_type_str = config.resample_freq or config.bar_type

        for freq, seqs in freq_to_seqs.items():
            # Skip if this freq matches the primary bar freq (already computed)
            if freq == bar_type_str:
                continue

            # Resample source bars to this timeframe
            resampled = resample_bars(primary_bars_df, freq)
            if resampled.is_empty():
                for seq in seqs:
                    stage_bundles[seq] = [EMPTY_BUNDLE] * n_primary
                continue

            resampled_ts = resampled["timestamp"].to_list()

            # Pre-add _et_ts/_timestamp_ns if not present
            if "_timestamp_ns" not in resampled.columns:
                resampled = resampled.with_columns(
                    (pl.col("timestamp").cast(pl.Int64) * (
                        1 if resampled["timestamp"].dtype == pl.Datetime("ns") else 1000
                    )).alias("_timestamp_ns"),
                )

            # Compute signals for each resampled bar
            signal_bar_window: list[BarEvent] = []
            resampled_bundles: list[SignalBundle] = []

            for row in resampled.iter_rows(named=True):
                bar_event = BarEvent(
                    symbol=config.symbol,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    bar_type=freq,
                    timestamp_ns=row["_timestamp_ns"],
                    avg_bid_size=float(row.get("avg_bid_size", 0.0) or 0.0),
                    avg_ask_size=float(row.get("avg_ask_size", 0.0) or 0.0),
                    avg_bid_price=float(row.get("avg_bid_price", 0.0) or 0.0),
                    avg_ask_price=float(row.get("avg_ask_price", 0.0) or 0.0),
                    aggressive_buy_vol=float(row.get("aggressive_buy_vol", 0.0) or 0.0),
                    aggressive_sell_vol=float(row.get("aggressive_sell_vol", 0.0) or 0.0),
                )
                signal_bar_window.append(bar_event)
                if len(signal_bar_window) > 500:
                    signal_bar_window = signal_bar_window[-500:]
                resampled_bundles.append(signal_engine.compute(signal_bar_window))

            # Time-align: for each primary bar, find the most recent resampled bar
            # that started at or before this primary bar's timestamp.
            aligned: list[SignalBundle] = []
            rs_idx = 0
            n_resampled = len(resampled_ts)

            for p_ts in primary_timestamps:
                # Advance rs_idx to the latest resampled bar <= p_ts
                while rs_idx + 1 < n_resampled and resampled_ts[rs_idx + 1] <= p_ts:
                    rs_idx += 1

                if rs_idx < n_resampled and resampled_ts[rs_idx] <= p_ts:
                    aligned.append(resampled_bundles[rs_idx])
                else:
                    aligned.append(EMPTY_BUNDLE)

            for seq in seqs:
                stage_bundles[seq] = aligned

            logger.info(
                "stage_bundles_built",
                freq=freq,
                seqs=seqs,
                resampled_bars=n_resampled,
                primary_bars=n_primary,
            )

        return stage_bundles

    def _evaluate_filters(
        self, config: BacktestConfig, bundle: SignalBundle, bar_index: int
    ) -> bool:
        """Evaluate filter engine, including multi-seq stages.

        For single-seq filters (all seq=1), this is equivalent to
        config.filter_engine.evaluate(bundle).

        For multi-seq filters, evaluates seq=1 against the primary bundle,
        then seq=2+ against pre-computed bundles from filter_stage_bundles.
        Short-circuits on first failing seq.

        Returns True if all sequences pass.
        """
        fe = config.filter_engine
        sequences = fe.sequences

        if not sequences:
            return True

        # If no multi-seq stages configured, evaluate all rules at once
        if config.filter_stage_bundles is None:
            return fe.evaluate(bundle).passes

        # Multi-seq: evaluate each sequence in order
        for seq in sequences:
            if seq == 1:
                # seq=1 uses the primary bundle (from signal_engine)
                result = fe.evaluate_seq(bundle, seq=1)
            else:
                # Higher seqs use pre-computed bundles
                stage_bundles = config.filter_stage_bundles.get(seq)
                if stage_bundles is None or bar_index >= len(stage_bundles):
                    continue  # No data for this seq — skip
                result = fe.evaluate_seq(stage_bundles[bar_index], seq=seq)

            if not result.passes:
                return False

        return True

    def _build_early_exit_fn(
        self,
        strategies: list,
        bar: BarEvent,
        bundle: SignalBundle,
    ):
        """Build an early exit callback for the current bar.

        Returns a callable (order, bar, bar_index) -> str | None, or None
        if no strategy supports early exits.
        """
        # Find strategies with check_early_exit method
        exit_strategies = {
            _get_strategy_id(s): s
            for s in strategies
            if hasattr(s, "check_early_exit")
        }
        if not exit_strategies:
            return None

        def _fn(order: PendingOrder, bar: BarEvent, bar_index: int) -> str | None:
            strat = exit_strategies.get(order.strategy_id)
            if strat is None:
                return None
            bars_in_trade = bar_index - order.fill_bar_index
            return strat.check_early_exit(
                bar=bar,
                bundle=bundle,
                bars_in_trade=bars_in_trade,
                direction=order.direction,
                fill_price=order.fill_price,
            )

        return _fn

    def _get_atr(self, strategies: list) -> float:
        """Read ATR from the first strategy's feature hub (if available)."""
        if strategies:
            s = strategies[0]
            # StrategyBase stores _last_snapshot with atr_ticks
            snap = getattr(s, "_last_snapshot", None)
            if snap is not None and hasattr(snap, "atr_ticks"):
                return snap.atr_ticks
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
            "strategies": [_get_strategy_id(s) for s in config.strategies],
        }


def _sub_time(t: time, seconds: int = 1) -> time:
    """Subtract seconds from a time object."""
    dt = datetime.combine(date.today(), t) - timedelta(seconds=seconds)
    return dt.time()
