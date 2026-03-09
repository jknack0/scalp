"""L2 tick-by-tick replay engine for order-book strategies.

Reads either raw DataBento MBP-10 DBN files or pre-filtered RTH Parquet files,
feeds L2 snapshots and trade events to a strategy, and simulates position
management with target/stop/expiry exits.

Usage:
    from src.backtesting.l2_replay import L2ReplayConfig, L2ReplayEngine
    from src.strategies.iceberg_strategy import L2Strategy, L2StrategyConfig

    strategy = L2Strategy(L2StrategyConfig(...))
    # From raw DBN (slow, includes non-RTH):
    config = L2ReplayConfig(strategy=strategy, dbn_dir="data/l2")
    # From pre-filtered RTH Parquet (fast):
    config = L2ReplayConfig(strategy=strategy, dbn_dir="data/l2_rth", use_parquet=True)
    engine = L2ReplayEngine()
    result = engine.run(config)
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from src.backtesting.metrics import (
    BacktestMetrics,
    BacktestResult,
    MetricsCalculator,
    Trade,
)
from src.filters import L2Snapshot
from src.filters.iceberg_absorption import TradeEvent
from src.strategies.base import TICK_SIZE, TICK_VALUE, Direction
from src.strategies.iceberg_strategy import L2Strategy

_ET = ZoneInfo("US/Eastern")
_UTC = ZoneInfo("UTC")

# MES point multiplier
POINT_VALUE = 5.0

RTH_START = dt_time(9, 30)
RTH_END = dt_time(16, 0)


@dataclass
class L2ReplayConfig:
    """Configuration for L2 replay backtest."""

    strategy: L2Strategy
    dbn_dir: str = "data/l2"
    rth_only: bool = True
    use_parquet: bool = False  # Use pre-filtered RTH Parquet files
    slippage_ticks: int = 1
    commission_per_side: float = 0.35
    initial_capital: float = 10_000.0
    # Print progress every N messages
    progress_interval: int = 5_000_000


@dataclass
class _Position:
    """Active position being tracked."""

    direction: Direction
    entry_price: float
    entry_time: datetime
    target_price: float
    stop_price: float
    expiry_time: datetime
    signal_confidence: float
    metadata: dict = field(default_factory=dict)


class L2ReplayEngine:
    """Replays MBP-10 data tick-by-tick through a strategy."""

    def run(self, config: L2ReplayConfig) -> BacktestResult:
        """Run the full L2 replay backtest."""
        if config.use_parquet:
            return self._run_parquet(config)
        return self._run_dbn(config)

    def _run_parquet(self, config: L2ReplayConfig) -> BacktestResult:
        """Run replay from pre-filtered RTH Parquet files (fast path)."""
        import polars as pl

        parquet_files = sorted(Path(config.dbn_dir).glob("*.parquet"))
        if not parquet_files:
            print(f"No .parquet files found in {config.dbn_dir}")
            return self._empty_result(config)

        print(f"L2 Replay (Parquet): {len(parquet_files)} files")
        for f in parquet_files:
            size_mb = f.stat().st_size / (1024 ** 2)
            print(f"  {f.name} ({size_mb:.0f} MB)")

        strategy = config.strategy
        trades: list[Trade] = []
        position: _Position | None = None
        msg_count = 0
        trade_count = 0
        signal_count = 0
        ts_et = None

        for pq_path in parquet_files:
            print(f"\nProcessing {pq_path.name}...")
            df = pl.read_parquet(str(pq_path))
            n_rows = len(df)
            print(f"  {n_rows:,} RTH messages")

            # Convert to numpy for fast row iteration
            ts_arr = df["ts_event"].to_numpy()
            action_arr = df["action"].to_list()
            side_arr = df["side"].to_list()
            price_arr = df["price"].to_numpy()
            size_arr = df["size"].to_numpy()

            # Level arrays
            bid_px = {}
            bid_sz = {}
            ask_px = {}
            ask_sz = {}
            for i in range(1, 11):
                bid_px[i] = df[f"bid_px_{i}"].to_numpy()
                bid_sz[i] = df[f"bid_sz_{i}"].to_numpy()
                ask_px[i] = df[f"ask_px_{i}"].to_numpy()
                ask_sz[i] = df[f"ask_sz_{i}"].to_numpy()

            for row_idx in range(n_rows):
                msg_count += 1

                if msg_count % config.progress_interval == 0:
                    print(
                        f"  {msg_count:>12,} msgs | "
                        f"{signal_count} signals | "
                        f"{len(trades)} trades | "
                        f"pos={'YES' if position else 'no'}"
                    )

                # Parse timestamp
                ts_ns = int(ts_arr[row_idx])
                ts_utc = datetime.fromtimestamp(ts_ns / 1e9, tz=_UTC)
                ts_et = ts_utc.astimezone(_ET)

                # Build L2Snapshot
                bids: list[tuple[float, int]] = []
                asks: list[tuple[float, int]] = []
                for lvl in range(1, 11):
                    bp = bid_px[lvl][row_idx] / 1e9
                    bs = int(bid_sz[lvl][row_idx])
                    ap = ask_px[lvl][row_idx] / 1e9
                    as_ = int(ask_sz[lvl][row_idx])
                    if bp > 0 and bs > 0:
                        bids.append((bp, bs))
                    if ap > 0 and as_ > 0:
                        asks.append((ap, as_))
                snapshot = L2Snapshot(timestamp=ts_et, bids=bids, asks=asks)

                # Check if trade
                act = action_arr[row_idx]
                if act in ("T", "F"):
                    px = price_arr[row_idx] / 1e9
                    sz = int(size_arr[row_idx])
                    if px > 0 and sz > 0:
                        sd = side_arr[row_idx]
                        if sd == "A":
                            aggressor = "buy"
                        elif sd == "B":
                            aggressor = "sell"
                        else:
                            aggressor = "buy"

                        trade_event = TradeEvent(
                            timestamp=ts_et, price=px,
                            size=sz, aggressor=aggressor,
                        )
                        strategy.on_trade(trade_event)
                        trade_count += 1

                        if position is not None:
                            exit_result = self._check_exit(
                                position, trade_event, ts_et
                            )
                            if exit_result is not None:
                                exit_price, exit_reason = exit_result
                                t = self._close_position(
                                    position, exit_price, ts_et,
                                    exit_reason, config,
                                )
                                trades.append(t)
                                position = None

                # Feed L2 to strategy
                signal = strategy.on_l2(snapshot)

                # Check level broken
                if position is not None:
                    level_price = position.metadata.get("level_price", 0.0)
                    level_side = position.metadata.get("level_side", "")
                    for ab_sig in strategy.get_absorption_signals():
                        if (
                            ab_sig.status == "broken"
                            and ab_sig.price == level_price
                            and ab_sig.side == level_side
                        ):
                            if snapshot.bids and snapshot.asks:
                                if position.direction == Direction.LONG:
                                    exit_px = snapshot.bids[0][0]
                                else:
                                    exit_px = snapshot.asks[0][0]
                            else:
                                exit_px = position.entry_price
                            t = self._close_position(
                                position, exit_px, ts_et,
                                "level_broken", config,
                            )
                            trades.append(t)
                            position = None
                            break

                # Check expiry
                if position is not None and ts_et >= position.expiry_time:
                    if snapshot.bids and snapshot.asks:
                        mid = (snapshot.bids[0][0] + snapshot.asks[0][0]) / 2
                    else:
                        mid = position.entry_price
                    t = self._close_position(
                        position, mid, ts_et, "expiry", config,
                    )
                    trades.append(t)
                    position = None

                # Enter new position
                if signal is not None and position is not None:
                    signal_count += 1
                    continue

                if signal is not None and position is None:
                    signal_count += 1
                    slip = config.slippage_ticks * TICK_SIZE
                    if signal.direction == Direction.LONG:
                        fill_price = signal.entry_price + slip
                    else:
                        fill_price = signal.entry_price - slip

                    position = _Position(
                        direction=signal.direction,
                        entry_price=fill_price,
                        entry_time=ts_et,
                        target_price=signal.target_price,
                        stop_price=signal.stop_price,
                        expiry_time=signal.expiry_time,
                        signal_confidence=signal.confidence,
                        metadata=signal.metadata,
                    )

        # Close remaining position
        if position is not None and ts_et is not None:
            t = self._close_position(
                position, position.entry_price, ts_et,
                "end_of_data", config,
            )
            trades.append(t)

        return self._build_result(
            trades, config, msg_count, trade_count, signal_count, 0
        )

    def _run_dbn(self, config: L2ReplayConfig) -> BacktestResult:
        """Run replay from raw DBN files (original path)."""
        import databento as db

        dbn_files = self._find_dbn_files(config.dbn_dir)
        if not dbn_files:
            print(f"No .dbn.zst files found in {config.dbn_dir}")
            return self._empty_result(config)

        print(f"L2 Replay (DBN): {len(dbn_files)} files")
        for f in dbn_files:
            size_gb = os.path.getsize(f) / (1024 ** 3)
            print(f"  {Path(f).name} ({size_gb:.1f} GB)")

        strategy = config.strategy
        trades: list[Trade] = []
        position: _Position | None = None
        msg_count = 0
        trade_count = 0
        signal_count = 0
        skipped_rth = 0
        ts_et = None

        for dbn_path in dbn_files:
            print(f"\nProcessing {Path(dbn_path).name}...")
            store = db.DBNStore.from_file(dbn_path)

            for msg in store:
                msg_count += 1

                if msg_count % config.progress_interval == 0:
                    print(
                        f"  {msg_count:>12,} msgs | "
                        f"{signal_count} signals | "
                        f"{len(trades)} trades | "
                        f"pos={'YES' if position else 'no'}"
                    )

                # Parse timestamp (UTC)
                ts_utc = datetime.fromtimestamp(
                    msg.ts_event / 1e9,
                    tz=_UTC,
                )
                ts_et = ts_utc.astimezone(_ET)

                # RTH filter
                if config.rth_only:
                    t = ts_et.time()
                    if t < RTH_START or t >= RTH_END:
                        skipped_rth += 1
                        continue

                # Build L2Snapshot from 10-level book
                snapshot = self._build_snapshot(msg, ts_et)

                # Check if this message is a trade
                trade_event = self._extract_trade(msg, ts_et)
                if trade_event is not None:
                    strategy.on_trade(trade_event)
                    trade_count += 1

                    # Check position exits on trade
                    if position is not None:
                        exit_result = self._check_exit(
                            position, trade_event, ts_et
                        )
                        if exit_result is not None:
                            exit_price, exit_reason = exit_result
                            t = self._close_position(
                                position, exit_price, ts_et,
                                exit_reason, config,
                            )
                            trades.append(t)
                            position = None

                # Feed L2 to strategy
                signal = strategy.on_l2(snapshot)

                # Check if our level broke (absorption "broken" signal)
                if position is not None:
                    level_price = position.metadata.get("level_price", 0.0)
                    level_side = position.metadata.get("level_side", "")
                    for ab_sig in strategy.get_absorption_signals():
                        if (
                            ab_sig.status == "broken"
                            and ab_sig.price == level_price
                            and ab_sig.side == level_side
                        ):
                            # Our level broke — exit at market
                            if snapshot.bids and snapshot.asks:
                                if position.direction == Direction.LONG:
                                    exit_px = snapshot.bids[0][0]  # sell at bid
                                else:
                                    exit_px = snapshot.asks[0][0]  # buy at ask
                            else:
                                exit_px = position.entry_price
                            t = self._close_position(
                                position, exit_px, ts_et,
                                "level_broken", config,
                            )
                            trades.append(t)
                            position = None
                            break

                # Check expiry (on every message, not just trades)
                if position is not None and ts_et >= position.expiry_time:
                    # Exit at mid price
                    if snapshot.bids and snapshot.asks:
                        mid = (snapshot.bids[0][0] + snapshot.asks[0][0]) / 2
                    else:
                        mid = position.entry_price
                    t = self._close_position(
                        position, mid, ts_et, "expiry", config,
                    )
                    trades.append(t)
                    position = None

                # Enter new position if signal and no current position
                if signal is not None and position is not None:
                    signal_count += 1
                    # Already in position, skip this signal
                    continue

                if signal is not None and position is None:
                    signal_count += 1
                    # Apply entry slippage
                    slip = config.slippage_ticks * TICK_SIZE
                    if signal.direction == Direction.LONG:
                        fill_price = signal.entry_price + slip
                    else:
                        fill_price = signal.entry_price - slip

                    position = _Position(
                        direction=signal.direction,
                        entry_price=fill_price,
                        entry_time=ts_et,
                        target_price=signal.target_price,
                        stop_price=signal.stop_price,
                        expiry_time=signal.expiry_time,
                        signal_confidence=signal.confidence,
                        metadata=signal.metadata,
                    )

        # Close any remaining position at last known price
        if position is not None and ts_et is not None:
            t = self._close_position(
                position, position.entry_price, ts_et,
                "end_of_data", config,
            )
            trades.append(t)

        return self._build_result(
            trades, config, msg_count, trade_count, signal_count, skipped_rth
        )

    def _build_result(
        self,
        trades: list[Trade],
        config: L2ReplayConfig,
        msg_count: int,
        trade_count: int,
        signal_count: int,
        skipped_rth: int,
    ) -> BacktestResult:
        """Print summary and build BacktestResult."""
        print(f"\n{'=' * 70}")
        print(f"L2 Replay Complete")
        print(f"  Messages processed: {msg_count:,}")
        print(f"  Trades in data:     {trade_count:,}")
        if skipped_rth:
            print(f"  Skipped (non-RTH):  {skipped_rth:,}")
        print(f"  Signals generated:  {signal_count}")
        print(f"  Trades taken:       {len(trades)}")

        if trades:
            metrics, equity, daily = MetricsCalculator.from_trades(
                trades, config.initial_capital
            )
        else:
            metrics, equity, daily = MetricsCalculator.from_trades(
                [], config.initial_capital
            )

        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            daily_pnl=daily,
            metrics=metrics,
            config_summary={
                "strategy": "l2_book",
                "dbn_dir": config.dbn_dir,
                "rth_only": config.rth_only,
                "use_parquet": config.use_parquet,
                "slippage_ticks": config.slippage_ticks,
                "commission_per_side": config.commission_per_side,
                "messages_processed": msg_count,
                "signals_generated": signal_count,
            },
        )

    def _build_snapshot(self, msg, ts_et: datetime) -> L2Snapshot:
        """Extract 10-level book from MBP-10 message."""
        bids: list[tuple[float, int]] = []
        asks: list[tuple[float, int]] = []
        for level in msg.levels:
            bid_px = level.bid_px / 1e9
            ask_px = level.ask_px / 1e9
            if bid_px > 0 and level.bid_sz > 0:
                bids.append((bid_px, level.bid_sz))
            if ask_px > 0 and level.ask_sz > 0:
                asks.append((ask_px, level.ask_sz))
        return L2Snapshot(timestamp=ts_et, bids=bids, asks=asks)

    def _extract_trade(
        self, msg, ts_et: datetime
    ) -> TradeEvent | None:
        """Extract trade event if this message is a trade action.

        DataBento MBP-10 action field: 'T' = trade, 'F' = fill.
        Side field: 'A' = ask (buyer aggressor), 'B' = bid (seller aggressor).
        """
        # Check action — accept Trade or Fill
        action = msg.action
        # Handle both enum objects and char values
        action_val = getattr(action, "value", action)
        if action_val not in ("T", "F"):
            return None

        price = msg.price / 1e9
        size = msg.size
        if price <= 0 or size <= 0:
            return None

        # Determine aggressor from side
        side = msg.side
        side_val = getattr(side, "value", side)
        if side_val == "A":
            aggressor = "buy"  # buyer lifts ask
        elif side_val == "B":
            aggressor = "sell"  # seller hits bid
        else:
            aggressor = "buy"  # default

        return TradeEvent(
            timestamp=ts_et,
            price=price,
            size=size,
            aggressor=aggressor,
        )

    def _check_exit(
        self,
        position: _Position,
        trade: TradeEvent,
        ts_et: datetime,
    ) -> tuple[float, str] | None:
        """Check if a trade triggers target or stop exit."""
        price = trade.price

        if position.direction == Direction.LONG:
            if price >= position.target_price:
                return (position.target_price, "target")
            if price <= position.stop_price:
                return (position.stop_price, "stop")
        else:  # SHORT
            if price <= position.target_price:
                return (position.target_price, "target")
            if price >= position.stop_price:
                return (position.stop_price, "stop")

        return None

    def _close_position(
        self,
        position: _Position,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        config: L2ReplayConfig,
    ) -> Trade:
        """Close a position and create a Trade record."""
        # Apply exit slippage
        slip = config.slippage_ticks * TICK_SIZE
        if position.direction == Direction.LONG:
            fill_exit = exit_price - slip
        else:
            fill_exit = exit_price + slip

        # P&L calculation
        if position.direction == Direction.LONG:
            gross_pnl = (fill_exit - position.entry_price) * POINT_VALUE
        else:
            gross_pnl = (position.entry_price - fill_exit) * POINT_VALUE

        commission = config.commission_per_side * 2
        slippage_cost = config.slippage_ticks * 2 * TICK_VALUE
        net_pnl = gross_pnl - commission

        # Duration
        duration = (exit_time - position.entry_time).total_seconds()

        return Trade(
            trade_id=str(uuid.uuid4()),
            strategy_id="l2_book",
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=fill_exit,
            entry_time=position.entry_time,
            exit_time=exit_time,
            size=1,
            gross_pnl=gross_pnl,
            slippage_cost=slippage_cost,
            commission=commission,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            bars_held=int(duration),  # seconds held (no bars in L2)
            entry_slippage_ticks=config.slippage_ticks,
            exit_slippage_ticks=config.slippage_ticks,
            metadata={
                **position.metadata,
                "confidence": position.signal_confidence,
                "hold_seconds": duration,
            },
        )

    def _find_dbn_files(self, dbn_dir: str) -> list[str]:
        """Find all .dbn.zst files recursively."""
        files = []
        for root, _, filenames in os.walk(dbn_dir):
            for f in sorted(filenames):
                if f.endswith(".dbn.zst"):
                    files.append(os.path.join(root, f))
        return files

    def _empty_result(self, config: L2ReplayConfig) -> BacktestResult:
        metrics, equity, daily = MetricsCalculator.from_trades(
            [], config.initial_capital
        )
        return BacktestResult(
            trades=[], equity_curve=equity, daily_pnl=daily,
            metrics=metrics, config_summary={},
        )
