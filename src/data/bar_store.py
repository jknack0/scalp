"""Postgres bar persistence — aggregates live 1s bars into 1m bars and stores them.

Subscribes to BAR events on the EventBus, accumulates 1s bars into 1m OHLCV,
and batch-inserts to Postgres at each minute boundary.  This gives the
auto-retrain pipeline a rolling window of recent market data without
needing to sync parquet files to the VPS.

Schema (created by migrate()):
    bars_1m(timestamp TIMESTAMPTZ, symbol TEXT, open, high, low, close, volume, vwap)
"""

from __future__ import annotations

import os
import time
import threading
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

from src.core.events import BarEvent, EventBus, EventType
from src.core.logging import get_logger

logger = get_logger("bar_store")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS bars_1m (
    timestamp   TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT          NOT NULL,
    vwap        DOUBLE PRECISION,
    PRIMARY KEY (symbol, timestamp)
);
"""

_UPSERT = """
INSERT INTO bars_1m (timestamp, symbol, open, high, low, close, volume, vwap)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (symbol, timestamp) DO UPDATE SET
    open   = EXCLUDED.open,
    high   = GREATEST(bars_1m.high, EXCLUDED.high),
    low    = LEAST(bars_1m.low, EXCLUDED.low),
    close  = EXCLUDED.close,
    volume = bars_1m.volume + EXCLUDED.volume,
    vwap   = EXCLUDED.vwap;
"""


class BarStore:
    """Accumulates 1s bars into 1m bars and persists to Postgres.

    Usage:
        store = BarStore(bus, dsn=os.environ["DATABASE_URL"])
        bus.subscribe(EventType.BAR, store.on_bar)
    """

    def __init__(
        self,
        event_bus: EventBus,
        dsn: str | None = None,
    ) -> None:
        self._bus = event_bus
        self._dsn = dsn or os.environ.get("DATABASE_URL", "")

        # Current 1m accumulator: keyed by (symbol, minute_ts)
        self._accum: dict[tuple[str, int], _MinuteAccum] = {}
        self._lock = threading.Lock()

        # Connect and migrate
        self._conn: psycopg2.extensions.connection | None = None
        if self._dsn:
            try:
                self._conn = psycopg2.connect(self._dsn)
                self._conn.autocommit = True
                self._migrate()
                logger.info("bar_store_connected", dsn=self._dsn.split("@")[-1])
            except Exception as e:
                logger.error("bar_store_connect_failed", error=str(e))
                self._conn = None

    def _migrate(self) -> None:
        """Create the bars_1m table if it doesn't exist."""
        if self._conn is None:
            return
        with self._conn.cursor() as cur:
            cur.execute(_CREATE_TABLE)
        logger.info("bar_store_migrated")

    async def on_bar(self, bar: BarEvent) -> None:
        """Accumulate a 1s bar into the current 1m bucket."""
        if self._conn is None:
            return

        # Compute minute boundary from bar timestamp
        bar_ts = bar.timestamp_ns / 1e9
        minute_ts = int(bar_ts // 60) * 60

        key = (bar.symbol, minute_ts)

        with self._lock:
            if key not in self._accum:
                # Check if previous minute needs flushing
                self._flush_completed(bar.symbol, minute_ts)
                self._accum[key] = _MinuteAccum(
                    symbol=bar.symbol,
                    minute_ts=minute_ts,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    vwap_num=bar.close * bar.volume,
                    vwap_den=bar.volume,
                )
            else:
                acc = self._accum[key]
                acc.high = max(acc.high, bar.high)
                acc.low = min(acc.low, bar.low)
                acc.close = bar.close
                acc.volume += bar.volume
                acc.vwap_num += bar.close * bar.volume
                acc.vwap_den += bar.volume

    def _flush_completed(self, symbol: str, current_minute: int) -> None:
        """Flush any completed minute bars (prior to current_minute) to Postgres."""
        to_flush = []
        to_remove = []
        for key, acc in self._accum.items():
            if key[0] == symbol and key[1] < current_minute:
                to_flush.append(acc)
                to_remove.append(key)

        for key in to_remove:
            del self._accum[key]

        if to_flush:
            self._write_bars(to_flush)

    def _write_bars(self, bars: list[_MinuteAccum]) -> None:
        """Insert completed 1m bars into Postgres (sync, called from lock)."""
        if self._conn is None or not bars:
            return

        rows = []
        for acc in bars:
            ts = datetime.fromtimestamp(acc.minute_ts, tz=timezone.utc)
            vwap = acc.vwap_num / acc.vwap_den if acc.vwap_den > 0 else acc.close
            rows.append((
                ts, acc.symbol,
                acc.open, acc.high, acc.low, acc.close,
                acc.volume, round(vwap, 4),
            ))

        try:
            with self._conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, _UPSERT, rows)
            logger.debug("bar_store_flushed", count=len(rows))
        except Exception as e:
            logger.error("bar_store_write_failed", error=str(e))
            # Try to reconnect
            try:
                self._conn = psycopg2.connect(self._dsn)
                self._conn.autocommit = True
            except Exception:
                self._conn = None

    def flush_all(self) -> None:
        """Flush all remaining accumulators (call on shutdown)."""
        with self._lock:
            all_bars = list(self._accum.values())
            self._accum.clear()
        if all_bars:
            self._write_bars(all_bars)
            logger.info("bar_store_shutdown_flush", count=len(all_bars))

    def close(self) -> None:
        """Flush and close the connection."""
        self.flush_all()
        if self._conn:
            self._conn.close()
            self._conn = None


class _MinuteAccum:
    """Working state for a 1m bar being built from 1s bars."""

    __slots__ = (
        "symbol", "minute_ts", "open", "high", "low", "close",
        "volume", "vwap_num", "vwap_den",
    )

    def __init__(
        self,
        symbol: str,
        minute_ts: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        vwap_num: float,
        vwap_den: int,
    ) -> None:
        self.symbol = symbol
        self.minute_ts = minute_ts
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.vwap_num = vwap_num
        self.vwap_den = vwap_den
