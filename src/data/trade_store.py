"""Postgres trade persistence — records full round-trip trades.

Captures signal submission (entry) and bracket exit (target/stop/expiry)
into a single `trades` table for performance tracking and analysis.

Two-phase write:
1. record_entry() — called when a signal is submitted to OMS
2. record_exit()  — called when the bracket closes (target/stop/expiry)

Schema (created by migrate()):
    trades(order_id PK, strategy_id, symbol, direction, entry/exit prices+times,
           exit_reason, pnl, commission, regime, confidence, metadata JSONB)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import psycopg2

from src.core.logging import get_logger

logger = get_logger("trade_store")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    order_id        TEXT PRIMARY KEY,
    strategy_id     TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,

    -- Entry
    entry_price     DOUBLE PRECISION NOT NULL,
    entry_time      TIMESTAMPTZ NOT NULL,
    target_price    DOUBLE PRECISION,
    stop_price      DOUBLE PRECISION,
    confidence      DOUBLE PRECISION,
    risk_reward     DOUBLE PRECISION,
    regime          TEXT,

    -- Exit (NULL until trade closes)
    exit_price      DOUBLE PRECISION,
    exit_time       TIMESTAMPTZ,
    exit_reason     TEXT,

    -- P&L
    pnl_ticks       DOUBLE PRECISION,
    pnl_usd         DOUBLE PRECISION,
    commission      DOUBLE PRECISION,

    -- Duration
    duration_seconds DOUBLE PRECISION,

    -- Strategy-specific metadata (JSONB for flexible querying)
    metadata        JSONB,

    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy_id);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades (entry_time);
"""

_INSERT_ENTRY = """
INSERT INTO trades (
    order_id, strategy_id, symbol, direction,
    entry_price, entry_time, target_price, stop_price,
    confidence, risk_reward, regime, metadata
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (order_id) DO NOTHING;
"""

_UPDATE_EXIT = """
UPDATE trades SET
    exit_price = %s,
    exit_time = %s,
    exit_reason = %s,
    pnl_ticks = %s,
    pnl_usd = %s,
    commission = %s,
    duration_seconds = %s
WHERE order_id = %s;
"""


class TradeStore:
    """Persists round-trip trades to Postgres."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or os.environ.get("DATABASE_URL", "")
        self._conn: psycopg2.extensions.connection | None = None

        if self._dsn:
            try:
                self._conn = psycopg2.connect(self._dsn)
                self._conn.autocommit = True
                self._migrate()
                logger.info("trade_store_connected", dsn=self._dsn.split("@")[-1])
            except Exception as e:
                logger.error("trade_store_connect_failed", error=str(e))
                self._conn = None

    def _migrate(self) -> None:
        if self._conn is None:
            return
        with self._conn.cursor() as cur:
            cur.execute(_CREATE_TABLE)
        logger.info("trade_store_migrated")

    def _ensure_conn(self) -> bool:
        """Reconnect if needed. Returns True if connected."""
        if self._conn is not None:
            try:
                self._conn.cursor().execute("SELECT 1")
                return True
            except Exception:
                pass
        try:
            self._conn = psycopg2.connect(self._dsn)
            self._conn.autocommit = True
            return True
        except Exception:
            self._conn = None
            return False

    def record_entry(
        self,
        order_id: str,
        strategy_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        entry_time: datetime,
        target_price: float,
        stop_price: float,
        confidence: float,
        risk_reward: float,
        regime: str,
        metadata: dict | None = None,
    ) -> None:
        """Record a new trade entry (signal submitted to OMS)."""
        if not self._ensure_conn():
            return

        meta_json = json.dumps(metadata) if metadata else None

        try:
            with self._conn.cursor() as cur:  # type: ignore[union-attr]
                cur.execute(_INSERT_ENTRY, (
                    order_id, strategy_id, symbol, direction,
                    entry_price, entry_time, target_price, stop_price,
                    confidence, risk_reward, regime, meta_json,
                ))
            logger.info("trade_entry_recorded", order_id=order_id,
                        strategy=strategy_id, direction=direction,
                        entry=entry_price)
        except Exception as e:
            logger.error("trade_entry_failed", order_id=order_id, error=str(e))

    def record_exit(
        self,
        order_id: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        pnl_ticks: float,
        pnl_usd: float,
        commission: float,
        duration_seconds: float,
    ) -> None:
        """Record trade exit (bracket closed)."""
        if not self._ensure_conn():
            return

        try:
            with self._conn.cursor() as cur:  # type: ignore[union-attr]
                cur.execute(_UPDATE_EXIT, (
                    exit_price, exit_time, exit_reason,
                    pnl_ticks, pnl_usd, commission,
                    duration_seconds, order_id,
                ))
            logger.info("trade_exit_recorded", order_id=order_id,
                        exit_reason=exit_reason, pnl_usd=round(pnl_usd, 2))
        except Exception as e:
            logger.error("trade_exit_failed", order_id=order_id, error=str(e))

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
