-- MES 1-Second Bar Schema for Supabase Postgres
-- Run this manually if needed, or let db_loader.py create it automatically.

-- Main table: 1-second bars
CREATE TABLE IF NOT EXISTS mes_bars_1s (
    timestamp   TIMESTAMPTZ NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT NOT NULL,
    vwap        DOUBLE PRECISION
);

-- Primary lookup index
CREATE INDEX IF NOT EXISTS idx_mes_bars_1s_ts
    ON mes_bars_1s (timestamp);

-- Composite index for price queries
CREATE INDEX IF NOT EXISTS idx_mes_bars_1s_ts_close
    ON mes_bars_1s (timestamp, close);

-- Pre-aggregated 1-minute bars (materialized view)
-- Refresh after loading new data: REFRESH MATERIALIZED VIEW mes_bars_1m;
CREATE MATERIALIZED VIEW IF NOT EXISTS mes_bars_1m AS
SELECT
    date_trunc('minute', timestamp) AS timestamp,
    (array_agg(open ORDER BY timestamp))[1] AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    (array_agg(close ORDER BY timestamp DESC))[1] AS close,
    SUM(volume) AS volume,
    SUM(vwap * volume) / NULLIF(SUM(volume), 0) AS vwap
FROM mes_bars_1s
GROUP BY date_trunc('minute', timestamp)
ORDER BY timestamp;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mes_bars_1m_ts
    ON mes_bars_1m (timestamp);

-- Useful queries:
--
-- Get 1 day of 1s bars:
--   SELECT * FROM mes_bars_1s
--   WHERE timestamp >= '2025-01-06' AND timestamp < '2025-01-07'
--   ORDER BY timestamp;
--
-- Get 1-minute bars for a date range:
--   SELECT * FROM mes_bars_1m
--   WHERE timestamp >= '2025-01-06' AND timestamp < '2025-01-10'
--   ORDER BY timestamp;
--
-- Daily OHLCV summary:
--   SELECT
--     DATE(timestamp) as date,
--     (array_agg(open ORDER BY timestamp))[1] AS open,
--     MAX(high) AS high,
--     MIN(low) AS low,
--     (array_agg(close ORDER BY timestamp DESC))[1] AS close,
--     SUM(volume) AS volume
--   FROM mes_bars_1s
--   WHERE timestamp >= '2025-01-01'
--   GROUP BY DATE(timestamp)
--   ORDER BY date;
--
-- Row count by year:
--   SELECT EXTRACT(YEAR FROM timestamp) AS year, COUNT(*) AS rows
--   FROM mes_bars_1s
--   GROUP BY year ORDER BY year;
