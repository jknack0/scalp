# Data Pipeline Guide — MES 1-Second Bars

> Phase 1, Task 4 | Last updated: 2026-03-03

---

## Overview

This pipeline takes your existing 15-year MES 1-second bar CSV and makes it queryable for backtesting:

```
CSV (5+ GB)  →  Parquet (partitioned by year)  →  Supabase Postgres (recent 3 years)
```

- **Parquet files**: Full 15 years, compressed (~70% smaller), fast to read with polars
- **Supabase**: Recent 3 years in Postgres for SQL queries and strategy backtesting
- **Databento MBO**: Not needed yet — add later for CVD aggressor-side data

## Prerequisites

Install dependencies:

```bash
uv sync
```

## Pipeline Steps

### Step 1: Convert CSV → Parquet

```bash
uv run python scripts/run_pipeline.py --convert --csv C:\Dev\bayesian\data\ESMES_15yr_databento.csv
```

This reads the CSV in chunks (handles 5+ GB without blowing up memory), partitions by year, and writes compressed Parquet files to `data/parquet/`:

```
data/parquet/
├── year=2011/data.parquet
├── year=2012/data.parquet
├── ...
└── year=2026/data.parquet
```

Idempotent — skips years that are already converted.

### Step 2: Load into Postgres

Set your database connection string:

```bash
export DATABASE_URL='postgresql://botuser:PASSWORD@localhost:5432/scalp'
```

Load the recent 3 years:

```bash
uv run python scripts/run_pipeline.py --load --start-year 2023 --end-year 2026
```

This creates the `mes_bars_1s` table, bulk-inserts with `COPY`, and builds a materialized `mes_bars_1m` view.

### Step 3: Validate

```bash
uv run python scripts/run_pipeline.py --validate --start-year 2023 --end-year 2026
```

Checks for:
- Gaps > 30s during RTH (9:30-16:00 ET)
- Duplicate timestamps
- Price outliers (> 5 std from rolling mean)
- Missing trading days

### Run Everything

```bash
uv run python scripts/run_pipeline.py --all --csv path/to/mes_1s_bars.csv --start-year 2023 --end-year 2026
```

---

## Querying Data

### From Supabase (SQL)

```sql
-- 1 day of 1s bars
SELECT * FROM mes_bars_1s
WHERE timestamp >= '2025-01-06' AND timestamp < '2025-01-07'
ORDER BY timestamp;

-- 1-minute bars (pre-aggregated, fast)
SELECT * FROM mes_bars_1m
WHERE timestamp >= '2025-01-06' AND timestamp < '2025-01-10'
ORDER BY timestamp;

-- Row count by year
SELECT EXTRACT(YEAR FROM timestamp) AS year, COUNT(*) AS rows
FROM mes_bars_1s
GROUP BY year ORDER BY year;
```

### From Parquet (Python — for older data or local analysis)

```python
from src.data.csv_to_parquet import read_parquet_range
from src.data.bars import resample_bars, build_dollar_bars

# Read 2 years of 1s bars
df = read_parquet_range("data/parquet", 2024, 2025)

# Resample to 5-minute bars
bars_5m = resample_bars(df, "5m")

# Build dollar bars ($50k threshold)
dollar_bars = build_dollar_bars(df, dollar_threshold=50_000)
```

---

## Storage Estimates

| What | Size |
|---|---|
| Raw CSV | 5+ GB |
| Parquet (all 15 years) | ~1.5-2 GB |
| Postgres (3 years) | ~1-2 GB |
| Postgres indexes | ~0.5-1 GB |
| **Total Supabase usage** | **~2-3 GB** (within 8 GB Pro limit) |

---

## Adding Databento MBO Data Later

When you're ready to build the CVD Divergence strategy (Phase 3), you'll need aggressor-side trade data that 1s bars don't have. At that point:

1. Sign up at [databento.com](https://databento.com)
2. Pull MES MBO data for the GLBX.MDP3 dataset (6-12 months is enough)
3. The `src/data/databento_downloader.py` stub is ready to be filled in
4. Estimated cost: $50-100 for 6 months of MES MBO data

---

## Exit Criteria

- [ ] CSV converted to Parquet (all 15 years)
- [ ] Recent 3 years loaded into Supabase
- [ ] Quality validation passes (or known issues documented)
- [ ] Can query 1 day of bars from Supabase in < 2 seconds
