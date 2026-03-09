# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MES micro E-mini S&P 500 futures scalping bot. Event-driven async Python architecture targeting Tradovate API (REST + WebSocket). Phases 1-5 complete; live paper trading operational.

## Commands

```bash
# Install dependencies
uv sync

# Run the bot (paper trading, ORB strategy, 5s bars)
python main.py

# Run the bot (live trading, specific strategy)
python main.py --live --strategy orb --bar-interval 5

# Run all tests (291 tests)
python -m pytest -v

# Run a single test file
python -m pytest tests/test_risk_manager.py -v

# Data pipeline (CSV -> Parquet + validation)
python scripts/run_pipeline.py --all --csv path/to/mes_1s.csv

# CPCV validation
python scripts/run_cpcv.py --strategy orb --start 2020-01-01 --end 2024-01-01

# DSR validation
python scripts/run_dsr.py --strategy orb --start 2020-01-01 --end 2024-01-01

# Walk-forward analysis
python scripts/run_wfa.py --strategy orb --start 2020-01-01 --end 2024-01-01

# Train HMM regime classifier
python scripts/train_hmm.py --parquet-dir data/parquet --output models/hmm.pkl

# 3-tier tuning pipeline
python scripts/tune_strategy.py --strategy orb                          # Tier 1: strategy params (15yr)
python scripts/tune_strategy.py --strategy orb --start 2020-01-01       # Tier 1: with date filter
python scripts/tune_l1_filters.py --strategy orb --tier1 results/orb/tier1_strategy_*.json  # Tier 2: L1 filters (1yr)
python scripts/tune_l2_filters.py --strategy orb --tier2 results/orb/tier2_l1_filters_*.json  # Tier 3: L2 filters (3mo)
python scripts/convert_l2_parquet.py                                    # prereq for Tier 3
```

No linter or formatter is configured yet.

## Architecture

**Event-driven async bot** — 7 concurrent asyncio tasks (EventBus, SessionManager, HealthMonitor, TickAggregator, FillMonitor, TradovateFeed, signal handling) orchestrated in `main.py` via `asyncio.gather`.

### Data flow

```
TradovateFeed (WebSocket) -> TickEvent -> EventBus
  -> TickAggregator -> BarEvent -> EventBus
    -> SignalHandler -> Strategy.on_bar() -> Signal
      -> RiskManager.check_order()
        -> TradovateOMS.submit_order()
          -> FillMonitor -> FillEvent -> RiskManager.record_fill()
```

### Core loop

`EventBus` (`src/core/events.py`) is the central pub/sub system. Components communicate through typed events (TICK, BAR, SIGNAL, FILL, RISK) using asyncio queues. Subscribers register callbacks per event type; exceptions in one subscriber don't crash others.

### Configuration cascade

`BotConfig` (`src/core/config.py`) uses Pydantic Settings with load order: field defaults -> `config/bot-config.yaml` -> `.env` -> environment variables. Credentials live in `.env` (never committed); runtime tuning lives in the YAML.

### Strategies (2 surviving)

- **ORB** (`src/strategies/orb_strategy.py`) — Opening range breakout. State machine: WAITING -> COLLECTING -> WATCHING -> SIGNAL/INACTIVE. Volume/VWAP/HMM/time filters.
- **VWAP** (`src/strategies/vwap_strategy.py`) — VWAP reversion/pullback. Three modes: REVERSION, PULLBACK, NEUTRAL. First-kiss confidence boost.

Killed strategies (kept for reference): CVD Divergence (too few trades), Vol Regime (all configs negative).

All strategies inherit from `StrategyBase` (`src/strategies/base.py`) which provides session gating, HMM regime checks, and signal construction.

### Feature library

`FeatureHub` (`src/features/feature_hub.py`) composes all calculators and produces a `FeatureVector` (15 features):
- **VWAP**: session VWAP, deviation SD, slope, flatness
- **ATR**: rolling ATR, vol regime (LOW/NORMAL/HIGH), semi-variance up/down
- **CVD**: cumulative volume delta, slope, z-score (from L1 TBBO trade classification)
- **Volume Profile**: POC distance, price-above-POC, proximity flag

Additional feature modules (not in FeatureVector but available):
- `order_book_imbalance.py` — OBI raw/smoothed/z-score from L1 bars (kept as filter, not a strategy)
- `l2_book.py` — L2 weighted mid, depth deterioration, absorption, spoof scoring

### Filters (`src/filters/`)

Filters gate trade entry signals — they don't generate signals, they block bad ones. Organized by data tier:

**L1 filters** (use tick/bar data, available for 1+ year backtest):
- `SpreadMonitor` (`spread_monitor.py`) — blocks when bid-ask spread z-score exceeds threshold (wide spread = bad fill). Config: `SpreadConfig(z_threshold=2.0)`.
- `VPINMonitor` (`vpin_monitor.py`) — volume-synchronized probability of informed trading. BVC bar approximation. Blocks trending trades when VPIN high, mean-reversion when low. Config: `VPINConfig(trending_threshold=0.55, mean_reversion_threshold=0.38)`.

**L2 filters** (use order book depth, available for ~3 months of MBP-10 data):
- `DepthMonitor` (`depth_monitor.py`) — blocks when liquidity is thinning on the trade side. Config: `DepthConfig(thin_threshold=0.6)`.
- `WeightedMidMonitor` (`weighted_mid.py`) — blocks when size-weighted mid leans against signal direction. Config: `WeightedMidConfig(lean_threshold=1.0)`.
- `MidMomentumMonitor` (`mid_momentum.py`) — blocks when mid-price drift contradicts signal direction. Linear regression slope normalized by stddev. Config: `MomentumConfig(neutral_threshold=0.3)`.
- `IcebergDetector` (`iceberg_absorption.py`) — detects hidden size replenishment at a level (iceberg orders).
- `AbsorptionDetector` (`iceberg_absorption.py`) — detects aggressive volume being absorbed by passive resting orders.
- `HiddenLiquidityDetector` / `HiddenLiquidityMap` (`hidden_liquidity.py`) — tracks levels where hidden liquidity has been observed.
- `QuoteFadeDetector` (`quote_fade.py`) — detects levels being pulled (quote stuffing/fading).

All filters provide `push()` (async, with Parquet persistence) and `push_sync()` (synchronous, for backtesting). L1 and L2 filters are wired into `BacktestEngine` via `BacktestConfig` fields (`vpin_monitor`, `spread_monitor`, `depth_monitor`, `weighted_mid_monitor`, `mid_momentum_monitor`).

### 3-tier tuning pipeline

Strategy parameters and filters are tuned separately on data windows matched to their depth:

1. **Tier 1** (`scripts/tune_strategy.py`) — Strategy params on 15yr 1s bars resampled to 60s. No filters. Ranked by composite score `sharpe × √trades` (min 50 trades). Output: `results/<strategy>/tier1_strategy_*.json`.
2. **Tier 2** (`scripts/tune_l1_filters.py`) — L1 filter combos (none/VPIN/spread/both + thresholds) on 1yr L1 data. Locks strategy params from Tier 1. Output: `results/<strategy>/tier2_l1_filters_*.json`.
3. **Tier 3** (`scripts/tune_l2_filters.py`) — L2 filter combos (depth/weighted_mid/momentum + thresholds) on 3mo L2 data. Locks strategy + L1 filters from Tier 2. Output: `results/<strategy>/tier3_l2_filters_*.json`.

Prerequisite for L2: `python scripts/convert_l2_parquet.py` (converts DataBento .dbn.zst to Parquet).

### Broker-agnostic adapters

Feed and OMS layers use abstract base classes (`src/feeds/base.py`, `src/oms/base.py`). Concrete Tradovate implementations are swappable. Strategy code consumes typed events, never broker-specific types.

### Risk manager

`RiskManager` (`src/risk/risk_manager.py`) runs sequential pre-trade checks: halted -> session valid -> position limit -> signal limit. Tracks daily P&L, position sizes, and fill history. Resets on session open.

### Session manager

`SessionManager` (`src/core/session.py`) tracks RTH 9:30 AM-4:00 PM ET, detects weekends, fires session open/close callbacks. Polls every 30 seconds.

### Live trading pipeline

- **Paper mode** (default): `python main.py` — instant fills at signal price, target/stop monitored tick-by-tick, $0.35/side commission applied.
- **Live mode**: `python main.py --live` — orders submitted via Tradovate REST API, FillMonitor polls order status every 2s.
- `SignalHandler` (`src/core/signal_handler.py`) bridges strategies to risk to OMS.
- `TickAggregator` (`src/core/tick_aggregator.py`) converts tick stream to time-based bars with L1 approximation.
- `FillMonitor` (`src/oms/fill_monitor.py`) tracks order lifecycle.

### Backtesting & validation

- `BacktestEngine` (`src/backtesting/engine.py`) — bar-replay simulator with SimulatedOMS and volatility slippage model.
- `CPCVValidator` (`src/backtesting/cpcv.py`) — combinatorial purged cross-validation with PBO.
- `DeflatedSharpeCalculator` (`src/backtesting/dsr.py`) — PSR + DSR for multiple testing correction.
- `WFARunner` (`src/backtesting/wfa.py`) — rolling walk-forward with efficiency ratio and param drift.
- `DecisionEngine` (`src/backtesting/decision_engine.py`) — 4-gate strategy approval (expectancy, Sharpe, DSR, correlation).

### Data pipeline

`src/data/` handles historical data: CSV -> year-partitioned Parquet (Polars + zstd). `src/data/bars.py` provides bar resampling (1s -> any timeframe) and dollar bar construction. `src/data/quality.py` validates gaps, duplicates, and outliers. Primary data source: `data/parquet/year=YYYY/data.parquet` (~1 GB, 15 years).

### Health monitoring

FastAPI endpoint (`src/monitoring/health.py`) at `GET /health` returns position, daily P&L, last tick age, event counts, and uptime. Runs as a background uvicorn task on port 8080.

### Logging

Structured logging via `structlog` (`src/core/logging.py`). Dual output: JSON rotating file (10 MB, 5 backups) + colored console. Noisy library loggers (uvicorn, fastapi) are suppressed.

## Key conventions

- Python 3.13, managed with `uv`
- Polars (not pandas) for all DataFrame operations
- All event dataclasses are frozen (immutable)
- Async-first: use `async def` and `await` for I/O-bound work
- Tests use `pytest-asyncio` with `asyncio_mode = "auto"` -- no need for `@pytest.mark.asyncio`
- MES tick size: $1.25 per tick (0.25 index points x $5 multiplier)
- Commission: $0.35/side (Tradovate Free plan)
- 291 tests across 32 test files -- run `python -m pytest -v` to verify
