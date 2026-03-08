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

# Run all tests (284 tests)
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

### Strategies (4 active)

- **ORB** (`src/strategies/orb_strategy.py`) — Opening range breakout. State machine: WAITING -> COLLECTING -> WATCHING -> SIGNAL/INACTIVE. Volume/VWAP/HMM/time filters.
- **VWAP** (`src/strategies/vwap_strategy.py`) — VWAP reversion/pullback. Three modes: REVERSION, PULLBACK, NEUTRAL. First-kiss confidence boost.
- **CVD** (`src/strategies/cvd_divergence_strategy.py`) — CVD divergence + POC proximity. Swing-point detection with price/CVD divergence.
- **Vol Regime** (`src/strategies/vol_regime_strategy.py`) — Volatility regime switcher. HIGH_VOL momentum pullbacks, LOW_VOL VWAP fades, TRANSITIONING no-trade.

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

### L2 / microstructure filters

`src/filters/` contains 8 order book analysis modules: depth monitoring, weighted mid, iceberg/absorption detection, hidden liquidity, quote fade, OBI monitoring, mid momentum, spread regime. These are available as gating filters for strategies but not yet wired into the live pipeline.

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
- 284 tests across 32 test files -- run `python -m pytest -v` to verify
