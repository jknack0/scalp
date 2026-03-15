# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MES micro E-mini S&P 500 futures scalping bot. Event-driven async Python architecture targeting Tradovate API (REST + WebSocket). Phases 1-5 complete; live paper trading operational.

## Commands

Always use `python -u` (unbuffered) when running scripts so output streams in real time.

```bash
# Install dependencies
uv sync

# Run the bot (paper trading, 1s bars)
python main.py

# Run the bot (live trading, specific strategy)
python main.py --live --strategy orb

# Run all tests
python -m pytest -v

# Run a single test file
python -m pytest tests/test_risk_manager.py -v

# Data pipeline (CSV -> Parquet + validation)
python scripts/build/pipeline.py --all --csv path/to/mes_1s.csv

# Build bars
python scripts/build/bars.py --freq 5s --rth          # 5s RTH bars from 1s source

# CPCV validation
python scripts/backtest/cpcv.py --strategy orb --start 2020-01-01 --end 2024-01-01

# DSR validation
python scripts/backtest/dsr.py --strategy orb --start 2020-01-01 --end 2024-01-01

# Walk-forward analysis
python scripts/backtest/wfa.py --strategy orb --start 2020-01-01 --end 2024-01-01

# Train HMM regime classifier
python scripts/train/hmm.py --parquet-dir data/parquet --output models/hmm.pkl

# 2-tier tuning pipeline (reads strategy YAML for bar freq, signals, session times)
python scripts/tune/strategy.py --strategy orb                              # Tier 1: strategy params (15yr)
python scripts/tune/strategy.py --strategy orb --start 2020-01-01           # Tier 1: with date filter
python scripts/tune/filters.py --strategy orb --tier1 results/orb/tier1_strategy_*.json  # Tier 2: filters (1yr)

# VPS deployment (Vultr Chicago, alias "bot" in ~/.ssh/config)
ssh bot                                              # SSH to VPS
scp .env bot:/opt/mes-bot/.env                       # Copy credentials to VPS
ssh bot 'cd /opt/mes-bot && sudo -u botuser git pull && sudo systemctl restart mes-bot'
ssh bot 'journalctl -u mes-bot -f'                   # Tail live logs
ssh bot 'curl -s localhost:8080/health'              # Health check
scp bot:/opt/mes-bot/logs/bot.log ./logs/vps-bot.log # Pull logs for analysis
```

No linter or formatter is configured yet.

## Mandatory checks

### Before running any backtest or sweep
1. **Read the strategy's YAML** (`config/strategies/*.yaml`) and match the `bar:` section to the BacktestConfig. If YAML says `type: dollar`, build dollar bars. If YAML says `interval_seconds: 300`, resample to 5m. Never default to raw 1s bars without checking.
2. **Verify signal geometry** after writing or modifying entry/target/stop logic:
   - LONG: `stop < entry < target`
   - SHORT: `stop > entry > target`
   Add an assertion or print-check before trusting backtest results. If WR > 90% or Sharpe > 10, something is almost certainly broken — investigate before reporting results.
3. **Check the bar cache** (`data/<name>/<name>.parquet`) matches what the strategy needs. Dollar bars, L1-enriched bars, and time-resampled bars are different things.

### Before modifying strategy geometry (entry/target/stop formulas)
1. **Draw the number line** — write out concrete values (e.g. "LONG entry=5600, VWAP=5610, SD=5, stop_sd=2 → stop = 5610 - (2+2)*5 = 5590 ✓ below entry") before coding.
2. **Test both directions** — LONG and SHORT must both have stops on the adverse side.
3. **Run a single-trade sanity check** with logging before sweeping 100+ configs.

## Script organization

```
scripts/
  build/        — data prep: bar resampling, Parquet conversion, pipelines
  download/     — data acquisition: L1, L2, Tradovate history
  tune/         — optimization: strategy params (Tier 1), filters (Tier 2)
  backtest/     — backtests & validation: CPCV, DSR, WFA, combined runs
  test/         — manual integration tests: feeds, orders, strategy smoke
  train/        — model training: HMM regime classifier
  report/       — report generation
  infra/        — VPS provisioning & deployment
```

## Architecture

**Event-driven async bot** — 7 concurrent asyncio tasks (EventBus, SessionManager, HealthMonitor, TickAggregator, FillMonitor, DatabentoFeed, signal handling) orchestrated in `main.py` via `asyncio.gather`.

### Data flow

```
DatabentoFeed (MBP-1 callback) -> TickEvent -> EventBus
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

- **ORB** (`src/strategies/orb.py`) — Opening range breakout / double-break fade. State machine: WAITING -> COLLECTING -> WATCHING -> SIGNAL/INACTIVE. Volume/VWAP/time filters.
- **VWAP** (`src/strategies/vwap_reversion.py`) — VWAP reversion/pullback. Three modes: REVERSION, PULLBACK, NEUTRAL. First-kiss confidence boost.

Killed strategies: CVD Divergence (too few trades), Vol Regime (all configs negative).

### Signal → Filter → Strategy pipeline

Each strategy's YAML config (`config/strategies/*.yaml`) declares:
- `bar:` — bar type and frequency (e.g. 5s time bars for ORB, dollar bars for VWAP)
- `signals:` — list of signal names to pre-compute (e.g. `[atr, vwap_session, spread]`)
- `filters:` — declarative filter rules (e.g. `spread: "< 2.0"`)

`SignalEngine` computes all declared signals per bar window, `FilterEngine` evaluates YAML rules against the `SignalBundle`, strategies only run if filters pass.

### Filters (`src/filters/`)

Filters gate trade entry signals via `FilterEngine` (declarative YAML expressions).

**Available filter signals:**
- `SpreadMonitor` (`spread_monitor.py`) — bid-ask spread z-score. Config: `SpreadConfig(z_threshold=2.0)`.
- `VPINMonitor` (`vpin_monitor.py`) — volume-synchronized probability of informed trading.

### 2-tier tuning pipeline

Strategy parameters and filters are tuned separately:

1. **Tier 1** (`scripts/tune/strategy.py`) — Strategy params on 15yr bars. Reads YAML for bar freq + signals. No filters. Ranked by `sharpe × √trades`. Output: `results/<strategy>/tier1_strategy_*.json`.
2. **Tier 2** (`scripts/tune/filters.py`) — Filter threshold sweep on 1yr enriched data via FilterEngine. Locks strategy params from Tier 1. Output: `results/<strategy>/tier2_filters_*.json`.

### Broker-agnostic adapters

Feed and OMS layers use abstract base classes (`src/feeds/base.py`, `src/oms/base.py`). Concrete Tradovate implementations are swappable. Strategy code consumes typed events, never broker-specific types.

### Risk manager

`RiskManager` (`src/risk/risk_manager.py`) runs sequential pre-trade checks: halted -> session valid -> position limit -> signal limit. Tracks daily P&L, position sizes, and fill history. Resets on session open.

### Session manager

`SessionManager` (`src/core/session.py`) tracks RTH 9:30 AM-4:00 PM ET, detects weekends, fires session open/close callbacks. Polls every 30 seconds.

### Live trading pipeline

- **Market data**: `DatabentoFeed` (`src/feeds/databento_feed.py`) — MBP-1 top-of-book via Databento Live. Callback API with thread-safe publish. 30s heartbeat logging.
- **Paper mode** (default): `python main.py` — bracket orders (entry/target/stop/expiry), tick-by-tick monitoring, $0.295/side ($0.59 RT) commission.
- **Live mode**: `python main.py --live` — orders submitted via Tradovate REST API, FillMonitor polls order status every 2s.
- `SignalHandler` (`src/core/signal_handler.py`) bridges strategies to risk to OMS.
- `TickAggregator` (`src/core/tick_aggregator.py`) converts tick stream to 1s bars with L1 approximation.
- `FillMonitor` (`src/oms/fill_monitor.py`) tracks order lifecycle.
- **VPS**: Vultr Chicago (`ssh bot`), systemd service `mes-bot`, logs at `/opt/mes-bot/logs/bot.log`.

### Backtesting & validation

- `BacktestEngine` (`src/backtesting/engine.py`) — bar-replay simulator with SimulatedOMS and volatility slippage model.
- `CPCVValidator` (`src/backtesting/cpcv.py`) — combinatorial purged cross-validation with PBO.
- `DeflatedSharpeCalculator` (`src/backtesting/dsr.py`) — PSR + DSR for multiple testing correction.
- `WFARunner` (`src/backtesting/wfa.py`) — rolling walk-forward with efficiency ratio and param drift.
- `DecisionEngine` (`src/backtesting/decision_engine.py`) — 4-gate strategy approval (expectancy, Sharpe, DSR, correlation).

### Data pipeline

`src/data/` handles historical data: CSV -> year-partitioned Parquet (Polars + zstd). `src/data/bars.py` provides bar resampling (1s -> any timeframe) and dollar bar construction. `src/data/quality.py` validates gaps, duplicates, and outliers.

**Canonical data directories:**
```
data/
  l1/                              — raw L1 tick data (year-partitioned)
  l2/                              — raw L2 order book data
  parquet/year=YYYY/data.parquet   — 1s OHLCV bars (~1 GB, 15 years)
  parquet_1m/year=YYYY/data.parquet — 1m OHLCV bars
  bars_<name>/bars_<name>.parquet  — persistent bar cache (auto-generated)
  enriched_<name>/enriched_<name>.parquet — cached enriched bars with signal columns
```

**Bar cache** (`src/data/bar_cache.py`): `BacktestEngine._load_bars` and `_load_l1_bars` check for cached files before rebuilding. First run builds and caches; subsequent runs load from disk. Cache names encode the bar type: `bars_5s`, `bars_l1_5s`, `bars_dollar_50k`, `enriched_5s_atr_spread_vwap_session`.

**Dollar bars**: VWAP strategy uses dollar bars (`bar.type: dollar`). These must be built from 1s source via `src/data/bars.build_dollar_bars()`. Do NOT feed raw 1s bars to a dollar-bar strategy.

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
- Commission: $0.295/side, $0.59 round trip (Tradovate Free plan)

<<<<<<< HEAD

## Remote execution (training/backtesting/optimization)

When running training, backtesting, or optimization scripts, **always write results to a file** so they can be read back via SSH:

- **Logs**: Write to `logs/<script_name>.log` (use `tee` or Python logging to file)
- **Results/metrics**: Write JSON summaries to `results/<strategy>/`
- **Model artifacts**: Save to `models/`
- **Console output**: Pipe through `tee` so output goes to both terminal and file, e.g.:
  ```bash
  python -u scripts/backtest/cpcv.py --strategy orb 2>&1 | tee logs/cpcv_orb.log
  ```

This allows progress and results to be checked remotely without an interactive terminal.
=======
## Remote Execution Workflow

- Development happens on a ThinkPad T14s (Arch Linux + Hyprland)
- Heavy training/backtesting runs on a Windows desktop (i7-10700KF, 32GB, RTX 4070 Super)
- SSH alias `ssh desktop` connects ThinkPad → Windows OpenSSH → drops straight into Arch WSL2
- Repo lives at `/mnt/c/Dev/scalp` on the desktop and `/dev/scalp` on the ThinkPad
- Model artifacts (.pkl, .joblib, .pt, .h5, .parquet) are tracked via Git LFS

### When executing training or backtest jobs:
1. Run `ssh desktop` to connect to the desktop
2. `cd /mnt/c/Dev/scalp`
3. Run the appropriate script
4. Tail logs, report results back
5. Commit artifacts and push so ThinkPad can pull them
>>>>>>> cd840ab (laptop to desktop)
