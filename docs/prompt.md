You are assisting with an MES (Micro E-mini S&P 500) futures scalping bot. This document describes the full system architecture.

## Overview

Event-driven async Python bot targeting Tradovate API (REST for orders) + Databento (live market data). Paper and live trading modes. 15 years of historical 1s bar data for backtesting. Python 3.13, managed with `uv`. Polars for all DataFrames. Structured logging via `structlog`.

## Live Data Flow

```
DatabentoFeed (OHLCV-1s bars)
  → EventBus publishes BarEvent + TickEvent (close price for paper fills)
    → BarResampler (1s → 5m, clock-aligned)
      → SignalHandler.on_bar()
        → SignalEngine.compute(bar_window) → SignalBundle
        → FilterEngine.evaluate(bundle) → pass/block
        → Strategy.on_bar(bar, bundle) → Signal | None
          → RiskManager.check_order() → approved/rejected
            → TradovateOMS.submit_order(signal) → bracket order
              → FillMonitor tracks lifecycle
```

The feed subscribes to Databento `ohlcv-1s` with `stype_in="raw_symbol"` for a single MES contract (e.g. `MESH6`). It emits ~1 bar/second during RTH. TickEvents carry the close price for paper bracket fill checking. BarResampler accumulates 1s bars into 5m bars aligned to clock boundaries.

## Core Components

### EventBus (`src/core/events.py`)
Central async pub/sub. Queue-based (maxsize 10,000). 5 event types:
- **TICK** — close price from 1s bar (for paper fill monitoring)
- **BAR** — OHLCV bar (1s from feed, 5m from resampler)
- **SIGNAL** — strategy trading signal
- **FILL** — order fill confirmation
- **RISK** — risk alerts/halts

All event dataclasses are `frozen=True, slots=True`. Subscribers register async callbacks per event type. Exceptions in one subscriber don't crash others.

### SignalEngine (`src/signals/signal_bundle.py`)
Computes all declared signals for a bar window (list of BarEvents, max 500). Returns a `SignalBundle` — frozen dict of `SignalResult` objects keyed by signal name. Each signal implements `SignalBase` and is registered in `SignalRegistry`. Signals are instantiated once at startup from strategy YAML `signals:` list.

Available signals (38 total): `adx`, `atr`, `bollinger`, `cvd_divergence`, `donchian_channel`, `ema_crossover`, `ema_ribbon`, `hmm_regime`, `initial_balance`, `keltner_channel`, `macd`, `mfi`, `obv`, `orb_breakout`, `orb_range_size`, `poc_distance`, `prior_day_bias`, `prior_day_levels`, `regime_v2`, `relative_volume`, `rsi_momentum`, `session_time`, `sma_trend`, `spread`, `stochastic`, `value_area`, `vectorized`, `volume_exhaustion`, `vpin`, `vwap_bias`, `vwap_deviation`, `vwap_session`, `vwap_slope`.

### FilterEngine (`src/filters/filter_engine.py`)
Declarative YAML expressions that gate entries. Each strategy has its own filters defined in its YAML config. Example: `{signal: adx, expr: "> 30.0"}`, `{signal: session_time, expr: ">= 600"}`.

### ExitEngine (`src/exits/exit_engine.py`)
Declarative YAML-driven exit conditions evaluated each bar while a position is open. OR logic — any single condition triggers an exit. Exit types:
- `bracket_target` / `bracket_stop` — ATR-based TP/SL from fill price
- `static_target` / `static_stop` — fixed ATR-multiple exits
- `trailing_stop` — ATR trailing with activation threshold
- `time_stop` — exit after N bars in trade
- `vwap_reversion_target` — exit when price reverts to within N SD of VWAP
- `adverse_signal_exit` — exit when signal field flips adverse
- `adverse_momentum` — exit on unrealized loss threshold
- `signal_bound_exit` — exit if signal outside bounds (e.g. ADX collapse)
- `price_vs_signal_exit` — exit if price crosses dynamic signal level
- `regime_exit` — exit when HMM regime becomes hostile
- `volatility_expansion_exit` — exit on ATR expansion

### RiskManager (`src/risk/risk_manager.py`)
Sequential pre-trade checks: halted → session valid → position limit → signal limit. Tracks daily P&L, position sizes, fill history. Resets on session open. Default: $150 max daily loss, 1 contract max position, 10 signals/day.

### SessionManager (`src/core/session.py`)
Tracks RTH 9:30 AM–4:00 PM ET. Detects weekends. Fires session open/close callbacks. Polls every 30s.

### SignalHandler (`src/core/signal_handler.py`)
Bridges strategies to OMS via risk checks. On each bar:
1. Accumulates bar into window (max 500)
2. SignalEngine computes signals → SignalBundle
3. FilterEngine evaluates entry gates
4. Each strategy processes bar (may emit Signal)
5. Signal → RiskManager validation
6. Approved signals → OMS as bracket orders

Also handles warmup: pulls 350 5m bars from Databento Historical API at startup, runs signal engine to initialize regime detector, ADX, ATR, etc. Blocks until async signals (regime_v2) complete.

### BarResampler (`src/core/bar_resampler.py`)
Accumulates 1s BarEvents into clock-aligned higher-timeframe bars (e.g. 5m). When a new window starts, emits the completed bar via callback to SignalHandler.

## Strategies

Duck-typed interface: `on_bar(bar, bundle) → Signal | None`, `reset()`. Some inherit from `StrategyBase` ABC, others duck-type directly. Each strategy loads from YAML via `ClassName.from_yaml(yaml_path)`.

### Signal dataclass (`src/strategies/base.py`)
```python
Signal(strategy_id, direction, entry_price, target_price, stop_price,
       signal_time, expiry_time, confidence, regime_state, metadata)
```
`validate_geometry()` enforces LONG: stop < entry < target, SHORT: stop > entry > target.

MES constants: TICK_SIZE=0.25, TICK_VALUE=$1.25, POINT_VALUE=$5.00, Commission=$0.295/side.

### 16 Strategies Implemented
| CLI name | Strategy | Bar |
|---|---|---|
| `vwap_band` | VWAP Band Reversion | 5m |
| `gap` | Opening Gap Fill | 5m |
| `va` | Value Area Reversion | 5m |
| `orb` | ORB Breakout | 1m |
| `cvd` | CVD Divergence Fade | 1m |
| `micro` | Micro-Pullback Scalp | 1m |
| `regime` | Regime Switcher (HMM) | 5m |
| `ttm` | TTM Squeeze Breakout | 5m |
| `macd` | MACD Zero-Line Rejection | 5m |
| `stoch_bb` | Stochastic+BB Fade | 5m |
| `donchian` | Donchian Breakout | 5m |
| `poc_va` | POC/VA Bounce | 5m |
| `ema_ribbon` | EMA Ribbon Pullback | 5m |
| `pdh_pdl` | PDH/PDL Fade | 5m |
| `mfi_obv` | MFI+OBV Divergence | 5m |
| `ib` | Initial Balance Fade | 5m |

**Validated**: `vwap_band` (64 trades, 70.3% WR, Sharpe 3.27, PF 2.04 over 10yr).
**Dead**: `donchian` (0/2,304 configs profitable on 10yr sweep).
**Remaining 14**: unvalidated.

### Strategy YAML Config Structure
```yaml
strategy:
  strategy_id: vwap_band
  max_signals_per_day: 5
  session_start: "09:45"
  session_end: "15:30"

bar:
  type: time
  freq: 5m

signals: [atr, vwap_session, adx, relative_volume, session_time]
signal_configs:
  adx:
    period: 14
    threshold: 20.0

filters:
  - signal: adx
    expr: "< 20.0"
  - signal: session_time
    expr: ">= 585"

exit:
  time_stop_minutes: 90

exits:
  - type: bracket_target
    enabled: true
  - type: bracket_stop
    enabled: true
  - type: time_stop
    enabled: true
    max_bars: 18
```

## Backtesting

### BacktestEngine (`src/backtesting/engine.py`)
Bar-replay simulator. Loads historical bars (1s, 5m, dollar) from year-partitioned Parquet files. SimulatedOMS with volatility slippage model. Entry: limit fill (no slippage). Stop: market fill (slippage applied). Ambiguity rule: if both target and stop in same bar, stop wins.

### Validation Suite
- **CPCV** (`src/backtesting/cpcv.py`) — Combinatorial purged cross-validation with PBO
- **DSR** (`src/backtesting/dsr.py`) — Deflated Sharpe Ratio for multiple testing correction
- **WFA** (`src/backtesting/wfa.py`) — Rolling walk-forward with efficiency ratio
- **DecisionEngine** (`src/backtesting/decision_engine.py`) — 4-gate approval: expectancy, Sharpe, DSR, correlation

### 2-Tier Tuning Pipeline
1. **Tier 1** (`scripts/tune/strategy.py`) — Strategy params on 15yr bars, ranked by `sharpe * sqrt(trades)`
2. **Tier 2** (`scripts/tune/filters.py`) — Filter threshold sweep on 1yr enriched data

## Data Pipeline

```
data/
  parquet/year=YYYY/data.parquet     — 1s OHLCV bars (~1GB, 15 years)
  parquet_1m/year=YYYY/data.parquet  — 1m bars
  bars_<name>/bars_<name>.parquet    — cached resampled bars
  enriched_<name>/enriched_<name>.parquet — cached enriched bars with signal columns
```

`BarCache` (`src/data/bar_cache.py`) handles persistent caching. First run builds and caches; subsequent runs load from disk. Enriched bars include pre-computed signal columns for fast backtesting via `enrich_bars()` (vectorized Polars).

## Deployment

- **VPS**: Vultr Chicago, systemd service `mes-bot`
- **Feed**: Databento OHLCV-1s, single contract subscription
- **Orders**: Tradovate REST API (paper mode default)
- **Health**: FastAPI `/health` endpoint on port 8080
- **Warmup**: 350 5m bars from Databento Historical at startup
- **Logging**: structlog JSON rotating file + colored console

## Concurrent Tasks (main.py)

```python
tasks = [
    EventBus.run(),           # dispatch loop
    SessionManager.run(),     # RTH tracking
    HealthMonitor.start(),    # /health endpoint
    FillMonitor.run(),        # order lifecycle
    DatabentoFeed.run(),      # market data stream
]
```

Graceful shutdown via SIGINT/SIGTERM. Session close flattens position, cancels working orders, resets strategies.

## Key Conventions

- All timestamps in nanoseconds (`timestamp_ns`)
- US/Eastern timezone for session logic
- Polars (never pandas) for DataFrames
- `python -u` for unbuffered output in scripts
- Backtest scripts write results to files for remote SSH access
- Signal geometry validation: LONG stop < entry < target, SHORT stop > entry > target
- If WR > 90% or Sharpe > 10, something is broken — investigate before trusting
