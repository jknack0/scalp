# MES Scalping Bot — Development Phases Roadmap

**Retail Algo Trader  ·  12 Phases  ·  58 Tasks  ·  ~20 Weeks to First Live Trade**

---

> **How to Use This Document**
> 
> For each task: (1) Read the **Why** and **Steps** to understand the goal. (2) Check if the task is marked `⚠ INPUT REQUIRED` — if so, gather that data first. (3) Open Claude Code in your repo root. (4) Paste the Claude Code prompt. (5) Claude Code will create all files, ask for any missing data, and implement the task end-to-end. Do not proceed to the next task until the current exit criterion is met.

---

## Phase Overview

| # | Phase | Duration | Tasks | Status | Primary Deliverable |
| --- | --- | --- | --- | --- | --- |
| 1 | Infrastructure & Cost Foundation | 2 Weeks | 6 | ✅ DONE | Broker account, live tick feed, Databento pipeline, commission math workbook |
| 2 | Tick Data Exploration & Regime Baseline | 3 Weeks | 5 | ✅ DONE | Dollar bars, Hurst/autocorr analysis, feature library, HMM regime classifier |
| 3 | Strategy Research & Signal Development | 4 Weeks | 6 | ✅ DONE | 5 strategy classes: ORB, VWAP Reversion, CVD Divergence, Vol Switcher, OBI |
| 4 | Rigorous Backtesting & Validation | 4 Weeks | 5 | ✅ DONE | CPCV + DSR validation, WFA, 2–3 survivor strategies with locked parameters |
| 5 | Paper Trading & Live Preparation | 4 Weeks | 5 | ✅ DONE | OMS, safety systems, live pipeline wired end-to-end |
| 6 | Live Deployment & Continuous Improvement | Ongoing | 5 | 🔲 TODO | 1 live contract, Grafana dashboard, monthly re-validation cadence |
| 7 | Filter Wiring & L1 Validation | 2 Weeks | 4 | 🔶 IN PROGRESS | Spread + VPIN filters validated, HMM vs Filters comparison |
| 8 | L2 Data Acquisition & Parsing | 1 Week | 1 | 🔲 TODO | 3 months L2 MBP-10 in Parquet |
| 9 | L2 Signal Wiring & Validation | 2 Weeks | 3 | 🔲 TODO | L2 vs L1 lift quantified, depth deterioration wired |
| 10 | SignalAggregator & Full Integration | 1 Week | 1 | 🔲 TODO | Unified entry gate combining all validated filters |
| 11 | Operational Hardening | 2 Weeks | 4 | 🔲 TODO | Kill switch, correlation monitor, trade journal, drift alerting |
| 12 | L1/L2 Order Book Strategies | 4 Weeks | 10 | 🔲 TODO | OFI, CVD+L2, absorption, sweep reversal, footprint delta, composite signal system |

## Commission Math Reference — The Non-Negotiable Foundation

Every strategy must clear this hurdle before research begins. MES tick = $1.25. Assumes 1-tick avg slippage per side.
NT = NinjaTrader Lifetime ($0.09/side). Std = standard retail broker ($0.62/side).

| Target / Stop | Gross Win | NT Breakeven WR | Std Breakeven WR | Verdict |
| --- | --- | --- | --- | --- |
| 2 tick / 2 tick | $2.50 | 57.5% | 74.8% | ❌ Not viable |
| 4 tick / 2 tick | $5.00 | 37.2% | 49.9% | ⚠️ Marginal |
| 8 tick / 4 tick | $10.00 | 29.5% | 41.6% | ✅ Viable |
| 12 tick / 6 tick | $15.00 | 26.2% | 37.6% | ✅ Good |
| 16 tick / 8 tick | $20.00 | 24.5% | 35.4% | ✅ Solid |

_\* Annual commission at 5 trades/day × 250 days × NT rate ($0.18 RT). Standard broker: $1,550/year at same frequency._

---


---

# Phase 1: Infrastructure & Cost Foundation

**Duration:** 2 Weeks  |  **Goal:** Get the math right before writing a single line of signal logic.

## Overview

This phase is the most important two weeks of the entire project — and the most commonly skipped by retail algo traders who pay for it in real money later. Before researching a single strategy, you need to lock in your cost structure, data pipeline, and execution environment. Every decision you make here has compounding effects across all remaining phases.

The core insight driving this phase: at MES tick value of $1.25, a round-trip commission of $1.24 (common at retail brokers) consumes 99.2% of a 1-tick profit. This isn't a small headwind — it makes 1–3 tick scalping mathematically impossible before accounting for slippage. The broker you choose, the commission tier you negotiate, and the data feed you select will determine which strategies are even worth building.

By the end of Phase 1, you will have a live Rithmic tick feed streaming to your Chicago VPS, 12+ months of MES tick data in TimescaleDB, a documented commission math analysis showing your exact breakeven win rates, and a production-ready asyncio execution skeleton ready to receive signal logic in later phases.

## Key Concepts

| Concept | What It Means for Your Bot |
| --- | --- |
| **MES Contract Specs** | Tick size: 0.25 index points = $1.25/tick. Point value: $5.00. Margin: ~$40 intraday (varies by broker). Trading hours: CME Globex nearly 24/5, RTH 9:30 AM–4:00 PM ET. Settlement: cash, quarterly contracts (H, M, U, Z). |
| **Round-Trip Commission** | Total cost of entering AND exiting a position. Includes: broker fee (both sides), exchange fee, NFA fee. At NinjaTrader Lifetime: $0.09 × 2 = $0.18 RT. At Schwab Futures: $0.62 × 2 = $1.24 RT. Difference over 10 trades/day × 250 days = $2,650/year per contract. |
| **Rithmic R|Protocol** | CME-certified market data and order routing platform. Provides L1 (BBO), L2 (10-level order book), and MBO (Market By Order / L3) data. Used by prop firms. Accessed via Protocol Buffers over WebSocket. Python library: pyrithmic (asyncio-based). |
| **Databento MBO Data** | Market By Order data captures every individual order insert, modify, and cancel at the exchange level. Enables reconstruction of the full order book, queue position modeling, and precise trade direction attribution. This is the data HFT firms use — now available retail. |
| **Dollar Bars vs. Time Bars** | Dollar bars sample a new bar every time $X notional trades, not every N seconds. This normalizes bar variance across volatility regimes (high vol = more frequent bars, low vol = fewer). López de Prado demonstrates dollar bars produce more stationary, IID-like return series than time bars — critical for ML model validity. |
| **Chicago VPS Latency** | CME Globex matching engine is in Aurora, IL (close to you). A Chicago-area VPS achieves 0.3–2ms round-trip. Your home connection adds 15–50ms. For strategies with 30-second+ hold times, this difference is mostly irrelevant. For sub-5-second strategies, Chicago VPS is mandatory. |

## Tasks (6 total)

### Broker Research & Account Setup

**Effort:** 3 days

#### Why This Matters

Commission structure determines which strategies are mathematically viable. This is not a cosmetic difference — NinjaTrader's $0.18 RT vs. a standard broker's $1.24 RT changes the breakeven win rate on an 8-tick target from 41.6% to 29.5%. That gap is the difference between a viable strategy and an impossible one.

#### Steps

1. Compare three brokers side-by-side: NinjaTrader Lifetime ($1,499 one-time, then $0.09/side MES), Edge Clear (Rithmic-connected, $0.10/side), AMP Futures ($0.25/side, Rithmic or Tradovate). Document all fees including exchange ($0.02), NFA ($0.02), clearing.

2. Calculate your exact breakeven win rate table for 4, 8, 12, 16 tick targets at each broker's commission rate. Include 1-tick average slippage assumption in the model. This becomes your 'minimum viable target' reference throughout the project.

3. Open account with chosen broker. Fund with $5,000–$10,000 minimum ($10,000 recommended to survive learning-curve drawdown without hitting margin limits). Enable futures trading permissions.

4. Request Rithmic data credentials separately — most brokers bundle this but it must be activated. Verify you have L1 + L2 (order book depth) permissions, not just L1.

5. Test a manual MES paper trade end-to-end: entry, bracket order (target + stop), exit. Confirm fills appear in account statement with correct commission amounts.

#### ✅ Exit Criteria

> Account open, API credentials in hand, RT commission confirmed in writing

#### ⚠️ Input Required Before Running

Claude Code will ask you for data before generating files. Have the following ready:

_Before I generate the broker comparison document, I need a few pieces of info from you:_
1. Which brokers are you currently considering or already have accounts with? (NinjaTrader, Edge Clear, AMP Futures, Tradovate, IBKR, or others?)
2. What commission rate did each quote you per side for MES? (e.g. NinjaTrader Lifetime = $0.09/side)
3. Do you have the NinjaTrader Lifetime license already, or are you evaluating it?
4. Any brokers you want to exclude from comparison?
Once you share this, I'll build the full broker comparison doc in your repo.

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create a markdown file at docs/phase1/broker-comparison.md in this repo.

The file should be a complete broker comparison analysis with the following sections:

## Structure
1. **Executive Summary** — which broker wins and why (1 paragraph)
2. **Commission Math Table** — for each broker: per-side rate, round-trip total, annual cost at 5 trades/day × 250 days × 1 MES contract, and breakeven win rate for 8-tick target with 1-tick slippage assumption
3. **Broker Profiles** — for each broker: platform, API type, data feed quality (L1/L2/L3), margin requirements for MES intraday, pros, cons
4. **Go/No-Go Decision** — chosen broker with explicit reasoning, next steps to open account

## Brokers to compare
[USER_BROKER_DATA — filled in from user input]

## Commission math formula to use
- Gross win per trade = profit_target_ticks × $1.25
- Net win = gross_win - (round_trip_commission + 2 × slippage_cost)  [slippage = 1 tick = $1.25 each way]
- Net loss = gross_loss + round_trip_commission + 2 × slippage_cost
- Breakeven win rate = net_loss / (net_win + net_loss)

Use the actual commission numbers the user provided. If any are missing, flag them with [VERIFY] in the doc.

At the end, add a section: **## What To Ask Your Broker** — a list of exact questions to send via email/chat to confirm fees before funding.

Save as docs/phase1/broker-comparison.md
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Chicago VPS Provisioning

**Effort:** 1 day

#### Why This Matters

For strategies with sub-60-second hold times, execution latency from your home connection (15–50ms) vs. a Chicago VPS (0.3–2ms) can be the difference between filling at your signal price or 2–3 ticks worse. Even for longer holds, running your bot on a VPS eliminates home internet outages as a risk.

#### Steps

1. Provision a VPS from QuantVPS Chicago ($60–$100/mo), Vultr Chicago ($20–$40/mo for lighter workloads), or Contabo Chicago. Prioritize providers advertising Aurora, IL or Chicago, IL datacenter location (closest to CME).

2. Install Ubuntu 22.04 LTS. Set up SSH key authentication, disable password auth, configure UFW firewall (allow only SSH + your Rithmic port).

3. Install Python 3.11+, pip, virtualenv. Install uvloop: pip install uvloop. Verify with a simple asyncio benchmark — uvloop should show 2–4× throughput improvement over default loop.

4. Measure round-trip latency to CME Globex using: ping 198.105.251.100 (CME Chicago IP). Target < 5ms. If > 10ms, contact provider — you may be routed to wrong datacenter.

5. Set up systemd service for your bot process so it auto-restarts on crash and survives VPS reboots. Test: sudo systemctl restart your-bot && verify it reconnects cleanly.

#### ✅ Exit Criteria

> VPS live, ping to CME Globex < 5ms, uvloop running

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create a shell script and documentation for Chicago VPS provisioning.

Create the following files in this repo:

**1. scripts/infra/provision-vps.sh**
A bash setup script that:
- Updates apt packages
- Installs Python 3.11, pip, virtualenv, git, htop, tmux
- Installs uvloop and verifies it works: python3 -c "import uvloop; print('uvloop ok')"
- Creates a non-root user 'botuser' with sudo privileges
- Configures UFW firewall: allow SSH (22), deny everything else inbound
- Installs fail2ban for SSH brute-force protection
- Creates /opt/mes-bot directory with correct ownership
- Sets up a systemd service template at /etc/systemd/system/mes-bot.service
- Runs a latency test: ping -c 10 198.105.251.100 (CME Globex IP) and saves output to /tmp/latency-test.txt
- Prints final summary: Python version, uvloop version, UFW status, ping avg ms

**2. docs/phase1/vps-setup.md**
Documentation covering:
- Recommended VPS providers (QuantVPS Chicago, Vultr Chicago, Contabo Chicago) with current pricing estimates
- How to SSH in and run the provision script
- How to verify latency < 5ms to CME
- How to set up SSH key authentication (step by step)
- Systemd service management commands (start/stop/restart/logs)
- Latency troubleshooting: what to do if ping > 10ms

Make the shell script idempotent (safe to run multiple times). Add set -e at the top so it fails fast on any error. Add clear echo statements before each major step.
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Rithmic Live Tick Feed Connection

**Effort:** 3 days

#### Why This Matters

Your live data feed is the foundation everything else runs on. Bugs here — timestamp drift, dropped ticks, reconnection failures — will corrupt your real-time signal calculations in ways that are hard to diagnose later. Build it right and test it thoroughly before adding any strategy logic.

#### Steps

1. Install pyrithmic: pip install pyrithmic. Review the library's RithmicClient class and its async callback architecture. Rithmic uses Protocol Buffers (protobuf) messages over WebSocket — you don't need to handle these directly, but understand the message flow.

2. Implement connection handler with exponential backoff reconnection: attempt reconnect after 1s, 2s, 4s, 8s, up to 60s max. Log every reconnect event with timestamp. A dropped connection during a live position is a major risk.

3. Subscribe to MES front-month contract (ESM5 format — verify current expiration). Subscribe to: best_bid_offer (L1), market_depth_by_price (L2 top 10 levels), trade_print (last trade with volume and aggressor side).

4. Timestamp every tick on receipt using time.perf_counter_ns() for nanosecond precision. Compare to exchange timestamp in the message. Track median and 99th percentile latency. Alert (log warning) if any tick exceeds 100ms.

5. Run the feed for 2 full RTH sessions (9:30 AM–4:00 PM). Verify: no gaps > 5 seconds during active hours, reconnections work cleanly, L2 book updates in sync with L1. Log tick counts per minute and verify they match expected volume profile.

#### ✅ Exit Criteria

> Live MES L1 + L2 tick stream printing to console, latency-stamped, < 50ms end-to-end

#### ⚠️ Input Required Before Running

Claude Code will ask you for data before generating files. Have the following ready:

_For the Rithmic connection setup, I need to know:_
1. Do you have Rithmic credentials already (from your broker)? If yes, which broker provided them?
2. Which Python async style do you prefer — pure asyncio callbacks, or do you want it wrapped in a cleaner class interface?
3. What's your repo structure? (e.g. src/feeds/, feeds/, or something else?)
Share whatever you have and I'll generate the full Rithmic connection module.

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the Rithmic tick feed connection module.

Create the following files:

**1. src/feeds/rithmic_feed.py**
A production-quality asyncio Rithmic feed client with:
- RithmicFeed class using pyrithmic library (pip install pyrithmic)
- Async connect() method with exponential backoff reconnection: delays of 1s, 2s, 4s, 8s, 16s, 32s, 60s (cap)
- Subscription to MES front-month: L1 (best bid/offer), L2 (market depth top 10), trade prints
- On every tick: attach receipt timestamp using time.perf_counter_ns()
- TickData dataclass: symbol, timestamp_exchange_ns, timestamp_received_ns, bid, ask, bid_size, ask_size, last_price, last_size, aggressor_side (BUY/SELL/UNKNOWN), depth_bids (list of 10 price/size tuples), depth_asks (list of 10 price/size tuples)
- Latency tracking: maintain a deque of last 1000 latency values (received_ns - exchange_ns). Expose p50_latency_ms and p99_latency_ms properties.
- Connection watchdog: if no tick received for 30 seconds during RTH (9:30-16:00 ET), emit a WARNING log and attempt reconnect
- Event emission: uses asyncio.Queue to publish TickData to subscribers (fan-out to multiple consumers)
- Clean shutdown: async close() method that cancels subscriptions before disconnecting

**2. src/feeds/models.py**
Dataclasses for: TickData, OrderBookLevel (price, size, num_orders), ConnectionStatus (enum: CONNECTING, CONNECTED, RECONNECTING, DISCONNECTED)

**3. src/feeds/config.py**
RithmicConfig dataclass loaded from environment variables: RITHMIC_USER, RITHMIC_PASSWORD, RITHMIC_SERVER_URL, RITHMIC_SYSTEM_NAME, RITHMIC_EXCHANGE (default "CME"), RITHMIC_SYMBOL (default "MESH5" — update to current front month)

**4. scripts/test_feed.py**
A standalone test script that:
- Connects to Rithmic
- Subscribes to MES
- Prints every tick for 60 seconds with: timestamp, bid/ask spread, last price, p50/p99 latency
- Prints a summary: total ticks received, avg ticks/second, p50 latency ms, p99 latency ms, any reconnection events
- Exits cleanly with Ctrl+C

**5. docs/phase1/rithmic-connection.md**
Setup guide: how to install pyrithmic, required env vars, how to run the test script, how to interpret the latency output, what good vs. bad latency looks like.

Use structlog for all logging. Make all credentials come from environment variables only — never hardcoded.
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Databento Historical Pipeline

**Effort:** 3 days

#### Why This Matters

Your backtest quality is only as good as your historical data quality. Databento provides the same MBO data as your live Rithmic feed, meaning your backtest environment precisely mirrors production — the single most important property of a reliable backtest pipeline.

#### Steps

1. Sign up for Databento (databento.com). For MES backtesting, you need the GLBX.MDP3 dataset (CME Globex). Check current pricing — Standard plan allows MBO data downloads. Budget $100–$200 for initial 12-month pull.

2. Install Databento Python client: pip install databento. Pull 12 months of MES MBO data using the historical API. Use DBN format (Databento's native binary format) for storage efficiency — it's 5–10× more compact than CSV.

3. Write a Parquet conversion script: read DBN files, extract fields (ts_event, action, side, price, size, order_id, bid_px, ask_px, bid_sz, ask_sz), write to Parquet partitioned by date. This enables fast date-range queries.

4. Load Parquet files into TimescaleDB hypertables partitioned by time. Create indexes on: (symbol, ts_event), (ts_event, price). Verify a 1-day query (millions of rows) returns in < 2 seconds.

5. Write a data quality validation script: check for gaps > 30 seconds during RTH hours, duplicate timestamps, price outliers (> 5 ATR from rolling mean), missing volume. Log all anomalies. Flag any dates with > 5 gaps for manual review.

6. Validate continuity between historical data (Databento) and live feed (Rithmic) by overlapping 1 day. Compare tick counts, OHLCV for each 1-minute bar. Differences < 0.1% are acceptable (Rithmic and CME timestamps differ slightly).

#### ✅ Exit Criteria

> 12 months MES MBO tick data in TimescaleDB, queryable in < 5 seconds

#### ⚠️ Input Required Before Running

Claude Code will ask you for data before generating files. Have the following ready:

_For the Databento pipeline, I need to know:_
1. Do you have a Databento account already? If yes, what's your subscription tier?
2. What date range do you want for the initial MES data pull? (Recommend: 2023-01-01 to today for ~2 years)
3. What's your TimescaleDB connection string / Supabase project URL? (I'll use it in the config — you can replace with a placeholder if you prefer)
4. How much local disk space is available on your VPS for raw tick data storage?
Share what you have and I'll build the full pipeline.

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the Databento historical data pipeline.

Create the following files:

**1. src/data/databento_pipeline.py**
Complete pipeline with these classes:

DatabentoConfig (dataclass): api_key (from env DATABENTO_API_KEY), dataset="GLBX.MDP3", schema="mbo" (Market By Order / L3), symbols=["MES.FUT"] (continuous front-month), start_date, end_date, output_dir

DatabentoDownloader class:
- download(config) → downloads DBN files to output_dir, one file per day, skips existing files (idempotent)
- Uses databento Python client: import databento as db
- Prints progress bar using tqdm: "Downloading 2024-01-03... [===>    ] 3/252 days"
- validate_download() → checks each file exists and is non-zero size, returns list of missing/corrupt dates

DBNToParquetConverter class:
- convert(dbn_dir, parquet_dir) → converts each DBN file to Parquet
- Parquet schema: ts_event (int64 nanoseconds), action (str: A=add, C=cancel, M=modify, T=trade, F=fill), side (str: B=bid, A=ask), price (float64, in dollars), size (int64), order_id (int64), bid_px (float64), ask_px (float64), bid_sz (int64), ask_sz (int64), date (date, partition column)
- Partitions output by date: parquet_dir/date=2024-01-03/data.parquet
- Adds computed columns: is_trade (bool, action=='T'), dollar_value (price × size × 5.0 for MES)

TimescaleDBLoader class:
- load(parquet_dir, conn_string, table_name="mes_ticks") → bulk-loads Parquet files into TimescaleDB
- Creates hypertable if not exists: CREATE TABLE IF NOT EXISTS mes_ticks (...); SELECT create_hypertable(...)
- Uses COPY command via psycopg2 for fast bulk insert (not row-by-row)
- Tracks which dates are already loaded (idempotent: skip loaded dates)
- Creates indexes: (symbol, ts_event), (ts_event, price), (date, is_trade)

DataQualityValidator class:
- validate(conn_string, start_date, end_date) → runs quality checks:
  - Gaps > 30s during RTH (9:30-16:00 ET) on any date
  - Duplicate ts_event values
  - Price outliers: price > rolling_5min_mean ± 5 × rolling_5min_std
  - Missing dates (trading days with zero rows)
  - Returns ValidationReport dataclass with: gap_count, duplicate_count, outlier_count, missing_dates[]
- Prints a human-readable summary table

**2. scripts/run_pipeline.py**
CLI script using argparse:
- --download: run download step
- --convert: run DBN→Parquet conversion
- --load: load to TimescaleDB
- --validate: run quality validation
- --all: run all steps in sequence
- --start-date, --end-date, --symbol flags
- Prints timing for each step

**3. sql/create_mes_ticks.sql**
Complete SQL for creating the TimescaleDB hypertable with all indexes and a materialized view for 1-minute OHLCV bars.

**4. docs/phase1/databento-pipeline.md**
- How to get your Databento API key
- What MBO (Market By Order) data is and why it matters
- How to run the pipeline end-to-end
- Expected file sizes (MES MBO data is ~500MB-2GB per year)
- How to query the data: example SQL for pulling 1 day of ticks
- Cost estimate: what 12-24 months of MES MBO data costs on Databento

Install requirements: databento, pyarrow, psycopg2-binary, tqdm, polars
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Commission Math Workbook

**Effort:** 1 day

#### Why This Matters

This document becomes your strategy research filter. Any strategy with a profit target below your minimum viable threshold gets rejected before you write a line of code. It's a 1-day investment that saves weeks of research on doomed strategies.

#### Steps

1. Build a Python dataclass for your cost model: broker_commission_per_side, exchange_fee, nfa_fee, avg_slippage_ticks. Instantiate it with your actual broker's numbers.

2. For each combination of (profit_target_ticks, stop_loss_ticks) from 1–20 ticks each: compute gross_win, net_win (after RT commission + spread), net_loss (after RT commission), and breakeven_win_rate = net_loss / (net_win + net_loss). Plot as a heatmap.

3. Identify your minimum viable profit target: the smallest target where breakeven win rate < 50% (meaning a coin-flip strategy breaks even). For most brokers this is 8–12 ticks. Mark this clearly in the notebook.

4. Compute annual commission budget: at 5 trades/day × 250 days = 1,250 trades/year. At your RT commission, that's X dollars in commissions regardless of P&L. This is your annual 'overhead' that profit must exceed.

5. Model slippage scenarios: 0-tick (ideal), 1-tick (realistic calm), 2-tick (event days), 3-tick (high volatility). Show how each scenario shifts breakeven win rates. Conclude: always assume 1-tick slippage in backtest.

#### ✅ Exit Criteria

> Python notebook documenting minimum viable target size, breakeven win rates, annual commission budget

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create a commission math analysis module and Jupyter notebook.

Create the following files:

**1. src/analysis/commission_model.py**
CostModel dataclass:
- broker_commission_per_side: float
- exchange_fee: float = 0.02
- nfa_fee: float = 0.02
- avg_slippage_ticks: float = 1.0
- tick_value: float = 1.25

Methods:
- round_trip_cost() → total RT cost in dollars including slippage
- gross_win(target_ticks) → target_ticks × tick_value
- net_win(target_ticks) → gross_win - round_trip_cost
- gross_loss(stop_ticks) → stop_ticks × tick_value
- net_loss(stop_ticks) → gross_loss + round_trip_cost
- breakeven_win_rate(target_ticks, stop_ticks) → net_loss / (net_win + net_loss)
- annual_commission_cost(trades_per_day=5, trading_days=250) → total annual commission spend
- min_viable_target(max_breakeven_wr=0.50) → smallest target where breakeven_wr < max_breakeven_wr
- profit_expectancy(target_ticks, stop_ticks, win_rate) → expected P&L per trade

BrokerComparison class:
- Takes list of CostModel instances with broker names
- compare_breakeven_matrix(targets=[4,6,8,10,12,16], stops=[2,3,4,5,6,8]) → pandas DataFrame
- annual_cost_comparison(trades_per_day=5) → DataFrame comparing annual costs
- plot_breakeven_heatmap(broker_name) → matplotlib heatmap of breakeven win rates
- plot_broker_comparison() → bar chart of annual costs by broker

**2. notebooks/phase1-commission-analysis.ipynb**
Complete Jupyter notebook with:

Cell 1: Setup — import all modules, define the 3 brokers:
  - NinjaTrader Lifetime: $0.09/side
  - Edge Clear: $0.10/side  
  - Standard Broker: $0.62/side

Cell 2: Breakeven win rate table for each broker (target 4–16 ticks, stop 2–8 ticks)

Cell 3: Heatmap visualization — breakeven win rate as color, green=achievable, red=impossible

Cell 4: Annual commission cost comparison — bar chart showing annual commission drag at 5, 10, 15 trades/day

Cell 5: Slippage sensitivity — show how 0, 0.5, 1.0, 1.5, 2.0 tick slippage shifts the viability threshold

Cell 6: **MINIMUM VIABLE STRATEGY PARAMETERS** — explicit table showing:
  - Minimum profit target for each broker
  - Required win rate at 1:1 RR for each broker
  - Required win rate at 1.5:1 RR for each broker
  - Recommended trading frequency based on annual commission budget

Cell 7: Conclusion — written analysis of which broker to use and why

**3. docs/phase1/commission-math.md**
Summary of findings with the minimum viable target table. This doc is referenced by all Phase 3 strategy specs.

Requirements: pandas, numpy, matplotlib, seaborn, jupyter
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Asyncio Bot Skeleton

**Effort:** 2 days

#### Why This Matters

Starting with clean architecture prevents the common failure mode of having a working prototype that's impossible to extend safely. Signal logic, order management, risk controls, and data feeds need clean separation from day one.

#### Steps

1. Set up Python project structure: /src/feeds/ (Rithmic connection), /src/signals/ (strategy logic), /src/oms/ (order management), /src/risk/ (position limits, daily loss), /src/monitoring/ (logging, alerts), /config/ (YAML configs), /tests/.

2. Implement config management using pydantic BaseSettings: load broker credentials, risk limits (max_daily_loss, max_position_size), strategy parameters from environment variables + YAML. Never hardcode credentials.

3. Build structured logging with structlog: every log entry includes timestamp, component, event_type, and relevant fields (price, quantity, order_id). Write logs to both console and rotating file. This is critical for post-trade analysis.

4. Implement a simple event bus: asyncio.Queue-based pub/sub where the tick feed publishes TickEvent objects and strategy subscribers consume them. This decouples feed from signal logic cleanly.

5. Write a health check endpoint (FastAPI, 2 lines): GET /health returns {status: ok, position: 0, daily_pnl: 0.0, last_tick_age_ms: 23}. Useful for monitoring from Telegram.

#### ✅ Exit Criteria

> Runnable Python project with clean module structure, config management, and logging — ready to receive signal logic

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the complete asyncio bot skeleton — the production architecture that all future phases plug into.

Create the following project structure:

**src/__init__.py** — empty

**src/core/events.py**
Event dataclasses using Python dataclasses:
- TickEvent(symbol, tick_data, timestamp_ns)
- BarEvent(symbol, bar, bar_type, timestamp_ns)  
- SignalEvent(strategy_id, signal, timestamp_ns)
- FillEvent(order_id, symbol, direction, fill_price, fill_size, commission, timestamp_ns)
- RiskEvent(event_type: str, message: str, timestamp_ns)

EventBus class:
- asyncio.Queue-based pub/sub
- subscribe(event_type, callback) → registers async callback
- publish(event) → puts event on queue, immediately returns (non-blocking)
- run() → async loop that dispatches events to subscribers
- Tracks event counts by type for monitoring

**src/core/config.py**
BotConfig loaded via pydantic BaseSettings from environment + YAML:
- Symbol: str = "MESH5"
- SessionStart: time = "09:30"
- SessionEnd: time = "16:00"
- MaxDailyLossUsd: float = 150.0
- MaxPositionContracts: int = 1
- MaxSignalsPerDay: int = 10
- SlippageAssumptionTicks: float = 1.0
- VpsLatencyTargetMs: float = 50.0
- TelegramBotToken: str (from env)
- TelegramChatId: str (from env)
- RithmicUser: str (from env)
- RithmicPassword: str (from env)
- DatabaseUrl: str (from env)
- DatabentoApiKey: str (from env)
- Load from config/bot-config.yaml with env var overrides

**src/core/logging.py**
Structured logging setup using structlog:
- configure_logging(log_level, log_file) → sets up structlog with:
  - JSON renderer for file output (machine-readable)
  - ConsoleRenderer for stdout (human-readable with colors)
  - Automatic fields: timestamp, level, component, event
  - Log rotation: 10MB max, keep 5 backups
- get_logger(component_name) → returns bound logger with component field

**src/core/session.py**
SessionManager class:
- is_rth() → bool: is current time in regular trading hours?
- is_excluded_window(excluded_windows: list) → bool: is now in a blocked period (FOMC etc.)?
- time_to_open() → timedelta: how long until next RTH open?
- seconds_in_session() → float: seconds elapsed since session open
- on_session_open(callback) → registers async callback for 9:30 AM trigger
- on_session_close(callback) → registers async callback for 4:00 PM trigger
- Uses pytz for timezone handling (America/New_York)

**src/risk/risk_manager.py**
RiskManager class:
- check_order(signal, current_position, daily_pnl) → RiskCheckResult(approved: bool, reason: str)
- Checks: daily loss limit, max position size, session validity, signal expiry
- record_fill(fill_event) → updates daily P&L tracking
- daily_pnl property → float
- is_halted property → bool (set True when daily loss limit hit, reset on new session)
- halt(reason: str) → sets is_halted=True, logs reason, triggers Telegram alert

**src/monitoring/health.py**
HealthMonitor class (FastAPI endpoint):
- GET /health → {status, position, daily_pnl_usd, last_tick_age_ms, is_halted, hmm_state, uptime_seconds}
- Uses FastAPI + uvicorn running in background task
- Updates last_tick_time on each tick event via EventBus subscription

**config/bot-config.yaml**
Template config file with all settings documented via comments. Values should be safe defaults (not real credentials).

**config/.env.example**
All required environment variables with placeholder values and descriptions.

**.github/workflows/test.yml** (if .github exists) OR **Makefile**
Commands: make test, make lint, make run-paper, make run-live

**requirements.txt**
All dependencies: asyncio, uvloop, pydantic, pydantic-settings, structlog, fastapi, uvicorn, pytz, python-telegram-bot, pyrithmic, databento, psycopg2-binary, pyarrow, polars, pandas, numpy, hmmlearn, mlfinlab, pypbo, vectorbt, hftbacktest, pytest, pytest-asyncio

**tests/test_risk_manager.py**
Unit tests for RiskManager:
- test_daily_loss_limit_triggers_halt
- test_max_position_blocks_order
- test_approved_order_passes_all_checks
- test_halt_resets_on_new_session
Use pytest-asyncio for async tests.

Make all classes importable from src package. Follow the separation: feeds → core → signals → risk → oms → monitoring.
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

## Phase 1 Gate

All exit criteria must pass before advancing to the next phase:

- [ ] **Broker Research & Account Setup:** Account open, API credentials in hand, RT commission confirmed in writing
- [ ] **Chicago VPS Provisioning:** VPS live, ping to CME Globex < 5ms, uvloop running
- [ ] **Rithmic Live Tick Feed Connection:** Live MES L1 + L2 tick stream printing to console, latency-stamped, < 50ms end-to-end
- [ ] **Databento Historical Pipeline:** 12 months MES MBO tick data in TimescaleDB, queryable in < 5 seconds
- [ ] **Commission Math Workbook:** Python notebook documenting minimum viable target size, breakeven win rates, annual commission budget
- [ ] **Asyncio Bot Skeleton:** Runnable Python project with clean module structure, config management, and logging — ready to receive signal logic


---

# Phase 2: Tick Data Exploration & Regime Baseline

**Duration:** 3 Weeks  |  **Goal:** Understand the statistical DNA of MES before building any signals.

## Overview

This phase is where your existing BayesBot HMM work becomes a direct competitive advantage. Most retail traders skip statistical characterization entirely and jump straight to building indicators — they end up with strategies that work in one regime and blow up in another. You're going to do this properly.

The foundational finding you're confirming from academic research (Safari & Schmidhuber 2025, applied to MES): there is a 'crossing point' between mean reversion and momentum at the 2–30 minute horizon. Below 2 minutes, apparent mean reversion is just bid-ask bounce — not tradable. Between 2 and 30 minutes, mean reversion is the dominant force. Above 30 minutes, momentum takes over. Your strategies will be designed to exploit the 2–30 minute zone specifically.

Beyond the reversion/momentum transition, you'll map the intraday volatility profile (confirming the U-shape and the dead zone), build your real-time feature library (CVD, VWAP deviation, order book imbalance), and extend your HMM to classify intraday regimes. The HMM state output from this phase becomes an input to every strategy in Phase 3.

## Key Concepts

| Concept | What It Means for Your Bot |
| --- | --- |
| **Hurst Exponent** | Measures the long-range dependence of a time series. H < 0.5: mean-reverting (anti-persistent). H = 0.5: random walk (no edge). H > 0.5: trending (momentum). For MES at 1-minute bars, H typically ranges 0.45–0.52 depending on time of day. Below 0.5 confirms the mean reversion regime we're targeting. |
| **Autocorrelation of Returns** | Measures whether a positive return at lag k predicts a positive or negative return now. Negative autocorrelation = mean reverting (bounce after a move). Positive = momentum. At 1-tick lags on MES, autocorrelation is strongly negative (bid-ask bounce). At 5–15 minute lags, it transitions. Plotting this across lags reveals your tradable horizon. |
| **Cumulative Volume Delta (CVD)** | Running sum of (buy volume − sell volume) over a session. Buy volume = trades that hit the ask (aggressive buyers). Sell volume = trades that hit the bid (aggressive sellers). When price rises but CVD falls, buyers are losing conviction — bearish divergence. CVD requires trade direction attribution from MBO data. |
| **VWAP Standard Deviation Bands** | Volume-Weighted Average Price anchored to session open. Standard deviation bands calculated from volume-weighted price variance. 2-SD band contains ~95% of price action on most days. Price at ±2 SD signals statistical overextension — the core entry trigger for VWAP reversion strategies. |
| **Volume Profile & POC** | Histogram of volume traded at each price level during a session or rolling window. Point of Control (POC) = price with most volume traded. Value Area = price range containing 70% of volume. POC acts as a strong mean reversion magnet — price tends to revisit high-volume nodes. Prior session POC is particularly reliable. |
| **HMM Regime States** | Hidden Markov Model classifies unobserved market 'states' from observable features (returns, volatility, volume). Your 5-state model: (1) High-Vol Trend Up, (2) High-Vol Trend Down, (3) Low-Vol Range, (4) Breakout Expansion, (5) Mean Reversion. Each state has different optimal strategy behavior — this is the intelligence layer that routes signal logic. |

## Tasks (5 total)

### Dollar Bar Construction

**Effort:** 3 days

#### Why This Matters

Standard time bars (1-min, 5-min) have a critical flaw: they sample the same number of bars during low-volatility midnight sessions as during the high-volatility open — even though the information content is wildly different. Dollar bars normalize for this, producing more stationary return series that ML models can learn from more reliably.

#### Steps

1. Define your dollar bar threshold. MES notional = price × $5. At MES ~5500, one contract = $27,500. A reasonable dollar bar threshold is $12,500–$25,000 (0.5–1 contract notional), producing ~200–400 bars per RTH session. Tune based on desired bar frequency.

2. Implement the dollar bar aggregator: iterate tick data chronologically, accumulate traded dollar volume (price × size × $5), emit a new bar when threshold crossed. Each bar: open, high, low, close, volume, vwap, num_trades, buy_volume, sell_volume (from aggressor side), timestamp_open, timestamp_close.

3. Compute bar returns (log returns preferred: ln(close/prev_close)). Run the Ljung-Box test for autocorrelation on dollar bar returns vs. 1-minute bar returns. Dollar bars should show lower autocorrelation (closer to IID). Verify with scipy.stats.kstest against normal distribution.

4. Run augmented Dickey-Fuller (ADF) test on dollar bar price series and return series. Price should be non-stationary (ADF p > 0.05 — as expected). Returns should be stationary (ADF p < 0.05). This confirms the bar type is valid for downstream analysis.

5. Store dollar bars in TimescaleDB as a hypertable with columns matching your bar schema. Create a materialized view that auto-populates new bars from the raw tick stream in near-real-time (update every 5 seconds during live trading).

#### ✅ Exit Criteria

> Dollar bar series built from MES tick data, IID test passing, stored in TimescaleDB

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the dollar bar construction module.

Create the following files:

**src/data/dollar_bars.py**
DollarBarConfig dataclass: symbol, dollar_threshold (default 12500.0), tick_value (5.0 for MES)

DollarBar dataclass: open, high, low, close, volume, vwap, num_trades, buy_volume, sell_volume, dollar_volume, timestamp_open (ns), timestamp_close (ns), bar_index

DollarBarBuilder class:
- on_tick(tick: TickData) → Optional[DollarBar]: returns completed bar or None
- Accumulates: dollar_volume, buy_volume (aggressor=BUY), sell_volume (aggressor=SELL), high/low tracking
- VWAP per bar: sum(price × size × tick_value) / sum(size × tick_value)
- Emits bar when accumulated dollar_volume >= dollar_threshold
- reset() called after each bar emission

DollarBarBackfiller class:
- build_from_parquet(parquet_dir, dollar_threshold, output_path) → builds dollar bar DataFrame from historical ticks
- Uses polars for performance (NOT pandas — too slow for tick data)
- Writes output as Parquet: date-partitioned dollar bar file
- Progress bar with tqdm showing: date being processed, bars generated so far

**src/analysis/bar_statistics.py**
BarStatistics class:
- autocorrelation_test(bars_df, lags=[1,2,5,10,20,50]) → DataFrame of autocorrelation by lag with p-values (scipy.stats.acf + Ljung-Box test)
- hurst_exponent(returns_series) → float using R/S method (hurst library)
- stationarity_test(series) → dict with ADF statistic, p-value, is_stationary (p < 0.05)
- bar_frequency_stats(bars_df) → dict: bars_per_day mean/std, bars_per_hour by time slot
- compare_bar_types(tick_df, time_bar_df, dollar_bar_df) → comparison of autocorrelation and variance ratios

**scripts/build_dollar_bars.py**
CLI script:
- Loads tick data from TimescaleDB for a date range
- Builds dollar bars using DollarBarBackfiller
- Runs BarStatistics tests: autocorrelation, ADF stationarity, Ljung-Box
- Saves bars to TimescaleDB table: mes_dollar_bars
- Prints summary: total bars built, avg bars/day, ADF test result, Ljung-Box p-value

**sql/create_dollar_bars.sql**
Hypertable for dollar bars with indexes.

**tests/test_dollar_bars.py**
Unit tests:
- test_bar_emits_at_threshold: synthetic ticks, verify bar emits exactly when dollar_volume >= threshold
- test_buy_sell_volume_attribution: ticks with known aggressor_side, verify buy/sell volumes correct
- test_vwap_calculation: known price/size sequence, verify VWAP matches hand calculation
- test_bar_reset_after_emission: verify state resets correctly for next bar
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Mean Reversion / Momentum Transition Analysis

**Effort:** 4 days

#### Why This Matters

This is the academic foundation for every strategy in Phase 3. If mean reversion on MES peaks at 4–8 minutes, your profit targets should be sized to capture 4–8 minutes of mean reversion. If momentum dominates at 30+ minutes, your VWAP reversion strategy needs to exit before that regime. Without this map, you're guessing.

#### Steps

1. Compute rolling autocorrelation of 1-tick MES returns at lags: 1, 2, 5, 10, 20, 50, 100, 200 ticks. Then convert to time by computing average seconds-per-tick during RTH. Plot autocorrelation vs. approximate time lag. Expect negative values (mean reversion) at short lags, crossing zero at 10–30 minutes.

2. Compute the Hurst exponent using the R/S method (hurst Python library) on windows of: 30s, 2m, 5m, 15m, 30m, 60m return series. For each window size, compute H over rolling 5-day periods. Plot H distribution by window size. Expect H < 0.5 at 2–15 min, H approaching 0.5 at 30+ min.

3. Reproduce the Safari & Schmidhuber analysis specifically on MES: fit AR(1) model at each timescale, extract the mean reversion coefficient. Positive coefficient = mean reverting, negative = momentum. Find the sign change — this is your 'crossing point' and your maximum hold time for mean reversion strategies.

4. Segment by time of day: compute autocorrelation separately for 9:30–11:00 AM, 11:00 AM–2:00 PM, and 2:00–4:00 PM. The dead zone (11 AM–2 PM) likely shows stronger mean reversion (tighter ranges) while the open shows mixed momentum/reversion depending on the day type.

5. Document findings in a Jupyter notebook with charts: (1) Autocorrelation vs. lag heatmap by time of day, (2) Hurst exponent vs. window size, (3) Mean reversion coefficient vs. timescale. This becomes your strategy design bible — every entry and exit parameter in Phase 3 should reference this analysis.

#### ✅ Exit Criteria

> Chart showing autocorrelation and Hurst exponent across horizons 30s–60m, reversion peak confirmed at 4–8 min on MES

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the mean reversion / momentum transition analysis module.

Create the following files:

**src/analysis/regime_characterization.py**
MeanReversionAnalyzer class:
- compute_autocorrelation_profile(returns_series, lags, confidence=0.95) → DataFrame: lag, autocorr, lower_ci, upper_ci, is_significant
- compute_hurst_by_window(returns_series, windows_seconds=[30,120,300,900,1800,3600]) → DataFrame: window_s, hurst_exp, interpretation
- compute_ar1_coefficient(returns_series, window_bars=100, step_bars=20) → rolling AR(1) coefficient — positive=momentum, negative=mean reversion
- find_crossing_point(ar1_by_lag_df) → the lag at which AR(1) coefficient crosses zero (reversion→momentum transition)
- segment_by_time_of_day(bars_df, segments={'open':(930,1100),'midday':(1100,1400),'close':(1400,1600)}) → dict of DataFrames

**notebooks/phase2-regime-characterization.ipynb**
Complete notebook with:

Cell 1: Load 6 months of MES dollar bars from TimescaleDB

Cell 2: Autocorrelation profile — plot autocorr vs. lag (in minutes) with 95% confidence bands. Shade the mean-reversion zone (negative autocorr) in green, momentum zone in blue.

Cell 3: Hurst exponent by window size — bar chart showing H for each window. Draw reference line at H=0.5 (random walk). Label: "Mean Reversion Zone" for H<0.5.

Cell 4: AR(1) coefficient rolling analysis — line chart over 6 months showing rolling AR(1) by different window sizes. When does the sign flip from negative to positive?

Cell 5: Time-of-day segmentation — compute autocorrelation separately for open (9:30-11AM), midday (11AM-2PM), and close (2-4PM). Show as 3-panel chart.

Cell 6: **FINDINGS TABLE** — explicit table:
| Timescale | Dominant Effect | Peak Effect At | Retail Tradable? |
Including the specific crossing point in minutes where MES transitions from reversion to momentum.

Cell 7: Strategy implications — written section interpreting findings for strategy design:
"Based on this analysis, VWAP reversion entries should target exits within X minutes. ORB momentum holds can extend to Y minutes. The dead zone shows Z character."

**docs/phase2/regime-characterization-findings.md**
Auto-generated from notebook conclusions. Template that gets filled in after running the notebook.
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Intraday Volatility & Volume Profile Mapping

**Effort:** 2 days

#### Why This Matters

Time-of-day filtering is one of the few 'free' edges available to retail traders. The U-shaped volatility profile is one of the most robustly documented patterns in academic finance (30+ years of consistent findings). Trading only during active hours dramatically improves signal quality and reduces commission drag from choppy dead-zone trades.

#### Steps

1. Compute 15-minute realized volatility (sum of squared 1-minute log returns, annualized) for every 15-minute window across 12 months of data. Average across all days by time slot. Plot as a heatmap: x-axis = time of day (9:30–16:00), y-axis = calendar month, color = realized vol.

2. Identify the U-shape inflection points: the time at which morning vol falls below 1.5× session average (typically 10:45–11:15 AM), and the time at which afternoon vol rises above 1.5× session average (typically 14:00–14:30). These are your session boundaries.

3. Compute average spread cost as a percentage of 15-min ATR by time slot. During the dead zone, ATR shrinks but tick spread stays constant — effective spread cost as % of move increases dramatically. This quantifies why dead-zone trading destroys P&L even for 'winning' strategies.

4. Separately analyze FOMC days (8 per year), NFP Fridays (12 per year), and CPI release days. These have dramatically different intraday volatility profiles. Decide your policy: sit out entirely, or build event-specific strategy variants. Document the decision.

5. Compute volume profile (volume at each price tick) for a rolling 5-day window. Identify the Point of Control (POC) and value area boundaries for each session. Store these as daily reference levels in TimescaleDB — your Phase 3 strategies will query prior-day POC as a signal input.

#### ✅ Exit Criteria

> 15-minute volatility heatmap by hour documented, dead zone boundaries defined with statistical confidence

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the intraday volatility mapping and volume profile module.

**src/analysis/intraday_profile.py**
IntradayVolatilityMapper class:
- compute_realized_vol_heatmap(bars_df, window_minutes=15) → DataFrame indexed by (time_slot, calendar_month): each cell = annualized realized vol
- compute_spread_cost_pct(bars_df, tick_value=1.25) → realized vol vs. 1-tick spread as % of ATR, by time slot
- identify_dead_zone(vol_heatmap, threshold_multiplier=1.5) → returns (dead_zone_start, dead_zone_end) times
- compute_u_shape_metrics(vol_heatmap) → dict: open_vol_mean, midday_vol_mean, close_vol_mean, u_shape_ratio

VolumeProfileBuilder class:
- build_session_profile(ticks_df, date) → dict: {price: volume_traded} for that session
- compute_value_area(profile_dict, pct=0.70) → (vah, val, poc) — value area high/low and point of control
- compute_rolling_profile(ticks_df, lookback_days=5) → rolling multi-day volume profile
- store_daily_levels(conn_string, date, vah, val, poc) → saves to TimescaleDB table: mes_volume_levels
- load_prior_session_levels(conn_string, date) → loads yesterday's VAH/VAL/POC

EventDayCalendar class:
- is_fomc_day(date) → bool
- is_nfp_day(date) → bool  
- is_cpi_day(date) → bool
- get_event_type(date) → Optional[str]
- load_event_calendar(csv_path) → loads from docs/data/economic-calendar.csv

**sql/create_volume_levels.sql**
Table for daily VAH/VAL/POC storage with index on date.

**scripts/build_volume_profiles.py**
- Computes daily volume profiles for all dates in the database
- Stores VAH/VAL/POC for each date in mes_volume_levels
- Prints: dates processed, avg value area width (ticks), avg POC stability (does POC move day to day?)

**notebooks/phase2-intraday-profile.ipynb**
Cell 1: Load data

Cell 2: Volatility heatmap — seaborn heatmap, x=time_slot (9:30, 9:45, ..., 15:45), y=month. Color=realized vol. Title: "MES 15-Min Realized Volatility by Time of Day"

Cell 3: U-shape chart — average realized vol by 15-min slot across all days. Overlay with 1-tick spread cost as % of ATR. Show dead zone shaded in gray.

Cell 4: Event day overlay — separate lines for normal days, FOMC days, NFP days. Show how event days distort the intraday profile.

Cell 5: Value area statistics — histogram of daily VAH-VAL width in ticks. Average POC stability. How often does price visit the prior POC?

Cell 6: **TRADING WINDOW DECISION** — explicit decision table:
| Session Window | Trade? | Reason |
| 9:30-10:00 | YES | Highest vol, ORB setup |
| 10:00-11:00 | YES | Active, good for VWAP |
| 11:00-14:00 | NO | Dead zone |
| 14:00-15:30 | YES | Afternoon pickup |
| 15:30-16:00 | CAUTION | MOC flow |
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Real-Time Feature Library

**Effort:** 4 days

#### Why This Matters

Your strategy signals in Phase 3 are functions of these features. Building the feature library as a separate, well-tested module means strategy logic stays clean and features can be reused across multiple strategies. This also enables fast parameter changes without touching data pipeline code.

#### Steps

1. Implement CVD (Cumulative Volume Delta): for each trade print, classify as buy (aggressor hit ask) or sell (aggressor hit bid) using the aggressor_side field from MBO data. Maintain a running sum: CVD += (buy_size - sell_size). Reset at session open. Expose: cvd_current, cvd_delta_per_bar (change in CVD per dollar bar), cvd_slope_20bar.

2. Implement VWAP with standard deviation bands: VWAP = sum(price × volume) / sum(volume) from session open. Variance = volume-weighted variance of price from VWAP. Bands at ±1, ±2, ±3 SD. Expose: vwap, vwap_deviation_sd (how many SDs current price is from VWAP), vwap_slope_20bar.

3. Implement order book imbalance ratio: using L2 data, compute: imbalance = (bid_volume_top5 - ask_volume_top5) / (bid_volume_top5 + ask_volume_top5). Range: -1 (full ask pressure) to +1 (full bid pressure). Compute rolling 20-tick average. Flag when current imbalance > 1.5× rolling average as 'elevated imbalance'.

4. Implement volume profile POC distance: load prior session's volume profile POC from TimescaleDB. Compute: poc_distance_ticks = abs(current_price - prior_poc) / tick_size. Update every dollar bar. Expose: poc_distance_ticks, price_above_poc (boolean), poc_proximity (True when within 4 ticks).

5. Implement session ATR: compute True Range for each 5-minute bar (high - low, with gaps accounted for). Rolling 14-bar ATR. Normalize stop losses as ATR multiples rather than fixed ticks. Expose: atr_5m, atr_5m_ticks.

6. Write unit tests for each feature: construct synthetic tick sequences with known properties (e.g., 100 buy ticks of size 1 = CVD +100) and assert feature values. Test edge cases: session open reset, reconnection after gap, negative CVD. Run tests in CI before every live deployment.

#### ✅ Exit Criteria

> All features computing in < 10ms from tick receipt, unit tested against known values

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the complete real-time feature library.

**src/features/__init__.py** — exports all features

**src/features/cvd.py**
CumulativeVolumeDelta class:
- on_tick(tick: TickData) → None: updates running state
- reset() → called at session open
- Properties: cvd (float), cvd_delta_per_bar (change since last bar), cvd_slope_20bar (linear regression slope of last 20 bar deltas), cvd_zscore (cvd normalized by rolling 20-bar std)
- divergence_from_price(price_high, cvd_at_high, current_price, current_cvd) → float: magnitude of price/CVD divergence (0.0 = no divergence, 1.0 = maximum)

**src/features/vwap.py**
VWAPCalculator class:
- on_tick(tick: TickData) → None
- reset() → session open
- Properties: vwap (float), deviation_sd (current price deviation in std devs), band_1sd_upper/lower, band_2sd_upper/lower, band_3sd_upper/lower, slope_20bar (float), is_flat (bool: abs(slope) < flat_threshold), is_trending (bool)
- first_kiss_detected(lookback_bars=6, sd_threshold=2.0) → bool: has price been >2SD then returned to within 0.5SD?

**src/features/orderbook.py**
OrderBookImbalance class:
- on_tick(tick: TickData) → None: updates with new L2 depth
- Properties: imbalance_ratio (float -1 to +1), imbalance_5tick_avg, imbalance_zscore, is_elevated (abs(imbalance) > 1.5 × rolling avg)
- depth_absorption(side: str, price_levels: int = 5) → float: bid or ask absorption at top N levels

**src/features/volume_profile_tracker.py**
LiveVolumeProfileTracker class:
- on_tick(tick: TickData) → None: updates session profile
- load_prior_session(conn_string, date) → loads VAH/VAL/POC from TimescaleDB
- Properties: poc_distance_ticks (from live price to prior POC), price_above_poc (bool), poc_proximity (bool: within 4 ticks), session_vah, session_val, session_poc (from prior day), live_poc (current session developing POC)

**src/features/atr.py**
ATRCalculator class:
- on_bar(bar: DollarBar) → None
- Properties: atr_14 (14-bar ATR), atr_ticks (ATR in ticks), vol_regime (str: LOW/NORMAL/HIGH based on ATR percentile), semi_variance_up, semi_variance_down, dominant_direction (str: UP/DOWN/NEUTRAL based on semi-variance comparison)

**src/features/feature_vector.py**
FeatureVector dataclass — all features combined for HMM and strategy consumption:
- timestamp_ns, symbol
- cvd, cvd_slope, cvd_zscore
- vwap, vwap_dev_sd, vwap_slope, vwap_is_flat
- book_imbalance, book_imbalance_zscore
- poc_distance_ticks, price_above_poc, poc_proximity
- atr_ticks, vol_regime, semi_var_up, semi_var_down, dominant_dir
- to_array() → np.ndarray of normalized features (for HMM input)
- to_dict() → dict (for logging/storage)

FeatureEngine class:
- Composes all feature calculators
- on_tick(tick) and on_bar(bar) → updates all features
- get_feature_vector() → FeatureVector
- reset() → session reset for all calculators

**tests/test_features.py**
Unit tests for each feature with synthetic data:
- test_cvd_buy_sell_attribution
- test_vwap_at_known_prices
- test_imbalance_ratio_bounds
- test_atr_matches_manual_calculation
- test_feature_vector_normalization
- test_session_reset_clears_state

**docs/phase2/feature-library.md**
Description of each feature, update frequency, valid range, and how strategies use it.
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### HMM Intraday Regime Classifier

**Effort:** 5 days

#### Why This Matters

This is the intelligence layer that makes your bot adaptive rather than brittle. A static strategy that always runs mean reversion will get destroyed during trending days. A strategy that always runs breakout will get chopped apart during range days. The HMM tells you which mode the market is in right now — and which strategy to deploy.

#### Steps

1. Define the 5 regime states based on your Phase 1 analysis: State 1 = High-Vol Trend Up (high ATR, positive returns, positive CVD slope), State 2 = High-Vol Trend Down (high ATR, negative returns, negative CVD slope), State 3 = Low-Vol Range (low ATR, VWAP deviation < 1 SD, flat CVD), State 4 = Breakout Expansion (rising ATR from low base, price leaving value area), State 5 = Mean Reversion (moderate ATR, price at VWAP ±2 SD, CVD divergence).

2. Build feature vector per dollar bar: [realized_vol_20bar, return_20bar, cvd_slope_20bar, vwap_deviation_sd, atr_5m, book_imbalance_20tick_avg, poc_distance_ticks]. Normalize each feature to z-score using rolling 250-bar window (prevents lookahead bias).

3. Extend your existing BayesBot HMM using hmmlearn's GaussianHMM. Train on 6 months of dollar bar features. Use 5 hidden states. Initialize with KMeans clustering to avoid random initialization instability. Set n_iter=100, covariance_type='full'.

4. Implement Viterbi decoding for most likely state sequence (offline, for backtesting). Implement online forward algorithm for real-time state probability updates — this gives a probability distribution over states, not just a point estimate. Expose: hmm_state (most likely), hmm_state_probs (5-dim array), hmm_state_age (bars in current state).

5. Evaluate on 2-month held-out test set. Primary metric: regime persistence accuracy (when in State X, what fraction of next N bars remain in State X?). Target > 65% persistence at 5-bar horizon. Secondary: Sharpe improvement from state-conditional vs. unconditional trading (does trading only in States 3 and 5 improve Sharpe vs. always trading?).

6. Implement rolling re-training: every 20 trading days, re-fit HMM on most recent 6 months of data. Store model snapshots with timestamps. During re-training (takes ~30 seconds), use previous model for inference. Log any state labeling changes between model versions.

#### ✅ Exit Criteria

> 5-state HMM with > 65% regime persistence accuracy on 2-month held-out test set, real-time inference < 5ms

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the HMM intraday regime classifier, extending the BayesBot architecture.

**src/models/hmm_regime.py**
RegimeState enum: HIGH_VOL_UP=0, HIGH_VOL_DOWN=1, LOW_VOL_RANGE=2, BREAKOUT=3, MEAN_REVERSION=4

HMMRegimeConfig dataclass: n_states=5, n_iter=100, covariance_type='full', lookback_bars_for_init=250, retrain_interval_days=20, model_dir='models/hmm/'

HMMRegimeClassifier class (extends BayesBot pattern):
- fit(feature_vectors: List[FeatureVector], dates: List[date]) → trains GaussianHMM using hmmlearn
  - Normalizes features using rolling z-score (no lookahead: uses expanding window during training)
  - Initializes emission means via KMeans(n_clusters=5) for stability
  - After fit, runs label_states() to assign semantic labels to HMM states by inspecting state emission means
- label_states() → maps internal HMM state indices to RegimeState enum by:
  - HIGH vol (high atr_ticks) + positive returns → HIGH_VOL_UP
  - HIGH vol + negative returns → HIGH_VOL_DOWN
  - LOW vol + flat VWAP → LOW_VOL_RANGE
  - Rising vol from low base → BREAKOUT
  - Moderate vol + high VWAP deviation → MEAN_REVERSION
- predict_viterbi(feature_vectors) → List[RegimeState]: offline most likely state sequence
- predict_online(feature_vector) → (RegimeState, np.ndarray): current state + probability distribution
- save(path) → pickle model + label mapping + normalization params
- load(path) → restores from disk

RollingRetrainer class:
- Runs in background asyncio task
- Every retrain_interval_days: loads most recent 6 months of feature vectors from TimescaleDB, retrains HMM, validates (accuracy >= 60%), if passes → atomically swaps model file, logs: "Model updated. State mapping: {old} → {new}"
- Validates by checking: regime persistence on test set (in state X, what % of next 5 bars also in state X?) >= 60%

**src/models/feature_store.py**
FeatureStore class:
- save_bar_features(feature_vector, conn_string) → stores to TimescaleDB table: mes_features
- load_features(conn_string, start_date, end_date) → returns List[FeatureVector]
- sql/create_feature_store.sql → TimescaleDB hypertable for features

**scripts/train_hmm.py**
CLI: python scripts/train_hmm.py --start 2023-01-01 --end 2024-06-01 --output models/hmm/v1/
- Loads feature vectors from TimescaleDB
- Trains HMM
- Prints: state label assignments, mean feature values per state, regime persistence accuracy on held-out 2 months
- Saves model
- Generates: docs/phase2/hmm-state-profiles.md with state characterization table

**notebooks/phase2-hmm-validation.ipynb**
Cell 1: Load model and test set features
Cell 2: State distribution — pie chart of time spent in each regime
Cell 3: Regime persistence matrix — heatmap: P(next state | current state)
Cell 4: Feature profiles per state — boxplots of key features by regime state
Cell 5: Regime vs. returns — did trading only in States 2 and 4 (LOW_VOL_RANGE, MEAN_REVERSION) improve Sharpe vs. always trading?
Cell 6: Regime timeline — gantt-style chart showing regime transitions over 2 months

**tests/test_hmm.py**
- test_model_trains_without_error
- test_label_states_assigns_all_enum_values
- test_online_prediction_matches_viterbi_on_same_data
- test_save_load_round_trip
- test_regime_persistence_above_threshold
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

## Phase 2 Gate

All exit criteria must pass before advancing to the next phase:

- [ ] **Dollar Bar Construction:** Dollar bar series built from MES tick data, IID test passing, stored in TimescaleDB
- [ ] **Mean Reversion / Momentum Transition Analysis:** Chart showing autocorrelation and Hurst exponent across horizons 30s–60m, reversion peak confirmed at 4–8 min on MES
- [ ] **Intraday Volatility & Volume Profile Mapping:** 15-minute volatility heatmap by hour documented, dead zone boundaries defined with statistical confidence
- [ ] **Real-Time Feature Library:** All features computing in < 10ms from tick receipt, unit tested against known values
- [ ] **HMM Intraday Regime Classifier:** 5-state HMM with > 65% regime persistence accuracy on 2-month held-out test set, real-time inference < 5ms


---

# Phase 3: Strategy Research & Signal Development

**Duration:** 4 Weeks  |  **Goal:** Build 5 clean, parameterizable strategy classes. No optimization yet — just correct logic.

## Overview

Phase 3 is the most intellectually engaging part of the project — and the most dangerous if you're not disciplined. The temptation is to immediately start tweaking parameters when something doesn't look right in early backtests. Resist this completely. The goal of this phase is correct logic implementation, not optimization. Optimization happens in Phase 4 under CPCV.

Each of the five strategies targets a distinct structural edge: ORB exploits the directional momentum of the opening range, VWAP Reversion exploits statistical overextension, CVD Divergence exploits informed-vs-uninformed order flow imbalances, the Volatility Regime Switcher adapts risk-reward dynamically, and the 80% Rule exploits the gravitational pull of the prior session's value area. Together they cover multiple market conditions — ensuring you always have at least one deployable strategy.

Every strategy class must follow the same interface: __init__(config), on_tick(tick), on_bar(bar), generate_signal() → Signal | None. This uniformity means the OMS, risk system, and backtester don't need to know which strategy they're running. It also enables running multiple strategies simultaneously in Phase 6.

## Key Concepts

| Concept | What It Means for Your Bot |
| --- | --- |
| **Signal Object** | Standardized output from every strategy: {direction: LONG\|SHORT, entry_price: float, target_price: float, stop_price: float, signal_time: datetime, strategy_id: str, confidence: float, regime_state: int}. The OMS consumes this object without knowing the strategy internals. |
| **Opening Range** | The high and low of the first N minutes of RTH trading (9:30 AM ET). The 15-minute opening range (9:30–9:45 AM) has the strongest academic evidence for breakout validity. The range represents the initial price discovery battle between buyers and sellers — a clean break suggests one side has won. |
| **VWAP First Kiss** | After price has deviated significantly from VWAP (2+ SD) and begins to revert, the 'first kiss' is the initial return to the VWAP line. This tends to be a high-probability entry point because: (1) mean reversion pressure is at maximum from the deviation, (2) institutional traders often use VWAP as a benchmark and will trade against deviation, (3) the reversal has already begun (momentum confirmation). |
| **Semi-Variance** | Measures variance using only returns above (upside semi-variance) or below (downside semi-variance) the mean. Useful for characterizing asymmetric volatility. When upside semi-variance >> downside, the market is making sharp upward moves and grinding down — different strategy behavior needed than symmetric volatility. |
| **Value Area** | The price range containing 70% of a session's volume. Derived from volume profile. The 80% Rule states: when price opens outside the prior session's value area and trades back inside, there is approximately 80% historical probability of traversing the entire value area. This is one of the most consistently documented patterns in equity index futures. |
| **Filter Stacking** | Combining multiple independent conditions before entry. Example ORB filter stack: (1) breakout bar closes outside range, (2) volume on breakout bar > 1.5× 20-day average, (3) VWAP is above breakout level (for longs), (4) HMM state is not Low-Vol Range (State 3), (5) not within 30 minutes of FOMC. Each filter reduces trade frequency but increases per-trade edge. |

## Tasks (6 total)

### Strategy Base Class & Interface

**Effort:** 1 day

#### Why This Matters

Every strategy in this phase must conform to the same interface. This isn't just good software engineering — it directly enables Phase 4's backtester to run all strategies identically, Phase 5's paper trader to switch strategies with a config change, and Phase 6's live bot to run multiple strategies simultaneously.

#### Steps

1. Define Signal dataclass: direction (Enum: LONG/SHORT), entry_price, target_price, stop_price, signal_time, expiry_time (signal valid until), strategy_id (str), confidence (0.0–1.0), regime_state (int), metadata (dict for debug info). Use Python dataclasses with frozen=True for immutability.

2. Define abstract StrategyBase class with abstract methods: on_tick(tick: TickData) → None, on_bar(bar: DollarBar) → None, generate_signal() → Optional[Signal], reset() → None (called at session open/close). Add concrete methods: is_active_session() → bool, get_metrics() → dict.

3. Implement StrategyConfig base dataclass with shared fields: symbol, max_signals_per_day (default 5), session_start ('09:30'), session_end ('16:00'), excluded_windows (list of datetime ranges for FOMC etc.), require_hmm_states (list of valid HMM states for entry).

4. Write a MockStrategy that implements the interface and generates random signals. Run it through a mini event loop for 100 ticks. Verify: signals have all required fields, expiry is honored, session boundaries are respected. This template ensures all real strategies implement the interface correctly.

#### ✅ Exit Criteria

> Abstract base class implemented, Signal dataclass defined, mock strategy passing all interface tests

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the strategy base class and signal interface that all 5 strategies implement.

**src/strategies/__init__.py**

**src/strategies/models.py**
Direction enum: LONG, SHORT
SignalStatus enum: PENDING, ACTIVE, FILLED, EXPIRED, CANCELLED

Signal dataclass (frozen=True):
- id: str (uuid4)
- strategy_id: str
- direction: Direction
- entry_price: float
- target_price: float
- stop_price: float
- signal_time: datetime
- expiry_time: datetime (signal invalid after this — don't enter)
- confidence: float (0.0–1.0)
- regime_state: RegimeState
- metadata: dict (for debug — include all relevant feature values at signal time)
- risk_reward_ratio property → abs(target_price - entry_price) / abs(stop_price - entry_price)
- ticks_to_target property → abs(target_price - entry_price) / 1.25
- ticks_to_stop property → abs(stop_price - entry_price) / 1.25

StrategyConfig dataclass:
- strategy_id: str
- symbol: str = "MESH5"
- max_signals_per_day: int = 5
- session_start: str = "09:30"
- session_end: str = "16:00"
- excluded_windows: List[Tuple[str, str]] = field(default_factory=list)  # [(start, end), ...]
- require_hmm_states: List[RegimeState] = field(default_factory=list)  # empty = all states ok
- min_confidence: float = 0.6

**src/strategies/base.py**
StrategyBase abstract class:
- __init__(config: StrategyConfig, feature_engine: FeatureEngine, hmm: HMMRegimeClassifier)
- Abstract methods:
  - on_tick(tick: TickData) → None
  - on_bar(bar: DollarBar) → None
  - generate_signal() → Optional[Signal]
  - reset() → None (called at session open/close)
- Concrete methods:
  - is_active_session() → bool (checks current time vs. session window)
  - is_valid_hmm_state() → bool (checks current HMM state against require_hmm_states)
  - can_generate_signal() → bool (is_active_session AND is_valid_hmm_state AND signals_today < max_signals_per_day AND not halted)
  - _make_signal(direction, entry, target, stop, confidence, metadata) → Signal (handles id, timestamps)
  - get_daily_metrics() → dict: {signals_today, wins, losses, avg_confidence}
  - _log_signal(signal) → structured log entry

**src/strategies/mock_strategy.py**
MockStrategy(StrategyBase) for testing:
- Generates a LONG signal every 10 bars with random confidence
- Used to test OMS and backtester without real signal logic

**tests/test_strategy_base.py**
- test_signal_creation_has_required_fields
- test_cannot_generate_outside_session
- test_hmm_state_filter_blocks_wrong_states  
- test_max_signals_per_day_limit
- test_signal_expiry_honored
- test_risk_reward_calculation
- Use MockStrategy for all tests
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### ORB Strategy (Opening Range Breakout)

**Effort:** 4 days

#### Why This Matters

ORB is the most academically studied intraday strategy and the most robust starting point. It requires no real-time order flow data (just OHLCV), is easy to reason about, and has clear, testable logic. It's also a natural filter for HMM state — ORB works best in high-volatility trending regimes (States 1 and 2).

#### Steps

1. Implement opening range tracking: from 9:30:00 to 9:44:59 ET, track the highest high and lowest low across all 1-minute bars. At 9:45:00, freeze the range. Log: orb_high, orb_low, orb_width (in ticks), orb_volume (total volume during range period).

2. Implement breakout detection: on each completed 5-minute bar after 9:45 AM, check if close > orb_high (long breakout) or close < orb_low (short breakout). Require the close to be beyond the range, not just a wick — this eliminates false breaks that reverse immediately.

3. Apply filter stack: (a) Breakout bar volume must be > volume_multiplier (default 1.5) times 20-day average volume for that 5-minute slot. (b) For long breakouts: session VWAP must be below current price (price above VWAP). (c) HMM state must not be State 3 (Low-Vol Range) — ORB fails in ranges. (d) Signal must be generated before 11:00 AM — late ORBs have poor reliability.

4. Set risk parameters: target = entry + (orb_width × target_multiplier, default 0.5). Stop = entry - orb_width (for longs) or entry + orb_width (for shorts). This gives a 1:0.5 risk-reward, but win rate should be 55–65% if filtered properly. Maximum hold time: 90 minutes — exit at market if not stopped or targeted.

5. Implement the signal object construction with all metadata: include orb_width, breakout_volume, vwap_at_signal, hmm_state, time_since_open. This metadata is critical for Phase 4 analysis — you'll use it to understand which conditions produced winning vs. losing trades.

6. Backtest over 3 months of unoptimized historical data (no parameter changes). Log all signals. Manually verify 10 random signals against charts: does the signal fire at the right bar? Is the target/stop at the right price? Fix any discrepancies before proceeding.

#### ✅ Exit Criteria

> ORB class generating signals on historical data matching manual chart analysis on 10 randomly selected days

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the Opening Range Breakout (ORB) strategy.

**src/strategies/orb_strategy.py**
ORBConfig(StrategyConfig):
- orb_window_minutes: int = 15 (9:30–9:45)
- breakout_bar_type: str = "5min" (bar that must close outside range)
- target_multiplier: float = 0.5 (target = entry + orb_width × multiplier)
- volume_multiplier: float = 1.5 (breakout bar must have this × 20-day avg volume)
- max_signal_time: str = "11:00" (no ORB signals after this time)
- require_vwap_alignment: bool = True
- require_hmm_states: List[RegimeState] = [HIGH_VOL_UP, HIGH_VOL_DOWN] (ORB doesn't work in ranges)

ORBStrategy(StrategyBase):
State machine:
- COLLECTING_RANGE (9:30–9:45): track high/low/volume of each 1-min bar
- RANGE_SET (after 9:45): orb_high, orb_low, orb_width, orb_volume frozen
- WATCHING_BREAKOUT: checking each 5-min bar close for breakout
- SIGNAL_GENERATED: signal emitted, waiting for fill or expiry
- INACTIVE: after 11:00 or daily limit hit

on_bar(bar):
- During COLLECTING_RANGE: update range_high, range_low, range_volume
- At 9:45:00 transition: freeze range, compute 20-day avg volume for this time slot (query feature store), set state=RANGE_SET
- During WATCHING_BREAKOUT: check each 5-min bar
  - Long breakout: bar.close > orb_high
  - Apply filters (see below)
  - If pass: generate_signal() → LONG signal

Filter implementation:
1. Volume check: bar.volume > volume_multiplier × avg_volume_for_this_slot
2. VWAP alignment: for longs, feature_engine.vwap.vwap < bar.close
3. HMM state: self.is_valid_hmm_state()
4. Time gate: current_time < max_signal_time
5. Single signal: only 1 ORB signal per day

Signal construction:
- entry_price = bar.close (assume fill at close of breakout bar + 1 tick slippage)
- target_price = entry + (orb_width × target_multiplier)
- stop_price = orb_low (for longs) — the opposite range boundary
- expiry_time = signal_time + timedelta(minutes=90)
- confidence = min(1.0, (bar.volume / avg_volume) / volume_multiplier) × hmm_probability

metadata = {orb_high, orb_low, orb_width, breakout_volume, avg_volume, volume_ratio, vwap_at_signal, hmm_state, hmm_state_probs, time_since_open_minutes}

**tests/test_orb.py**
Use synthetic bar sequences to test:
- test_range_correctly_collected_9_30_to_9_45
- test_no_signal_on_wick_break_only_close
- test_long_signal_above_orb_high
- test_short_signal_below_orb_low
- test_volume_filter_blocks_low_volume_breakout
- test_no_signal_after_11am
- test_only_one_signal_per_day
- test_target_at_correct_price (entry + orb_width × 0.5)
- test_stop_at_opposite_range_boundary
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### VWAP Reversion Strategy

**Effort:** 4 days

#### Why This Matters

VWAP reversion exploits the strongest documented mean reversion effect on equity index futures: statistical overextension from the volume-weighted average. The regime switch (flat VWAP = reversion, steep VWAP = pullback) is what separates a sophisticated implementation from a naive one that gets destroyed on trending days.

#### Steps

1. Implement real-time VWAP calculation anchored to RTH open (9:30 AM). Update every tick: vwap = running_dollar_volume / running_volume. Compute volume-weighted variance: var = running_weighted_sq_deviation / running_volume. Standard deviation = sqrt(var). Expose bands at ±1, ±2, ±3 SD.

2. Implement VWAP slope detection: compute VWAP change per dollar bar over last 20 bars. Classify: abs(slope) < threshold_flat → range/reversion mode. abs(slope) > threshold_trending → pullback mode. Tune thresholds using the volatility characterization from Phase 2 (dead zone vs. active hours).

3. Reversion mode entry logic: price crosses below –2 SD VWAP (or above +2 SD for shorts). Wait for a reversal bar confirmation (next bar closes back toward VWAP, or a bullish engulfing pattern at –2 SD). Generate signal: entry = current close, target = VWAP, stop = –3.5 SD VWAP.

4. First-kiss variant: detect when price has been > 2 SD from VWAP for 6+ bars (30+ minutes if using 5-min bars). When price closes within 0.5 SD of VWAP for the first time, enter in the direction of VWAP from the extreme. This entry has higher probability because the reversion is already confirmed.

5. Pullback mode entry logic (trending session): enter longs on pullbacks to VWAP from above (price touches VWAP, then shows a reversal bar back in trend direction). Target: 2× distance from VWAP to entry. Stop: 0.5 SD below VWAP. This converts the strategy from fade to trend-following in trending sessions.

6. Add filter: skip VWAP signals within 15 minutes of session open (VWAP not yet meaningful) and within 30 minutes of major economic releases. Add HMM state filter: reversion mode only in States 3 and 5. Pullback mode only in States 1 and 2.

#### ✅ Exit Criteria

> VWAP class generating correct SD band signals, regime-switching between reversion and pullback modes based on VWAP slope

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the VWAP Reversion strategy with regime-switching between reversion and pullback modes.

**src/strategies/vwap_strategy.py**
VWAPConfig(StrategyConfig):
- entry_sd_reversion: float = 2.0 (enter when price at this many SDs from VWAP)
- stop_sd: float = 3.5 (stop loss at this SD level)
- flat_slope_threshold: float = 0.002 (abs VWAP slope below this = flat/range session)
- trending_slope_threshold: float = 0.005 (above this = trending)
- first_kiss_lookback_bars: int = 6
- min_session_age_minutes: int = 15 (don't trade first 15 min)
- require_reversal_candle: bool = True

VWAPMode enum: REVERSION, PULLBACK, NEUTRAL

VWAPStrategy(StrategyBase):
Properties:
- current_mode: VWAPMode (updates each bar based on VWAP slope)
- last_mode_change_bar: int

on_bar(bar):
1. Update mode based on vwap.slope_20bar:
   - abs(slope) < flat_threshold → REVERSION
   - abs(slope) > trending_threshold → PULLBACK
   - else → NEUTRAL (no trade)
2. If mode changed: log mode transition, don't trade for 2 bars (transition noise)
3. If REVERSION mode and session age > min_session_age_minutes:
   - Check: abs(vwap.deviation_sd) >= entry_sd_reversion
   - Check: require_reversal_candle → bar reverses toward VWAP
   - Check: HMM state in [LOW_VOL_RANGE, MEAN_REVERSION]
   - Generate reversion signal
4. If PULLBACK mode:
   - Check: price within 0.25 SD of VWAP (pulled back to VWAP)
   - Check: VWAP slope direction matches dominant market direction
   - Check: bar shows continuation (close in direction of slope)
   - Check: HMM state in [HIGH_VOL_UP, HIGH_VOL_DOWN]
   - Generate pullback signal

Reversion signal:
- entry = bar.close
- target = feature_engine.vwap.vwap (revert to VWAP)
- stop = -3.5 SD VWAP level (for long) or +3.5 SD (for short)
- confidence based on: abs(deviation) / 3.0 (capped at 1.0) × hmm_state_prob

First-kiss variant (additional check):
- If feature_engine.vwap.first_kiss_detected(lookback=6, threshold=2.0):
  - Boost confidence by 0.15
  - Tighten stop to -3.0 SD (higher conviction)

Pullback signal:
- entry = bar.close
- target = entry + 2 × (entry - vwap) in trend direction (2× extension beyond VWAP)
- stop = 0.5 SD beyond VWAP level

**tests/test_vwap.py**
- test_mode_switches_on_slope_change
- test_no_trade_during_mode_transition
- test_reversion_entry_at_correct_sd_level
- test_pullback_entry_only_when_price_at_vwap
- test_first_kiss_boosts_confidence
- test_no_signal_first_15_minutes
- test_stop_at_correct_sd_level
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### CVD Divergence + POC Strategy

**Effort:** 5 days

#### Why This Matters

CVD divergence is the most technically demanding strategy to implement — it requires MBO tick data and careful real-time calculation — but it captures genuine informed-vs-uninformed order flow imbalances. When price makes a new high but aggressive buying is drying up, it reveals that the price move was driven by passive buyers being hit, not by new aggressive demand entering. This is an early exhaustion signal.

#### Steps

1. Implement trade direction classification from MBO data: for each trade, if aggressor_side = BUY (trade hit ask), add volume to buy_volume. If aggressor_side = SELL (trade hit bid), add to sell_volume. Compute delta = buy_volume - sell_volume per dollar bar. Cumulative delta = running sum since session open.

2. Implement swing detection for both price and CVD: track local highs (bars where close > max of surrounding 3 bars) and local lows. Maintain a rolling window of last 5 swing highs and 5 swing lows for both price series and CVD series.

3. Bearish divergence: price makes a new swing high AND CVD fails to make a new swing high (or makes a lower swing high). The divergence must be significant: CVD swing high must be at least cvd_divergence_threshold% lower than previous CVD swing high while price is higher. Typical threshold: 15–25%.

4. Bullish divergence: price makes a new swing low AND CVD fails to confirm (makes a higher low while price lower). Same significance threshold applies. Only generate signal when divergence occurs within poc_proximity_ticks (default 6 ticks) of the prior 4-hour volume profile POC.

5. Risk parameters: target = 12–16 ticks (3–4 points). Stop = beyond the swing extreme that produced the divergence + 2 ticks buffer. Maximum hold: 4 dollar bars (~20 minutes at typical bar frequency). This strategy generates fewer signals (1–3/day) with higher confidence — do not force trades.

6. Build divergence visualization: for every generated signal, save a chart showing price, CVD, and the POC level for visual verification. Review 20 signals manually — divergences should be visually obvious, not marginal. If many signals look questionable, increase the divergence significance threshold.

#### ✅ Exit Criteria

> CVD divergence detector generating 1–3 signals per day on historical data, verified against manual order flow analysis

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the CVD Divergence + Volume Profile POC strategy.

**src/strategies/cvd_divergence_strategy.py**
CVDDivergenceConfig(StrategyConfig):
- swing_lookback_bars: int = 5 (bars each side for swing detection)
- divergence_threshold_pct: float = 0.15 (CVD swing must differ by this % to count)
- poc_proximity_ticks: int = 6 (divergence must occur within this many ticks of POC)
- target_ticks: int = 14
- max_hold_bars: int = 4
- max_signals_per_day: int = 3

SwingDetector class (internal):
- on_bar(bar) → Optional[Swing]: returns Swing if current bar is a swing high/low
- Swing detection: bar.close > max(surrounding n bars) = swing high
- Maintains rolling window of last 5 swing highs and 5 swing lows (both price and CVD value at each swing)
- Swing dataclass: bar_index, price, cvd_value, timestamp, swing_type (HIGH/LOW)

DivergenceDetector class (internal):
- check_bearish(recent_price_highs: List[Swing], recent_cvd_highs: List[Swing]) → Optional[DivergenceSignal]
  - Bearish: price makes higher high but CVD makes lower high
  - Magnitude: (prev_cvd_high - current_cvd_high) / prev_cvd_high > divergence_threshold_pct
- check_bullish(recent_price_lows, recent_cvd_lows) → Optional[DivergenceSignal]

CVDDivergenceStrategy(StrategyBase):
- on_bar(bar):
  1. Update swing detector with current bar (price and CVD value)
  2. Run divergence checks
  3. If divergence detected: check POC proximity (poc_distance_ticks <= poc_proximity_ticks)
  4. If proximity passes: generate signal
  5. Track open signal age; if max_hold_bars exceeded, expire

Signal construction:
- direction: SHORT for bearish div, LONG for bullish
- entry: bar.close
- target: entry ± (target_ticks × 1.25)
- stop: swing_extreme + 2 ticks buffer (beyond the diverging swing)
- expiry: signal_time + estimated_bar_duration × max_hold_bars
- confidence: divergence_magnitude (capped 0.0–1.0) × poc_proximity_factor (1.0 at POC, 0.5 at max distance)
- metadata: {divergence_magnitude, price_swing_high, cvd_at_high, poc_distance_ticks, prior_poc, prior_vah, prior_val}

Visualization helper:
- plot_divergence(bars_df, signal: Signal, output_path) → saves matplotlib chart showing price, CVD, POC line, and divergence markers. Used for manual verification of signal quality.

**scripts/verify_cvd_signals.py**
Runs strategy on historical data, generates charts for 20 random signals, saves to docs/phase3/cvd-signal-verification/

**tests/test_cvd_divergence.py**
- test_swing_detected_at_correct_bar
- test_bearish_divergence_detected
- test_bullish_divergence_detected
- test_below_threshold_not_detected
- test_poc_proximity_filter_blocks_far_divergence
- test_max_signals_per_day_limit
- test_signal_expires_after_max_hold_bars
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Volatility Regime Switcher Strategy

**Effort:** 4 days

#### Why This Matters

Market microstructure changes dramatically between high-volatility and low-volatility periods — not just the magnitude of moves, but their character. In high-vol periods, momentum dominates and wide targets with tight stops capture trending moves. In low-vol periods, mean reversion dominates and high win-rate scalping with tight targets is viable. This strategy explicitly adapts rather than assuming static conditions.

#### Steps

1. Implement upside and downside semi-variance: over a rolling 20-bar window, compute: upside_var = mean of squared positive returns, downside_var = mean of squared negative returns. Classify regime: if max(upside_var, downside_var) > 75th percentile of rolling 250-bar history → high-vol. Else → low-vol.

2. High-volatility mode: trade in the direction of the dominant semi-variance (upside_var > downside_var → long bias, vice versa). Entry on a 3-bar pullback against the dominant direction (buy dips in uptrend). Target: 8 ticks. Stop: 2 ticks. This produces a negatively skewed strategy (frequent small losses, occasional large wins) — acceptable because win rate is not the goal.

3. Low-volatility mode: fade extensions from session VWAP. Entry when price moves 1.5 SD from VWAP in either direction. Target: 2 ticks. Stop: 30 ticks. Win rate target: 85%+. The wide stop is intentional — in true range conditions, the stop almost never hits. The danger is a regime shift mid-trade, which the HMM state check helps prevent.

4. Regime transition handling: when semi-variance crosses the 75th percentile threshold (high-vol to low-vol or vice versa), immediately cancel any open signals from the previous mode. Do not enter new signals for 2 bars after a regime transition — the signal-to-noise ratio is poorest during transitions.

5. Add HMM guard: cross-validate semi-variance regime against HMM state. High-vol mode should align with HMM States 1, 2, or 4. Low-vol mode should align with States 3 or 5. If the two models disagree (semi-var says high-vol, HMM says low-vol), do not trade — conflicting signals indicate ambiguous conditions.

#### ✅ Exit Criteria

> Semi-variance classifier correctly identifying high-vol vs. low-vol regimes, strategy generating appropriate risk-reward for each

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the Volatility Regime Switcher strategy.

**src/strategies/vol_regime_strategy.py**
VolRegimeConfig(StrategyConfig):
- semi_var_lookback_bars: int = 20
- semi_var_high_vol_percentile: float = 0.75
- semi_var_history_bars: int = 250
- high_vol_target_ticks: int = 8
- high_vol_stop_ticks: int = 2
- low_vol_target_ticks: int = 2
- low_vol_stop_ticks: int = 30
- low_vol_entry_sd: float = 1.5 (enter when price at this many SDs from VWAP)
- pullback_bars: int = 3 (high-vol: enter on N-bar pullback against dominant direction)
- transition_cooldown_bars: int = 2

VolatilityRegime enum: HIGH_VOL, LOW_VOL, TRANSITIONING

SemiVarianceClassifier class:
- on_bar(bar) → None: updates rolling return history
- current_regime property → VolatilityRegime
- compute_semi_variances(returns) → (upside_var, downside_var)
- percentile_threshold property → 75th percentile of last semi_var_history_bars max(semi_var)
- dominant_direction property → "UP" if upside_var > downside_var else "DOWN"
- transition_detected property → did regime change on last bar?

VolRegimeSwitcherStrategy(StrategyBase):
State:
- current_regime: VolatilityRegime
- regime_age_bars: int (bars in current regime)
- bars_since_transition: int
- pullback_counter: int (for high-vol pullback detection)

on_bar(bar):
1. Update SemiVarianceClassifier
2. Cross-validate with HMM: if semi_var says HIGH_VOL but HMM says LOW_VOL_RANGE → set regime TRANSITIONING, skip signal
3. If TRANSITIONING: skip for transition_cooldown_bars, then adopt new regime
4. If HIGH_VOL regime:
   - Track pullback_counter: bars moving against dominant_direction
   - If pullback_counter >= pullback_bars: pullback has occurred → generate HIGH_VOL signal
   - Reset pullback_counter on signal
5. If LOW_VOL regime:
   - Check: abs(vwap.deviation_sd) >= low_vol_entry_sd
   - Generate LOW_VOL signal (fade the extension)

HIGH_VOL signal:
- direction = dominant_direction
- entry = bar.close
- target = entry ± (high_vol_target_ticks × 1.25)
- stop = entry ∓ (high_vol_stop_ticks × 1.25)
- confidence = (upside_var - downside_var) / max(upside_var, downside_var) (normalized asymmetry)

LOW_VOL signal:
- direction = opposite of deviation (long if below VWAP)
- entry = bar.close
- target = entry ± (low_vol_target_ticks × 1.25)
- stop = entry ∓ (low_vol_stop_ticks × 1.25)
- confidence = abs(vwap_deviation_sd) / 3.0 (capped 1.0)

**tests/test_vol_regime.py**
- test_regime_classified_high_vol_above_75th_percentile
- test_regime_classified_low_vol_below_threshold
- test_transitioning_blocks_signals
- test_high_vol_signal_requires_n_bar_pullback
- test_low_vol_signal_at_vwap_extension
- test_hmm_conflict_creates_transitioning_state
- test_direction_follows_dominant_semi_variance
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### 80% Rule (Value Area) Strategy

**Effort:** 3 days

#### Why This Matters

The 80% Rule is one of the most consistently documented patterns in futures trading, taught in Market Profile methodology. It has a clear structural explanation: when price re-enters the prior session's value area, the institutions who established that value area during the prior session will often resume similar behavior — pushing price back to the opposite boundary.

#### Steps

1. Load prior session value area from TimescaleDB: prior_vah (value area high), prior_val (value area low), prior_poc (point of control). These are computed end-of-day and stored for the next session. At session open, determine: is price inside or outside the prior value area?

2. Outside-open detection: if the 9:30 AM open is outside the prior value area, monitor for re-entry. Re-entry = a 30-minute bar closes inside the value area. Start hold confirmation counter.

3. Two-period hold confirmation: track whether price holds inside the value area for 2 consecutive 30-minute periods. If it fails to hold (closes back outside), reset — the rule is invalidated. If it holds both periods, generate a signal toward the opposite value area boundary.

4. Signal construction: entry = close of the second confirming 30-minute bar. Target = opposite value area boundary (prior_val if opened above, prior_vah if opened below). Stop = re-break of the value area boundary (exit if price closes back outside). This is a wide stop by design — the 80% pattern implies price should not exit the VA again if the pattern is valid.

5. Session timing: this strategy almost always triggers in the first 2 hours of RTH (outside opens re-enter quickly or not at all). If no confirming signal by 11:30 AM, invalidate for the day. Add filter: require at least 60% of prior session's volume to have occurred within the value area (ensures VA is meaningful).

#### ✅ Exit Criteria

> 80% Rule class correctly identifying value area breaches and 2-period hold confirmations, generating signals on historical data

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the 80% Value Area Rule strategy.

**src/strategies/value_area_strategy.py**
ValueAreaConfig(StrategyConfig):
- hold_periods_required: int = 2 (consecutive 30-min periods inside VA)
- period_minutes: int = 30
- outside_open_tolerance_ticks: int = 2 (open must be at least this far outside VA)
- min_prior_session_va_pct: float = 0.60 (60% of prior session volume must be inside VA)
- invalidation_time: str = "11:30" (invalidate if no signal by this time)

ValueAreaState enum: WATCHING_OPEN, WAITING_FOR_REENTRY, CONFIRMING_HOLD, SIGNAL_GENERATED, INVALIDATED

ValueAreaStrategy(StrategyBase):
State:
- current_state: ValueAreaState
- open_price: float
- open_outside_direction: str (ABOVE or BELOW prior VA)
- hold_period_count: int
- current_30min_bar_start: datetime
- current_30min_bar_close: Optional[float]

Initialization (session open 9:30 AM):
1. Load prior session VAH/VAL/POC from FeatureStore
2. Determine open_price (first tick of session)
3. If open outside VA by >= outside_open_tolerance_ticks: set WAITING_FOR_REENTRY
4. Else: set INVALIDATED (inside open, no setup)
5. Validate: prior session had >= min_prior_session_va_pct volume inside VA

on_bar(bar) — using 30-minute synthetic bars (aggregated from dollar bars):
- Build 30-min bars internally from dollar bar stream
- WAITING_FOR_REENTRY: check if 30-min bar.close is inside VA
  - If yes: start hold confirmation, hold_period_count = 1
  - Set current_state = CONFIRMING_HOLD
- CONFIRMING_HOLD:
  - If bar.close still inside VA: hold_period_count += 1
  - If hold_period_count >= hold_periods_required: generate_signal()
  - If bar.close outside VA again: reset to WAITING_FOR_REENTRY (setup invalidated)
  - If current_time >= invalidation_time and no signal: INVALIDATED

Signal construction:
- direction: if open was ABOVE prior VA → SHORT (expecting traverse to VAL)
            if open was BELOW prior VA → LONG (expecting traverse to VAH)
- entry = close of second confirming bar
- target = opposite VA boundary (VAL for shorts, VAH for longs)
- stop = re-break of VA on a 30-min close (dynamic — not a fixed price at signal time)
- confidence based on: distance open was outside VA + prior session VA quality

**tests/test_value_area.py**
- test_inside_open_immediately_invalidated
- test_outside_open_above_va_detected
- test_reentry_starts_hold_count
- test_two_period_hold_generates_signal
- test_one_period_hold_then_exit_resets
- test_invalidation_after_1130
- test_target_is_opposite_va_boundary
- test_direction_correct_for_above_and_below_opens
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

## Phase 3 Gate

All exit criteria must pass before advancing to the next phase:

- [ ] **Strategy Base Class & Interface:** Abstract base class implemented, Signal dataclass defined, mock strategy passing all interface tests
- [ ] **ORB Strategy (Opening Range Breakout):** ORB class generating signals on historical data matching manual chart analysis on 10 randomly selected days
- [ ] **VWAP Reversion Strategy:** VWAP class generating correct SD band signals, regime-switching between reversion and pullback modes based on VWAP slope
- [ ] **CVD Divergence + POC Strategy:** CVD divergence detector generating 1–3 signals per day on historical data, verified against manual order flow analysis
- [ ] **Volatility Regime Switcher Strategy:** Semi-variance classifier correctly identifying high-vol vs. low-vol regimes, strategy generating appropriate risk-reward for each
- [ ] **80% Rule (Value Area) Strategy:** 80% Rule class correctly identifying value area breaches and 2-period hold confirmations, generating signals on historical data


---

# Phase 4: Rigorous Backtesting & Validation

**Duration:** 4 Weeks  |  **Goal:** Most strategies die here. That is the point.

## Overview

Phase 4 is where you find out whether your strategies have real edges or are statistical artifacts of the historical period you trained on. The goal is not to 'save' any particular strategy — it's to find the 2–3 strategies that genuinely survive rigorous out-of-sample testing. The ones that don't survive are not failures; they're expensive lessons that cost you nothing but time.

The two core validation tools in this phase — Combinatorial Purged Cross-Validation (CPCV) and Deflated Sharpe Ratio (DSR) — are from Marcos López de Prado's work at AQR and are used by institutional quant funds. They correct for the most common backtesting pathology: selecting the best-looking strategy from many trials, which inflates the apparent Sharpe Ratio by pure chance. López de Prado demonstrated that after 1,000 independent backtests on random noise, the expected maximum Sharpe is 3.26 — purely by luck. DSR corrects for this.

The exit criterion from this phase is deliberately strict: 2–3 strategies with PBO < 0.10, DSR ≥ 0.95, OOS Sharpe ≥ 50% of IS Sharpe across 6+ WFO cycles. If only 1 strategy survives, that's fine — go live with 1. If zero survive, go back to Phase 3 and build new hypotheses. Do not modify validation criteria to rescue a strategy you like.

## Key Concepts

| Concept | What It Means for Your Bot |
| --- | --- |
| **Combinatorial Purged Cross-Validation (CPCV)** | An improved k-fold cross-validation for time series. Generates multiple backtest paths by treating different combinations of time periods as train/test sets, while 'purging' training observations whose labeling horizon overlaps the test period to prevent leakage. Produces a Probability of Backtest Overfitting (PBO) metric. |
| **Deflated Sharpe Ratio (DSR)** | Corrects the observed Sharpe Ratio for: (1) the number of strategies tested (selection bias), (2) non-normality of returns (skewness and kurtosis), (3) the length of the track record. DSR < 0.95 means the strategy's Sharpe is not statistically distinguishable from noise after accounting for multiple testing. Implemented in the pypbo Python library. |
| **Probability of Backtest Overfitting (PBO)** | From CPCV: the fraction of backtest paths where the strategy's in-sample best configuration ranked below median in out-of-sample performance. PBO = 0 means the in-sample best always outperforms out-of-sample. PBO = 0.5 means the strategy is essentially noise. Target: PBO < 0.10. |
| **Walk-Forward Analysis (WFA)** | Sequential train/test splits that roll forward through time: train on months 1–3, test on month 4. Then train on months 2–4, test on month 5. Etc. Simulates adaptive re-optimization. The ratio of OOS Sharpe to IS Sharpe (the 'efficiency ratio') should be > 0.5. Lower means the strategy is overfitted to each training window. |
| **Slippage Modeling** | Realistic backtest slippage must account for: (1) bid-ask spread cost (1 tick minimum for market orders), (2) market impact (your order moves price — small for 1-2 MES contracts), (3) adverse selection (limit orders fill when price moves against you), (4) event-day spikes (FOMC, NFP). Flat 1-tick slippage assumption is standard for retail MES backtesting. |
| **Queue Position Modeling** | For limit order strategies, your fill probability depends on your position in the FIFO queue. If you submit a limit buy at the best bid, but 500 contracts are already queued, your order fills only if 500 contracts trade first. hftbacktest's LogProbQueueModel2 estimates fill probability based on queue depth — much more realistic than assuming all limit orders fill instantly. |

## Tasks (5 total)

### Tick-Level Backtest Engine

**Effort:** 5 days

#### Why This Matters

The most common backtest failure mode is using bar-level data (1-minute OHLCV) for strategies that are sensitive to intra-bar price paths. A 1-minute bar that went high then low is very different from one that went low then high — your stop would have been hit differently. Tick-level simulation eliminates this ambiguity completely.

#### Steps

1. Evaluate hftbacktest (GitHub: kronos-io/hftbacktest). This Python/Rust library provides full tick-by-tick order book reconstruction with queue position modeling. Install: pip install hftbacktest. Review their MES example if available, or adapt the equity futures example.

2. Implement volatility-conditioned slippage model: during 'calm' periods (ATR < 75th percentile), assume 1-tick slippage. During 'active' periods (ATR > 75th percentile), assume 2-tick slippage. During high-impact events (FOMC, NFP — flag these dates explicitly), assume 3-tick slippage. Apply to both market order entry and limit order stops.

3. Implement realistic limit order fill simulation: for limit orders at the best bid/ask, use hftbacktest's SquareProbQueueModel or LogProbQueueModel2. If queue depth > 100 contracts ahead of your order, assume 40% fill probability on that bar (the rest fill on subsequent bars or expire). Never assume 100% fill on limit orders at the touch.

4. Implement commission deduction per trade: subtract round-trip commission immediately on fill. Include a separate 'commission_paid' tracker so you can analyze gross P&L vs. net P&L — this tells you whether your edge is in the signal or in your cost structure.

5. Validate the engine against 3 days of manual paper trading (if you have it from Phase 1 testing) or against a known-simple strategy (buy at open, sell at close). The engine should reproduce P&L within 5% of the expected value before you trust it with complex strategies.

#### ✅ Exit Criteria

> Engine producing equity curves that match manual paper trading P&L within 5% across 3 test days

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the tick-level backtest engine with realistic fill simulation.

**src/backtesting/engine.py**
BacktestConfig dataclass:
- start_date, end_date
- strategies: List[StrategyBase]
- initial_capital: float = 10000.0
- slippage_model: str = "volatility_conditioned" (or "flat")
- flat_slippage_ticks: float = 1.0
- commission_model: CostModel (from Phase 1)
- use_queue_position_model: bool = True

BacktestEngine class:
- run(config) → BacktestResult
- Replay tick data from TimescaleDB chronologically
- On each tick: feed to FeatureEngine → update features → feed to strategies → collect signals
- On each bar: feed to strategies → generate signals
- Route signals to SimulatedOMS for fill simulation

SimulatedOMS class:
- on_signal(signal) → None: places simulated order
- on_tick(tick) → Optional[SimulatedFill]: checks if pending orders fill on this tick
- Fill logic for LIMIT orders:
  - LONG limit: fills if tick.last_price <= entry_price (price trades at or below limit)
  - Apply slippage: actual_fill = entry_price + slippage_ticks × 1.25 (assume slight adverse)
  - If use_queue_position_model: estimate queue depth, apply fill probability
- Fill logic for STOP orders:
  - Fill when price crosses stop level
  - Slippage: volatile_slippage_ticks during events
- Apply commission on fill

VolatilitySlippageModel class:
- compute_slippage_ticks(tick, atr) → float
- Calm (ATR < 75th pct): 1.0 tick
- Active (ATR > 75th pct): 2.0 ticks
- Event day (from EventDayCalendar): 3.0 ticks

BacktestResult dataclass:
- trades: List[Trade]
- equity_curve: pd.Series (indexed by timestamp)
- daily_pnl: pd.Series
- metrics: BacktestMetrics

BacktestMetrics dataclass:
- sharpe_ratio, sortino_ratio, profit_factor, win_rate, avg_win, avg_loss
- max_drawdown, max_drawdown_duration_days
- total_trades, avg_hold_time_bars
- gross_pnl, net_pnl, total_commission, avg_slippage_ticks

**src/backtesting/metrics.py**
MetricsCalculator class:
- from_trades(trades: List[Trade]) → BacktestMetrics
- sharpe(daily_returns, risk_free=0.0, periods=252) → float
- sortino(daily_returns, target=0.0) → float
- max_drawdown(equity_curve) → (drawdown_pct, duration_days)
- profit_factor(trades) → gross_wins / gross_losses
- validate_engine(engine_result, manual_pnl_dict) → dict comparing engine vs. manual for test dates

**tests/test_backtest_engine.py**
- test_simple_long_fills_and_targets
- test_stop_loss_fills_correctly
- test_commission_deducted_per_trade
- test_slippage_applied_to_fill_price
- test_equity_curve_monotonic_within_single_trade
- test_no_lookahead_bias (signal can only use data available at signal time)
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### CPCV Implementation

**Effort:** 3 days

#### Why This Matters

Standard k-fold cross-validation is invalid for time series because it allows training on future data and creates autocorrelation between train/test splits. CPCV from mlfinlab handles both problems with temporal purging and embargoing — making it the gold standard for financial strategy validation.

#### Steps

1. Install mlfinlab: pip install mlfinlab. Review the CombinatorialPurgedKFold class documentation. Set up parameters: N=8 groups (each ~25 trading days with 6 months of data), k=2 test groups per split. This generates C(8,2)=28 unique backtest paths.

2. Implement the purging logic: for each training/test split, identify any training observations whose label formation window overlaps with the test period. Remove these observations from training. Standard purge = 1 bar before test period start (since we use next-bar entry, the signal label uses next bar's open price).

3. Implement embargoing: after each test period, add an embargo of 5% × test_period_length bars to the next training period. This prevents information leakage from test back into training through autocorrelated features.

4. Run CPCV for each strategy: for each of the 28 backtest paths, (a) fit strategy parameters (if any) on training data, (b) run backtest on test data, (c) record cumulative return. Compute PBO: fraction of paths where OOS rank < median rank. Store all 28 equity curves per strategy.

5. Plot CPCV results: for each strategy, show all 28 equity curves overlaid. Strategies with genuine edge show mostly upward-sloping curves with some variance. Overfitted strategies show high variance — some paths strongly positive, some strongly negative. Visual inspection is as informative as the PBO number.

#### ✅ Exit Criteria

> CPCV running on all 5 strategies, PBO computed, strategies ranked by PBO

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the CPCV validation pipeline.

**src/validation/cpcv.py**
CPCVConfig dataclass:
- n_groups: int = 8 (split data into N groups of complete trading sessions)
- k_test: int = 2 (test on k groups per split)
- embargo_pct: float = 0.05 (5% of dataset as embargo after each test period)
- purge_bars: int = 1 (bars to purge from training that overlap test label formation)

CPCVValidator class:
- run(strategy_class, strategy_config, feature_vectors, bars, dates) → CPCVResult
- Generates C(n_groups, k_test) unique train/test splits = C(8,2) = 28 paths
- For each path:
  1. Determine train/test date ranges
  2. Apply purge: remove training bars whose labels overlap test period
  3. Apply embargo: remove bars immediately after each test period from next training window
  4. Fit strategy parameters on training data (if any tunable parameters)
  5. Run BacktestEngine on test data with fitted parameters
  6. Record: equity_curve, final_return, sharpe
- Compute PBO: Probability of Backtest Overfitting
  - Rank all 28 OOS test Sharpes
  - PBO = fraction of paths where IS best config ranked below median OOS

PBOCalculator class:
- compute_pbo(is_sharpes: List[float], oos_sharpes: List[float]) → float
- plot_cpcv_paths(paths: List[EquityCurve], output_path) → matplotlib chart of all 28 equity curves overlaid
- generate_cpcv_report(result: CPCVResult) → markdown string with PBO, path statistics, interpretation

CPCVResult dataclass:
- strategy_id: str
- pbo: float
- n_paths: int
- oos_sharpes: List[float]
- is_sharpes: List[float]
- equity_curves: List[pd.Series]
- verdict: str ("PASS" if pbo < 0.10 else "FAIL")
- interpretation: str (human-readable explanation)

**scripts/run_cpcv.py**
CLI: python scripts/run_cpcv.py --strategy orb --start 2023-01-01 --end 2024-06-01
- Loads data, instantiates strategy, runs CPCV
- Saves: results/cpcv/{strategy_id}/cpcv_result.json, equity_curves.png, report.md
- Prints: PBO score, verdict, path statistics

**docs/phase4/cpcv-methodology.md**
Explanation of CPCV for your own reference:
- What purging and embargoing prevent
- How to interpret PBO
- What the equity curve plots tell you
- Common misinterpretations
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Deflated Sharpe Ratio Computation

**Effort:** 2 days

#### Why This Matters

After running 5 strategies through many backtests, the best-looking Sharpe Ratio is almost certainly inflated by selection bias. DSR corrects for this. A strategy with apparent Sharpe 2.0 that fails DSR < 0.95 is statistically indistinguishable from a lucky random strategy — and should not receive real capital.

#### Steps

1. Install pypbo: pip install pypbo. Review the prob_sharpe_ratio and deflated_sharpe_ratio functions. Inputs needed: observed Sharpe Ratio, number of trials (total backtests run across all strategies and parameter sets), skewness and kurtosis of strategy returns, number of trading periods.

2. Count total trials across all parameter combinations tested: if ORB tested 3 target multipliers × 2 volume filters × 2 HMM state requirements = 12 variants, count all 12. Sum across all 5 strategies. This 'trials' count is what makes DSR conservative — it accounts for the fishing expedition.

3. Compute DSR for each strategy's best configuration: pypbo.deflated_sharpe_ratio(sr_hat, sr_0, T, skewness, kurtosis, n_trials). Threshold: DSR ≥ 0.95. A DSR of 0.95 means there is a 95% probability the true Sharpe Ratio is positive after accounting for selection bias.

4. Document go/no-go decision for each strategy with explicit reasoning. Template: 'ORB: PBO=0.08 (PASS), DSR=0.97 (PASS), WFA efficiency=0.71 (PASS) → Proceed to paper trading.' or 'CVD Divergence: PBO=0.22 (FAIL) → Strategy retired. Failure mode: divergence signals were too infrequent for statistical confidence. Hypothesis for future research: combine with volume profile filter to increase signal quality.'

#### ✅ Exit Criteria

> DSR computed for all 5 strategies, clear go/no-go documented for each

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the Deflated Sharpe Ratio computation and strategy go/no-go decision framework.

**src/validation/dsr.py**
DSRConfig dataclass:
- min_dsr_threshold: float = 0.95
- benchmark_sharpe: float = 0.0 (null hypothesis: strategy has zero Sharpe)
- n_trading_periods: int = 252 (annualized)

DSRCalculator class:
- compute_dsr(observed_sharpe, n_trials, returns_series, n_periods) → DSRResult
  - Uses pypbo.prob_sharpe_ratio and pypbo.deflated_sharpe_ratio
  - n_trials: total backtests run across ALL strategies and parameter sets
  - Computes: skewness, kurtosis of returns_series
  - Returns DSRResult with: dsr, psr (probabilistic SR), verdict

TrialCounter class:
- track_trial(strategy_id, param_set_hash) → None (call each time you run a backtest)
- total_trials property → int
- trials_by_strategy property → dict
- save(path) / load(path) → persist trial count between sessions
- IMPORTANT: trial counter must be started at beginning of Phase 4 and never reset

DSRResult dataclass:
- strategy_id: str
- observed_sharpe: float
- dsr: float
- psr: float
- n_trials_used: int
- verdict: str ("PASS" if dsr >= 0.95 else "FAIL")
- interpretation: str

StrategyGoNoGo class:
- evaluate(strategy_id, cpcv_result, dsr_result, wfa_result) → GoNoGoDecision
- GoNoGoDecision dataclass: strategy_id, go (bool), pbo (float), dsr (float), wfa_efficiency (float), reasoning (str), next_steps (str)
- generate_decision_document(decisions: List[GoNoGoDecision]) → markdown
  - For each strategy: full table of all three validation metrics, PASS/FAIL per metric, final decision, failure analysis if failed

**scripts/run_dsr.py**
CLI: python scripts/run_dsr.py --strategy orb --n-trials 150
- Loads CPCV result (must be run first)
- Computes DSR with provided trial count
- Outputs: results/dsr/{strategy_id}/dsr_result.json, decision.md

**docs/phase4/strategy-decisions.md** (template)
Template that gets filled in after running all validations:
| Strategy | PBO | DSR | WFA Eff | Decision | Notes |
|---|---|---|---|---|---|
| ORB | [AUTO] | [AUTO] | [AUTO] | [AUTO] | |
...
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Walk-Forward Analysis

**Effort:** 5 days

#### Why This Matters

WFA tests whether your strategy's parameters remain stable across time — or whether each re-optimization is essentially curve-fitting to a new regime. An efficiency ratio of 0.7 means your OOS Sharpe is 70% of your IS Sharpe — acceptable. An efficiency ratio of 0.2 means the strategy's apparent edge mostly disappears out-of-sample — a red flag.

#### Steps

1. Set WFA parameters: training window = 3 months (63 trading days), test window = 1 month (21 trading days). With 18 months of data, this produces 6 WFO cycles. Prefer rolling windows over expanding windows — recent data is more representative of current microstructure than 2-year-old data.

2. For each WFO cycle: (a) optimize strategy parameters on training window using a coarse grid search (not Bayesian optimization — too easy to overfit), (b) record best in-sample configuration, (c) run that configuration unchanged on test window, (d) record OOS P&L, Sharpe, drawdown.

3. Compute efficiency ratio per strategy: OOS Sharpe / IS Sharpe. Average across all 6 cycles. Target > 0.5. Also compute IS/OOS correlation: are the months where IS Sharpe is highest also the months where OOS Sharpe is highest? High correlation suggests the parameter optimization is capturing real signal, not noise.

4. Analyze parameter drift: plot the optimal parameter value (e.g., ORB target multiplier) across 6 WFO cycles. If it varies wildly (0.3, 0.8, 0.4, 0.9, 0.3, 0.7), the 'optimal' parameter is not stable and the strategy is curve-fitting. If it's relatively stable (0.45, 0.50, 0.48, 0.52, 0.47, 0.51), the parameter represents a genuine structural feature.

5. Conduct parameter stability analysis independently: take the final optimal parameters from the last WFO cycle. Vary each parameter ±20% in 5% increments. For each variation, compute OOS Sharpe. Plot Sharpe as a function of each parameter. Goal: smooth plateau around optimal, not a sharp spike. Spikes indicate overfitting; plateaus indicate robustness.

#### ✅ Exit Criteria

> 6+ WFO cycles completed per strategy, efficiency ratio (OOS/IS Sharpe) documented

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the Walk-Forward Analysis pipeline.

**src/validation/wfa.py**
WFAConfig dataclass:
- train_months: int = 3
- test_months: int = 1
- min_cycles: int = 6
- use_rolling_windows: bool = True (rolling preferred over expanding)
- param_grid: dict (strategy-specific parameter search space)
- coarse_grid_only: bool = True (avoid Bayesian optimization — too easy to overfit)

WFARunner class:
- run(strategy_class, param_grid, bars_df, start_date, end_date) → WFAResult
- For each cycle:
  1. Slice training window (rolling: most recent train_months)
  2. Grid search on training window: for each param combination → run BacktestEngine → record IS Sharpe
  3. Select best IS param set
  4. Evaluate best params on test window (fixed, no changes) → record OOS Sharpe, P&L
  5. Log: cycle_num, train_start, train_end, test_start, test_end, best_params, is_sharpe, oos_sharpe
- Compute efficiency_ratio = mean(oos_sharpes) / mean(is_sharpes)
- Compute IS/OOS correlation (pearsonr of IS vs OOS Sharpe across cycles)

ParameterStabilityAnalyzer class:
- analyze(strategy_class, best_params, bars_df, variation_pct=0.20, steps=5) → StabilityResult
- For each parameter: vary ± variation_pct in steps increments
- For each variation: run OOS backtest with that single param changed
- StabilityResult: {param_name → {value → oos_sharpe}} dict
- plot_stability(result, output_dir) → one chart per parameter showing Sharpe vs. param value
- stability_score(result) → float: fraction of neighbor combinations that remain profitable

WFAResult dataclass:
- cycles: List[WFACycle]
- efficiency_ratio: float
- is_oos_correlation: float
- param_drift: dict (optimal param value by cycle — measures stability)
- verdict: str ("PASS" if efficiency_ratio >= 0.5 else "FAIL")

**scripts/run_wfa.py**
CLI with strategy-specific param grids:
- ORB: target_multiplier=[0.4,0.5,0.6,0.7], volume_multiplier=[1.3,1.5,1.7,2.0]
- VWAP: entry_sd=[1.8,2.0,2.2,2.5], flat_threshold=[0.001,0.002,0.003,0.005]
- CVD: divergence_threshold=[0.10,0.15,0.20,0.25], poc_proximity=[4,6,8,10]
- VolRegime: semi_var_percentile=[0.70,0.75,0.80], pullback_bars=[2,3,4]
- ValueArea: hold_periods=[2,3], min_va_pct=[0.55,0.60,0.65,0.70]

Saves: results/wfa/{strategy_id}/wfa_result.json, param_stability/*.png, summary.md
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Strategy Kill & Survivor Documentation

**Effort:** 2 days

#### Why This Matters

Documentation of why strategies failed is as valuable as documentation of why they succeeded. Future research hypotheses come from understanding failure modes. And the explicit go/no-go decision — written down before seeing Phase 5 paper trading results — prevents the common cognitive bias of adjusting your standards after the fact.

#### Steps

1. Write a one-page strategy assessment for each of the 5 strategies: summary of validation results (PBO, DSR, WFA efficiency), primary failure mode if retired (overfitting, insufficient edge after costs, parameter instability, too infrequent signals), and forward research hypothesis if retired.

2. For surviving strategies: document final parameters locked for Phase 5 (these parameters cannot be changed in Phase 5 — paper trading tests the fixed strategy, not an ongoing optimization). Include: entry conditions, filter thresholds, target/stop levels, session windows, HMM state requirements.

3. Compute combined strategy correlation: if 2 strategies both survive, compute the correlation of their daily P&L from the CPCV paths. Low correlation (< 0.3) confirms they are capturing different edges and are genuinely additive when run together. High correlation (> 0.7) means they're essentially the same strategy — only run the one with better DSR.

4. Set Phase 5 entry criteria clearly in writing: 'Strategy X will be declared fit for live trading if paper trading produces Sharpe ≥ 0.8 and profit factor ≥ 1.3 over minimum 200 trades.' Any deviation from these criteria before seeing results is data manipulation.

#### ✅ Exit Criteria

> 2–3 survivor strategies documented with parameters, performance profiles, and failure analysis for retired strategies

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the strategy kill/survivor decision system and documentation generator.

**src/validation/decision_engine.py**
ValidationSummary dataclass (aggregates all Phase 4 results):
- strategy_id: str
- cpcv_pbo: float
- dsr: float
- wfa_efficiency: float
- wfa_is_oos_correlation: float
- param_stability_score: float
- total_oos_trades: int
- oos_sharpe: float
- oos_win_rate: float
- oos_profit_factor: float

ValidationDecision dataclass:
- strategy_id: str
- decision: str ("PROCEED" | "RETIRE")
- passed_pbo: bool (pbo < 0.10)
- passed_dsr: bool (dsr >= 0.95)
- passed_wfa: bool (efficiency >= 0.50)
- passed_stability: bool (stability_score >= 0.65)
- failure_modes: List[str] (which checks failed and why)
- research_hypothesis: str (if retired: what to try next time)
- locked_parameters: dict (if proceeding: final params for Phase 5, DO NOT CHANGE)
- correlation_with_survivors: dict (pairwise correlation with other proceeding strategies)

DecisionEngine class:
- evaluate_all(summaries: List[ValidationSummary]) → List[ValidationDecision]
- check_strategy_correlation(decisions: List[ValidationDecision], bars_df) → correlation matrix
- generate_final_report(decisions, output_path) → full markdown report

**scripts/generate_phase4_report.py**
Aggregates all results from results/cpcv/, results/dsr/, results/wfa/ and generates:

**docs/phase4/validation-report.md** (auto-generated)
Complete Phase 4 report with:
1. Executive summary: X of 5 strategies proceeding
2. Full validation table for all 5 strategies
3. Per-strategy sections:
   - CPCV equity curve image
   - Parameter stability charts
   - WFA cycle table
   - Final decision with reasoning
4. Strategy correlation matrix (for survivors)
5. Locked parameters for Phase 5 (in code block — copy-paste ready)
6. Retired strategy postmortems

**docs/phase4/locked-params.yaml** (auto-generated)
YAML file with final locked parameters for each proceeding strategy:
```yaml
orb:
  target_multiplier: 0.52
  volume_multiplier: 1.6
  # DO NOT CHANGE - locked by WFA cycle 6, 2024-09-15
vwap:
  entry_sd: 2.1
  flat_threshold: 0.002
  # DO NOT CHANGE - locked by WFA cycle 6, 2024-09-15
```

Paper trading gate criteria saved to docs/phase5/entry-criteria.md (also auto-generated):
```
Strategy X goes live only if:
- Sharpe >= 0.8 over 200+ paper trades
- Profit factor >= 1.3
- Average slippage within 0.5 ticks of backtest assumption
- Fill rate >= 70% on limit orders
These criteria are fixed. Any change to criteria after seeing paper results = data manipulation.
```
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

## Phase 4 Gate

All exit criteria must pass before advancing to the next phase:

- [ ] **Tick-Level Backtest Engine:** Engine producing equity curves that match manual paper trading P&L within 5% across 3 test days
- [ ] **CPCV Implementation:** CPCV running on all 5 strategies, PBO computed, strategies ranked by PBO
- [ ] **Deflated Sharpe Ratio Computation:** DSR computed for all 5 strategies, clear go/no-go documented for each
- [ ] **Walk-Forward Analysis:** 6+ WFO cycles completed per strategy, efficiency ratio (OOS/IS Sharpe) documented
- [ ] **Strategy Kill & Survivor Documentation:** 2–3 survivor strategies documented with parameters, performance profiles, and failure analysis for retired strategies


---

# Phase 5: Paper Trading & Live Preparation

**Duration:** 4 Weeks  |  **Goal:** The incubation phase that Kevin Davey credits for 85-90% of his survival as an algo trader.

## Overview

Paper trading is not a formality — it is the most important gate between theory and real capital. The difference between a backtest and paper trading exposes three categories of problems that backtests cannot catch: implementation bugs (your signal fires but the order routing has a bug), fill rate assumptions (you assumed 90% fill on limit orders but only getting 60% in practice), and latency issues (your signal arrives 200ms after the optimal entry because feature calculation is slower than expected).

Kevin Davey, who has won the World Cup Championship of Futures Trading three times with verified returns, attributes 85–90% of his survival as an algo trader to the incubation (paper trading) phase. He runs every strategy in paper trading for a minimum of 3 months — even strategies that show excellent backtests. He has said explicitly: 'I've seen amazing-looking backtest strategies fail catastrophically in paper trading. I've never seen a strategy succeed in paper trading that I wouldn't also trade live.'

Your exit criteria from this phase are non-negotiable: Sharpe ≥ 0.8, profit factor ≥ 1.3, over minimum 200 paper trades. If after 4 weeks you have only 80 trades and a Sharpe of 1.2, you keep paper trading until you have 200 trades. The statistical significance threshold requires 200+ trades — below that, results are noise.

## Key Concepts

| Concept | What It Means for Your Bot |
| --- | --- |
| **Profit Factor** | Gross profit / Gross loss. A profit factor of 1.0 means you make exactly what you lose — break even. 1.3 means for every $1 lost, you make $1.30 — a 30% edge. 1.5+ is considered strong for a scalping strategy. Profit factor is more informative than win rate alone because it accounts for payoff asymmetry. |
| **Fill Rate** | The percentage of limit orders that actually fill vs. expire unfilled. A backtest might assume 95% fill rate on limit orders at the touch. Paper trading might reveal 60% — especially during fast-moving periods when your limit order gets skipped by aggressive flow. Tracking fill rate discrepancy between backtest and paper is critical. |
| **Slippage vs. Assumption** | Compare actual fill prices to signal prices in paper trading. If your signal generates a long entry at 5500.00 and your paper fill is 5500.25 (1 tick worse), that's 1 tick of slippage. Over 200 trades, if average slippage is 1.5 ticks vs. your 1-tick backtest assumption, your real edge is 0.5 tick per trade worse than modeled — potentially strategy-killing. |
| **Server-Side Bracket Orders** | Orders submitted directly to the broker's server with built-in take profit and stop loss. If your client disconnects, the bracket orders remain active at the exchange. Critical safety feature: without server-side stops, a client crash during an open position means no protection. Rithmic supports native bracket orders (OCO: one-cancels-other). |
| **Telegram Bot Integration** | A simple Telegram bot can send you real-time alerts for fills, errors, and daily P&L summaries. The bot can also receive commands: /status (current position, daily P&L), /halt (flatten all positions and disable new signals), /resume. This gives you a mobile kill switch without needing to SSH into your VPS. |
| **Position State Machine** | A formal state machine for position management: FLAT → PENDING_ENTRY → LONG/SHORT → PENDING_EXIT → FLAT. Every transition has explicit conditions and guards. This prevents bugs like double-entering a position or submitting duplicate stop orders. Implement as a Python enum with transition validation. |

## Tasks (5 total)

### Live Signal Pipeline Integration

**Effort:** 4 days

#### Why This Matters

The signal pipeline connects your Phase 2 features and Phase 3 strategy logic to real-time tick data. Bugs here can cause missed entries, wrong prices, or signals that fire at incorrect times. Test it exhaustively before connecting it to any order management system.

#### Steps

1. Connect the Phase 1 Rithmic tick feed to the Phase 2 feature library: on each tick event, update CVD, VWAP deviation, book imbalance. On each dollar bar close, update ATR, HMM state, volume profile POC distance. Use the asyncio event bus from the Phase 1 skeleton.

2. Connect feature updates to strategy on_tick() and on_bar() callbacks. Each strategy's generate_signal() method is called after every bar close during valid session hours. Implement a signal queue: if a signal is already active (waiting to be entered or in a position), discard new signals from the same strategy.

3. Instrument every step with nanosecond timestamps: tick_received_ns, features_updated_ns, signal_generated_ns. Compute latency = signal_generated_ns - tick_received_ns. Log every bar. Alert (Telegram message) if any single signal takes > 100ms. The target P99 latency is < 50ms.

4. Implement session management: at 9:30 AM ET, call reset() on all strategies (clear opening range, reset CVD, reload prior-day value area from TimescaleDB). At 4:00 PM ET, flatten all positions (market order), disable signal generation, log daily summary.

5. Run the signal pipeline for 3 full RTH sessions in 'dry run' mode (signals generated but not routed to OMS). Verify: signal timestamps are correct, no signals fire during excluded windows (FOMC, dead zone if filtered), HMM state updates are happening correctly by comparing to post-hoc analysis.

#### ✅ Exit Criteria

> All strategy signals generating in < 50ms end-to-end on Chicago VPS, verified with latency logging

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the live signal pipeline that connects the tick feed to strategy signal generation with full latency instrumentation.

**src/pipeline/signal_pipeline.py**
LatencyTracker class:
- record(stage: str, timestamp_ns: int) → None
- end_to_end_ns property → int (from tick_received to signal_generated)
- stages property → dict of stage→timestamp
- to_log_dict() → dict for structlog

SignalPipeline class (the main orchestrator):
- __init__(feed, feature_engine, strategies, oms, risk_manager, hmm_classifier)
- async run() → main event loop
- async _on_tick(tick: TickData):
  1. Record tick_received timestamp
  2. Update FeatureEngine: feature_engine.on_tick(tick)
  3. Record features_updated timestamp
  4. Update HMM online inference: hmm.predict_online(feature_engine.get_feature_vector())
  5. For each strategy: strategy.on_tick(tick)
  6. Emit TickEvent on EventBus
- async _on_bar(bar: DollarBar):
  1. For each strategy: strategy.on_bar(bar)
  2. For each strategy: signal = strategy.generate_signal()
  3. If signal: validate with risk_manager.check_order(signal, position, daily_pnl)
  4. If approved: route to OMS, record signal_generated timestamp
  5. Log LatencyTracker dict: all stage timestamps + end_to_end_ms
  6. Alert if end_to_end_ms > 100 (via Telegram)
- Session management: subscribe to SessionManager.on_session_open/close callbacks
  - On open: reset all strategies, load prior session levels, log "Session open"
  - On close: flatten all positions via OMS, log daily summary, store feature vectors to FeatureStore

**src/pipeline/session_orchestrator.py**
SessionOrchestrator class:
- Manages the full trading day lifecycle
- Pre-session (9:25 AM): connect feed, warm up feature engine on last 30 min of overnight data, load volume levels from DB
- Session open (9:30 AM): reset strategies, enable signal generation, log session start
- Intraday: forward events through pipeline
- Session close (4:00 PM): disable signals, flatten all positions, compute daily summary, store data
- Post-session: archive logs, store feature vectors, update HMM retrain check

**scripts/run_pipeline_dryrun.py**
Dry run mode — signals generated but NOT routed to OMS:
- Connects to live Rithmic feed (or replays historical data if --replay flag)
- Runs full pipeline for one session
- Logs all signals that would have been generated with: time, strategy, direction, entry, target, stop, confidence, HMM state, all feature values
- Latency report at end: p50/p90/p99 end-to-end latency, slowest step identified
- Compare signal times to what backtest expected for same bars
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Order Management System (OMS)

**Effort:** 5 days

#### Why This Matters

The OMS is the most safety-critical component of your bot. A bug in the OMS — duplicate order submission, missed stop, wrong quantity — causes real financial losses in live trading that are difficult to recover from. Over-engineer this module. Test every edge case.

#### Steps

1. Implement the Position state machine: FLAT → ENTERING (bracket submitted, awaiting fill) → LONG or SHORT (filled, bracket active) → EXITING (target or stop hit, awaiting close fill) → FLAT. Every state transition logs: state, price, time, reason. No transition is allowed without explicit logging.

2. Implement bracket order submission via Rithmic: on signal receipt, submit a bracket order with entry (limit at signal price or market), take profit (limit), and stop loss (stop-limit or stop-market). Use server-side OCO: when target fills, stop is automatically cancelled by exchange — and vice versa.

3. Implement duplicate order prevention: maintain a set of active order IDs. Before submitting any order, verify the corresponding position is in the correct state. If FLAT, submit entry. If ENTERING, log warning and skip (order already pending). If LONG/SHORT, log error and skip.

4. Implement fill confirmation: when a fill message arrives from Rithmic, update position state and log fill price, fill time, fill quantity. Compute slippage: fill_price - signal_price. Store per-trade. This is your primary data source for comparing paper trading vs. backtest assumptions.

5. Implement order timeout: if an entry limit order is not filled within max_entry_wait_bars (default 3 dollar bars), cancel it. This prevents stale limit orders from filling at prices that are no longer contextually valid — the market has moved on.

#### ✅ Exit Criteria

> OMS successfully routing paper orders to Rithmic for 5 consecutive days with zero duplicate orders or missed stops

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the production Order Management System with full state machine and server-side bracket orders.

**src/oms/state_machine.py**
PositionState enum: FLAT, ENTERING, LONG, SHORT, EXITING

StateTransition dataclass: from_state, to_state, trigger, price, timestamp, reason

PositionStateMachine class:
- current_state: PositionState = FLAT
- current_signal: Optional[Signal] = None
- entry_price: Optional[float] = None
- entry_time: Optional[datetime] = None
- active_order_ids: Set[str]
- transitions: List[StateTransition] (audit log)
- transition(new_state, trigger, price, reason) → None: validates transition is legal, logs it
- Legal transitions: FLAT→ENTERING, ENTERING→LONG/SHORT (on fill), ENTERING→FLAT (cancel), LONG/SHORT→EXITING (target/stop hit), EXITING→FLAT (close fill)
- Illegal transitions raise StateMachineError — never silently ignored

**src/oms/order_manager.py**
OrderManager class (live OMS using Rithmic):
- on_signal(signal: Signal) → str (order_id): routes signal to broker
  1. Check state_machine.current_state == FLAT (else reject + log)
  2. Build bracket: entry limit order + OCO (take profit limit + stop loss)
  3. Submit via Rithmic API
  4. Register order IDs in active_order_ids
  5. Transition state machine: FLAT → ENTERING
  6. Set timeout: if not filled in max_entry_wait_bars → cancel
- on_fill(fill_event: FillEvent) → None:
  1. Identify fill type (entry / target / stop)
  2. Transition state machine appropriately
  3. Compute slippage = fill_price - signal.entry_price (for entry fills)
  4. Log fill with all details
  5. Notify Telegram: "FILLED LONG MES @ 5523.50 | Target: 5533.50 | Stop: 5519.50"
- on_cancel(order_id) → None: handle rejected/cancelled orders
- flatten_all() → None: market order to close any open position (emergency)
- cancel_pending() → None: cancel ENTERING state orders (for safety halt)

**src/oms/paper_oms.py**
PaperOrderManager (implements same interface as OrderManager):
- Simulates fills without routing to broker
- Uses same SimulatedOMS logic from BacktestEngine (reuse code)
- Records paper fills to TimescaleDB table: paper_trades
- Makes switching between paper and live a config flag (not a code change)

**src/oms/fill_tracker.py**
FillTracker class:
- record_fill(fill_event, signal) → None: stores to TimescaleDB
- compute_slippage_stats(start_date, end_date) → dict: avg, p50, p90, p99 slippage vs. backtest assumption
- compare_paper_vs_backtest(paper_trades_df, backtest_trades_df) → DataFrame with per-bar comparison
- generate_fill_report(start_date, end_date) → markdown with slippage analysis

**tests/test_oms.py**
- test_flat_to_entering_on_signal
- test_entering_to_long_on_fill
- test_illegal_transition_raises_error
- test_duplicate_signal_rejected
- test_timeout_cancels_pending_order
- test_flatten_all_works_from_long_and_short
- test_paper_oms_records_fill_to_db
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Safety Systems & Kill Switches

**Effort:** 3 days

#### Why This Matters

Your safety systems are not for paper trading — they're for the live trading that follows. Building and testing them in paper trading ensures they work before real money is at risk. A safety system you haven't tested is a safety system that doesn't exist.

#### Steps

1. Hard daily loss limit: track cumulative daily P&L in real-time. When daily_pnl < -max_daily_loss (set to $150 for 1 MES contract, ~12 ticks), immediately: (1) flatten all open positions at market, (2) cancel all pending orders, (3) disable signal generation for the rest of the day, (4) send Telegram alert. Test this by manually triggering paper trades until the limit is hit.

2. Maximum position size guard: before every order submission, verify: abs(current_position) + new_order_size <= max_position_size (set to 1 MES contract). If this check fails for any reason, reject the order and alert. This prevents runaway position building from OMS bugs.

3. Stale data detector: monitor time since last tick during RTH hours (9:30 AM–4:00 PM ET). If no tick for 30 seconds, send Telegram warning. If no tick for 60 seconds, assume data feed issue, flatten all positions at market, halt. On reconnect, wait for 2 minutes of stable feed before resuming.

4. Connection watchdog: Rithmic WebSocket connection health check every 5 seconds. If connection drops, attempt reconnect immediately. If reconnect fails, flatten via backup REST API (if available) or Telegram alert for manual intervention. During the reconnection gap, assume worst-case position and prepare to flatten.

5. Telegram bot implementation: use python-telegram-bot library. Implement commands: /status → current position, daily P&L, last tick age, HMM state. /halt → flatten all positions, disable signals (requires confirmation). /resume → re-enable signals. /metrics → today's trade count, win rate, avg slippage. Test each command from mobile.

#### ✅ Exit Criteria

> All safety systems tested: daily loss limit triggers correctly, Telegram kill switch works, stale data halt works

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create all safety systems and the Telegram kill switch.

**src/safety/daily_loss_limit.py**
DailyLossMonitor class:
- __init__(max_loss_usd: float, oms: OrderManager, telegram: TelegramAlerter)
- record_fill(fill_event) → None: updates running daily P&L
- daily_pnl property → float
- check() → None: called after every fill — if daily_pnl < -max_loss_usd: trigger halt
- async halt(reason: str):
  1. Log: "DAILY LOSS LIMIT HIT: {daily_pnl:.2f} vs limit {max_loss_usd:.2f}"
  2. oms.flatten_all() (flatten any open position)
  3. oms.cancel_pending() (cancel any pending entries)
  4. risk_manager.halt(reason)
  5. Telegram: "🚨 DAILY LOSS LIMIT HIT\n P&L: -$X\n All positions flattened. Bot halted for today."
  6. Disable signal pipeline for remainder of session
- reset() → called at session open to reset daily P&L and re-enable

**src/safety/position_guard.py**
PositionGuard class:
- max_contracts: int
- check_order(signal, current_position) → (bool, str): returns (approved, reason)
- Always called BEFORE OMS.on_signal — hard gate
- Log every rejection with reason

**src/safety/data_watchdog.py**
DataWatchdog class:
- last_tick_time: datetime
- on_tick(tick) → None: updates last_tick_time
- async monitor() → runs every 5 seconds:
  - If now - last_tick_time > 30s during RTH: log WARNING + Telegram "⚠️ No ticks for 30s"
  - If now - last_tick_time > 60s during RTH: trigger emergency halt
- stale_data_age_ms property → int

**src/safety/connection_watchdog.py**
ConnectionWatchdog class:
- async monitor(feed: RithmicFeed):
  - Ping Rithmic every 5 seconds (heartbeat)
  - On disconnect: immediately call oms.flatten_all() + attempt reconnect
  - If reconnect succeeds: wait 2 minutes before re-enabling signals
  - If reconnect fails after 3 attempts: alert Telegram, remain halted

**src/monitoring/telegram_bot.py**
TelegramBot class using python-telegram-bot:
- Commands:
  - /status → "Position: FLAT | Daily P&L: +$47.50 | Trades: 3 | Last tick: 0.3s ago | HMM: MEAN_REVERSION"
  - /halt → "Are you sure? Reply /confirm to halt." → on /confirm: risk_manager.halt("manual halt")
  - /resume → re-enable signals (if halted manually, not by loss limit)
  - /metrics → today's detailed metrics: win rate, avg slippage, fill rate, trade count by strategy
  - /health → VPS uptime, memory, CPU, ping to CME
- Alerts (bot sends proactively, no command needed):
  - Fill notification with P&L
  - Daily loss limit hit
  - Stale data warning
  - Connection drop/restore
  - Daily P&L summary at 4:05 PM ET

**tests/test_safety.py**
Critical tests — ALL must pass:
- test_daily_loss_limit_triggers_halt_at_threshold
- test_daily_loss_limit_does_not_trigger_before_threshold
- test_halt_flattens_open_position
- test_halt_cancels_pending_entry
- test_position_guard_blocks_over_limit_order
- test_data_watchdog_alerts_at_30s
- test_data_watchdog_halts_at_60s
- test_reset_enables_trading_next_session
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Paper Trading Execution & Monitoring

**Effort:** 15 days

#### Why This Matters

Paper trading is not optional. It is the empirical test of whether your implementation matches your backtest. Every day of paper trading produces data that either confirms your model or reveals implementation gaps. The gate criteria are deliberately demanding — passing them with statistical confidence requires genuine edge, not luck.

#### Steps

1. Run paper trading for minimum 4 weeks (approximately 20 RTH sessions). Log every trade: signal time, entry time, entry price, signal price, slippage (entry_price - signal_price), exit time, exit price, exit reason (target/stop/timeout/daily limit), P&L (gross and net), HMM state at entry, strategy_id.

2. Daily review (15 minutes): open your Grafana dashboard. Check: daily P&L vs. expectation, trade count vs. expected rate (3–10/day per strategy), fill rate for limit orders, average slippage vs. 1-tick assumption. Flag any day where slippage > 2 ticks average for investigation.

3. Weekly review (1 hour): compare rolling Sharpe (last 20 days) to backtest Sharpe. If live Sharpe is consistently < 50% of backtest Sharpe after 3 weeks, there is a structural gap — likely slippage, fill rate, or implementation issue. Investigate before continuing.

4. Fill rate analysis: for each limit order entry, record whether it filled within max_entry_wait_bars. Compute fill rate by strategy and by time of day. If fill rate < 70% for any strategy, increase the entry price aggressively by 1 tick (e.g., if VWAP reversion entry is at exactly –2 SD, move to –2 SD + 1 tick). Re-evaluate.

5. Backtest vs. live comparison: for each day of paper trading, replay the same day through your backtest engine. Compare signals: should generate at same bars. Compare fills: paper fills should be within 1–2 ticks of backtest fills. Discrepancies > 3 ticks indicate either data misalignment or implementation bugs requiring investigation.

#### ✅ Exit Criteria

> 200+ paper trades executed, Sharpe ≥ 0.8, profit factor ≥ 1.3 — OR extended paper trading continues until criteria met

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the paper trading monitoring system and daily review tooling.

**src/monitoring/paper_trading_monitor.py**
PaperTradingMonitor class:
- Connects to TimescaleDB paper_trades table
- compute_rolling_metrics(window_days=20) → PaperMetrics
- PaperMetrics dataclass: sharpe, sortino, profit_factor, win_rate, avg_win_usd, avg_loss_usd, max_drawdown, total_trades, avg_hold_bars, avg_slippage_ticks, fill_rate
- compare_to_backtest(backtest_metrics: BacktestMetrics) → ComparisonReport
  - For each metric: backtest_value, paper_value, delta_pct, is_within_threshold (20% tolerance)
- flag_discrepancies(comparison: ComparisonReport) → List[str]: list of metrics outside tolerance

**scripts/daily_paper_review.py**
Run after each session close (can be automated via cron at 4:15 PM ET):
- Load today's paper trades from TimescaleDB
- Compute today's P&L, trade count, win rate, avg slippage
- Compare to rolling 20-day baseline
- Flag any trade with slippage > 2 ticks for manual review
- Print summary table to console + save to docs/paper-trading/daily-logs/YYYY-MM-DD.md
- If rolling Sharpe drops below 0.4 for 3 consecutive days: send Telegram alert

**scripts/paper_trading_report.py**
Weekly/end-of-paper-period report:
- Load all paper trades from start of paper period
- Compute all PaperMetrics
- Load backtest metrics for the same strategies
- Generate: docs/paper-trading/paper-trading-report.md with:
  - Executive summary: PASS or FAIL against Phase 5 gate criteria
  - Full metrics table (paper vs. backtest)
  - Per-strategy breakdown
  - Worst 10 trades (for analysis)
  - Best 10 trades
  - Slippage distribution chart (histogram)
  - Fill rate by time of day chart
  - Rolling Sharpe chart

**docs/paper-trading/gate-checklist.md** (template)
```
Phase 5 Gate Checklist — Do not proceed to live until ALL are checked:
[ ] Total paper trades >= 200
[ ] Rolling Sharpe (all trades) >= 0.8
[ ] Profit factor >= 1.3
[ ] Average slippage <= backtest assumption + 0.5 ticks
[ ] Fill rate on limit orders >= 70%
[ ] All safety systems tested and verified
[ ] Daily loss limit tested (triggered correctly in paper)
[ ] Telegram kill switch tested from mobile
[ ] No days with > 3 ticks average slippage in last 2 weeks
[ ] Paper trading Sharpe consistent across both active strategies
```
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Live Readiness Assessment

**Effort:** 2 days

#### Why This Matters

The decision to go live must be made before you look at the paper trading results — otherwise you'll rationalize away a failed gate. Write down your decision criteria at the start of paper trading (we already did this in Phase 4), then apply them mechanically at the end.

#### Steps

1. Compute final paper trading statistics: Sharpe ratio (annualized), profit factor, win rate, average win/loss, maximum drawdown, drawdown recovery time, average slippage, fill rate, trade count. Compare each to Phase 4 predictions — document any significant discrepancies.

2. Apply gate criteria mechanically: Sharpe ≥ 0.8? Profit factor ≥ 1.3? 200+ trades? Average slippage within 0.5 ticks of backtest assumption? All safety systems functional? If all pass: PROCEED TO LIVE. If any fail: either fix the specific issue and restart the 4-week paper period, or retire the strategy.

3. Write the live trading risk budget: max daily loss ($150), max position (1 MES), max drawdown from peak before full halt (20%, approximately $1,000–$2,000 depending on account size), expected daily P&L range ($-75 to +$100 based on paper trading distribution). This document is reviewed monthly in Phase 6.

4. Final infrastructure checklist: VPS running on Chicago provider ✓, Rithmic L2 data active ✓, Telegram bot commands tested ✓, daily loss limit tested ✓, bracket orders tested ✓, stale data detector tested ✓, backup halt mechanism documented ✓. Get someone else to review this list if possible.

#### ✅ Exit Criteria

> Written go/no-go decision for live trading with documented performance profile and risk budget

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the live readiness assessment system and go-live documentation.

**scripts/live_readiness_check.py**
Automated checklist runner — run before first live trade:
Checks (each prints PASS/FAIL):
1. Broker account funded (prompt: "Is your broker account funded with >= $5,000? [y/n]")
2. Rithmic L2 data: attempt to connect and subscribe, verify tick received within 10s
3. TimescaleDB connection: query mes_ticks table, verify data present for last trading day
4. Safety systems: run unit tests for daily loss limit, position guard, data watchdog
5. Telegram bot: send test message, verify it appears in chat
6. Kill switch: send /halt command, verify bot stops accepting signals, send /resume
7. VPS latency: ping CME IP, report ms (PASS if < 5ms)
8. Paper OMS → live OMS config: verify PAPER_MODE=false in environment
9. Locked parameters: verify config matches docs/phase4/locked-params.yaml exactly
10. Daily loss limit: verify max_daily_loss_usd matches risk budget in docs/phase5/risk-budget.md
Print: "X/10 checks passed. [PROCEED TO LIVE | FIX ISSUES BEFORE PROCEEDING]"

**docs/phase5/risk-budget.md** (template — fill in after paper trading gate)
```markdown
# Live Trading Risk Budget
Date: [FILL AFTER PASSING PAPER GATE]
Based on paper trading period: [START DATE] to [END DATE]

## Position Sizing
- Contracts: 1 MES (non-negotiable for first 60 days)
- Notional per contract: ~$[CURRENT MES PRICE × 5]

## Daily Risk Limits
- Max daily loss: $150 (12 ticks on 1 MES)
- Daily loss limit coded as automatic halt: YES
- Max signals per day: [FROM PAPER TRADING AVG + 50%]

## Drawdown Protocol
- Peak equity: $[STARTING CAPITAL]
- 20% drawdown halt trigger: $[STARTING CAPITAL × 0.80]
- Post-drawdown: 2-week minimum pause + re-validation

## Expected Performance (from paper trading)
- Daily P&L range: $[P10] to $[P90] based on paper distribution
- Expected monthly net: $[MEAN MONTHLY P&L FROM PAPER]
- Max consecutive losing days observed: [FROM PAPER]

## Strategy 2 Unlock Criteria
- 60 consecutive profitable days on Strategy 1
- Strategy 2 passes independent paper trading gate
```
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

## Phase 5 Gate

All exit criteria must pass before advancing to the next phase:

- [ ] **Live Signal Pipeline Integration:** All strategy signals generating in < 50ms end-to-end on Chicago VPS, verified with latency logging
- [ ] **Order Management System (OMS):** OMS successfully routing paper orders to Rithmic for 5 consecutive days with zero duplicate orders or missed stops
- [ ] **Safety Systems & Kill Switches:** All safety systems tested: daily loss limit triggers correctly, Telegram kill switch works, stale data halt works
- [ ] **Paper Trading Execution & Monitoring:** 200+ paper trades executed, Sharpe ≥ 0.8, profit factor ≥ 1.3 — OR extended paper trading continues until criteria met
- [ ] **Live Readiness Assessment:** Written go/no-go decision for live trading with documented performance profile and risk budget


---

# Phase 6: Live Deployment & Continuous Improvement

**Duration:** Ongoing (Monthly Review Cadence)  |  **Goal:** Deploy one contract. Scale only after 60 days of profitable live trading.

## Overview

Live trading is not the finish line — it's the beginning of a continuous research and monitoring cycle. The most common failure mode for algo traders who make it to live trading is: (1) scaling too fast before demonstrating consistent edge, (2) ignoring early warning signs of strategy decay, and (3) failing to maintain the monthly re-validation cadence that keeps the strategy adapted to current market microstructure.

Your Phase 6 philosophy is: one contract, one strategy, maximum caution for the first 60 days. You are not trying to maximize returns in month 1 — you are verifying that the live environment behaves like the paper environment. If it does, you can begin scaling (adding the second strategy, eventually moving to 2 contracts on the best performer). If it doesn't, you need to understand why before committing more capital.

The monthly re-validation cadence is non-negotiable. Market microstructure evolves continuously — the HFT algorithms that provide liquidity change their behavior, the participant composition shifts, volatility regimes come and go. A strategy that was validated on 2024 data may decay meaningfully by Q3 2025. Monthly re-validation catches decay early, when you can re-optimize or switch strategies, rather than after a catastrophic drawdown.

## Key Concepts

| Concept | What It Means for Your Bot |
| --- | --- |
| **Strategy Decay** | The gradual deterioration of a strategy's edge as market conditions change. Causes: HFT algorithm changes (a major liquidity provider changes behavior), participant composition shifts (more/less retail flow), volatility regime change, crowding (too many traders running the same strategy). Detected by: rolling Sharpe dropping below threshold, win rate drift, slippage increase. |
| **Rolling Sharpe Monitoring** | Computing the annualized Sharpe ratio on a rolling window (20 trading days) rather than all-time. This is your early warning system for strategy decay. A healthy strategy should show a relatively stable rolling Sharpe. Persistent downward trend in rolling Sharpe over 4+ weeks is a decay signal. |
| **Scaling Protocol** | The process of increasing position size or adding strategies. Recommended approach: (1) 60 days profitable on Strategy 1 with 1 contract before adding Strategy 2. (2) 90 days profitable on both strategies before moving to 2 contracts on the best performer. (3) Never increase position size after a losing period — scale up only from equity peaks. |
| **Maximum Adverse Excursion (MAE)** | The worst intra-trade drawdown from entry to the eventual exit. If your strategy's backtested MAE distribution shows 95th percentile of $30, but your live trades are regularly seeing $45+ MAE before eventual wins, your stop is being tested more aggressively than modeled. This signals regime change or implementation issue. |
| **Regime Invalidation Protocol** | Pre-defined conditions under which a strategy is immediately moved back to paper trading: (1) rolling 20-day Sharpe drops below 0.3, (2) 3 consecutive losing weeks, (3) average slippage increases > 1 tick above baseline, (4) HMM model shows > 70% time in a state the strategy wasn't designed for. |
| **Monthly Re-Validation** | Each month: re-run WFA on most recent 3 months including live data. If parameters drift significantly from deployed parameters, re-optimize in paper trading first. If OOS Sharpe from WFA drops below 0.5, halt live trading and return to paper. Never re-optimize deployed parameters using live trading results directly — always paper first. |

## Tasks (5 total)

### Go-Live Checklist & First Trade

**Effort:** 1 day

#### Why This Matters

The first live trade is a system integration test — verifying that every component (feed → signal → OMS → broker → confirmation) works end-to-end with real money. Start with the minimum size (1 MES contract) and treat the first week as continued validation, not profit-seeking.

#### Steps

1. Run through the complete go-live checklist: broker account funded and margin available, Rithmic L2 data subscription active and verified, VPS running in Chicago with latest code deployed, all safety systems active (daily loss limit, position guard, stale data detector), Telegram bot commands tested from mobile.

2. Set live trading parameters conservatively for the first week: max_daily_loss = $100 (lower than normal $150), max_signals_per_day = 3 (lower than normal 5–10). This limits downside during the critical first-week system validation period.

3. Place your first trade intentionally: wait for a strong, clear signal (confidence > 0.8, all filters passing). Accept that the first trade might not be the highest-quality setup — the goal is system validation, not the best trade. Confirm via Telegram: fill message received? Correct price? Commission deducted correctly?

4. Review the first week daily: compare live fills to what paper trading produced in comparable conditions. Track: slippage consistency with paper, fill rates, bracket order behavior (target and stop routing correctly). Any significant discrepancy from paper trading behavior requires investigation before continuing.

#### ✅ Exit Criteria

> First live trade placed, confirmed in broker account, correct commission charged

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the go-live automation and first trade verification system.

**scripts/go_live.py**
Interactive go-live script:
1. Runs live_readiness_check.py — must be 10/10 before continuing
2. Prompts: "Ready to go live with 1 MES contract? Type CONFIRM to proceed:"
3. Sets environment: PAPER_MODE=false, MAX_CONTRACTS=1, MAX_DAILY_LOSS=100 (conservative first week)
4. Starts the main bot process as a systemd service
5. Opens Telegram monitoring session: sends "🟢 Bot started LIVE. Trading 1 MES. Daily loss limit: $100."
6. Writes to docs/phase6/go-live-log.md: datetime, starting equity, config hash, strategies active

**scripts/verify_first_trade.py**
Run after the first live trade fills:
- Queries broker fills via Rithmic API (last fill)
- Compares: fill_price vs. signal_price (slippage check), commission deducted (matches broker statement?), order_id in our trade log (fill confirmation received?)
- Prints verification report
- Saves to docs/phase6/first-trade-verification.md

**docs/phase6/go-live-log.md** (template)
```
# Go-Live Log

## Initial Deployment
- Date: [AUTO-FILLED]
- Starting equity: $[AUTO-FILLED]
- Strategies active: [AUTO-FILLED]
- Config hash: [AUTO-FILLED — git commit hash]
- Paper trading period: [START] to [END]
- Paper trading final Sharpe: [AUTO-FILLED]
- Paper trading final profit factor: [AUTO-FILLED]

## First Trade
- Date/time: [FILL AFTER FIRST TRADE]
- Strategy: [FILL]
- Direction: [FILL]
- Signal price: [FILL]
- Fill price: [FILL]
- Slippage: [FILL] ticks
- Commission: $[FILL]
- Outcome: [WIN/LOSS/OPEN]

## Week 1 Notes
[FILL DAILY]
```
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Performance Monitoring Dashboard

**Effort:** 3 days

#### Why This Matters

You cannot manage what you cannot measure. The dashboard gives you instant visibility into whether the strategy is performing as expected, whether slippage is drifting, and whether the HMM model is operating in expected states. It turns abstract bot behavior into observable, actionable data.

#### Steps

1. Set up Grafana connected to TimescaleDB. Create a 'Live Trading' dashboard with panels: (1) Daily P&L bar chart (current month), (2) Rolling 20-day Sharpe (line chart), (3) Win rate by strategy (gauge), (4) Average slippage vs. baseline (line chart with reference line), (5) HMM state distribution pie chart (last 5 days), (6) Trade count by hour (histogram).

2. Configure Grafana alerts: rolling Sharpe drops below 0.4 for 3 consecutive days → Telegram alert. Average slippage exceeds 1.5 ticks for 5 consecutive days → Telegram alert. Zero trades in a session when session was valid (9:30–11:00 AM ET) → Telegram alert (possible signal pipeline issue).

3. Add a 'health' panel: last tick timestamp, last signal timestamp, current position, today's trade count, today's P&L, daily loss limit utilization (%). This should be the first thing you check every morning.

4. Set up weekly automated report: every Friday at 4:30 PM ET, generate and email/Telegram a summary: week's P&L, trade count, win rate, rolling Sharpe, largest winner, largest loser, most common exit reason. This creates a consistent weekly review cadence.

#### ✅ Exit Criteria

> Grafana dashboard live with all metrics, alerts configured and tested

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the Grafana monitoring dashboard configuration and TimescaleDB views.

**sql/monitoring_views.sql**
Create these TimescaleDB views and continuous aggregates:
1. daily_pnl_summary: date, gross_pnl, net_pnl, commission_paid, trade_count, win_count, loss_count, avg_slippage_ticks
2. rolling_sharpe_20d: date, rolling_sharpe (annualized), rolling_sortino
3. hourly_trade_distribution: hour_of_day, avg_trade_count, avg_win_rate (for heatmap)
4. strategy_performance: strategy_id, date, trades, pnl, win_rate, avg_hold_bars
5. hmm_state_distribution: date, state_name, pct_time_in_state

**config/grafana/dashboard.json**
Grafana dashboard JSON (import-ready) with panels:
1. Daily P&L Bar Chart — last 30 days, green=positive, red=negative
2. Rolling 20-Day Sharpe — line chart with reference lines at 0.8 (target) and 0.4 (danger)
3. Win Rate Gauge — current month win rate, 0-100%
4. Avg Slippage vs. Baseline — line chart: actual slippage tick avg vs. backtest assumption (1.0)
5. HMM State Distribution — pie chart: % time in each regime last 5 days
6. Trade Count by Hour — histogram showing when trades occur
7. Health Status Panel — last tick age, position, daily P&L, daily loss limit utilization %
8. Strategy Breakdown Table — per-strategy: trades, P&L, Sharpe this month

**src/monitoring/metrics_exporter.py**
MetricsExporter class — runs as background task:
- Every 5 minutes: computes rolling metrics from TimescaleDB, writes to TimescaleDB metrics table
- Grafana reads from metrics table (not raw trade table — avoids slow queries on dashboard)

**docs/phase6/grafana-setup.md**
Step-by-step:
- Installing Grafana on VPS
- Connecting TimescaleDB as data source
- Importing dashboard.json
- Setting up alert notifications to Telegram channel
- Setting up Grafana Cloud (free tier) for remote access from mobile
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Monthly Re-Validation Process

**Effort:** 4 hours/month

#### Why This Matters

Market microstructure evolves continuously. Strategy decay is real and can happen within 2–3 months of deployment. Monthly re-validation catches decay early, when you have options — re-optimize, switch to the backup strategy, or take a research break. Catching it after 4 months of bleeding is much more expensive.

#### Steps

1. On the first Monday of each month, run the re-validation pipeline: pull the last 3 months of live + paper trading data. Append to TimescaleDB. Re-run WFA using the same parameters as Phase 4 validation. The only difference: include the most recent month of live trading in the OOS test window.

2. Compare current month's performance to Phase 4 baseline: current Sharpe vs. validated Sharpe, current win rate vs. validated win rate, current slippage vs. paper trading slippage. Document deltas. If any metric is > 30% worse than baseline, elevate to full investigation.

3. Re-optimization decision: if WFA shows parameter drift (optimal parameters from current data differ significantly from deployed parameters), create a paper trading variant with the new parameters. Run both in parallel for 2 weeks. Only switch to new parameters if paper variant shows better performance — never re-optimize based on live trading results alone.

4. Write the monthly review document: date, strategy performance summary, WFA results, parameter status (unchanged / updated to new values after paper validation), regime analysis (dominant HMM states this month vs. design intent), any incidents (safety systems triggered, connection issues, unexpected fills), forward plan for next month.

#### ✅ Exit Criteria

> Documented monthly review for each month of live trading, parameters re-confirmed or updated

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the monthly re-validation automation.

**scripts/monthly_revalidation.py**
Run on first Monday of each month (automate via cron: 0 9 1-7 * 1 for first Monday at 9 AM):
1. Pull last 3 months of live + paper trade data from TimescaleDB
2. Pull last 3 months of feature vectors (for WFA)
3. Re-run WFA using WFARunner from Phase 4 with same config
4. Compare: current WFA efficiency vs. Phase 4 baseline
5. Compare: current OOS Sharpe vs. Phase 5 gate Sharpe
6. Check for parameter drift: optimal params this month vs. locked params
7. Generate report: docs/phase6/monthly-reviews/YYYY-MM.md
8. Send Telegram summary: "📊 Monthly Re-Validation: Strategy ORB Sharpe=1.2 (baseline 1.4, -14%). Params stable. No action needed."
   Or: "⚠️ Monthly Re-Validation: Strategy VWAP Sharpe=0.4 (below 0.5 threshold). Consider moving to paper."

**docs/phase6/monthly-reviews/template.md**
```markdown
# Monthly Re-Validation — [MONTH YEAR]
Generated: [AUTO]

## Performance vs. Baseline
| Metric | Phase 5 Baseline | This Month | Delta | Status |
|---|---|---|---|---|
| Sharpe | [AUTO] | [AUTO] | [AUTO] | [AUTO] |
| Win Rate | [AUTO] | [AUTO] | [AUTO] | [AUTO] |
| Profit Factor | [AUTO] | [AUTO] | [AUTO] | [AUTO] |
| Avg Slippage | [AUTO] | [AUTO] | [AUTO] | [AUTO] |

## WFA Results
- Efficiency ratio: [AUTO]
- OOS Sharpe: [AUTO]
- Parameter drift: [AUTO — list any params that shifted > 15%]

## HMM Regime Analysis
- Dominant regime this month: [AUTO]
- % time in mean reversion states: [AUTO]
- % time in trending states: [AUTO]
- Notable: [flags if regime distribution is unusual]

## Incidents
[Manual — fill in any safety system triggers, connection issues, unexpected fills]

## Decision
[ ] No action — continue as-is
[ ] Monitor closely — metrics borderline
[ ] Move to paper — Sharpe below threshold
[ ] Re-optimize in paper — parameter drift detected
[ ] Retire strategy — [reason]

## Next Month Focus
[Manual]
```
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Strategy 2 Onboarding

**Effort:** 2 weeks (parallel activity after 60 live days)

#### Why This Matters

Running two uncorrelated strategies significantly improves risk-adjusted returns — the same logic as portfolio diversification. But Strategy 2 must pass the paper trading gate independently, without any benefit of doubt from Strategy 1's performance. Each strategy earns its live deployment separately.

#### Steps

1. After 60 profitable days on Strategy 1 live, reactivate paper trading for Strategy 2 (the second survivor from Phase 4). Run both simultaneously: Strategy 1 live with real money, Strategy 2 in paper mode on the same machine.

2. Apply identical Phase 5 gate criteria to Strategy 2: Sharpe ≥ 0.8, profit factor ≥ 1.3, 200+ paper trades. This takes another 4–6 weeks of paper trading. Do not rush this gate.

3. When Strategy 2 passes the paper gate, compute the correlation between Strategy 1 and Strategy 2 daily P&L from paper trading (should be < 0.3 — confirmed in Phase 4 but re-verify with live data). If correlation is now > 0.5, they're too similar to deploy simultaneously — only deploy the one with better DSR.

4. Go live with Strategy 2 on 1 MES contract alongside Strategy 1. Monitor both independently on the Grafana dashboard. Track combined account-level metrics: total daily P&L, combined maximum position (should not exceed 2 MES contracts combined), combined daily loss limit ($250 for 2 strategies).

#### ✅ Exit Criteria

> Strategy 2 passing paper trading gate independently, live deployment on separate account or alongside Strategy 1

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the Strategy 2 onboarding system — running a second strategy in parallel paper trading while Strategy 1 is live.

**scripts/run_dual_mode.py**
Dual-mode runner: Strategy 1 LIVE + Strategy 2 PAPER simultaneously:
- Receives same tick feed
- Strategy 1: routes to live OrderManager (real broker)
- Strategy 2: routes to PaperOrderManager (simulated fills)
- Completely isolated P&L tracking: live_pnl vs. paper2_pnl
- Shared safety systems: both strategies contribute to the single daily_loss_limit (combined)
- Config flag: STRATEGY_2_LIVE=false (flip to true when paper gate passes)

**src/monitoring/dual_strategy_monitor.py**
DualStrategyMonitor class:
- track_correlation(live_fills, paper2_fills, window_days=20) → float: daily P&L correlation
- alert_if_correlated(threshold=0.5) → sends Telegram if strategies too correlated to deploy simultaneously
- combined_metrics() → dict: combined Sharpe, combined max drawdown, diversification ratio

**scripts/strategy2_gate_check.py**
Same as Phase 5 paper trading gate, but runs independently for Strategy 2:
- Loads paper_trades WHERE strategy_id = 'strategy_2_id'
- Applies same gate: Sharpe >= 0.8, profit factor >= 1.3, 200+ trades
- Also checks: correlation with live Strategy 1 daily P&L < 0.3
- Outputs: docs/phase6/strategy2-gate-result.md with PASS/FAIL
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

### Drawdown Protocol & Strategy Retirement

**Effort:** Ongoing

#### Why This Matters

The biggest killer of retail algo traders is not bad strategies — it is good strategies that decay, combined with the psychological inability to stop trading them. A written drawdown protocol removes the human decision from the process. You define the rules in advance, when you're clear-headed. You follow them mechanically when you're under pressure.

#### Steps

1. Define and code the drawdown protocol: if account drawdown from peak exceeds 20% (approximately $1,000–$2,000 depending on account size), all strategies automatically halt. Implementation: compute running peak equity; if current_equity < peak_equity × 0.80, trigger the same halt sequence as the daily loss limit.

2. Post-drawdown protocol: after a 20% drawdown halt, minimum 2-week pause before resuming trading. During the pause: (1) re-run the full Phase 4 validation on all currently deployed strategies using the most recent data, (2) identify whether the drawdown was regime-related (HMM model in unexpected states) or implementation-related (slippage increase, fill rate drop), (3) return to paper trading for 2 weeks minimum before re-deploying live.

3. Strategy retirement criteria: if a strategy triggers the 20% drawdown halt twice in 6 months, it is permanently retired from live trading. It may be re-researched in Phase 3 with a new hypothesis, but the current implementation is closed. Document the retirement with full P&L history and postmortem analysis.

4. Keep a trading journal (10 minutes per day): not a feelings journal, but a data journal. Note: did the market behave as the HMM model predicted? Were there any signals that looked right on the chart but didn't generate? Any fills that seemed off? This qualitative data catches issues that quantitative metrics miss.

#### ✅ Exit Criteria

> Written drawdown protocol followed exactly — no discretionary overrides

#### Claude Code Prompt

Open Claude Code in your repo root and paste the following:

```text
You are helping build a MES futures scalping bot. Create the drawdown monitoring system and strategy retirement documentation.

**src/safety/drawdown_monitor.py**
DrawdownMonitor class:
- peak_equity: float (initialized to starting_capital, updated on new equity highs)
- current_equity property → float (computed from starting_capital + sum of fills)
- drawdown_pct property → (peak_equity - current_equity) / peak_equity
- on_fill(fill_event) → None: updates current_equity, checks drawdown threshold
- check() → None: if drawdown_pct >= 0.20: trigger full halt
- async halt_all(reason: str):
  1. oms.flatten_all()
  2. risk_manager.halt("20% drawdown threshold breached")
  3. Telegram: "🔴 DRAWDOWN HALT\nPeak: $X\nCurrent: $Y\nDrawdown: 20.X%\nAll strategies halted. 2-week minimum pause begins."
  4. Write to docs/phase6/incidents/drawdown-YYYY-MM-DD.md
  5. Set flag: REQUIRES_REVALIDATION=true

**scripts/post_drawdown_analysis.py**
Run after a drawdown halt, before resuming:
1. Load all trades from the drawdown period
2. Identify: was drawdown regime-related (HMM shows unusual state distribution?) or implementation-related (slippage spike, fill rate drop)?
3. Re-run CPCV on most recent 6 months
4. Re-run WFA on most recent 6 months
5. Generate: docs/phase6/incidents/drawdown-analysis-YYYY-MM-DD.md
6. Decision output: "RESUME PAPER" or "RETIRE STRATEGY" with reasoning

**docs/phase6/drawdown-protocol.md**
```markdown
# Drawdown Protocol

## Automatic Halt Triggers
1. Daily loss limit: $150 (coded, automatic)
2. Peak-to-trough drawdown: 20% of starting capital (coded, automatic)
3. 3 consecutive losing weeks (manual review trigger — Telegram alert, not auto-halt)

## Post-Halt Process (20% drawdown)
- [ ] Minimum 2-week trading pause (no exceptions)
- [ ] Run post_drawdown_analysis.py
- [ ] Review HMM state distribution during drawdown period
- [ ] Review slippage and fill rate during drawdown period
- [ ] Determine cause: regime change, implementation bug, or random variance
- [ ] Decision: resume paper, re-optimize, or retire strategy
- [ ] Paper trade for minimum 2 weeks before re-deploying live

## Strategy Retirement Criteria
A strategy is permanently retired from live trading if:
- 20% drawdown halt occurs twice within 6 months
- Monthly re-validation shows OOS Sharpe < 0.3 for 2 consecutive months
- Root cause analysis cannot identify recoverable failure mode

## Retired Strategy Archive
Retired strategies are NOT deleted. They are moved to strategies/retired/ with:
- Full validation results
- Live trading P&L history
- Postmortem document
- Research hypotheses for future improvement
```
```

> 💡 **Tip:** In Claude Code, use `/` to paste multi-line prompts. All file paths are included — Claude Code will create them automatically in your repo.

---

## Phase 6 Gate

All exit criteria must pass before advancing to the next phase:

- [ ] **Go-Live Checklist & First Trade:** First live trade placed, confirmed in broker account, correct commission charged
- [ ] **Performance Monitoring Dashboard:** Grafana dashboard live with all metrics, alerts configured and tested
- [ ] **Monthly Re-Validation Process:** Documented monthly review for each month of live trading, parameters re-confirmed or updated
- [ ] **Strategy 2 Onboarding:** Strategy 2 passing paper trading gate independently, live deployment on separate account or alongside Strategy 1
- [ ] **Drawdown Protocol & Strategy Retirement:** Written drawdown protocol followed exactly — no discretionary overrides

---

# Appendix: Strategy Cheat Sheet

### 1. Opening Range Breakout (ORB)

| Parameter | Value / Logic |
| --- | --- |
| **Entry** | 5-min bar closes outside 9:30–9:45 AM high/low |
| **Direction** | Long if closes above ORB high; Short if closes below ORB low |
| **Target** | 50% of ORB width from breakout bar close |
| **Stop** | Opposite ORB boundary |
| **Max Hold** | 90 minutes — exit at market if unresolved |
| **Filters** | Volume > 1.5× 20-day avg; VWAP aligned; Not HMM State 3; Signal before 11 AM |
| **Best In** | HMM States: High-Vol Up and High-Vol Down |

### 2. VWAP Reversion

| Parameter | Value / Logic |
| --- | --- |
| **Entry (Reversion)** | Price touches –2 SD or –3 SD VWAP + reversal candle confirmation |
| **Entry (Pullback)** | Trending session: buy pullback to VWAP from above |
| **Target** | Return to VWAP (reversion) or 2× entry distance (pullback) |
| **Stop** | –3.5 SD VWAP (reversion) or 0.5 SD beyond VWAP (pullback) |
| **Mode Switch** | VWAP slope determines mode: flat → reversion, steep → pullback |
| **Filters** | Skip first 15 min of session; skip 30 min around major releases |
| **Best In** | HMM States 3 and 5 for reversion; States 1 and 2 for pullback |

### 3. CVD Divergence + POC

| Parameter | Value / Logic |
| --- | --- |
| **Bearish Setup** | Price makes higher high; CVD makes lower high (≥15% divergence) |
| **Bullish Setup** | Price makes lower low; CVD makes higher low (≥15% divergence) |
| **Confluence** | Divergence must occur within 6 ticks of 4-hour volume profile POC |
| **Target** | 12–16 ticks (3–4 points) |
| **Stop** | Beyond the diverging swing extreme + 2 tick buffer |
| **Max Hold** | 4 dollar bars (~20 minutes) |
| **Frequency** | 1–3 high-conviction signals per day only |

### 4. Volatility Regime Switcher

| Parameter | Value / Logic |
| --- | --- |
| **Regime Detection** | 20-bar upside vs. downside semi-variance vs. rolling 250-bar 75th percentile |
| **High-Vol Entry** | 3-bar pullback against dominant semi-variance direction |
| **High-Vol R/R** | 8 ticks target / 2 ticks stop |
| **Low-Vol Entry** | Price moves 1.5 SD from VWAP in either direction |
| **Low-Vol R/R** | 2 ticks target / 30 ticks stop (high win rate mode) |
| **HMM Guard** | Semi-var and HMM must agree — no trade if conflicting |
| **Transition** | Cancel open signals, wait 2 bars before entering new mode |

### 5. 80% Rule (Value Area)

| Parameter | Value / Logic |
| --- | --- |
| **Setup** | Price opens outside prior session value area |
| **Trigger** | Price closes inside VA AND holds inside for 2 consecutive 30-min periods |
| **Entry** | Close of the second confirming 30-min bar |
| **Target** | Opposite value area boundary (VAH or VAL) |
| **Stop** | Price closes back outside value area on a 30-min bar |
| **Time Filter** | Invalidate if no confirming signal by 11:30 AM |
| **Quality Filter** | Prior session must have ≥60% of volume within the value area |

---

# Phase 7: Filter Wiring & L1 Historical Validation

**Duration:** 2 Weeks  |  **Goal:** Wire microstructure filters into the pipeline and validate they improve risk-adjusted returns using 1 year of L1 data.

**Context:** Phases 1-5 complete. Live paper trading operational on Tradovate. 1 year L1 historical data available. 8 filter modules fully implemented with tests but not all wired into the pipeline.

## Strategic Data Decisions

### L1 Data — Use Aggressively
1 year of L1 is enough to validate VPIN, spread filter, and mid momentum with statistical confidence. Do all exploratory work here first.

### OBI is Already Invalidated
OBI was tested as a standalone strategy via CPCV with actual L1 TBBO bid/ask sizes. Result: Sharpe -17 in both momentum and contrarian interpretations. The feature module is kept for potential use as a soft confidence modifier, but do not invest further validation time in OBI predictive power at 5s/4-tick scale.

## Tasks (4 total)

### 7.1 — Wire Spread Filter as Hard Gate ✅ DONE

Wire SpreadMonitor (`src/filters/spread_monitor.py`) into the live signal pipeline and backtest engine as a hard gate blocking all entries during anomalous spreads.

**What was done:**
- SpreadMonitor wired into BacktestConfig with `spread_monitor` field
- BacktestEngine calls `push_sync()` per bar, checks `is_spread_normal()` before signal evaluation
- Validated on ORB over 3 months: blocks ~4% of trades, improves Sharpe

### 7.2 — VPIN Regime Filter ✅ DONE

Implement VPIN as a regime filter using L1 data with Bulk Volume Classification.

**What was done:**
- `src/filters/vpin_monitor.py` — VPINMonitor with BVC bar approximation (not binary classification)
- Regime gating: VWAP blocked in "trending" (VPIN > 0.55), ORB/CVD blocked in "mean_reversion" (VPIN < 0.38)
- Thresholds calibrated empirically: VPIN mean=0.467, P50=0.464 on 1yr L1 data
- BacktestEngine integration: `vpin_monitor` field in BacktestConfig, calls `on_bar_approx()` with BVC per bar
- Validated on ORB: combined spread+VPIN over full year: 154→59 trades, Sharpe 1.56→6.69, WR 74.7→79.7%, max DD 4.19→2.05%

### 7.3 — HMM vs Filters Comparison 🔲 TODO

**The key question:** VPIN + spread filters are doing regime detection (market health, trending vs mean-reverting) — which overlaps heavily with the HMM classifier built in Phase 2. Are they redundant or complementary?

Run each survivor strategy through 3 configurations on 1 year of L1 data:
1. **Filters only** (spread + VPIN, no HMM)
2. **HMM only** (no spread/VPIN filters)
3. **Both** (HMM + spread + VPIN)

**Output:** For each strategy × configuration, compute: trades, win rate, Sharpe, Sortino, profit factor, max drawdown.

**Decision criteria:**
- If filters-only matches or beats HMM-only → drop HMM (simpler, more interpretable, directly grounded in microstructure)
- If both > filters-only by meaningful margin → keep both (complementary signal)
- If HMM-only > filters-only → investigate why (HMM captures something filters miss)

**Script:** `scripts/validate_hmm_vs_filters.py`

### 7.4 — Run All Strategies Through Combined Filters 🔲 TODO

Run VWAP, CVD Divergence, and Vol Regime Switcher through the 4-way combined filter validation (no filter, spread only, VPIN only, both) on full year L1 data.

**Requirements per strategy:**
- VWAP: needs `reversion_hmm_states=[], pullback_hmm_states=[], min_confidence=0.3`
- CVD: needs `require_hmm_states=[]`
- Vol Regime: needs `high_vol_hmm_states=[], low_vol_hmm_states=[]`

**Script:** Update `scripts/validate_combined_filters.py` to accept `--strategy` flag.

## Phase 7 Gate

- [ ] Spread + VPIN filters validated on ORB (full year L1) — ✅ DONE
- [ ] HMM vs Filters comparison completed for all strategies
- [ ] All strategies tested with combined filters
- [ ] Decision documented: keep HMM, drop HMM, or use both

---

# Phase 8: L2 Data Acquisition & Parsing

**Duration:** 1 Week  |  **Goal:** Download and parse 3 months of L2 MBP-10 data into queryable Parquet files.

*Only start after Phase 7 validation is complete.*

## Tasks (1 total)

### 8.1 — Download and Parse L2 Data

Download L2 MBP-10 data from DataBento for Sept-Nov (best mix of trending/mean-reversion regimes).

**Download:** Script exists at `scripts/download_l2.py`. Uses DataBento batch API (submit_job → poll → download). Monthly chunks. Resume support.

**Parse to Parquet:** Create `scripts/parse_l2.py`:
- Read `.dbn.zst` files using `databento.DBNStore`
- Extract 10-level bid/ask snapshots per timestamp
- Write to `data/l2/year=YYYY/month=MM/data.parquet`
- Columns: timestamp, bid_price_0..9, bid_size_0..9, ask_price_0..9, ask_size_0..9

**Derived signals:** Create `scripts/build_l2_signals.py`:
- Stream parsed L2 Parquet in time order
- Feed each snapshot through existing filter modules: OBIMonitor, WeightedMidMonitor, DepthMonitor, IcebergDetector
- Write computed signals to `data/l2_signals/data.parquet`

**Cost estimate (March 2026):** 3 months L2 ~$60. DataBento dataset: `GLBX.MDP3`, schema: `mbo`, symbol: `MES.c.0`.

---

# Phase 9: L2 Signal Wiring & Validation

**Duration:** 2 Weeks  |  **Goal:** Quantify actual L2 lift over L1-only baselines. Wire depth signals if validated.

*3 months of L2 data now in Parquet.*

## Tasks (3 total)

### 9.1 — Wire L2 Signals Into Live Pipeline

Wire L2 depth signals into the live pipeline with graceful L1 fallback. L2 signals are advisory — if L2 feed is down, fall back to L1 approximations.

- Add `L2SnapshotEvent` to `src/core/events.py`
- Add `l2_available`, `obi_true`, `weighted_mid` to FeatureVector
- Strategies use true OBI when available, fallback to L1 approx

### 9.2 — Validate L2 vs L1 Lift

Run VWAP and ORB strategies three ways:
1. No OBI filter (baseline)
2. L1 OBI as soft confidence modifier
3. True L2 OBI as soft confidence modifier

**Decision:** If L2 shows < 2% win rate improvement over L1 approx, do not subscribe to L2 feed in live trading. Use L1 approximation permanently.

### 9.3 — Wire Depth Deterioration Into ORB

Wire DepthMonitor into ORB as breakout confirmation (confidence modifier, not hard gate):
- Ask thinning on long breakout → boost confidence
- Bid thinning on short breakout → boost confidence
- Counter-side thinning → reduce confidence

---

# Phase 10: SignalAggregator & Full Integration

**Duration:** 1 Week  |  **Goal:** Combine all validated filters into a single entry gate.

*Only build after Phase 9 validation — only include signals that passed validation.*

## Tasks (1 total)

### 10.1 — Build SignalAggregator

`src/filters/signal_aggregator.py` — combines validated filters:

**Hard blocks (any = rejected):**
- Spread not normal
- VPIN regime conflicts with strategy type

**Soft confidence scoring (only for validated signals):**
- Weighted mid lean matches direction: +0.10
- Depth deterioration matches: +0.10

**EntryDecision dataclass:** `approved`, `confidence_adjustment`, `reasons`, `regime`

Wire into SignalHandler. Log full EntryDecision on every evaluation.

---

# Phase 11: Operational Hardening

**Duration:** 2 Weeks  |  **Goal:** Safety systems, monitoring, and trade journal before going live with real money.

## Tasks (4 total)

### 11.1 — Kill Switch and Position Flatten

Add to health FastAPI server:
- `POST /halt` — sets `RiskManager.halted = True`
- `POST /flatten` — halts + submits market order to close position
- `POST /resume` — clears halt flag
- Bearer token auth (KILL_SWITCH_TOKEN in .env)
- Telegram notifications on halt/flatten events

### 11.2 — Strategy Correlation Monitor

Detect when multiple strategies pile into the same direction simultaneously:
- Track last signal direction per strategy in rolling 15-minute window
- `compute_strategy_correlation()` → 0.0–1.0
- Block additional entries when correlation > 0.75
- Add to RiskManager

### 11.3 — Live Trade Journal

Structured Parquet journal capturing full signal context per trade:
- `data/journal/trades_YYYY-MM-DD.parquet`
- Columns: entry/exit time, strategy, direction, prices, PnL, spread_z, vpin, regime, aggregator confidence, exit reason, MAE/MFE
- Analysis script: `scripts/trade_report.py --days 30`
- Key metric: does aggregator confidence correlate with trade outcome?

### 11.4 — Parameter Drift Alerting

`src/monitoring/drift_monitor.py` — automated detection when live conditions drift outside backtest ranges:
- Load baseline feature statistics from backtest Parquet
- Every 30 min: compute rolling z-score for ATR, spread, VPIN, CVD slope
- Alert via Telegram if any feature > 3.0 sigma from baseline
- Do NOT halt automatically — human decision
- Cooldown: max 1 alert per hour per feature

## Phase 11 Gate

- [ ] Kill switch tested: halt/flatten/resume work correctly
- [ ] Correlation monitor blocks concentrated exposure
- [ ] Trade journal capturing full context for every paper trade
- [ ] Drift alerting tested with synthetic out-of-range data

---

# Critical Decision Gates

**After Phase 7.3:** If VPIN + spread filters match or beat HMM alone, drop the HMM. Simpler system = more robust in production. If HMM adds meaningful lift, keep both.

**After Phase 9.2:** If L2 signals show < 2% lift over L1 approximation, do not subscribe to L2 feed in live trading. Use L1 approximation permanently and save the L2 budget.

**After 4 weeks of live journal data (11.3):** If aggregator confidence does not correlate with trade outcome, the aggregator is adding complexity without value. Revert to simpler per-strategy hard gates (spread + VPIN only).

---

# Appendix: Tech Stack

| Layer | Tool | Notes |
| --- | --- | --- |
| Historical Data | Databento (GLBX.MDP3) | L1 TBBO + L2 MBP-10. Nanosecond timestamps. Same API for historical + live. |
| Live Data + Execution | Tradovate REST + WebSocket | Cloud-hosted API. Free plan $0.35/side. Paper + live modes. |
| Broker | Tradovate | Free plan, $0.35/side MES commission. REST auth + WS streaming. |
| Backtesting | Custom bar-replay engine | `src/backtesting/engine.py` — slippage model, simulated OMS, CPCV + WFA + DSR. |
| ML / Regimes | hmmlearn | GaussianHMM with 5 states. KMeans initialization. |
| Validation | Custom CPCV + DSR | `src/backtesting/cpcv.py`, `src/backtesting/dsr.py`, `src/backtesting/wfa.py`. |
| Event Loop | asyncio | Event-driven pub/sub via EventBus. |
| Storage | Parquet (Polars + zstd) | All persistence is local Parquet. No database. |
| Monitoring | FastAPI /health endpoint | Position, daily P&L, last tick age, event counts, uptime. |
| Alerting | Telegram Bot (planned) | /status, /halt, /resume commands. |
| VPS | Vultr Chicago $6/mo | IP: 66.42.124.72. SSH key auth, UFW active. |

---

# Non-Negotiable Rules

**Minimum 8-tick profit target** — No exceptions until broker RT commission is under $0.25. At 2–4 tick targets, the math does not work.

**Never skip CPCV + DSR** — If a strategy does not pass DSR ≥ 0.95, it does not go live. No exceptions based on feel or visual backtest quality.

**200 paper trades minimum** — Paper trading with fewer trades is statistically meaningless. Extend paper trading duration until the trade count is met.

**Hard daily loss limit is automated** — Not a manual check. Code it. Test it. If it is not in the code, it does not exist when you need it most.

**Monthly re-validation is non-negotiable** — Strategy decay happens at intraday timescales. Monthly WFA re-runs catch decay early.

**Scale only from equity peaks** — Never increase position size after a losing period. Scale up only when the account is at or near all-time high.

---

# Phase 12: L1/L2 Order Book Strategies

**Goal:** Implement and validate a suite of order-book-driven strategies that exploit L2 depth data, OFI signals, and microstructure patterns. These strategies require upgrading from Tradovate to Rithmic for proper MBO/L2 data access.

> **Infrastructure prerequisite:** Strategies ranked 3, 7, 9, and 10 require Rithmic MBO data. Strategies ranked 1, 2, 4, 5, 6, and 8 can begin with Tradovate L1 + Databento L2 for backtesting, but production deployment benefits significantly from Rithmic.

---

## The Foundational Signal: Order Flow Imbalance (OFI)

OFI (Cont, Kukanov & Stoikov, 2014) is the bedrock metric for every strategy in this phase. It measures net imbalance between supply and demand changes at the best bid/ask by tracking four state transitions:

```
e(t) = I{P_bid(t) ≥ P_bid(t-1)} × ΔV_bid - I{P_ask(t) ≤ P_ask(t-1)} × ΔV_ask
OFI = Σ e(t) over interval
```

Linear price impact model **ΔP ≈ β × OFI** holds with R² ~65%. Multi-level OFI (5 depth levels via PCA) pushes R² above **85%**. OFI exhibits positive autocorrelation — imbalance at time t predicts continued imbalance at t+1.

**Critical caveat:** High explanatory power ≠ profitability. Transaction costs ($12.50 per ES round-trip) destroy sub-tick edge. Penn State research found **Pearson correlation of -0.775** between exchange latency and OFI-strategy P&L. OFI is best deployed as an alpha component within a multi-signal system.

**Companion metrics to compute alongside OFI:**

| Metric | Formula | Use |
|--------|---------|-----|
| Book Imbalance Ratio | (ΣQ_bid - ΣQ_ask) / (ΣQ_bid + ΣQ_ask) | Directional bias [-1, +1] |
| Weighted Mid Price (VAMP) | (P_bid × Q_ask + P_ask × Q_bid) / (Q_bid + Q_ask) | Fair value estimate |
| Stoikov Microprice | Martingale limit of expected mid-prices conditional on book state | Superior fair value (requires Markov model training) |
| Queue Exhaustion | ΔQ_bid / Q_bid_initial over rolling window | Imminent level break detection |
| Distance-Weighted Imbalance | Exponential decay [1.0, 0.5, 0.25, 0.125, 0.0625] to levels 1–5 | Noise-reduced directional signal |

**Sampling:** Compute OFI in 1-second buckets with 5–60 second rolling windows. Process every tick for book state, evaluate strategy logic every 1–5 seconds. Top 5 depth levels capture nearly all marginal predictive improvement. Normalize via rolling Z-score (5-minute window).

---

## Strategy Rankings (by suitability for our setup)

### Rank 1: CVD Divergence with L2 Book Imbalance Confirmation

**Data:** L1 trade prints for CVD; L2 depth (5 levels) for book imbalance confirmation.

**Mechanism:** Detects exhaustion points where aggressive order flow weakens despite price extension. Bearish divergence = price higher high, CVD lower high = buyers losing momentum. L2 layer confirms: book imbalance shifting against price direction (e.g., ask depth building after price rises) strengthens reversal signal. Dual-confirmation (trade flow via CVD + resting orders via L2) reduces false signals.

**Entry/exit:** Identify CVD divergence over 3–5 bars on 512-tick or 5-minute charts. Require divergence at key level (VWAP, prior day H/L, value area boundary). Confirm with book imbalance ≥ |0.3| opposing price direction. Enter on price breaking below divergent bar's low (bearish) or above high (bullish). Stop 4–6 ticks beyond swing extreme; target 1.5–2× risk or next significant level. Time stop: 3–5 minutes.

**Proximity benefit:** Low-to-moderate. CVD divergences develop over multiple bars (minutes). L2 confirmation tolerates 5–20ms delay.

**Realistic edge:** 55–65% win rate, 1.5–2:1 R:R, estimated Sharpe 0.8–1.5 before costs. 2–4 high-quality signals per RTH session. Works in range-bound/exhaustion markets, fails on trend days.

**Implementation:** Medium-high. Trade classifier, CVD accumulator, swing detection (N-bar fractal), divergence comparator, book imbalance calculator, signal combiner.

**Data feed:** Tradovate insufficient (documented CVD discrepancies). Rithmic recommended. Databento excellent for backtesting.

**Key risk:** Absorption masquerading as divergence — large passive seller absorbing buying creates high CVD without follow-through (looks bullish, is bearish).

---

### Rank 2: Microstructure Mean Reversion at Overextension Points

**Data:** L1 trade data for CVD/VWAP; L2 depth (5 levels) for book state after sharp moves; trade velocity from time & sales.

**Mechanism:** Exploits temporary price dislocations from aggressive order bursts, stop-loss cascades, and liquidity vacuums. Market makers pull quotes during fast moves (wide spreads), then re-enter when move exhausts, normalizing price. Kinlay documents Sharpe ratios of 3–5 for well-implemented versions.

**Entry/exit:** Price moves ≥4 ticks from VWAP in <30 seconds AND CVD diverges AND book rebuilds on opposing side → enter fade. Alternative: 3+ failed attempts to break a level with high delta → enter opposing direction. Stop 2 ticks beyond extreme. Target VWAP or POC. Time stop 30–60 seconds.

**Proximity benefit:** Moderate. Better proximity = better fill prices on fade entry. Signal window (5–120 seconds) forgiving for ~2ms latency.

**Realistic edge:** Sharpe 2–4 achievable with regime filtering. Works best in range-bound/choppy sessions. Fails catastrophically when fading informed flow or pre-news leaks.

**Implementation:** Medium. VWAP calculator, CVD tracker, book depth monitor, absorption detector, overextension state machine. Critical: regime detector to avoid fading strong trends.

---

### Rank 3: Bid-Ask Absorption and Iceberg Detection

**Data:** L2/MBO for detecting hidden orders that repeatedly refill at a price level. Trade prints for volume-at-price. Native CME icebergs preserve OrderID across refills (MBO TRADE → MODIFY cycles). Synthetic icebergs: new orders of similar size at same price within ~300ms.

**Mechanism:** Absorption = aggressive market orders hit a level repeatedly without breaking through. Signal: traded volume at level exceeds displayed quantity by 5:1+, with consistent mechanical replenishment. Enter in direction supported by absorption.

**Entry/exit:** Identify key level (prior day H/L, VWAP, round number). Wait for 2–3 confirmed refills with significant volume. Limit order 1 tick above absorption level for queue position. Target 4–8 ticks; stop 2–3 ticks below absorption level. Time stop 30–60 seconds.

**Proximity benefit:** Moderate. MBO event sequencing benefits from lower latency, but confirmation requires 2–3 refill cycles (seconds).

**Realistic edge:** 55–65% win rate at established levels with correlated instrument confirmation. Short half-life (seconds to ~1 minute), false positives common.

**Implementation:** High. Requires MBO feed — order-ID-to-price map, TRADE → MODIFY cycle detection. Synthetic iceberg: heuristic (similar size, same price, <300ms).

**Data feed:** Tradovate insufficient (no MBO). **Rithmic required.**

---

### Rank 4: Level-Based Liquidity Sweep Reversal

**Data:** L1 trade prints + velocity for sweep detection. L2 depth for book thickness near key levels. Pre-session level identification.

**Mechanism:** Liquidity clusters at prior day extremes, overnight H/L, round numbers, VWAP — with stop orders beyond. Between clusters, liquidity is thin → fast traversal. When price sweeps through a level (triggering stops), forced liquidation creates temporary dislocation. **Trade the reversal after the sweep**, not the sweep itself.

**Entry/exit:** Pre-mark key levels. On sweep, start 5–15 second timer. If no sustained aggressive follow-through, absorption appears on other side, or delta diverges → enter opposite sweep direction. Limit order 1–2 ticks inside prior range. Target 6–12 ticks (next liquidity cluster); stop 3–4 ticks beyond sweep extreme. Time stop 2–3 minutes.

**Proximity benefit:** Low-to-moderate. Sweep events play out over seconds to minutes.

**Realistic edge:** 50–60% win rate, 1:1.5 to 1:3 R:R. One of the most reliable DOM patterns, grounded in stop-order placement behavior. Overnight trading regularly produces sweeps spanning 15–25 points.

**Implementation:** Moderate. Pre-session level parser, book thickness calculator, sweep detector (price crossing level + volume spike), failure/reversal detector, state machine.

**Data feed:** Tradovate partially sufficient (core signal visible in L1). Rithmic recommended for book thickness and absorption confirmation.

---

### Rank 5: OFI-Based Directional Scalping

**Data:** L1 BBO (bid/ask price and size changes) for OFI. L2 depth (5 levels) for multi-level OFI and market depth skew.

**Mechanism:** When normalized OFI exceeds threshold (Z-score > ±1.5–2.0), enter in direction of imbalance. OFI autocorrelation means imbalance predicts short-term continuation. Enhanced by combining with Market Depth Skew.

**Entry/exit:** Rolling OFI over 5–30 second windows, Z-score normalized. Enter when Z > +1.5 (long) or < -1.5 (short). Fixed tick target 3–5 ticks on ES. Hard stop 2 ticks. Time exit 5–30 seconds.

**Proximity benefit:** **High.** Signal half-life is 5–30 seconds. The -0.775 latency-P&L correlation means speed directly determines profitability. Most latency-sensitive strategy of all ten.

**Realistic edge:** Sharpe 1–3 standalone, but frequently negative after costs. Better as alpha signal within multi-signal system than standalone entry.

**Data feed:** Rithmic minimum. Tradovate L1 technically sufficient for BBO-based OFI but latency degrades signal.

---

### Rank 6: Footprint Delta Divergence

**Data:** L1 trade prints classified by aggressor side. Constructs footprint: bid_volume and ask_volume at each price level within each candle.

**Mechanism:** CVD divergence applied within individual bars. Detects exhaustion via stacked imbalances (3+ consecutive price levels with 3:1+ ratio), absorption (800–1500+ contracts at single level without breakthrough), and per-bar delta divergence from price.

**Entry/exit:** On 512-tick chart, identify delta divergence at key level with footprint confirmation (stacked imbalances or absorption). Enter on break below/above divergent bar. Stop 4–6 ticks; target 1.5–2× risk.

**Proximity benefit:** Low. Footprint uses aggregated bar data; signals develop over seconds to minutes.

**Realistic edge:** 55–65% win rate with proper level filtering. Best combined with CVD and level analysis.

**Data feed:** Rithmic recommended for proper trade classification. Tradovate/CQG CVD discrepancies make it unreliable for automated delta calculations.

---

### Rank 7: DOM Fade — Detecting and Fading Pulled Walls

**Data:** L2 depth (5–10 levels) for abnormally large resting orders. Rate of order additions/cancellations at each level.

**Mechanism:** Large visible limit orders (>2 SD above normal depth, typically >1,000–3,000 contracts on ES) often get pulled before execution. Fade strategy trades against apparent direction of wall, anticipating removal. Genuine orders: persist for minutes, increase in size as price approaches, get traded into. Fake orders: disappear as price approaches, appeared suddenly, high cancellation-to-fill ratios.

**Entry/exit:** Identify abnormally large order. Wait for price to approach within 2–3 ticks. If wall starts getting pulled (size shrinks >30% in <2 seconds), enter in direction of approaching price. Target 2–4 ticks; stop 1–2 ticks beyond wall level. Time stop 15–30 seconds.

**Proximity benefit:** Low. Signal timeframe (10–60 seconds) forgiving. Data quality matters more than raw speed.

**Realistic edge:** Low-to-moderate. Per Jigsaw Trading, not viable as standalone 1-rule system. Must combine with market context, time of day, and correlated instrument confirmation.

**Data feed:** Tradovate marginally sufficient for detecting large stacks. Rithmic recommended for faster pull detection.

---

### Rank 8: Tape Reading — Time & Sales Velocity Analysis

**Data:** L1 trade prints: timestamp, price, size, aggressor side. Measures trade velocity (trades/second), large trade clustering (≥50 contracts within 15 seconds), directional ratio (aggressive_buy_volume / total_volume).

**Mechanism:** Monitors raw trade stream for acceleration events. When trade velocity exceeds 2× baseline AND ≥65% of volume is directional AND 2+ large prints cluster in same direction within 15 seconds → enter in direction of flow.

**Entry/exit:** Enter market when velocity + directional clustering + large prints converge. Stop 4–6 ticks; trail stop behind each 2-tick advance if tape speed maintains. Time exit 60–90 seconds if no movement.

**Proximity benefit:** Moderate. Seeing trades quickly matters for velocity spikes, but signal develops over 10–30 seconds.

**Realistic edge:** Marginal as standalone. HFT firms exploit these signals faster. Best as **confirmation layer** for entries triggered by other signals, contributing ~0.5–1 tick average edge.

**Data feed:** Rithmic recommended. Tradovate tick data has higher latency and potential aggregation.

---

### Rank 9: Spoofing Detection — Fading Manipulative Walls

**Data:** MBO data for tracking individual order lifecycles (add/modify/cancel with timestamps). Behavioral metrics: quoting activity ratio, cancellation rate, execution-to-cancellation ratio, order lifespan distribution.

**Mechanism:** Spoofed orders show characteristic patterns: lifespan <3 seconds, concentrated on one side, high cancellation-to-fill ratio, cyclical place-cancel behavior. Trade opposite the spoofed wall — expect price reversal when wall is pulled.

**Realistic edge:** Moderate but unreliable. Post-Dodd-Frank enforcement has reduced spoofing frequency. ML models (Random Forest + LSTM) produce probability scores but false positive rates are high. Better as **filter** (avoid trading with spoofed orders) than primary signal.

**Implementation:** Very high complexity — most complex strategy here. Full order lifecycle tracking, ML model training on labeled spoofing examples, feature engineering for sliding-window behavioral metrics, continuous retraining.

**Data feed:** **Rithmic MBO required.** Tradovate completely insufficient.

---

### Rank 10: Queue Position Market Making

**Data:** MBO data for precise queue position tracking via OrderID/PriorityID. CME FIFO matching = earliest limit order fills first.

**Why it ranks last:** **Requires colocation.** Queue position value in ES is on the same order of magnitude as bid-ask spread (Moallemi & Yuan, 2017). Competing for front-of-queue requires sub-100μs latency — colocation, not 5-mile proximity. At ~2ms, you will consistently lose queue races to colocated HFTs.

**Salvageable element:** Queue *exhaustion detection* (remaining quantity at best bid drops below 50% of rolling average AND aggressive flow continues) can signal imminent level breaks. Viable at our latency as a supplementary signal.

---

## Infrastructure: Tradovate → Rithmic Migration

Tradovate's data feed is insufficient for 7 of the 10 strategies above.

| Limitation | Impact |
|-----------|--------|
| CQG intermediary adds latency | Slower signal detection, worse fills |
| Only 10 depth levels for MES | Inadequate for multi-level OFI and depth analysis |
| No MBO (Market-by-Order) data | Cannot detect icebergs, track order lifecycles, or detect spoofing |
| Documented CVD discrepancies | Delta calculations may be incorrect vs. direct CME feed |
| WebSocket reliability issues | Connection drops (1005/1006 errors) during critical moments |

**Rithmic recommended replacement:** Full-depth unfiltered L2, MBO data with individual order tracking, microsecond timestamps, direct market access, Python asyncio support via `async_rithmic` PyPI package. Cost: ~$50–150/month total including data fees.

**Databento ($199/month Standard):** Ideal complement for backtesting — direct CME MDP 3.0 feed with 42μs cross-connect latency, full MBO/MBP schemas, 15+ years historical, Python client. Data-only (no execution); pair with Rithmic for order routing.

**Optimal production stack:** Rithmic (live data + execution via `async_rithmic`) + Databento (historical research + model training) + Aurora-area VPS (~$60–150/month).

---

## Composite Strategy: The Target Architecture

The strongest approach is a **layered signal system** combining top-ranked components:

1. **Primary signal:** CVD divergence at key pre-identified levels (prior day H/L, VWAP, value area boundaries)
2. **Confirmation layer 1:** L2 book imbalance shifting against price direction (imbalance ≥ |0.3|)
3. **Confirmation layer 2:** Footprint internals showing stacked imbalances or absorption at the divergent bar
4. **Execution timing:** OFI direction and tape velocity for precise entry within signal window
5. **Risk filter:** Regime detector to avoid fading trend days; news calendar filter for high-impact events

**Expected performance:** 2–4 high-quality signals per RTH session, 55–65% win rate, 1.5:1 to 2:1 R:R, composite Sharpe 1.5–3.0 before costs.

## Academic References

- Cont, Kukanov & Stoikov (2014) — OFI explains ~65% of short-interval price variance
- Cont, Cucuringu & Zhang (2023) — Multi-level OFI via PCA pushes R² above 85%
- Gould & Bonart (2016) — Queue imbalance strongly predicts next mid-price movement
- Kolm, Turiel & Westray (2023) — LSTM on OFI features outperforms raw order book models; effective horizon ≈ two average price changes
- Moallemi & Yuan (2017) — Queue position value comparable to bid-ask spread
- Maven Securities — Microstructure alpha decay ~5.6% annualized in U.S., accelerating ~36bps/year
