# Broker Comparison: MES Futures Scalping Bot

> Last updated: 2026-03-03
> **Status: DECIDED — Tradovate (account opened)**

---

## Executive Summary

**Tradovate** is the chosen broker. Account is active. It offers a modern REST + WebSocket API, making it language-agnostic and straightforward to integrate from Python, Node, or any HTTP-capable stack. On the Lifetime plan its all-in commission drops to $0.09/side ($0.18 round-trip), matching the lowest in the industry, while providing $50 intraday margins and solid L2 data. If sub-millisecond execution ever becomes a bottleneck, the upgrade path would be opening a separate account at EdgeClear (which uses the Rithmic data/execution infrastructure). Rithmic cannot be added to Tradovate — they are entirely separate systems.

---

## Commission Math Table

### Assumptions

| Parameter | Value |
|---|---|
| Profit target | 8 ticks |
| Tick value (MES) | $1.25 |
| Slippage per side | 1 tick = $1.25 |
| Stop loss | 8 ticks (1:1 R:R) |
| Trades per day | 5 |
| Trading days per year | 250 |
| Round trips per year | 1,250 |

### Formulas

```
Gross win          = 8 ticks x $1.25                      = $10.00
Net win            = $10.00 - round_trip_commission - $2.50 (slippage both sides)
Net loss           = $10.00 + round_trip_commission + $2.50
Breakeven win rate = net_loss / (net_win + net_loss)
```

### Results

| Broker + Plan | Per-Side | Round-Trip | Net Win | Net Loss | Breakeven WR | Annual Commission | Notes |
|---|---|---|---|---|---|---|---|
| **AMP Futures** (CQG) | $0.42 | $0.84 | $6.66 | $13.34 | **66.7%** | $1,050 | All-in rate |
| **NinjaTrader** Free | $0.39 | $0.78 | $6.72 | $13.28 | **66.4%** | $975 | All-in rate |
| **NinjaTrader** Monthly | $0.29 | $0.58 | $6.92 | $13.08 | **65.4%** | $725 + $1,188 sub = **$1,913** | $99/mo subscription |
| **NinjaTrader** Lifetime | $0.09 | $0.18 | $7.32 | $12.68 | **63.4%** | $225 + $1,499 Y1 | One-time $1,499 license |
| **Tradovate** Free | $0.35 | $0.70 | $6.80 | $13.20 | **66.0%** | $875 | All-in rate [VERIFY] |
| **Tradovate** Lifetime | $0.09 | $0.18 | $7.32 | $12.68 | **63.4%** | $225 + lifetime fee Y1 | Lifetime fee [VERIFY] |
| **EdgeClear** Standard | $0.22 | $0.44 | $7.06 | $12.94 | **64.7%** | $550 | Broker+clearing only; exchange & NFA fees extra [VERIFY] |
| **EdgeClear** Rithmic API | $0.10 | $0.20 | $7.30 | $12.70 | **63.5%** | $250 + $240 data fee = **$490** | $20/mo Rithmic data; exchange fees may be extra [VERIFY] |

### Key Takeaways

- **Breakeven win rate ranges from 63.4% to 66.7%** across all options. The ~3% spread means commission choice alone shifts your edge requirement.
- **NinjaTrader Lifetime pays for itself in ~2.0 years** vs the Free plan at 5 trades/day ($1,499 / ($975 - $225) per year).
- **NinjaTrader Monthly plan is a trap** at 5 trades/day: you'd need 24+ trades/day to beat the Free plan's total cost.
- **EdgeClear rates need verification**: the stated $0.22/side is broker+clearing only. Add CME exchange fee (~$0.20-$0.25) + NFA ($0.02) for true all-in.

---

## Broker Profiles

### AMP Futures

| Attribute | Details |
|---|---|
| **Platform** | Multi-platform: NinjaTrader, Sierra Chart, MultiCharts, Bookmap, TradingView, 20+ others |
| **API type** | Via CQG or Rithmic (depends on data feed choice). No native REST API from AMP itself. |
| **Data feed** | CQG (default) or Rithmic. L1 + L2 (DOM) available. L3 not applicable for futures. |
| **MES intraday margin** | **$40** (lowest in this comparison) |
| **Overnight margin** | ~$2,465 (CME maintenance) |
| **Minimum deposit** | $100 |
| **Commission (all-in)** | $0.42/side with CQG routing |

**Pros**
- Lowest intraday margin ($40 MES) maximizes capital efficiency
- 20+ platform choices; bring your own front-end
- No platform lock-in; easy to switch front-ends
- Established clearing firm with long track record

**Cons**
- No native REST/WebSocket API; you must go through CQG or Rithmic APIs
- CQG data feed adds $0.10/side to cost
- Platform fees vary widely; some add $0.10-$0.25/side on top
- Commission is middle-of-pack once all fees are included

---

### NinjaTrader (Brokerage)

| Attribute | Details |
|---|---|
| **Platform** | NinjaTrader 8 (desktop, Windows only) |
| **API type** | NinjaScript (C# .NET). Strategies compile and run inside the NinjaTrader process. |
| **Data feed** | Proprietary (Continuum/CQG backend). L1 + L2 (DOM). |
| **MES intraday margin** | **$50** |
| **Overnight margin** | ~$1,320 |
| **Minimum deposit** | $50 (effectively; matches margin) |
| **Commission (all-in)** | $0.39/side (Free) / $0.29 (Monthly) / $0.09 (Lifetime) |

**Pros**
- Rock-bottom $0.09/side all-in on Lifetime plan
- Mature backtesting and strategy framework (NinjaScript)
- Built-in market replay for replay-based testing
- Now owns Tradovate; institutional backing

**Cons**
- **C# only** for strategy development; no Python, no REST API
- Windows-only platform; cannot run on Linux servers or containers
- Locked into NinjaTrader ecosystem
- NinjaScript strategies run in-process; crash the platform = crash your bot

---

### Tradovate

| Attribute | Details |
|---|---|
| **Platform** | Web-based + desktop app (Windows/Mac). Also accessible via NinjaTrader 8. |
| **API type** | **REST + WebSocket** (language-agnostic). OpenAPI/Swagger docs. Demo and live endpoints. |
| **Data feed** | Proprietary. L1 + L2 (DOM) via WebSocket streaming. |
| **MES intraday margin** | **$50** |
| **Overnight margin** | ~$1,320 |
| **Minimum deposit** | $0 (no minimum) [VERIFY] |
| **Commission (all-in)** | $0.35/side (Free) / $0.25 (Monthly) / $0.09 (Lifetime) [VERIFY exact tiers] |

**API Details for Bot Development**
```
Auth:     POST https://live.tradovateapi.com/v1/auth/accesstokenrequest
Orders:   POST /v1/order/placeorder
Stream:   wss://live.tradovateapi.com/v1/websocket
Demo:     https://demo.tradovateapi.com/v1/...
```
- Bearer token auth
- Real-time fills, DOM, quotes via WebSocket
- GitHub examples in C# and JavaScript
- Community forum with API developer section

**Pros**
- **Best API for custom bot development**: REST + WebSocket, any language
- Well-documented with OpenAPI specs and demo environment
- Web-based platform; monitor from anywhere
- Same parent company as NinjaTrader; shared clearing infrastructure
- $0.09/side all-in on Lifetime plan

**Cons**
- Lifetime plan pricing unclear / may have changed post-NinjaTrader acquisition [VERIFY]
- API documentation can lag behind actual behavior (check community forum)
- WebSocket reconnection logic needs careful implementation
- Less mature order-flow tooling vs Sierra Chart or Bookmap

---

### EdgeClear

| Attribute | Details |
|---|---|
| **Platform** | Sierra Chart, EdgeProX, Bookmap, TradingView, Jigsaw |
| **API type** | **Rithmic R\|API+** (C++) or **R\|Protocol API** (WebSocket + Google Protocol Buffers, language-agnostic) |
| **Data feed** | **Rithmic** (institutional-grade, lowest latency). L1 + L2 (full DOM). |
| **MES intraday margin** | **$50** |
| **Overnight margin** | ~$2,465 |
| **Minimum deposit** | $1,500 (micros only) / $5,000 (all contracts) |
| **Commission** | $0.22/side broker+clearing (exchange & NFA extra) [VERIFY all-in total] |

**Rithmic API Details**
- **R\|API+**: Native C++ library. Highest performance. Requires Rithmic SDK license.
- **R\|Protocol API**: WebSocket + protobuf. Language/OS independent. Good for Python/Node bots.
- **Colocation available**: Co-locate near CME matching engine for minimum latency.
- **Data fee**: $20/month for Rithmic API access + $0.10/contract transaction fee.

**Pros**
- **Best data quality**: Rithmic feed is the gold standard for futures
- Colocation option for ultra-low-latency execution
- Protocol Buffer API is modern and language-agnostic
- Great platform ecosystem (Sierra Chart, Bookmap, Jigsaw) for manual analysis
- Competitive broker fee ($0.22/side before exchange)

**Cons**
- **Highest minimum deposit** ($1,500 for micros)
- All-in cost unclear: exchange + NFA fees are extra [VERIFY total]
- Rithmic API has a steeper learning curve (protobuf, C++ SDK)
- Must request API access through EdgeClear; not self-service
- $20/month data fee adds fixed overhead

---

## Go/No-Go Decision

### Chosen Broker: Tradovate (Free plan to start, upgrade to Lifetime later)

**Reasoning:**

1. **API-first**: Tradovate is the only broker in this comparison with a native REST + WebSocket API. For building a custom scalping bot in Python or Node, this is the decisive factor. No C# requirement, no native SDK dependency, no platform lock-in.

2. **Commission trajectory**: Start on the Free plan ($0.35/side, $875/year). Once the bot is profitable and validated, upgrade to Lifetime ($0.09/side, $225/year) for a ~$650/year savings. The Lifetime plan pays for itself quickly.

3. **Demo environment**: `demo.tradovateapi.com` provides a full sandbox to develop and test the bot without risking capital or incurring fees.

4. **Adequate margins**: $50 MES intraday margin is standard. AMP's $40 is $10 lower but not a dealbreaker.

5. **Upgrade path**: If the bot proves profitable and latency becomes the bottleneck, migrate to EdgeClear + Rithmic Protocol API without changing the bot's architecture (both use WebSocket).

### Rejected Alternatives

| Broker | Why Not |
|---|---|
| AMP Futures | No native API; would require CQG or Rithmic SDK, adding complexity and cost. Good fallback if Tradovate API proves unreliable. |
| NinjaTrader | C# / NinjaScript only. Cannot deploy on Linux. Strategy runs inside platform process. Wrong architecture for a standalone bot. |
| EdgeClear | Higher minimum deposit ($1,500), unclear all-in costs, Rithmic API learning curve. Better suited as a Phase 2 upgrade once the bot is proven. |

### Next Steps to Open Account

1. **Go to** [tradovate.com/pricing](https://www.tradovate.com/pricing/) and select the Free plan
2. **Complete application** (SSN, ID verification, futures risk disclosure)
3. **Fund account** with minimum required amount [VERIFY minimum - may be $0]
4. **Enable API access** via account settings or by contacting support
5. **Get demo credentials** at `demo.tradovateapi.com` to start bot development immediately
6. **Subscribe to CME market data** for MES (required for live; demo data is free)

---

## What To Ask Your Broker

Send these exact questions to each broker's support before funding. Copy-paste and fill in.

### For Tradovate

1. "What is the **exact all-in per-side commission** for 1 MES (Micro E-mini S&P 500) contract on the Free plan? Please itemize: broker commission, exchange fee, NFA fee, clearing fee, and any other per-trade charges."
2. "What is the current **Lifetime plan price**, and what is the all-in per-side MES rate on that plan?"
3. "Is there a **minimum account balance** required to open an account and trade MES?"
4. "What is the current **MES intraday (day trade) margin**? At what time must positions be closed or covered to overnight margin?"
5. "Is **REST API and WebSocket access** included with all plans, or does it require a separate subscription or approval?"
6. "What **CME market data fees** apply for streaming MES quotes via the API? Is there a non-professional rate?"
7. "Are there any **monthly platform fees, inactivity fees, or minimum activity requirements**?"
8. "What is your **order routing latency** from API submission to exchange acknowledgment?"
9. "Do you support **bracket orders (OCO)** and **trailing stops** via the API?"
10. "What happens if my bot places an order **within 15 minutes of session close** while under day-trade margin? Will the order be rejected, or will margin be auto-upgraded?"

### For EdgeClear (backup / Phase 2)

1. "What is the **exact all-in per-side commission** for 1 MES contract? Please itemize broker fee, exchange fee, NFA fee, and clearing fee separately."
2. "For the **Rithmic R|Protocol API** (WebSocket + protobuf): what is the monthly data fee, per-contract transaction fee, and any setup costs?"
3. "What is the **MES intraday margin**, and does it differ between Rithmic and CQG routing?"
4. "Can I get **Rithmic API demo credentials** before funding a live account?"
5. "What is the process and timeline for **Rithmic API access approval**?"

### For AMP Futures (backup)

1. "What is the **exact all-in per-side cost** for 1 MES contract using CQG routing? Itemize: AMP clearing, CQG routing, exchange, NFA, and any other fees."
2. "Do you offer **Rithmic routing** as an alternative to CQG, and what is the all-in per-side cost with Rithmic?"
3. "What is the current **MES intraday margin**? Is it $40 or has it changed?"
4. "For automated trading via CQG or Rithmic API: are there any additional **API access fees or approval processes**?"

---

## Appendix: Commission Impact on Profitability

To put the commission differences in perspective for an 8-tick MES scalp:

```
                        Worst Case        Best Case
                        (AMP $0.84 RT)    (NT/Tradovate Lifetime $0.18 RT)
                        ──────────────    ──────────────────────────────────
Net win (8 ticks)       $6.66             $7.32
Net loss (8 ticks)      $13.34            $12.68
Breakeven WR            66.7%             63.4%

At 65% actual win rate:
  Per-trade EV          -$0.34            +$0.32
  Annual P&L (1250 RT)  -$425             +$400

At 68% actual win rate:
  Per-trade EV          +$0.26            +$0.92
  Annual P&L (1250 RT)  +$325             +$1,150
```

The difference between the cheapest and most expensive broker is **$825/year in commissions** and **3.3 percentage points of breakeven win rate**. For a marginal strategy operating near breakeven, broker choice can be the difference between profit and loss.
