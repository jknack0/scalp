# Commission Math — MES Scalping

> Phase 1, Task 5 | Last updated: 2026-03-03

---

## Your Numbers (Tradovate Free)

| Parameter | Value |
|---|---|
| Per-side commission | $0.35 (all-in) |
| Round-trip commission | $0.70 |
| Assumed slippage | 1 tick/side = $2.50 RT |
| **Total round-trip cost** | **$3.20** |
| MES tick value | $1.25 |
| MES point value | $5.00 |

## Breakeven Win Rates

At your current commission rate, the win rate you need **just to break even**:

| Target | Stop | Net Win | Net Loss | Breakeven WR |
|---|---|---|---|---|
| 4 ticks ($5.00) | 4 ticks | $1.80 | $8.20 | **82.0%** |
| 8 ticks ($10.00) | 4 ticks | $6.80 | $8.20 | **54.7%** |
| 8 ticks ($10.00) | 8 ticks | $6.80 | $13.20 | **66.0%** |
| 10 ticks ($12.50) | 8 ticks | $9.30 | $13.20 | **58.7%** |
| 12 ticks ($15.00) | 8 ticks | $11.80 | $13.20 | **52.8%** |
| 16 ticks ($20.00) | 8 ticks | $16.80 | $13.20 | **44.0%** |

## Key Findings

1. **Minimum viable target: 8 ticks.** Below 8 ticks, the required win rate exceeds 70%, which is unsustainable long-term.

2. **At 8t target / 8t stop (1:1 R:R): you need 66% win rate.** This is the baseline — every strategy must beat this in backtesting.

3. **At 12t target / 8t stop (1.5:1 R:R): you need 53% win rate.** Much more forgiving. Strategies with asymmetric R:R are strongly preferred.

4. **Slippage is the hidden cost.** At 2-tick slippage (volatile conditions), the breakeven WR at 8t/8t jumps to ~72%. Always assume 1-tick in backtests.

## Annual Commission Budget

| Trades/Day | Tradovate Free | Tradovate Lifetime | EdgeClear + Rithmic |
|---|---|---|---|
| 5 | $875 | $225 | $400 |
| 10 | $1,750 | $450 | $800 |
| 15 | $2,625 | $675 | $1,200 |

At 5 trades/day, upgrading to Lifetime saves **$650/year**.

## Strategy Filters

Use these as hard filters for Phase 2-3 strategy research:

- **REJECT** any strategy with profit target < 8 ticks
- **REJECT** any strategy with backtested win rate < 68% at 1:1 R:R (2% safety margin)
- **PREFER** strategies with 1.5:1+ R:R (requires only ~55% win rate)
- **ALWAYS** backtest with 1-tick slippage per side

## Upgrade Path

| When | Action | Impact |
|---|---|---|
| Bot is consistently profitable | Upgrade to Tradovate Lifetime | -2.6% breakeven WR, -$650/yr costs |
| Need sub-ms execution | Open EdgeClear + Rithmic account | Direct CME access, better fills |

---

## Source Code

- Model: `src/analysis/commission_model.py`
- Notebook: `notebooks/phase1-commission-analysis.ipynb`
- Broker comparison: `docs/phase1/broker-comparison.md`
