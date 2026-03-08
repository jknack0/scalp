# Phase 4 Validation Report

Generated: 2026-03-06 20:33

## Executive Summary

- **Strategies evaluated**: 5
- **Survivors (PROCEED)**: 0
- **Retired**: 5

**No strategies survived validation.** Review research hypotheses below before proceeding.

## Validation Gate Results

| Strategy | PBO | DSR | WFA Eff | Stability | Decision |
|----------|-----|-----|---------|-----------|----------|
| orb | FAIL (0.71) | FAIL (0.09) | PASS (1.54) | PASS (0.81) | **RETIRE** |
| vwap_reversion | FAIL (0.50) | PASS (1.00) | FAIL (-0.30) | FAIL (0.53) | **RETIRE** |
| cvd_divergence | FAIL (0.50) | FAIL (0.00) | PASS (0.87) | PASS (1.00) | **RETIRE** |
| vol_regime_switcher | FAIL (0.79) | FAIL (0.00) | FAIL (-55.24) | PASS (1.00) | **RETIRE** |
| obi | PASS (0.00) | FAIL (nan) | FAIL (0.00) | PASS (1.00) | **RETIRE** |

## Per-Strategy Assessment

### orb — RETIRE

- OOS Sharpe: 3.557
- OOS Trades: 117
- IS/OOS Correlation: -0.29

**Failure modes:**
- PBO 0.71 >= 0.10 — high probability of backtest overfitting
- DSR 0.09 < 0.95 — Sharpe not significant after multiple-testing correction

**Research hypothesis:** Reduce parameter count or simplify entry conditions to lower overfitting risk. Increase sample size or reduce strategy variants tested to improve deflated Sharpe.

### vwap_reversion — RETIRE

- OOS Sharpe: -1.159
- OOS Trades: 57
- IS/OOS Correlation: 0.04

**Failure modes:**
- PBO 0.50 >= 0.10 — high probability of backtest overfitting
- WFA efficiency -0.30 < 0.50 — poor out-of-sample transfer
- Param stability 0.53 < 0.65 — excessive parameter drift

**Research hypothesis:** Reduce parameter count or simplify entry conditions to lower overfitting risk. Narrow parameter grid or switch to adaptive parameters for better OOS transfer. Anchor key parameters or use regime-conditional defaults to reduce drift.

### cvd_divergence — RETIRE

- OOS Sharpe: -5.364
- OOS Trades: 181
- IS/OOS Correlation: -0.06

**Failure modes:**
- PBO 0.50 >= 0.10 — high probability of backtest overfitting
- DSR 0.00 < 0.95 — Sharpe not significant after multiple-testing correction

**Research hypothesis:** Reduce parameter count or simplify entry conditions to lower overfitting risk. Increase sample size or reduce strategy variants tested to improve deflated Sharpe.

### vol_regime_switcher — RETIRE

- OOS Sharpe: 12.501
- OOS Trades: 2884
- IS/OOS Correlation: -0.05

**Failure modes:**
- PBO 0.79 >= 0.10 — high probability of backtest overfitting
- DSR 0.00 < 0.95 — Sharpe not significant after multiple-testing correction
- WFA efficiency -55.24 < 0.50 — poor out-of-sample transfer

**Research hypothesis:** Reduce parameter count or simplify entry conditions to lower overfitting risk. Increase sample size or reduce strategy variants tested to improve deflated Sharpe. Narrow parameter grid or switch to adaptive parameters for better OOS transfer.

### obi — RETIRE

- OOS Sharpe: 0.000
- OOS Trades: 0
- IS/OOS Correlation: 0.00

**Failure modes:**
- DSR nan < 0.95 — Sharpe not significant after multiple-testing correction
- WFA efficiency 0.00 < 0.50 — poor out-of-sample transfer

**Research hypothesis:** Increase sample size or reduce strategy variants tested to improve deflated Sharpe. Narrow parameter grid or switch to adaptive parameters for better OOS transfer.

## Phase 5 Entry Criteria

Before proceeding to Phase 5 (Live Paper Trading):

- [ ] All surviving strategies have locked parameters (above)
- [ ] No parameter changes allowed without re-running Phase 4
- [ ] Ensemble weights computed from OOS Sharpe ratios
- [ ] Risk limits configured per strategy
- [ ] Paper trading infrastructure verified
