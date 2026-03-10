#!/usr/bin/env python3
"""Tier 1: Sweep VWAP Band Reversion params over 14 years of 5m bars.

Pre-computes signal bundles once, then iterates param combos re-using them.
No HMM — we want to find params that work on raw market structure.

Params swept:
  - deviation_sd: min VWAP deviation to enter (2.0 - 4.0)
  - slope_max: max |VWAP slope| for entry (0.3 - 1.0)
  - adx_max: max ADX for entry (20 - 40)
  - atr_mult: stop = ATR * multiplier (1.5 - 3.0)
  - time_stop: minutes until expiry (10 - 60)
  - rsi_long_max / rsi_short_min: RSI thresholds (20-40 / 60-80)
"""
import os, sys, time as _time, logging, math
from datetime import date
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Suppress ALL logging during sweep (stdlib + structlog)
logging.disable(logging.CRITICAL)

import structlog
structlog.configure(
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),  # route through stdlib (which is disabled)
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
)

import yaml
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.signals.signal_bundle import SignalEngine
from src.strategies.vwap_band_reversion import VWAPBandReversionStrategy

# ── Param grid ──
PARAM_GRID = {
    "deviation_sd": [2.0, 2.5, 3.0, 3.5, 4.0],
    "slope_max": [0.3, 0.5, 0.75, 1.0],
    "adx_max": [20.0, 25.0, 30.0, 40.0],
    "atr_mult": [1.5, 2.0, 2.5, 3.0],
    "time_stop": [10, 15, 20, 30, 45, 60],
    "rsi_long": [20.0, 30.0, 40.0],
    "rsi_short": [60.0, 70.0, 80.0],
}

# Total combos
total = 1
for v in PARAM_GRID.values():
    total *= len(v)
print(f"Total param combos: {total:,}")


def build_config(dev_sd, slope, adx, atr_m, tstop, rsi_l, rsi_s):
    """Build a strategy config dict from sweep params."""
    return {
        "strategy": {
            "strategy_id": "vwap_band_reversion",
            "max_signals_per_day": 999,
        },
        "signal_configs": {
            "rsi_momentum": {
                "period": 2,
                "long_threshold": rsi_l,
                "short_threshold": rsi_s,
            },
            "adx": {"period": 14, "threshold": adx},
        },
        "filters": [
            {"signal": "session_time", "expr": ">= 585"},
            {"signal": "session_time", "expr": "<= 900"},
            {"signal": "vwap_session", "field": "session_age_bars", "expr": ">= 30"},
            {"signal": "vwap_session", "field": "slope", "expr": f"abs <= {slope}"},
            {"signal": "vwap_session", "field": "deviation_sd", "expr": f"abs >= {dev_sd}"},
            {"signal": "adx", "expr": f"< {adx}"},
            {"signal": "relative_volume", "expr": ">= 0.5"},
        ],
        "exit": {
            "target": {"type": "vwap"},
            "stop": {"type": "atr_multiple", "multiplier": atr_m},
            "time_stop_minutes": tstop,
            "early_exit": [{"type": "vwap_slope", "threshold": 0.3}],
        },
    }


def main():
    # ── Load base YAML for signal config ──
    yaml_path = "config/strategies/vwap_band_reversion.yaml"
    with open(yaml_path) as f:
        base_cfg = yaml.safe_load(f)

    signal_names = [s for s in base_cfg.get("signals", []) if s != "hmm_regime"]
    signal_configs = base_cfg.get("signal_configs", {})
    signal_engine = SignalEngine(signal_names, signal_configs)

    config = BacktestConfig(
        strategies=[],
        start_date=date(2011, 3, 1),
        end_date=date(2025, 2, 28),
        parquet_dir="data/parquet_5m",
        resample_freq="5m",
        signal_engine=signal_engine,
    )

    engine = BacktestEngine()

    # ── Pre-compute bundles once ──
    print("Pre-computing signal bundles for 14 years...")
    t0 = _time.time()
    bundles = engine.precompute_bundles(config)
    print(f"  {len(bundles)} bundles in {_time.time() - t0:.1f}s")

    # ── Sweep ──
    results = []
    combos = list(product(
        PARAM_GRID["deviation_sd"],
        PARAM_GRID["slope_max"],
        PARAM_GRID["adx_max"],
        PARAM_GRID["atr_mult"],
        PARAM_GRID["time_stop"],
        PARAM_GRID["rsi_long"],
        PARAM_GRID["rsi_short"],
    ))

    print(f"\nSweeping {len(combos):,} combos...")
    t0 = _time.time()

    for i, (dev_sd, slope, adx, atr_m, tstop, rsi_l, rsi_s) in enumerate(combos):
        cfg = build_config(dev_sd, slope, adx, atr_m, tstop, rsi_l, rsi_s)
        strat = VWAPBandReversionStrategy(cfg)

        run_config = BacktestConfig(
            strategies=[strat],
            start_date=date(2011, 3, 1),
            end_date=date(2025, 2, 28),
            parquet_dir="data/parquet_5m",
            resample_freq="5m",
            prebuilt_bundles=bundles,
        )

        result = engine.run(run_config)
        m = result.metrics

        if m.total_trades >= 50:  # min trades for statistical significance
            score = m.sharpe_ratio * math.sqrt(m.total_trades) if m.sharpe_ratio > 0 else m.sharpe_ratio
            results.append({
                "deviation_sd": dev_sd,
                "slope_max": slope,
                "adx_max": adx,
                "atr_mult": atr_m,
                "time_stop": tstop,
                "rsi_long": rsi_l,
                "rsi_short": rsi_s,
                "trades": m.total_trades,
                "win_rate": m.win_rate,
                "net_pnl": m.net_pnl,
                "sharpe": m.sharpe_ratio,
                "profit_factor": m.profit_factor,
                "max_dd": m.max_drawdown_pct,
                "score": score,
            })

        if (i + 1) % 500 == 0:
            elapsed = _time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(combos) - i - 1) / rate
            print(f"  [{i+1:,}/{len(combos):,}] {rate:.1f} combos/s, ETA {eta/60:.0f}min, {len(results)} viable")

    elapsed = _time.time() - t0
    print(f"\nDone: {len(combos):,} combos in {elapsed/60:.1f}min ({len(combos)/elapsed:.1f}/s)")
    print(f"Viable results (>=50 trades): {len(results)}")

    if not results:
        print("No viable results found!")
        return

    # Sort by score (Sharpe × √trades)
    results.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n{'='*100}")
    print(f"TOP 20 RESULTS (ranked by Sharpe × √trades)")
    print(f"{'='*100}")
    print(f"{'dev_sd':>6} {'slope':>5} {'adx':>4} {'atr_m':>5} {'tstop':>5} "
          f"{'rsi_l':>5} {'rsi_s':>5} | {'trades':>6} {'WR':>6} {'PnL':>9} "
          f"{'Sharpe':>7} {'PF':>6} {'DD%':>6} {'score':>7}")
    print("-" * 100)
    for r in results[:20]:
        print(f"{r['deviation_sd']:6.1f} {r['slope_max']:5.2f} {r['adx_max']:4.0f} "
              f"{r['atr_mult']:5.1f} {r['time_stop']:5d} {r['rsi_long']:5.0f} "
              f"{r['rsi_short']:5.0f} | {r['trades']:6d} {r['win_rate']:5.1%} "
              f"${r['net_pnl']:8.2f} {r['sharpe']:7.2f} {r['profit_factor']:6.2f} "
              f"{r['max_dd']:5.2f}% {r['score']:7.2f}")

    # Also show top 10 by raw Sharpe (min 100 trades)
    high_n = [r for r in results if r["trades"] >= 100]
    if high_n:
        high_n.sort(key=lambda r: r["sharpe"], reverse=True)
        print(f"\n{'='*100}")
        print(f"TOP 10 BY SHARPE (min 100 trades)")
        print(f"{'='*100}")
        print(f"{'dev_sd':>6} {'slope':>5} {'adx':>4} {'atr_m':>5} {'tstop':>5} "
              f"{'rsi_l':>5} {'rsi_s':>5} | {'trades':>6} {'WR':>6} {'PnL':>9} "
              f"{'Sharpe':>7} {'PF':>6} {'DD%':>6}")
        print("-" * 100)
        for r in high_n[:10]:
            print(f"{r['deviation_sd']:6.1f} {r['slope_max']:5.2f} {r['adx_max']:4.0f} "
                  f"{r['atr_mult']:5.1f} {r['time_stop']:5d} {r['rsi_long']:5.0f} "
                  f"{r['rsi_short']:5.0f} | {r['trades']:6d} {r['win_rate']:5.1%} "
                  f"${r['net_pnl']:8.2f} {r['sharpe']:7.2f} {r['profit_factor']:6.2f} "
                  f"{r['max_dd']:5.2f}%")


if __name__ == "__main__":
    main()
