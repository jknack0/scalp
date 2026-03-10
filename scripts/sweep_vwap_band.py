"""Quick 3-month parameter sweep for VWAP Band Reversion strategy."""
import os, sys, logging, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from itertools import product
from pathlib import Path

import structlog

# Suppress all logging in workers
logging.disable(logging.CRITICAL)
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.signals.signal_bundle import SignalEngine
from src.strategies.vwap_band_reversion import VWAPBandReversionStrategy

# ── Sweep grid ──
DEVIATION_SD = [1.5, 2.0, 2.5, 3.0]
RSI_PAIRS = [(20, 80), (30, 70), (40, 60), (50, 50)]  # (long_max, short_min); 50/50 = disabled
ADX_MAX = [25, 35, 100]  # 100 = effectively disabled
STOP_ATR_MULT = [1.0, 1.5, 2.0, 3.0]
TIME_STOP = [15, 30, 60]
SLOPE_MAX = [0.5, 1.0]
RVOL_MIN = [0.5, 1.0]

START = date(2024, 3, 1)
END = date(2025, 2, 28)
WORKERS = 10

SIGNAL_NAMES = ["vwap_session", "rsi_momentum", "adx", "atr", "relative_volume", "session_time"]


def _run_one(dev_sd, rsi_long, rsi_short, adx_max, stop_mult, time_stop, slope_max, rvol_min):
    """Run a single config."""
    logging.disable(logging.CRITICAL)

    filters = [
        {"signal": "session_time", "expr": ">= 585"},
        {"signal": "session_time", "expr": "<= 900"},
        {"signal": "vwap_session", "field": "session_age_bars", "expr": ">= 30"},
        {"signal": "vwap_session", "field": "slope", "expr": f"abs <= {slope_max}"},
        {"signal": "vwap_session", "field": "deviation_sd", "expr": f"abs >= {dev_sd}"},
        {"signal": "adx", "expr": f"< {adx_max}"},
        {"signal": "relative_volume", "expr": f">= {rvol_min}"},
    ]

    cfg = {
        "strategy": {"strategy_id": "vwap_band_reversion", "max_signals_per_day": 3},
        "signals": SIGNAL_NAMES,
        "signal_configs": {
            "rsi_momentum": {"period": 2, "long_threshold": rsi_long, "short_threshold": rsi_short},
            "adx": {"period": 14, "threshold": adx_max},
        },
        "filters": filters,
        "exit": {
            "target": {"type": "vwap"},
            "stop": {"type": "atr_multiple", "multiplier": stop_mult},
            "time_stop_minutes": time_stop,
        },
    }

    strategy = VWAPBandReversionStrategy(cfg)
    signal_engine = SignalEngine(SIGNAL_NAMES, cfg.get("signal_configs", {}))

    config = BacktestConfig(
        strategies=[strategy],
        start_date=START,
        end_date=END,
        parquet_dir="data/parquet_5m",
        resample_freq="5m",
        signal_engine=signal_engine,
    )

    engine = BacktestEngine()
    r = engine.run(config)
    m = r.metrics

    return {
        "dev_sd": dev_sd, "rsi": f"{rsi_long}/{rsi_short}", "adx_max": adx_max,
        "stop_mult": stop_mult, "time_stop": time_stop,
        "slope_max": slope_max, "rvol_min": rvol_min,
        "trades": m.total_trades, "wr": m.win_rate, "pnl": m.net_pnl,
        "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
        "pf": m.profit_factor, "dd": m.max_drawdown_pct,
        "avg_win": m.avg_win, "avg_loss": m.avg_loss,
    }


def main():
    combos = list(product(
        DEVIATION_SD, RSI_PAIRS, ADX_MAX, STOP_ATR_MULT, TIME_STOP, SLOPE_MAX, RVOL_MIN
    ))
    total = len(combos)
    print(f"\nVWAP Band Reversion Sweep: {total} configs, {WORKERS} workers")
    print(f"Period: {START} to {END}")
    print(f"Grid: dev_sd={DEVIATION_SD} rsi={RSI_PAIRS} adx={ADX_MAX}")
    print(f"       stop_mult={STOP_ATR_MULT} time={TIME_STOP} slope={SLOPE_MAX} rvol={RVOL_MIN}\n")

    results = []
    done = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}
        for combo in combos:
            dev_sd, (rsi_l, rsi_s), adx, stop, ts, slope, rvol = combo
            f = pool.submit(_run_one, dev_sd, rsi_l, rsi_s, adx, stop, ts, slope, rvol)
            futures[f] = combo

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                continue
            results.append(result)
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  [{done:4d}/{total}] ...", flush=True)

    # Score: sharpe * sqrt(trades), min 5 trades for 3 months
    MIN_TRADES = 5
    for r in results:
        if r["trades"] >= MIN_TRADES and r["sharpe"] > 0:
            r["score"] = r["sharpe"] * math.sqrt(r["trades"])
        else:
            r["score"] = -999.0

    results.sort(key=lambda x: x["score"], reverse=True)

    qualified = [r for r in results if r["score"] > -999]
    profitable = [r for r in results if r["pnl"] > 0]

    print(f"\n{'='*100}")
    print(f"TOP 20 CONFIGS (score = Sharpe × sqrt(trades), min {MIN_TRADES} trades)")
    print(f"{'='*100}")
    for i, r in enumerate(results[:20]):
        s = "+" if r["pnl"] > 0 else "-"
        print(f"  {i+1:2d}. {s} dev={r['dev_sd']:.1f} rsi={r['rsi']:>5s} adx<{r['adx_max']:>3} "
              f"stop={r['stop_mult']:.1f}×ATR t={r['time_stop']:2d}m "
              f"slope<{r['slope_max']:.1f} rvol>{r['rvol_min']:.1f}  "
              f"trades={r['trades']:3d} WR={r['wr']:.0%} "
              f"Sharpe={r['sharpe']:>6.2f} PF={r['pf']:.2f} PnL=${r['pnl']:>8.2f}")

    print(f"\n  {len(profitable)}/{len(results)} profitable, "
          f"{len(qualified)}/{len(results)} qualified")

    # Also show best by raw PnL
    by_pnl = sorted(results, key=lambda x: x["pnl"], reverse=True)
    print(f"\n{'='*100}")
    print(f"TOP 10 BY RAW PnL")
    print(f"{'='*100}")
    for i, r in enumerate(by_pnl[:10]):
        print(f"  {i+1:2d}. dev={r['dev_sd']:.1f} rsi={r['rsi']:>5s} adx<{r['adx_max']:>3} "
              f"stop={r['stop_mult']:.1f}×ATR t={r['time_stop']:2d}m "
              f"slope<{r['slope_max']:.1f} rvol>{r['rvol_min']:.1f}  "
              f"trades={r['trades']:3d} WR={r['wr']:.0%} "
              f"Sharpe={r['sharpe']:>6.2f} PnL=${r['pnl']:>8.2f}")

    # Save
    out_dir = Path("results/vwap_band")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sweep_{datetime.now().strftime('%Y-%m-%d_%H%M')}.json"
    output = {
        "period": {"start": str(START), "end": str(END)},
        "total_configs": len(results),
        "qualified": len(qualified),
        "profitable": len(profitable),
        "top_20": results[:20],
        "top_10_pnl": by_pnl[:10],
        "all_results": results,
    }
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
