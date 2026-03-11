#!/usr/bin/env python3
"""Joint entry + exit sweep for VWAP Band Reversion over 10yr.

Sweeps BOTH entry filter thresholds and exit params simultaneously.
Signals are pre-computed once; only filter thresholds + exit configs vary per combo.

Entry params swept: deviation_sd, slope, adx, relative_volume
Exit params swept: stop_atr, target_sd, time_bars, trailing_stop (on/off)
"""
import os, sys, time as _time, logging, math
from collections import Counter
from datetime import date, datetime, time, timedelta
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Suppress ALL logging
logging.disable(logging.CRITICAL)
import structlog
structlog.configure(
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)

import polars as pl
import yaml

from src.backtesting.engine import BacktestConfig, BacktestEngine, SimulatedOMS, MetricsCalculator
from src.backtesting.slippage import VolatilitySlippageModel
from src.analysis.commission_model import tradovate_free
from src.core.events import BarEvent
from src.exits.exit_engine import ExitEngine
from src.signals.signal_bundle import SignalEngine, SignalBundle, EMPTY_BUNDLE
from src.strategies.vwap_band_reversion import VWAPBandReversionStrategy


def p(msg):
    print(msg, flush=True)


# ── COARSE GRID ──
# Entry filter thresholds
GRID_DEV_SD   = [2.0, 3.0, 4.0]       # abs deviation_sd >= X
GRID_SLOPE    = [0.2, 0.5]             # abs slope <= X
GRID_ADX      = [20.0, 30.0]           # adx < X
GRID_RVOL     = [0.3, 1.0]             # relative_volume >= X

# Exit params
GRID_STOP_ATR = [2.0, 4.0]             # hard stop ATR multiple
GRID_TGT_SD   = [0.3, 1.0]            # reversion target SD band
GRID_TIME     = [15, 45]               # time stop bars
GRID_TRAIL    = [(0, 0), (1.5, 4)]     # (atr_mult, activate_ticks) — 0=off

# Fixed params
SLOPE_EXIT_THRESH = 0.5                 # adverse slope exit threshold
SESSION_AGE_BARS = 30                   # VWAP age requirement
SESSION_START_MIN = 585                 # 9:45 AM
SESSION_END_MIN = 900                   # 3:00 PM


def build_filters(dev_sd, slope, adx, rvol):
    """Build filter list for a given entry config."""
    return [
        {"signal": "session_time", "expr": f">= {SESSION_START_MIN}"},
        {"signal": "session_time", "expr": f"<= {SESSION_END_MIN}"},
        {"signal": "vwap_session", "field": "session_age_bars", "expr": f">= {SESSION_AGE_BARS}"},
        {"signal": "vwap_session", "field": "slope", "expr": f"abs <= {slope}"},
        {"signal": "vwap_session", "field": "deviation_sd", "expr": f"abs >= {dev_sd}"},
        {"signal": "adx", "expr": f"< {adx}"},
        {"signal": "relative_volume", "expr": f">= {rvol}"},
        {"signal": "hmm_regime", "expr": "passes"},
    ]


def build_exits(stop_atr, target_sd, time_bars, trail_atr, trail_act):
    """Build exits list for ExitEngine."""
    return [
        {"type": "static_stop", "enabled": True, "atr_multiple": stop_atr},
        {"type": "vwap_reversion_target", "enabled": True,
         "target_sd_band": target_sd, "vwap_signal": "vwap_session",
         "deviation_field": "deviation_sd"},
        {"type": "time_stop", "enabled": True, "max_bars": time_bars},
        {"type": "adverse_signal_exit", "enabled": True,
         "signal": "vwap_session", "field": "slope",
         "long_threshold": -SLOPE_EXIT_THRESH, "short_threshold": SLOPE_EXIT_THRESH},
        {"type": "trailing_stop", "enabled": trail_atr > 0,
         "atr_multiple": trail_atr if trail_atr > 0 else 1.0,
         "activate_after_ticks": trail_act if trail_act > 0 else 99},
        {"type": "regime_exit", "enabled": True,
         "hmm_signal": "hmm_regime",
         "hostile_regimes_long": [1], "hostile_regimes_short": [1],
         "min_bars_before_active": 2},
    ]


def build_strategy_config(filters, exits):
    """Build full strategy config dict."""
    return {
        "strategy": {"strategy_id": "vwap_band_reversion", "max_signals_per_day": 999},
        "signal_configs": {"adx": {"period": 14, "threshold": 25.0}},
        "filters": filters,
        "exits": exits,
        "exit": {
            "target": {"type": "vwap"},
            "stop": {"type": "atr_multiple", "multiplier": 3.0},
            "time_stop_minutes": 120,
        },
    }


def _sub_time(t: time, seconds: int = 0) -> time:
    dt = datetime.combine(date.today(), t) - timedelta(seconds=seconds)
    return dt.time()


def fast_run(bars, bundles, dates, times, et_times,
             session_close_time, strategy, exit_engine):
    """Backtest loop using ExitEngine for all exit decisions."""
    oms = SimulatedOMS(
        commission_model=tradovate_free(),
        slippage_model=VolatilitySlippageModel(),
        max_position=1,
        exit_engine=exit_engine,
    )

    all_trades = []
    prev_date = None
    n_bars = len(bars)
    exit_reasons = Counter()

    for bar_index in range(n_bars):
        bar = bars[bar_index]
        bar_date = dates[bar_index]
        bar_time = et_times[bar_index]

        if prev_date is None or bar_date != prev_date:
            if prev_date is not None:
                trades = oms.close_all(
                    close_price=bar.close, close_time=bar_time,
                    bar_index=bar_index, bar_date=bar_date,
                    current_atr_ticks=0.0, reason="session_close",
                )
                all_trades.extend(trades)
            strategy.reset()

        prev_date = bar_date
        bundle = bundles[bar_index]

        signal = strategy.on_bar(bar, bundle)
        if signal is not None and oms.open_position_count < 1:
            oms.on_signal(signal, bar_index)

        trades = oms.on_bar(bar, bar_index, bar_time, bar_date, 0.0,
                           bundle=bundle)
        all_trades.extend(trades)

        if times[bar_index] >= session_close_time:
            trades = oms.close_all(
                close_price=bar.close, close_time=bar_time,
                bar_index=bar_index, bar_date=bar_date,
                current_atr_ticks=0.0, reason="session_close",
            )
            all_trades.extend(trades)

    if n_bars > 0:
        trades = oms.close_all(
            close_price=bars[-1].close, close_time=et_times[-1],
            bar_index=n_bars - 1, bar_date=dates[-1],
            current_atr_ticks=0.0, reason="session_close",
        )
        all_trades.extend(trades)

    for t in all_trades:
        exit_reasons[t.exit_reason] += 1

    metrics, _, _ = MetricsCalculator.from_trades(all_trades, 10000.0)
    return metrics, exit_reasons


def main():
    yaml_path = "config/strategies/vwap_band_reversion.yaml"
    with open(yaml_path) as f:
        base_cfg = yaml.safe_load(f)

    signal_names = [s for s in base_cfg.get("signals", []) if s != "hmm_regime"]
    signal_configs = base_cfg.get("signal_configs", {})
    signal_engine = SignalEngine(signal_names, signal_configs)

    config = BacktestConfig(
        strategies=[],
        start_date=date(2015, 3, 1),
        end_date=date(2025, 2, 28),
        parquet_dir="data/parquet_5m",
        resample_freq="5m",
        signal_engine=signal_engine,
    )

    engine = BacktestEngine()

    # ── Pre-compute bundles (fast signals only, no HMM) ──
    p("Pre-computing signal bundles for 10 years...")
    t0 = _time.time()
    bundles = engine.precompute_bundles(config)
    p(f"  {len(bundles)} bundles in {_time.time() - t0:.1f}s")

    # ── Pre-compute HMM regime states ──
    from src.models.hmm_regime import HMMRegimeClassifier, RegimeState, build_feature_matrix
    from src.signals.base import SignalResult

    hmm_cfg = signal_configs.get("hmm_regime", {})
    pass_states_str = hmm_cfg.get("pass_states", [])
    pass_states = [RegimeState[s] for s in pass_states_str]

    p("Pre-building BarEvent array + HMM features...")
    t1 = _time.time()
    bars_df = engine._load_bars(config)
    bars_df = bars_df.with_columns(
        pl.col("timestamp")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("US/Eastern")
            .alias("_et_ts"),
        (pl.col("timestamp").cast(pl.Int64) * 1000).alias("_timestamp_ns"),
    )
    bars_df = bars_df.filter(
        (pl.col("_et_ts").dt.time() >= config.session_start)
        & (pl.col("_et_ts").dt.time() < config.session_end)
    )
    bars_df = bars_df.with_columns(
        pl.col("_et_ts").dt.date().alias("_bar_date"),
        pl.col("_et_ts").dt.time().alias("_bar_time"),
    )

    if hmm_cfg.get("model_path"):
        p("  Loading HMM model and predicting regime states...")
        clf = HMMRegimeClassifier.load(hmm_cfg["model_path"])
        features, timestamps = build_feature_matrix(bars_df)
        states = clf.predict_sequence(features)
        p(f"  HMM: {features.shape} features, {len(states)} states")

        bar_ts = bars_df["timestamp"].dt.epoch("ns").to_numpy()
        ts_to_state = dict(zip(timestamps.tolist(), states))
        hmm_injected = 0

        for i in range(len(bundles)):
            bar_ns = int(bar_ts[i])
            state = ts_to_state.get(bar_ns)
            if state is not None:
                passes = state in pass_states if pass_states else True
                hmm_result = SignalResult(
                    value=float(state.value),
                    passes=passes,
                    direction="none",
                    metadata={
                        "regime": state.name,
                        "regime_value": state.value,
                        "pass_states": [s.name for s in pass_states],
                    },
                )
                merged = dict(bundles[i].results)
                merged["hmm_regime"] = hmm_result
                bundles[i] = SignalBundle(results=merged, bar_count=bundles[i].bar_count)
                hmm_injected += 1

        p(f"  Injected HMM into {hmm_injected}/{len(bundles)} bundles")

    # Build BarEvent array once
    bar_events = []
    bar_dates = []
    bar_times = []
    et_times = []
    for row in bars_df.iter_rows(named=True):
        bar_events.append(BarEvent(
            symbol="MES",
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
            bar_type="5m",
            timestamp_ns=row["_timestamp_ns"],
        ))
        bar_dates.append(row["_bar_date"])
        bar_times.append(row["_bar_time"])
        et_times.append(row["_et_ts"])

    session_close_time = _sub_time(config.session_end, seconds=1)
    p(f"  {len(bar_events)} 5m session bars in {_time.time() - t1:.1f}s")

    # ── Build combo grid ──
    entry_combos = list(product(GRID_DEV_SD, GRID_SLOPE, GRID_ADX, GRID_RVOL))
    exit_combos = list(product(GRID_STOP_ATR, GRID_TGT_SD, GRID_TIME, GRID_TRAIL))
    total = len(entry_combos) * len(exit_combos)
    p(f"\nJoint sweep: {len(entry_combos)} entry x {len(exit_combos)} exit = {total} combos over 10yr")

    # ── Run sweep ──
    t0 = _time.time()
    results = []

    for combo_idx, ((dev_sd, slope, adx, rvol), (stop_atr, tgt_sd, time_bars, (trail_atr, trail_act))) in enumerate(
        product(entry_combos, exit_combos)
    ):
        filters = build_filters(dev_sd, slope, adx, rvol)
        exits = build_exits(stop_atr, tgt_sd, time_bars, trail_atr, trail_act)
        cfg = build_strategy_config(filters, exits)
        strat = VWAPBandReversionStrategy(cfg)
        exit_engine = ExitEngine.from_list(exits)

        m, reasons = fast_run(
            bar_events, bundles, bar_dates, bar_times, et_times,
            session_close_time, strat, exit_engine,
        )

        score = m.sharpe_ratio * math.sqrt(m.total_trades) if m.sharpe_ratio > 0 else m.sharpe_ratio
        total_exits = sum(reasons.values()) or 1
        results.append({
            # Entry params
            "dev_sd": dev_sd, "slope": slope, "adx": adx, "rvol": rvol,
            # Exit params
            "stop_atr": stop_atr, "tgt_sd": tgt_sd,
            "time_bars": time_bars, "trail_atr": trail_atr, "trail_act": trail_act,
            # Metrics
            "trades": m.total_trades, "win_rate": m.win_rate,
            "net_pnl": m.net_pnl, "sharpe": m.sharpe_ratio,
            "profit_factor": m.profit_factor, "max_dd": m.max_drawdown_pct,
            "score": score,
            "exits": dict(reasons),
            "pct_tp": reasons.get("target", 0) / total_exits,
            "pct_stop": sum(v for k, v in reasons.items() if "stop" in k) / total_exits,
            "pct_trail": reasons.get("stop:trailing", 0) / total_exits,
            "pct_early": sum(v for k, v in reasons.items() if "early" in k) / total_exits,
            "pct_time": sum(v for k, v in reasons.items() if "time" in k) / total_exits,
            "pct_session": reasons.get("session_close", 0) / total_exits,
        })

        if (combo_idx + 1) % 20 == 0 or combo_idx == 0:
            elapsed = _time.time() - t0
            rate = (combo_idx + 1) / elapsed
            eta = (total - combo_idx - 1) / rate if rate > 0 else 0
            p(f"  [{combo_idx+1}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = _time.time() - t0
    p(f"\nDone: {total} configs in {elapsed:.0f}s ({elapsed/total:.1f}s/config)")

    if not results:
        p("No results!")
        return

    results.sort(key=lambda r: r["score"], reverse=True)

    # ── Helper to format a row ──
    def fmt(r):
        trail = f"{r['trail_atr']:.1f}/{r['trail_act']}" if r['trail_atr'] > 0 else "  off"
        return (f"{r['dev_sd']:5.1f} {r['slope']:5.2f} {r['adx']:5.0f} {r['rvol']:5.2f} | "
                f"{r['stop_atr']:5.1f} {r['tgt_sd']:6.2f} {r['time_bars']:5d} {trail:>7s} | "
                f"{r['trades']:6d} {r['win_rate']:5.1%} ${r['net_pnl']:8.2f} "
                f"{r['sharpe']:7.2f} {r['profit_factor']:6.2f} {r['max_dd']:6.2f}% {r['score']:7.2f} | "
                f"{r['pct_tp']:4.0%} {r['pct_stop']:5.0%} {r['pct_trail']:5.0%} "
                f"{r['pct_time']:5.0%} {r['pct_early']:5.0%} {r['pct_session']:5.0%}")

    hdr_entry = f"{'dv_sd':>5} {'slope':>5} {'adx':>5} {'rvol':>5}"
    hdr_exit = f"{'stop':>5} {'tgt_sd':>6} {'tbars':>5} {'trail':>7}"
    hdr_metrics = f"{'trades':>6} {'WR':>6} {'PnL':>9} {'Sharpe':>7} {'PF':>6} {'DD%':>7} {'score':>7}"
    hdr_exits = f"{'%TP':>5} {'%Stop':>5} {'%Trl':>5} {'%Time':>5} {'%Early':>6} {'%Sess':>5}"
    hdr = f"{hdr_entry} | {hdr_exit} | {hdr_metrics} | {hdr_exits}"

    # ── Top 20 overall ──
    p(f"\n{'='*160}")
    p(f"TOP 20 BY SCORE (Sharpe x sqrt(trades))")
    p(f"{'='*160}")
    p(hdr)
    p("-" * 155)
    for r in results[:20]:
        p(fmt(r))

    # ── Top 10 by Sharpe (min 30 trades) ──
    high_n = [r for r in results if r["trades"] >= 30]
    if high_n:
        high_n.sort(key=lambda r: r["sharpe"], reverse=True)
        p(f"\n{'='*160}")
        p(f"TOP 10 BY SHARPE (min 30 trades)")
        p(f"{'='*160}")
        p(hdr)
        p("-" * 155)
        for r in high_n[:10]:
            p(fmt(r))

    # ── Top 10 by net PnL (min 30 trades) ──
    if high_n:
        high_n.sort(key=lambda r: r["net_pnl"], reverse=True)
        p(f"\n{'='*160}")
        p(f"TOP 10 BY NET PNL (min 30 trades)")
        p(f"{'='*160}")
        p(hdr)
        p("-" * 155)
        for r in high_n[:10]:
            p(fmt(r))

    # ── Parameter sensitivity analysis ──
    p(f"\n{'='*80}")
    p(f"PARAMETER SENSITIVITY (avg Sharpe by param value, min 10 trades)")
    p(f"{'='*80}")
    viable = [r for r in results if r["trades"] >= 10]
    if viable:
        for param_name, grid in [
            ("dev_sd", GRID_DEV_SD), ("slope", GRID_SLOPE),
            ("adx", GRID_ADX), ("rvol", GRID_RVOL),
            ("stop_atr", GRID_STOP_ATR), ("tgt_sd", GRID_TGT_SD),
            ("time_bars", GRID_TIME), ("trail_atr", [0, 1.5]),
        ]:
            p(f"\n  {param_name}:")
            for val in grid:
                subset = [r for r in viable if r[param_name] == val]
                if subset:
                    avg_sharpe = sum(r["sharpe"] for r in subset) / len(subset)
                    avg_pnl = sum(r["net_pnl"] for r in subset) / len(subset)
                    avg_trades = sum(r["trades"] for r in subset) / len(subset)
                    avg_wr = sum(r["win_rate"] for r in subset) / len(subset)
                    p(f"    {val:>6}: avg_sharpe={avg_sharpe:+.3f}  avg_pnl=${avg_pnl:+.0f}  "
                      f"avg_trades={avg_trades:.0f}  avg_wr={avg_wr:.1%}  (n={len(subset)})")

    # ── Best config exit breakdown ──
    if results:
        best = results[0]
        p(f"\n{'='*60}")
        p(f"BEST CONFIG EXIT BREAKDOWN")
        p(f"{'='*60}")
        p(f"  Entry: dev_sd>={best['dev_sd']}, slope<={best['slope']}, "
          f"adx<{best['adx']}, rvol>={best['rvol']}")
        p(f"  Exit:  stop={best['stop_atr']}xATR, tgt={best['tgt_sd']}SD, "
          f"time={best['time_bars']}bars, trail={best['trail_atr']}/{best['trail_act']}")
        for reason, count in sorted(best["exits"].items(), key=lambda x: -x[1]):
            pct = count / best["trades"] * 100
            p(f"  {reason:<30s} {count:4d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
