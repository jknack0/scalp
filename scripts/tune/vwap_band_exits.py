#!/usr/bin/env python3
"""Sweep VWAP Band Reversion EXIT params over 5yr 5m bars with HMM gating.

Entry params are LOCKED to tuned values (dev_sd=3.0, adx<25, slope<=0.3).
Exit params are swept via ExitEngine: stop ATR mult, reversion target SD,
time stop bars, adverse slope threshold, and toggles for regime_exit +
volatility_expansion_exit.

Pre-computes bars, BarEvents, SignalBundles, and HMM regime states once.
Then for each exit param combo, only re-runs the strategy + OMS inner loop.
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


# ── Locked entry params (from Tier 1 sweep) ──
ENTRY_CONFIG = {
    "strategy": {"strategy_id": "vwap_band_reversion", "max_signals_per_day": 999},
    "signal_configs": {
        "adx": {"period": 14, "threshold": 25.0},
    },
    "filters": [
        {"signal": "session_time", "expr": ">= 585"},
        {"signal": "session_time", "expr": "<= 900"},
        {"signal": "vwap_session", "field": "session_age_bars", "expr": ">= 30"},
        {"signal": "vwap_session", "field": "slope", "expr": "abs <= 0.3"},
        {"signal": "vwap_session", "field": "deviation_sd", "expr": "abs >= 3.0"},
        {"signal": "adx", "expr": "< 25.0"},
        {"signal": "relative_volume", "expr": ">= 0.5"},
        {"signal": "hmm_regime", "expr": "passes"},
    ],
}


# ── 5yr validation of top 1yr configs ──
# Top configs from 1yr kitchen sink sweep:
#   1. stop=3.0, tgt=0.5, t=30, slope=0.5, trail=0.8/8  (Sharpe 10.37, 86% WR, $487)
#   2. stop=3.0, tgt=0.5, t=30, slope=0.5, trail=1.5/4  (Sharpe 10.15, 79% WR, $502)
#   3. stop=3.0, tgt=0.0, t=10, slope=0.3, trail=off     (Sharpe 10.11, 79% WR, $556)
VALIDATION_CONFIGS = [
    # (stop_atr, target_sd, time_bars, slope_thresh, trail_atr, trail_activate, tp_enabled)
    # 10yr validation on 5m bars — locked winner config
    (3.0, 0.5, 30, 0.5, 1.5, 4, True),      # THE CONFIG
]
LOCKED_REGIME_EXIT = True
LOCKED_VOL_EXPANSION = 999.0  # disabled

p(f"Validating {len(VALIDATION_CONFIGS)} configs over 5 years...")


def build_exits(stop_atr, target_sd, time_bars, slope_thresh, trail_atr, trail_activate, tp_enabled=True):
    """Build exits list for ExitEngine from sweep params."""
    exits = [
        {"type": "static_stop", "enabled": True, "atr_multiple": stop_atr},
        {"type": "vwap_reversion_target", "enabled": tp_enabled,
         "target_sd_band": target_sd, "vwap_signal": "vwap_session",
         "deviation_field": "deviation_sd"},
        {"type": "time_stop", "enabled": time_bars < 900, "max_bars": time_bars},
        {"type": "adverse_signal_exit", "enabled": slope_thresh < 900,
         "signal": "vwap_session", "field": "slope",
         "long_threshold": -slope_thresh, "short_threshold": slope_thresh},
        {"type": "trailing_stop", "enabled": trail_atr > 0,
         "atr_multiple": trail_atr, "activate_after_ticks": trail_activate},
        {"type": "regime_exit", "enabled": LOCKED_REGIME_EXIT,
         "hmm_signal": "hmm_regime",
         "hostile_regimes_long": [1], "hostile_regimes_short": [1],
         "min_bars_before_active": 2},
        {"type": "volatility_expansion_exit", "enabled": LOCKED_VOL_EXPANSION < 900,
         "atr_signal": "atr", "expansion_multiple": LOCKED_VOL_EXPANSION,
         "min_bars_before_active": 3},
    ]
    return exits


def build_config_with_exits(exits_list):
    """Build strategy config with locked entries + variable exits."""
    cfg = dict(ENTRY_CONFIG)
    cfg["exits"] = exits_list
    # Legacy exit config — still needed for Signal geometry (target/stop placeholders)
    cfg["exit"] = {
        "target": {"type": "vwap"},
        "stop": {"type": "atr_multiple", "multiplier": 3.0},
        "time_stop_minutes": 120,  # Long expiry — ExitEngine handles actual time stop
    }
    return cfg


def _sub_time(t: time, seconds: int = 0) -> time:
    dt = datetime.combine(date.today(), t) - timedelta(seconds=seconds)
    return dt.time()


def fast_run_exit_engine(bars, bundles, dates, times, et_times,
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

        # Session boundaries
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

        # ExitEngine handles all exits — pass bundle for snapshot + evaluation
        trades = oms.on_bar(bar, bar_index, bar_time, bar_date, 0.0,
                           bundle=bundle)
        all_trades.extend(trades)

        # Session close
        if times[bar_index] >= session_close_time:
            trades = oms.close_all(
                close_price=bar.close, close_time=bar_time,
                bar_index=bar_index, bar_date=bar_date,
                current_atr_ticks=0.0, reason="session_close",
            )
            all_trades.extend(trades)

    # Final close
    if n_bars > 0:
        trades = oms.close_all(
            close_price=bars[-1].close, close_time=et_times[-1],
            bar_index=n_bars - 1, bar_date=dates[-1],
            current_atr_ticks=0.0, reason="session_close",
        )
        all_trades.extend(trades)

    # Count exit reasons
    for t in all_trades:
        exit_reasons[t.exit_reason] += 1

    metrics, _, _ = MetricsCalculator.from_trades(all_trades, 10000.0)
    return metrics, exit_reasons


def build_timeframe_map(et_times_5m, et_times_1m):
    """Map each 5m bar index to (start, end) slice in the 1m bar array."""
    mapping = []
    j = 0
    n_1m = len(et_times_1m)
    for ts_5m in et_times_5m:
        ts_end = ts_5m + timedelta(minutes=5)
        start = j
        while j < n_1m and et_times_1m[j] < ts_end:
            j += 1
        mapping.append((start, j))
    return mapping


def fast_run_dual_timeframe(bars_5m, bundles_5m, dates_5m, times_5m, et_times_5m,
                             bars_1m, dates_1m, times_1m, et_times_1m,
                             tf_map, session_close_time, strategy, exit_engine):
    """Dual-timeframe backtest: entries on 5m, exits on 1m."""
    oms = SimulatedOMS(
        commission_model=tradovate_free(),
        slippage_model=VolatilitySlippageModel(),
        max_position=1,
        exit_engine=exit_engine,
    )

    all_trades = []
    prev_date = None
    n_5m = len(bars_5m)
    exit_reasons = Counter()

    for i5 in range(n_5m):
        bar_5m = bars_5m[i5]
        bar_date = dates_5m[i5]
        bar_time = et_times_5m[i5]

        # Session boundaries (detected on 5m)
        if prev_date is None or bar_date != prev_date:
            if prev_date is not None:
                trades = oms.close_all(
                    close_price=bar_5m.close, close_time=bar_time,
                    bar_index=i5 * 5, bar_date=bar_date,
                    current_atr_ticks=0.0, reason="session_close",
                )
                all_trades.extend(trades)
            strategy.reset()

        prev_date = bar_date
        bundle = bundles_5m[i5]

        # ── Entry evaluation on 5m ──
        signal = strategy.on_bar(bar_5m, bundle)
        if signal is not None and oms.open_position_count < 1:
            # Map entry to 1m index space: last 1m bar of this 5m period
            # so fill happens on the first 1m bar of the NEXT 5m period
            start_1m, end_1m = tf_map[i5]
            entry_1m_idx = end_1m - 1 if end_1m > start_1m else i5 * 5
            oms.on_signal(signal, entry_1m_idx)

        # ── Exit evaluation on 1m sub-bars ──
        start_1m, end_1m = tf_map[i5]
        for k in range(start_1m, end_1m):
            bar_1m = bars_1m[k]
            bar_1m_time = et_times_1m[k]
            bar_1m_date = dates_1m[k]

            # Pass 5m bundle (carried forward) for signal-based exits
            trades = oms.on_bar(bar_1m, k, bar_1m_time, bar_1m_date, 0.0,
                               bundle=bundle)
            all_trades.extend(trades)

        # Session close
        if times_5m[i5] >= session_close_time:
            # Use last 1m bar price if available
            if end_1m > start_1m:
                close_price = bars_1m[end_1m - 1].close
                close_time = et_times_1m[end_1m - 1]
                close_idx = end_1m - 1
            else:
                close_price = bar_5m.close
                close_time = bar_time
                close_idx = i5 * 5
            trades = oms.close_all(
                close_price=close_price, close_time=close_time,
                bar_index=close_idx, bar_date=bar_date,
                current_atr_ticks=0.0, reason="session_close",
            )
            all_trades.extend(trades)

    # Final close
    if n_5m > 0:
        last_1m_start, last_1m_end = tf_map[-1]
        if last_1m_end > last_1m_start:
            close_price = bars_1m[last_1m_end - 1].close
            close_time = et_times_1m[last_1m_end - 1]
            close_idx = last_1m_end - 1
        else:
            close_price = bars_5m[-1].close
            close_time = et_times_5m[-1]
            close_idx = n_5m * 5
        trades = oms.close_all(
            close_price=close_price, close_time=close_time,
            bar_index=close_idx, bar_date=dates_5m[-1],
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
    p("Pre-computing signal bundles for 5 years...")
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

    # ── Validate on 5m bars ──
    p(f"\nValidating {len(VALIDATION_CONFIGS)} config(s) over 10 years (5m bars)...")
    t0 = _time.time()
    results = []

    for i, (stop_atr, target_sd, time_bars, slope_thresh, trail_atr, trail_act, tp_on) in enumerate(VALIDATION_CONFIGS):
        exits_list = build_exits(stop_atr, target_sd, time_bars, slope_thresh, trail_atr, trail_act, tp_on)
        cfg = build_config_with_exits(exits_list)
        strat = VWAPBandReversionStrategy(cfg)
        exit_engine = ExitEngine.from_list(exits_list)

        m, reasons = fast_run_exit_engine(
            bar_events, bundles, bar_dates, bar_times, et_times,
            session_close_time, strat, exit_engine,
        )

        score = m.sharpe_ratio * math.sqrt(m.total_trades) if m.sharpe_ratio > 0 else m.sharpe_ratio
        total_exits = sum(reasons.values()) or 1
        results.append({
            "stop_atr": stop_atr, "target_sd": target_sd,
            "time_bars": time_bars, "slope_thresh": slope_thresh,
            "trail_atr": trail_atr, "trail_act": trail_act, "tp_on": tp_on,
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
        elapsed = _time.time() - t0
        p(f"  [{i+1}/{len(VALIDATION_CONFIGS)}] {elapsed:.0f}s")

    elapsed = _time.time() - t0
    p(f"\nDone: {len(VALIDATION_CONFIGS)} configs in {elapsed:.0f}s")

    if not results:
        p("No viable results found!")
        return

    results.sort(key=lambda r: r["score"], reverse=True)

    # ── Helper to format a row ──
    def fmt(r):
        trail = f"{r['trail_atr']:.1f}/{r['trail_act']}" if r['trail_atr'] > 0 else "  off"
        tp = "Y" if r['tp_on'] else "N"
        return (f"{r['stop_atr']:5.1f} {r['target_sd']:6.2f} {r['time_bars']:5d} "
                f"{r['slope_thresh']:5.1f} {trail:>7s} {tp:>2s} | "
                f"{r['trades']:6d} {r['win_rate']:5.1%} ${r['net_pnl']:8.2f} "
                f"{r['sharpe']:7.2f} {r['profit_factor']:6.2f} {r['max_dd']:5.2f}% {r['score']:7.2f} | "
                f"{r['pct_tp']:4.0%} {r['pct_stop']:5.0%} {r['pct_trail']:5.0%} "
                f"{r['pct_time']:5.0%} {r['pct_early']:5.0%} {r['pct_session']:5.0%}")

    # ── All results ──
    p(f"\n{'='*135}")
    p(f"ALL RESULTS (ranked by Sharpe x sqrt(trades))")
    p(f"{'='*135}")
    hdr = (f"{'stop':>5} {'tgt_sd':>6} {'tbars':>5} {'slope':>5} {'trail':>7} {'TP':>2} | "
           f"{'trades':>6} {'WR':>6} {'PnL':>9} {'Sharpe':>7} {'PF':>6} {'DD%':>6} {'score':>7} | "
           f"{'%TP':>5} {'%Stop':>5} {'%Trl':>5} {'%Time':>5} {'%Early':>6} {'%Sess':>5}")
    p(hdr)
    p("-" * 130)
    for r in results[:20]:
        p(fmt(r))

    # ── Top 10 by Sharpe (min 50 trades) ──
    high_n = [r for r in results if r["trades"] >= 50]
    if high_n:
        high_n.sort(key=lambda r: r["sharpe"], reverse=True)
        p(f"\n{'='*130}")
        p(f"TOP 10 BY SHARPE (min 50 trades)")
        p(f"{'='*130}")
        p(hdr)
        p("-" * 130)
        for r in high_n[:10]:
            p(fmt(r))

    # ── Top 10 with trailing stop active ──
    trail_results = [r for r in results if r["trail_atr"] > 0 and r["pct_trail"] > 0]
    if trail_results:
        trail_results.sort(key=lambda r: r["score"], reverse=True)
        p(f"\n{'='*130}")
        p(f"TOP 10 WITH TRAILING STOP (actually firing)")
        p(f"{'='*130}")
        p(hdr)
        p("-" * 130)
        for r in trail_results[:10]:
            p(fmt(r))

    # ── Top result exit breakdown ──
    if results:
        best = results[0]
        p(f"\n{'='*60}")
        p(f"BEST CONFIG EXIT BREAKDOWN")
        p(f"{'='*60}")
        for reason, count in sorted(best["exits"].items(), key=lambda x: -x[1]):
            pct = count / best["trades"] * 100
            p(f"  {reason:<30s} {count:4d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
