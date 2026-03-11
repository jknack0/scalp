#!/usr/bin/env python3
"""Tier 1: Sweep VWAP Band Reversion params over 5m bars with HMM gating.

Pre-computes bars, BarEvents, SignalBundles, and HMM regime states once.
Then for each param combo, only re-runs the strategy + OMS inner loop.
"""
import os, sys, time as _time, logging, math
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
from src.core.events import BarEvent
from src.signals.signal_bundle import SignalEngine, SignalBundle, EMPTY_BUNDLE
from src.strategies.vwap_band_reversion import VWAPBandReversionStrategy


def p(msg):
    print(msg, flush=True)


# ── Param grid ──
PARAM_GRID = {
    "deviation_sd": [2.0, 2.5, 3.0, 3.5, 4.0],
    "slope_max": [0.3, 0.5, 1.0],
    "adx_max": [25.0, 40.0],
    "atr_mult": [1.5, 2.0, 2.5, 3.0],
    "time_stop": [10, 15, 30, 60],
    "rsi_long": [30.0, 40.0],
    "rsi_short": [60.0, 70.0],
}

total = 1
for v in PARAM_GRID.values():
    total *= len(v)
p(f"Total param combos: {total:,}")


def build_config(dev_sd, slope, adx, atr_m, tstop, rsi_l, rsi_s):
    return {
        "strategy": {"strategy_id": "vwap_band_reversion", "max_signals_per_day": 999},
        "signal_configs": {
            "rsi_momentum": {"period": 2, "long_threshold": rsi_l, "short_threshold": rsi_s},
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
            {"signal": "hmm_regime", "expr": "passes"},
        ],
        "exit": {
            "target": {"type": "vwap"},
            "stop": {"type": "atr_multiple", "multiplier": atr_m},
            "time_stop_minutes": tstop,
            "early_exit": [{"type": "vwap_slope", "threshold": 0.3}],
        },
    }


def _sub_time(t: time, seconds: int = 0) -> time:
    dt = datetime.combine(date.today(), t) - timedelta(seconds=seconds)
    return dt.time()


def fast_run(bars, bundles, dates, times, et_times, session_close_time, strategy):
    """Stripped-down backtest loop — no logging, no data loading, minimal overhead."""
    from src.backtesting.engine import SimulatedOMS
    from src.backtesting.slippage import VolatilitySlippageModel
    from src.analysis.commission_model import tradovate_free

    oms = SimulatedOMS(
        commission_model=tradovate_free(),
        slippage_model=VolatilitySlippageModel(),
        max_position=1,
    )

    all_trades = []
    prev_date = None
    n_bars = len(bars)

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

        # Build early exit fn
        early_exit_fn = None
        if hasattr(strategy, "check_early_exit"):
            def _make_fn(s, b, bun):
                def _fn(order, bar, bar_index):
                    bars_in_trade = bar_index - order.fill_bar_index
                    return s.check_early_exit(
                        bar=bar, bundle=bun, bars_in_trade=bars_in_trade,
                        direction=order.direction, fill_price=order.fill_price,
                    )
                return _fn
            early_exit_fn = _make_fn(strategy, bar, bundle)

        trades = oms.on_bar(bar, bar_index, bar_time, bar_date, 0.0,
                           early_exit_fn=early_exit_fn)
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

    metrics, _, _ = MetricsCalculator.from_trades(all_trades, 10000.0)
    return metrics


def main():
    yaml_path = "config/strategies/vwap_band_reversion.yaml"
    with open(yaml_path) as f:
        base_cfg = yaml.safe_load(f)

    signal_names = [s for s in base_cfg.get("signals", []) if s != "hmm_regime"]
    signal_configs = base_cfg.get("signal_configs", {})
    signal_engine = SignalEngine(signal_names, signal_configs)

    config = BacktestConfig(
        strategies=[],
        start_date=date(2020, 3, 1),
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
    p(f"  {len(bar_events)} session bars in {_time.time() - t1:.1f}s")

    # ── Sweep ──
    combos = list(product(
        PARAM_GRID["deviation_sd"],
        PARAM_GRID["slope_max"],
        PARAM_GRID["adx_max"],
        PARAM_GRID["atr_mult"],
        PARAM_GRID["time_stop"],
        PARAM_GRID["rsi_long"],
        PARAM_GRID["rsi_short"],
    ))

    p(f"\nSweeping {len(combos):,} combos...")
    t0 = _time.time()
    results = []

    for i, (dev_sd, slope, adx, atr_m, tstop, rsi_l, rsi_s) in enumerate(combos):
        cfg = build_config(dev_sd, slope, adx, atr_m, tstop, rsi_l, rsi_s)
        strat = VWAPBandReversionStrategy(cfg)

        m = fast_run(bar_events, bundles, bar_dates, bar_times, et_times,
                     session_close_time, strat)

        if m.total_trades >= 50:
            score = m.sharpe_ratio * math.sqrt(m.total_trades) if m.sharpe_ratio > 0 else m.sharpe_ratio
            results.append({
                "deviation_sd": dev_sd, "slope_max": slope, "adx_max": adx,
                "atr_mult": atr_m, "time_stop": tstop,
                "rsi_long": rsi_l, "rsi_short": rsi_s,
                "trades": m.total_trades, "win_rate": m.win_rate,
                "net_pnl": m.net_pnl, "sharpe": m.sharpe_ratio,
                "profit_factor": m.profit_factor, "max_dd": m.max_drawdown_pct,
                "score": score,
            })

        if (i + 1) % 50 == 0:
            elapsed = _time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(combos) - i - 1) / rate
            p(f"  [{i+1:,}/{len(combos):,}] {rate:.1f}/s, ETA {eta/60:.0f}min, {len(results)} viable")

    elapsed = _time.time() - t0
    p(f"\nDone: {len(combos):,} combos in {elapsed/60:.1f}min ({len(combos)/elapsed:.1f}/s)")
    p(f"Viable results (>=50 trades): {len(results)}")

    if not results:
        p("No viable results found!")
        return

    results.sort(key=lambda r: r["score"], reverse=True)

    p(f"\n{'='*100}")
    p(f"TOP 20 RESULTS (ranked by Sharpe x sqrt(trades))")
    p(f"{'='*100}")
    p(f"{'dev_sd':>6} {'slope':>5} {'adx':>4} {'atr_m':>5} {'tstop':>5} "
      f"{'rsi_l':>5} {'rsi_s':>5} | {'trades':>6} {'WR':>6} {'PnL':>9} "
      f"{'Sharpe':>7} {'PF':>6} {'DD%':>6} {'score':>7}")
    p("-" * 100)
    for r in results[:20]:
        p(f"{r['deviation_sd']:6.1f} {r['slope_max']:5.2f} {r['adx_max']:4.0f} "
          f"{r['atr_mult']:5.1f} {r['time_stop']:5d} {r['rsi_long']:5.0f} "
          f"{r['rsi_short']:5.0f} | {r['trades']:6d} {r['win_rate']:5.1%} "
          f"${r['net_pnl']:8.2f} {r['sharpe']:7.2f} {r['profit_factor']:6.2f} "
          f"{r['max_dd']:5.2f}% {r['score']:7.2f}")

    high_n = [r for r in results if r["trades"] >= 100]
    if high_n:
        high_n.sort(key=lambda r: r["sharpe"], reverse=True)
        p(f"\n{'='*100}")
        p(f"TOP 10 BY SHARPE (min 100 trades)")
        p(f"{'='*100}")
        p(f"{'dev_sd':>6} {'slope':>5} {'adx':>4} {'atr_m':>5} {'tstop':>5} "
          f"{'rsi_l':>5} {'rsi_s':>5} | {'trades':>6} {'WR':>6} {'PnL':>9} "
          f"{'Sharpe':>7} {'PF':>6} {'DD%':>6}")
        p("-" * 100)
        for r in high_n[:10]:
            p(f"{r['deviation_sd']:6.1f} {r['slope_max']:5.2f} {r['adx_max']:4.0f} "
              f"{r['atr_mult']:5.1f} {r['time_stop']:5d} {r['rsi_long']:5.0f} "
              f"{r['rsi_short']:5.0f} | {r['trades']:6d} {r['win_rate']:5.1%} "
              f"${r['net_pnl']:8.2f} {r['sharpe']:7.2f} {r['profit_factor']:6.2f} "
              f"{r['max_dd']:5.2f}%")


if __name__ == "__main__":
    main()
