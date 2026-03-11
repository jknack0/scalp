#!/usr/bin/env python3
"""Walk-forward stability check for VWAP Band Reversion winning config.

Runs the locked config on 2-year rolling windows across 10yr to check
if the edge is consistent or clustered in one period.
"""
import os, sys, time as _time, logging, math
from collections import Counter
from datetime import date, datetime, time, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


# ── WINNING CONFIG (locked from joint sweep) ──
WINNING_FILTERS = [
    {"signal": "session_time", "expr": ">= 585"},
    {"signal": "session_time", "expr": "<= 900"},
    {"signal": "vwap_session", "field": "session_age_bars", "expr": ">= 30"},
    {"signal": "vwap_session", "field": "slope", "expr": "abs <= 0.5"},
    {"signal": "vwap_session", "field": "deviation_sd", "expr": "abs >= 3.0"},
    {"signal": "adx", "expr": "< 20.0"},
    {"signal": "relative_volume", "expr": ">= 0.3"},
    {"signal": "hmm_regime", "expr": "passes"},
]

WINNING_EXITS = [
    {"type": "static_stop", "enabled": True, "atr_multiple": 4.0},
    {"type": "vwap_reversion_target", "enabled": True,
     "target_sd_band": 1.0, "vwap_signal": "vwap_session",
     "deviation_field": "deviation_sd"},
    {"type": "time_stop", "enabled": True, "max_bars": 45},
    {"type": "adverse_signal_exit", "enabled": True,
     "signal": "vwap_session", "field": "slope",
     "long_threshold": -0.5, "short_threshold": 0.5},
    {"type": "regime_exit", "enabled": True,
     "hmm_signal": "hmm_regime",
     "hostile_regimes_long": [1], "hostile_regimes_short": [1],
     "min_bars_before_active": 2},
]

WINNING_CFG = {
    "strategy": {"strategy_id": "vwap_band_reversion", "max_signals_per_day": 999},
    "signal_configs": {"adx": {"period": 14, "threshold": 25.0}},
    "filters": WINNING_FILTERS,
    "exits": WINNING_EXITS,
    "exit": {
        "target": {"type": "vwap"},
        "stop": {"type": "atr_multiple", "multiplier": 4.0},
        "time_stop_minutes": 225,
    },
}

# 2-year rolling windows
WINDOWS = [
    ("2015-2017", date(2015, 3, 1), date(2017, 2, 28)),
    ("2017-2019", date(2017, 3, 1), date(2019, 2, 28)),
    ("2019-2021", date(2019, 3, 1), date(2021, 2, 28)),
    ("2021-2023", date(2021, 3, 1), date(2023, 2, 28)),
    ("2023-2025", date(2023, 3, 1), date(2025, 2, 28)),
]


def _sub_time(t: time, seconds: int = 0) -> time:
    dt = datetime.combine(date.today(), t) - timedelta(seconds=seconds)
    return dt.time()


def run_window(bar_events, bundles, dates, times, et_times,
               session_close_time, start_date, end_date):
    """Run winning config on a date-filtered subset."""
    strat = VWAPBandReversionStrategy(WINNING_CFG)
    exit_engine = ExitEngine.from_list(WINNING_EXITS)

    oms = SimulatedOMS(
        commission_model=tradovate_free(),
        slippage_model=VolatilitySlippageModel(),
        max_position=1,
        exit_engine=exit_engine,
    )

    all_trades = []
    prev_date = None
    n_bars = len(bar_events)
    exit_reasons = Counter()

    for bar_index in range(n_bars):
        bar_date = dates[bar_index]
        if bar_date < start_date or bar_date > end_date:
            continue

        bar = bar_events[bar_index]
        bar_time = et_times[bar_index]

        if prev_date is None or bar_date != prev_date:
            if prev_date is not None:
                trades = oms.close_all(
                    close_price=bar.close, close_time=bar_time,
                    bar_index=bar_index, bar_date=bar_date,
                    current_atr_ticks=0.0, reason="session_close",
                )
                all_trades.extend(trades)
            strat.reset()

        prev_date = bar_date
        bundle = bundles[bar_index]

        signal = strat.on_bar(bar, bundle)
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

    # Final close
    if all_trades:
        last_idx = n_bars - 1
        trades = oms.close_all(
            close_price=bar_events[last_idx].close,
            close_time=et_times[last_idx],
            bar_index=last_idx, bar_date=dates[last_idx],
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

    p("Pre-computing signal bundles for 10 years...")
    t0 = _time.time()
    bundles = engine.precompute_bundles(config)
    p(f"  {len(bundles)} bundles in {_time.time() - t0:.1f}s")

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

    # ── Run on each window ──
    p(f"\n{'='*90}")
    p(f"WALK-FORWARD STABILITY CHECK — winning config on 2yr rolling windows")
    p(f"{'='*90}")
    hdr = f"{'Window':<12} | {'Trades':>6} {'WR':>6} {'PnL':>9} {'Sharpe':>7} {'PF':>6} {'DD%':>6} | {'%TP':>5} {'%Stop':>5} {'%Early':>6} {'%Sess':>5}"
    p(hdr)
    p("-" * 88)

    total_trades = 0
    total_pnl = 0.0
    positive_windows = 0

    for label, start, end in WINDOWS:
        t0 = _time.time()
        m, reasons = run_window(bar_events, bundles, bar_dates, bar_times, et_times,
                                session_close_time, start, end)
        elapsed = _time.time() - t0

        total_exits = sum(reasons.values()) or 1
        pct_tp = reasons.get("target", 0) / total_exits
        pct_stop = sum(v for k, v in reasons.items() if "stop" in k) / total_exits
        pct_early = sum(v for k, v in reasons.items() if "early" in k) / total_exits
        pct_session = reasons.get("session_close", 0) / total_exits

        p(f"{label:<12} | {m.total_trades:6d} {m.win_rate:5.1%} ${m.net_pnl:8.2f} "
          f"{m.sharpe_ratio:7.2f} {m.profit_factor:6.2f} {m.max_drawdown_pct:5.2f}% | "
          f"{pct_tp:4.0%} {pct_stop:5.0%} {pct_early:5.0%} {pct_session:5.0%}  ({elapsed:.1f}s)")

        total_trades += m.total_trades
        total_pnl += m.net_pnl
        if m.net_pnl > 0:
            positive_windows += 1

    p(f"\n{'='*90}")
    p(f"SUMMARY: {positive_windows}/{len(WINDOWS)} windows profitable, "
      f"{total_trades} total trades, ${total_pnl:.2f} total PnL")
    p(f"{'='*90}")


if __name__ == "__main__":
    main()
