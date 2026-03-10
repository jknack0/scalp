"""Backtest VWAP Band Reversion — straight run, full logging to file."""
import os, sys, logging, time as _time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import yaml

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.signals.signal_bundle import SignalBundle, SignalEngine
from src.strategies.base import Direction
from src.strategies.vwap_band_reversion import VWAPBandReversionStrategy

# ── Set up file logging (DEBUG level — every signal, filter, bar) ──
LOG_FILE = "logs/vwap_band_backtest.log"
os.makedirs("logs", exist_ok=True)

# Remove any existing handlers, set root to DEBUG
root = logging.getLogger()
root.setLevel(logging.DEBUG)
# Clear existing handlers
for h in root.handlers[:]:
    root.removeHandler(h)

# File handler: every log line
fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
root.addHandler(fh)

# Console handler: INFO and above only
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s | %(message)s"))
root.addHandler(ch)

# Also capture structlog output to the file
import structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(colors=False),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

# ── Load strategy from YAML ──
yaml_path = "config/strategies/vwap_band_reversion.yaml"
with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

strategy = VWAPBandReversionStrategy(cfg)

# Build signal engine from YAML — exclude hmm_regime (pre-computed separately)
signal_names = cfg.get("signals", [])
signal_configs = cfg.get("signal_configs", {})
fast_signal_names = [s for s in signal_names if s != "hmm_regime"]
signal_engine = SignalEngine(fast_signal_names, signal_configs)

# NOTE: Do NOT pass filter_engine to BacktestConfig — the strategy
# evaluates filters internally via its own FilterEngine.
# Passing it to the engine would cause double-evaluation.

config = BacktestConfig(
    strategies=[strategy],
    start_date=date(2011, 3, 1),
    end_date=date(2025, 2, 28),
    parquet_dir="data/parquet_5m",
    resample_freq=cfg["bar"]["freq"],  # "5m"
    signal_engine=signal_engine,
    # filter_engine intentionally omitted — strategy handles filters
)

engine = BacktestEngine()

# ── Pre-compute HMM regime states vectorially (fast) ──
hmm_cfg = signal_configs.get("hmm_regime", {})
if "hmm_regime" in signal_names and hmm_cfg.get("model_path"):
    t0 = _time.time()
    print("Pre-computing HMM regime states...")

    from src.models.hmm_regime import HMMRegimeClassifier, RegimeState, build_feature_matrix
    from src.signals.base import SignalResult

    clf = HMMRegimeClassifier.load(hmm_cfg["model_path"])
    pass_states_str = hmm_cfg.get("pass_states", [])
    pass_states = [RegimeState[s] for s in pass_states_str]

    # Pre-compute bundles (fast signals only, no HMM)
    print("  Computing signal bundles (no HMM)...")
    bundles = engine.precompute_bundles(config)
    print(f"  {len(bundles)} bundles in {_time.time() - t0:.1f}s")

    # Load bars for HMM feature extraction
    print("  Building HMM feature matrix...")
    t1 = _time.time()
    bars_df = engine._load_bars(config)

    # Same preprocessing as engine.run()
    import polars as pl
    from datetime import time as dt_time
    bars_df = bars_df.with_columns(
        pl.col("timestamp")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("US/Eastern")
            .alias("_et_ts"),
    )
    bars_df = bars_df.filter(
        (pl.col("_et_ts").dt.time() >= config.session_start)
        & (pl.col("_et_ts").dt.time() < config.session_end)
    )

    # Build vectorized feature matrix from all session bars
    features, timestamps = build_feature_matrix(bars_df)
    print(f"  Features: {features.shape} in {_time.time() - t1:.1f}s")

    # Predict all states at once (Viterbi — fast)
    t2 = _time.time()
    states = clf.predict(features)
    print(f"  Predicted {len(states)} states in {_time.time() - t2:.1f}s")

    # Align states to bars: features are offset by warm-up rows
    # build_feature_matrix drops warm-up rows, so we need to map timestamps
    n_bars = len(bundles)
    ts_set = set(timestamps.tolist())

    # Get bar timestamps for alignment
    bar_ts = bars_df["timestamp"].dt.epoch("ns").to_numpy()

    # Build timestamp -> state lookup
    ts_to_state = dict(zip(timestamps.tolist(), states))

    # Merge HMM results into pre-computed bundles
    hmm_injected = 0
    for i in range(n_bars):
        bar_ns = int(bar_ts[i])
        state = ts_to_state.get(bar_ns)
        if state is not None:
            passes = state in pass_states if pass_states else True
            hmm_result = SignalResult(
                value=float(state.value),
                passes=passes,
                direction="long" if state == RegimeState.HIGH_VOL_UP else (
                    "short" if state == RegimeState.HIGH_VOL_DOWN else "none"
                ),
                metadata={
                    "regime": state.name,
                    "regime_value": state.value,
                    "pass_states": [s.name for s in pass_states],
                },
            )
            # Create new bundle with HMM result merged in
            merged = dict(bundles[i].results)
            merged["hmm_regime"] = hmm_result
            bundles[i] = SignalBundle(results=merged, bar_count=bundles[i].bar_count)
            hmm_injected += 1

    print(f"  Injected HMM into {hmm_injected}/{n_bars} bundles in {_time.time() - t0:.1f}s total")

    # Use prebuilt bundles (skip signal engine in the main loop)
    config.prebuilt_bundles = bundles
    config.signal_engine = None  # don't recompute signals

result = engine.run(config)

# ── Print results to console AND log file ──
def report(msg):
    print(msg)
    logging.info(msg)

report(f"\n{'='*60}")
report(f"VWAP Band Reversion Backtest")
report(f"Period: {config.start_date} to {config.end_date}")
report(f"Bar freq: {config.resample_freq or '5m'}")
report(f"{'='*60}")
m = result.metrics
report(f"Total trades: {m.total_trades}")
report(f"Win rate: {m.win_rate:.1%}")
report(f"Avg win: ${m.avg_win:.2f}")
report(f"Avg loss: ${m.avg_loss:.2f}")
report(f"Profit factor: {m.profit_factor:.2f}")
report(f"Net P&L: ${m.net_pnl:.2f}")
report(f"Gross P&L: ${m.gross_pnl:.2f}")
report(f"Commission: ${m.total_commission:.2f}")
report(f"Slippage: ${m.total_slippage:.2f}")
report(f"Sharpe: {m.sharpe_ratio:.2f}")
report(f"Sortino: {m.sortino_ratio:.2f}")
report(f"Max drawdown: {m.max_drawdown_pct:.2f}%")
report(f"Avg bars held: {m.avg_bars_held:.1f}")
report(f"Best trade: ${m.best_trade:.2f}")
report(f"Worst trade: ${m.worst_trade:.2f}")

if result.trades:
    report(f"\n--- Trade Log ({len(result.trades)} trades) ---")
    for t in result.trades:
        d = "LONG" if t.direction == Direction.LONG else "SHORT"
        report(f"  {t.entry_time.strftime('%Y-%m-%d %H:%M')} {d:5s} "
               f"entry={t.entry_price:.2f} exit={t.exit_price:.2f} "
               f"pnl=${t.net_pnl:.2f} slippage=${t.slippage_cost:.2f} "
               f"({t.exit_reason})")

report(f"\nFull logs written to: {LOG_FILE}")
