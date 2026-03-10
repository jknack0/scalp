"""VWAP Band Reversion: 1-month diagnostic — why no signals?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.disable(logging.CRITICAL)

from collections import Counter
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.core.events import BarEvent
from src.filters.filter_engine import FilterEngine
from src.signals.signal_bundle import SignalEngine

_ET = ZoneInfo("US/Eastern")

# Load bars
config = BacktestConfig(
    strategies=[],
    start_date=date(2025, 2, 1),
    end_date=date(2025, 2, 28),
    parquet_dir="data/parquet",
    resample_freq="5m",
)
engine = BacktestEngine()
bars_df = engine._load_bars(config)
print(f"Loaded {len(bars_df):,} 5m bars\n")

# Build signal engine matching the YAML
signal_engine = SignalEngine(
    ["vwap_session", "rsi_momentum", "adx", "atr", "relative_volume", "sma_trend", "session_time"],
    signal_configs={
        "rsi_momentum": {"period": 2, "long_threshold": 20.0, "short_threshold": 80.0},
        "adx": {"period": 14, "threshold": 25.0},
        "sma_trend": {"period": 200},
    },
)

# Build FilterEngine from the YAML filters
filters = [
    {"signal": "session_time", "expr": ">= 585"},
    {"signal": "session_time", "expr": "<= 900"},
    {"signal": "vwap_session", "field": "session_age_bars", "expr": ">= 30"},
    {"signal": "vwap_session", "field": "slope", "expr": "abs <= 0.5"},
    {"signal": "vwap_session", "field": "deviation_sd", "expr": "abs >= 2.0"},
    {"signal": "adx", "expr": "< 25.0"},
    {"signal": "relative_volume", "expr": ">= 1.0"},
]
filter_engine = FilterEngine.from_list(filters)

# Replay bars
bar_window: list[BarEvent] = []
gate_blocks = Counter()
gate_values = {"deviation_sd": [], "rsi": [], "adx": [], "slope": [], "rvol": [],
               "session_time": [], "session_age": []}
signals_generated = 0

for row in bars_df.iter_rows(named=True):
    ts = row["timestamp"]
    if hasattr(ts, "timestamp"):
        ts_ns = int(ts.timestamp() * 1e9)
    else:
        ts_ns = int(ts * 1e9) if ts > 1e15 else int(ts * 1e9)

    bar = BarEvent(
        symbol="MESM6",
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=int(row["volume"]),
        bar_type="5m",
        timestamp_ns=ts_ns,
    )
    bar_window.append(bar)
    if len(bar_window) > 500:
        bar_window = bar_window[-500:]

    bundle = signal_engine.compute(bar_window)

    # Check each filter individually to see what blocks
    vwap_r = bundle.get("vwap_session")
    if vwap_r is None:
        gate_blocks["no_vwap_signal"] += 1
        continue

    meta = vwap_r.metadata
    vwap = meta.get("vwap", 0.0)
    sd = meta.get("sd", 0.0)
    if vwap == 0 or sd == 0:
        gate_blocks["vwap_zero"] += 1
        continue

    # Record raw values
    deviation_sd = meta.get("deviation_sd", 0.0)
    slope = meta.get("slope", 0.0)
    session_age = meta.get("session_age_bars", 0)
    rsi_r = bundle.get("rsi_momentum")
    adx_r = bundle.get("adx")
    rvol_r = bundle.get("relative_volume")
    time_r = bundle.get("session_time")

    rsi_val = rsi_r.value if rsi_r else 50.0
    adx_val = adx_r.value if adx_r else 0.0
    rvol_val = rvol_r.value if rvol_r else 1.0
    time_val = time_r.value if time_r else 0.0

    gate_values["deviation_sd"].append(abs(deviation_sd))
    gate_values["rsi"].append(rsi_val)
    gate_values["adx"].append(adx_val)
    gate_values["slope"].append(abs(slope))
    gate_values["rvol"].append(rvol_val)
    gate_values["session_time"].append(time_val)
    gate_values["session_age"].append(session_age)

    # Check each filter rule individually
    for rule_dict in filters:
        single_fe = FilterEngine.from_list([rule_dict])
        result = single_fe.evaluate(bundle)
        if not result.passes:
            label = rule_dict["signal"]
            if "field" in rule_dict:
                label += f".{rule_dict['field']}"
            gate_blocks[f"{label} {rule_dict['expr']}"] += 1

    # Check all filters together
    result = filter_engine.evaluate(bundle)
    if not result.passes:
        continue

    # Directional RSI check (strategy-level, not filter)
    direction = "LONG" if deviation_sd < 0 else "SHORT"
    if rsi_r is not None:
        if direction == "LONG" and rsi_val > 20.0:
            gate_blocks["rsi_not_oversold (directional)"] += 1
            continue
        if direction == "SHORT" and rsi_val < 80.0:
            gate_blocks["rsi_not_overbought (directional)"] += 1
            continue

    # SMA trend alignment check (strategy-level) — LOG but don't block for now
    sma_r = bundle.get("sma_trend")
    sma_block = False
    if sma_r is not None and sma_r.direction != "none":
        if deviation_sd < 0 and sma_r.direction == "short":
            sma_block = True
        if deviation_sd > 0 and sma_r.direction == "long":
            sma_block = True

    signals_generated += 1
    now = datetime.fromtimestamp(ts_ns / 1e9, tz=_ET)
    sma_tag = " [COUNTER-TREND]" if sma_block else ""
    print(f"  SIGNAL: {now.strftime('%m/%d %H:%M')} {direction} dev={deviation_sd:+.2f}s "
          f"rsi={rsi_val:.1f} adx={adx_val:.1f} rvol={rvol_val:.1f} slope={slope:.4f}{sma_tag}")

print(f"\n=== Gate Block Summary (Feb 2025, 5m bars) ===")
print(f"Total bars processed: {len(bars_df):,}")
print(f"Bars with valid VWAP: {sum(len(v) for v in gate_values.values()) // len(gate_values):,}")
for gate, count in gate_blocks.most_common():
    print(f"  {gate:<45s}: {count:>6,} bars blocked")
print(f"  {'SIGNALS PASSED':<45s}: {signals_generated:>6,}")

print(f"\n=== Signal Value Distributions ===")
for name, vals in gate_values.items():
    if vals:
        arr = np.array(vals)
        print(f"  {name:<15s}: min={arr.min():.3f}  p25={np.percentile(arr, 25):.3f}  "
              f"median={np.median(arr):.3f}  p75={np.percentile(arr, 75):.3f}  "
              f"max={arr.max():.3f}  p95={np.percentile(arr, 95):.3f}")

# Threshold analysis
dev_arr = np.array(gate_values["deviation_sd"])
rsi_arr = np.array(gate_values["rsi"])
adx_arr = np.array(gate_values["adx"])
rvol_arr = np.array(gate_values["rvol"])
print(f"\n=== Threshold Analysis ===")
print(f"  Bars with |dev| >= 2.0s: {np.sum(dev_arr >= 2.0):,} / {len(dev_arr):,} ({np.mean(dev_arr >= 2.0)*100:.1f}%)")
print(f"  Bars with |dev| >= 1.5s: {np.sum(dev_arr >= 1.5):,} / {len(dev_arr):,} ({np.mean(dev_arr >= 1.5)*100:.1f}%)")
print(f"  Bars with RSI < 10 or > 90: {np.sum((rsi_arr < 10) | (rsi_arr > 90)):,}")
print(f"  Bars with RSI < 20 or > 80: {np.sum((rsi_arr < 20) | (rsi_arr > 80)):,}")
print(f"  Bars with ADX < 20: {np.sum(adx_arr < 20):,} / {len(adx_arr):,} ({np.mean(adx_arr < 20)*100:.1f}%)")
print(f"  Bars with ADX < 25: {np.sum(adx_arr < 25):,} / {len(adx_arr):,} ({np.mean(adx_arr < 25)*100:.1f}%)")
print(f"  Bars with rvol >= 1.5: {np.sum(rvol_arr >= 1.5):,} / {len(rvol_arr):,} ({np.mean(rvol_arr >= 1.5)*100:.1f}%)")
print(f"  Bars with rvol >= 1.0: {np.sum(rvol_arr >= 1.0):,} / {len(rvol_arr):,} ({np.mean(rvol_arr >= 1.0)*100:.1f}%)")
