"""Backtest TickDirectionPredictor as a standalone strategy.

Runs vectorized inference on historical bars, generates bracket orders
(TP/SL), and simulates fills with slippage + commission.

Usage:
    python -u scripts/backtest/tick_predictor_standalone.py \
        --start 2025-01-01 --end 2026-03-01 \
        --tp-ticks 24 --sl-ticks 12 \
        --threshold 0.60 --cooldown 60 \
        2>&1 | tee logs/tick_predictor_backtest.log

    # Sweep over thresholds and cooldowns
    python -u scripts/backtest/tick_predictor_standalone.py \
        --start 2025-01-01 --end 2026-03-01 \
        --tp-ticks 24 --sl-ticks 12 --sweep \
        2>&1 | tee logs/tick_predictor_sweep.log
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time as _time
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np
import polars as pl

from src.core.logging import configure_logging, get_logger
from src.signals.tick_predictor.features.feature_builder import FEATURE_NAMES

logger = get_logger("tick_predictor.backtest")

MES_TICK_SIZE = 0.25
MES_TICK_VALUE = 1.25  # $1.25 per tick
MES_POINT_VALUE = 5.0  # $5.00 per point
COMMISSION_RT = 0.59   # Tradovate Free round-trip


@dataclass(frozen=True)
class BacktestParams:
    tp_ticks: int = 24          # Take-profit in ticks
    sl_ticks: int = 12          # Stop-loss in ticks
    threshold: float = 0.60     # Min confidence to enter
    cooldown_bars: int = 60     # Min bars between entries
    max_position: int = 1       # Max concurrent positions
    slippage_ticks: float = 1.0 # Slippage on market orders (stops, expiry)


@dataclass
class SimTrade:
    entry_bar: int
    exit_bar: int
    direction: str             # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    exit_reason: str           # "target", "stop", "expiry", "session_close"
    confidence: float
    gross_pnl: float
    slippage_cost: float
    commission: float
    net_pnl: float
    bars_held: int


@dataclass
class BacktestResult:
    params: dict
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    net_pnl: float
    gross_pnl: float
    total_commission: float
    total_slippage: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    avg_bars_held: float
    avg_trades_per_day: float
    exit_reasons: dict[str, int]
    best_trade: float
    worst_trade: float


def load_model_and_calibrator(model_path: str, calibrator_path: str):
    """Load LightGBM model and temperature calibrator."""
    import lightgbm as lgb
    from src.signals.tick_predictor.model.calibrator import TemperatureCalibrator

    model = lgb.Booster(model_file=model_path)
    assert model.num_feature() == len(FEATURE_NAMES), (
        f"Model expects {model.num_feature()} features, got {len(FEATURE_NAMES)}"
    )

    calibrator = TemperatureCalibrator()
    calibrator.load(calibrator_path)

    return model, calibrator


def load_features(start_date: str, end_date: str, tp_ticks: int, sl_ticks: int) -> pl.DataFrame:
    """Load cached features or raise if not found."""
    # Try v4 cache first
    suffix = f"_v4_h300_tp{tp_ticks}_sl{sl_ticks}_rth"
    path = Path(f"data/tick_predictor/features_{start_date}_{end_date}{suffix}.parquet")
    if path.exists():
        print(f"    [Cache HIT] {path}")
        return pl.read_parquet(path)
    raise FileNotFoundError(
        f"No cached features at {path}. Run train.py first to generate features."
    )


def run_backtest(
    features_df: pl.DataFrame,
    model,
    calibrator,
    params: BacktestParams,
    horizon_bars: int = 300,
) -> tuple[BacktestResult, list[SimTrade]]:
    """Run vectorized backtest on pre-computed features."""
    t0 = _time.perf_counter()

    # ── Vectorized inference ─────────────────────────────────────
    X = features_df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    timestamps_ns = features_df["timestamp_ns"].to_numpy()
    n_bars = len(X)

    print(f"    Running inference on {n_bars:,} bars...")
    raw_proba = model.predict(X)  # [N, 3] — DOWN, FLAT, UP
    if raw_proba.ndim == 1:
        raw_proba = raw_proba.reshape(-1, 3)

    cal_proba = calibrator.predict_proba_calibrated(raw_proba)
    # Columns: 0=DOWN, 1=FLAT, 2=UP

    p_down = cal_proba[:, 0]
    p_up = cal_proba[:, 2]

    # We need close prices for entry/exit simulation
    # Load bars to get OHLC — join by timestamp_ns
    # Actually, we need the original bars. Let's load them.
    print(f"    Loading bar prices for simulation...")
    bars_df = _load_bars_for_sim(features_df)
    close = bars_df["close"].to_numpy()
    high = bars_df["high"].to_numpy()
    low = bars_df["low"].to_numpy()

    tp_pts = params.tp_ticks * MES_TICK_SIZE
    sl_pts = params.sl_ticks * MES_TICK_SIZE
    slippage_pts = params.slippage_ticks * MES_TICK_SIZE

    # ── Trade simulation ─────────────────────────────────────────
    print(f"    Simulating trades (threshold={params.threshold}, cooldown={params.cooldown_bars})...")

    trades: list[SimTrade] = []
    last_entry_bar = -params.cooldown_bars  # allow first trade immediately
    open_positions: list[dict] = []

    # Detect session boundaries (date changes in timestamps)
    bar_dates = (
        pl.Series("ts", timestamps_ns)
        .cast(pl.Datetime("ns"))
        .dt.date()
        .to_numpy()
    )

    for i in range(n_bars):
        # ── Close positions on session change ────────────────────
        if i > 0 and bar_dates[i] != bar_dates[i - 1]:
            for pos in open_positions:
                exit_price = close[i]
                if pos["direction"] == "LONG":
                    exit_price -= slippage_pts
                    gross = (exit_price - pos["entry_price"]) * MES_POINT_VALUE
                else:
                    exit_price += slippage_pts
                    gross = (pos["entry_price"] - exit_price) * MES_POINT_VALUE

                slip_cost = params.slippage_ticks * MES_TICK_VALUE
                net = gross - COMMISSION_RT - slip_cost
                trades.append(SimTrade(
                    entry_bar=pos["entry_bar"], exit_bar=i,
                    direction=pos["direction"], entry_price=pos["entry_price"],
                    exit_price=exit_price, exit_reason="session_close",
                    confidence=pos["confidence"],
                    gross_pnl=gross, slippage_cost=slip_cost,
                    commission=COMMISSION_RT, net_pnl=net,
                    bars_held=i - pos["entry_bar"],
                ))
            open_positions.clear()

        # ── Check exits on open positions ────────────────────────
        closed_indices = []
        for j, pos in enumerate(open_positions):
            bars_held = i - pos["entry_bar"]

            if pos["direction"] == "LONG":
                stop_hit = low[i] <= pos["stop_price"]
                target_hit = high[i] >= pos["target_price"]
            else:
                stop_hit = high[i] >= pos["stop_price"]
                target_hit = low[i] <= pos["target_price"]

            # Ambiguity: if both hit, assume stop (conservative)
            exit_reason = None
            exit_price = 0.0

            if stop_hit and target_hit:
                exit_reason = "stop"
            elif stop_hit:
                exit_reason = "stop"
            elif target_hit:
                exit_reason = "target"
            elif bars_held >= horizon_bars:
                exit_reason = "expiry"

            if exit_reason:
                if exit_reason == "target":
                    exit_price = pos["target_price"]  # limit, no slippage
                    slip_cost = 0.0
                elif exit_reason == "stop":
                    exit_price = pos["stop_price"]
                    if pos["direction"] == "LONG":
                        exit_price -= slippage_pts
                    else:
                        exit_price += slippage_pts
                    slip_cost = params.slippage_ticks * MES_TICK_VALUE
                else:  # expiry
                    exit_price = close[i]
                    if pos["direction"] == "LONG":
                        exit_price -= slippage_pts
                    else:
                        exit_price += slippage_pts
                    slip_cost = params.slippage_ticks * MES_TICK_VALUE

                if pos["direction"] == "LONG":
                    gross = (exit_price - pos["entry_price"]) * MES_POINT_VALUE
                else:
                    gross = (pos["entry_price"] - exit_price) * MES_POINT_VALUE

                net = gross - COMMISSION_RT - slip_cost
                trades.append(SimTrade(
                    entry_bar=pos["entry_bar"], exit_bar=i,
                    direction=pos["direction"], entry_price=pos["entry_price"],
                    exit_price=exit_price, exit_reason=exit_reason,
                    confidence=pos["confidence"],
                    gross_pnl=gross, slippage_cost=slip_cost,
                    commission=COMMISSION_RT, net_pnl=net,
                    bars_held=i - pos["entry_bar"],
                ))
                closed_indices.append(j)

        for j in reversed(closed_indices):
            open_positions.pop(j)

        # ── Check for new entry ──────────────────────────────────
        if len(open_positions) >= params.max_position:
            continue
        if i - last_entry_bar < params.cooldown_bars:
            continue
        # Don't enter on last horizon_bars of data
        if i >= n_bars - horizon_bars:
            continue

        direction = None
        confidence = 0.0

        if p_up[i] >= params.threshold and p_up[i] > p_down[i]:
            direction = "LONG"
            confidence = float(p_up[i])
        elif p_down[i] >= params.threshold and p_down[i] > p_up[i]:
            direction = "SHORT"
            confidence = float(p_down[i])

        if direction:
            entry_price = close[i]
            if direction == "LONG":
                target_price = entry_price + tp_pts
                stop_price = entry_price - sl_pts
            else:
                target_price = entry_price - tp_pts
                stop_price = entry_price + sl_pts

            open_positions.append({
                "entry_bar": i,
                "entry_price": entry_price,
                "target_price": target_price,
                "stop_price": stop_price,
                "direction": direction,
                "confidence": confidence,
            })
            last_entry_bar = i

    elapsed = _time.perf_counter() - t0

    # ── Compute metrics ──────────────────────────────────────────
    if not trades:
        print(f"    No trades generated!")
        return BacktestResult(
            params=asdict(params), total_trades=0, win_rate=0, avg_win=0,
            avg_loss=0, profit_factor=0, net_pnl=0, gross_pnl=0,
            total_commission=0, total_slippage=0, sharpe=0, sortino=0,
            max_drawdown_pct=0, avg_bars_held=0, avg_trades_per_day=0,
            exit_reasons={}, best_trade=0, worst_trade=0,
        ), trades

    net_pnls = np.array([t.net_pnl for t in trades])
    gross_pnls = np.array([t.gross_pnl for t in trades])
    wins = net_pnls[net_pnls > 0]
    losses = net_pnls[net_pnls <= 0]

    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
    pf = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else float("inf")

    # Daily P&L for Sharpe/Sortino
    trade_dates = [bar_dates[t.entry_bar] for t in trades]
    daily_pnl = {}
    for d, pnl in zip(trade_dates, net_pnls):
        daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl
    daily_returns = np.array(list(daily_pnl.values()))

    sharpe = 0.0
    sortino = 0.0
    if len(daily_returns) > 1:
        std = float(np.std(daily_returns, ddof=1))
        if std > 0:
            sharpe = float(np.mean(daily_returns)) / std * np.sqrt(252)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0:
            down_std = float(np.std(downside, ddof=1))
            if down_std > 0:
                sortino = float(np.mean(daily_returns)) / down_std * np.sqrt(252)

    # Max drawdown
    equity = np.cumsum(net_pnls)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak)
    max_dd_pct = float(np.min(dd) / (peak[np.argmin(dd)] + 1e-9) * 100) if len(dd) > 0 else 0

    # Exit reason counts
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    n_days = len(daily_pnl)
    avg_trades_per_day = len(trades) / n_days if n_days > 0 else 0

    result = BacktestResult(
        params=asdict(params),
        total_trades=len(trades),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=pf,
        net_pnl=float(np.sum(net_pnls)),
        gross_pnl=float(np.sum(gross_pnls)),
        total_commission=COMMISSION_RT * len(trades),
        total_slippage=float(sum(t.slippage_cost for t in trades)),
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_dd_pct,
        avg_bars_held=float(np.mean([t.bars_held for t in trades])),
        avg_trades_per_day=avg_trades_per_day,
        exit_reasons=exit_reasons,
        best_trade=float(np.max(net_pnls)),
        worst_trade=float(np.min(net_pnls)),
    )

    print(f"    Backtest done: {len(trades)} trades in {elapsed:.1f}s")
    return result, trades


def _load_bars_for_sim(features_df: pl.DataFrame) -> pl.DataFrame:
    """Load bar OHLC data matching the feature timestamps."""
    import datetime as dt

    timestamps_ns = features_df["timestamp_ns"].to_numpy()
    min_ts = int(timestamps_ns.min())
    max_ts = int(timestamps_ns.max())

    min_dt = dt.datetime.fromtimestamp(min_ts / 1e9, tz=dt.timezone.utc)
    max_dt = dt.datetime.fromtimestamp(max_ts / 1e9, tz=dt.timezone.utc)

    # Try L1 data first, then OHLCV
    years = range(min_dt.year, max_dt.year + 1)
    l1_paths = [f"data/l1/year={y}/data.parquet" for y in years
                if Path(f"data/l1/year={y}/data.parquet").exists()]

    if l1_paths:
        ticks = (
            pl.scan_parquet(l1_paths)
            .filter(
                (pl.col("timestamp") >= min_dt)
                & (pl.col("timestamp") <= max_dt + dt.timedelta(seconds=1))
            )
            .sort("timestamp")
            .collect()
        )
        bars = (
            ticks.with_columns(pl.col("timestamp").dt.truncate("1s").alias("bar_ts"))
            .group_by("bar_ts")
            .agg([
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
            ])
            .sort("bar_ts")
            .with_columns(pl.col("bar_ts").dt.epoch("ns").alias("timestamp_ns"))
        )
    else:
        paths = [f"data/parquet/year={y}/data.parquet" for y in years
                 if Path(f"data/parquet/year={y}/data.parquet").exists()]
        bars = (
            pl.scan_parquet(paths)
            .filter(
                (pl.col("timestamp") >= min_dt)
                & (pl.col("timestamp") <= max_dt + dt.timedelta(seconds=1))
            )
            .select(["timestamp", "open", "high", "low", "close"])
            .sort("timestamp")
            .collect()
            .with_columns(pl.col("timestamp").dt.epoch("ns").alias("timestamp_ns"))
        )

    # Join to features to get aligned OHLC
    result = features_df.select("timestamp_ns").join(
        bars.select(["timestamp_ns", "open", "high", "low", "close"]),
        on="timestamp_ns",
        how="left",
    )
    # Forward-fill any missing bars
    for col in ["open", "high", "low", "close"]:
        result = result.with_columns(pl.col(col).fill_null(strategy="forward"))

    return result


def print_result(result: BacktestResult) -> None:
    """Pretty-print backtest results."""
    p = result.params
    print(f"\n    {'-'*50}")
    print(f"    Threshold: {p['threshold']}  Cooldown: {p['cooldown_bars']}s  "
          f"TP/SL: {p['tp_ticks']}/{p['sl_ticks']} ticks")
    print(f"    {'-'*50}")
    print(f"    Trades:        {result.total_trades:,}")
    print(f"    Win rate:      {result.win_rate*100:.1f}%")
    print(f"    Avg win:       ${result.avg_win:.2f}")
    print(f"    Avg loss:      ${result.avg_loss:.2f}")
    print(f"    Profit factor: {result.profit_factor:.2f}")
    print(f"    Net P&L:       ${result.net_pnl:,.2f}")
    print(f"    Gross P&L:     ${result.gross_pnl:,.2f}")
    print(f"    Commission:    ${result.total_commission:,.2f}")
    print(f"    Slippage:      ${result.total_slippage:,.2f}")
    print(f"    Sharpe:        {result.sharpe:.2f}")
    print(f"    Sortino:       {result.sortino:.2f}")
    print(f"    Max DD:        {result.max_drawdown_pct:.1f}%")
    print(f"    Avg bars held: {result.avg_bars_held:.0f}")
    print(f"    Trades/day:    {result.avg_trades_per_day:.1f}")
    print(f"    Best trade:    ${result.best_trade:.2f}")
    print(f"    Worst trade:   ${result.worst_trade:.2f}")
    print(f"    Exit reasons:  {result.exit_reasons}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest TickDirectionPredictor")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--tp-ticks", type=int, default=24)
    parser.add_argument("--sl-ticks", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--cooldown", type=int, default=60)
    parser.add_argument("--max-position", type=int, default=1)
    parser.add_argument("--slippage", type=float, default=1.0)
    parser.add_argument("--model-path", default="models/tick_predictor/lgbm_latest.txt")
    parser.add_argument("--calibrator-path", default="models/tick_predictor/calibrator_latest.pkl")
    parser.add_argument("--sweep", action="store_true", help="Sweep thresholds and cooldowns")
    args = parser.parse_args()

    configure_logging(log_level="INFO")

    print(f"\n{'='*70}")
    print(f"  TICK PREDICTOR BACKTEST")
    print(f"  Data: {args.start} to {args.end}")
    print(f"  TP/SL: {args.tp_ticks}/{args.sl_ticks} ticks")
    print(f"{'='*70}\n")

    # Load model
    print("  Loading model...")
    model, calibrator = load_model_and_calibrator(args.model_path, args.calibrator_path)
    print(f"    Model: {args.model_path} ({model.num_trees()} trees)")

    # Load features
    print("  Loading features...")
    features_df = load_features(args.start, args.end, args.tp_ticks, args.sl_ticks)
    print(f"    Features: {len(features_df):,} rows")

    results_dir = Path("results/tick_predictor")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
        cooldowns = [0, 30, 60, 120, 300]

        print(f"\n  Sweeping {len(thresholds)} thresholds x {len(cooldowns)} cooldowns "
              f"= {len(thresholds)*len(cooldowns)} configs\n")

        all_results = []
        for thresh, cd in itertools.product(thresholds, cooldowns):
            params = BacktestParams(
                tp_ticks=args.tp_ticks, sl_ticks=args.sl_ticks,
                threshold=thresh, cooldown_bars=cd,
                max_position=args.max_position, slippage_ticks=args.slippage,
            )
            result, _ = run_backtest(features_df, model, calibrator, params)
            print_result(result)
            all_results.append(asdict(result))

        # Sort by net P&L
        all_results.sort(key=lambda x: x["net_pnl"], reverse=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = results_dir / f"backtest_sweep_{ts}.json"
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n{'='*70}")
        print(f"  SWEEP COMPLETE")
        print(f"  Best config: threshold={all_results[0]['params']['threshold']}, "
              f"cooldown={all_results[0]['params']['cooldown_bars']}")
        print(f"  Best Net P&L: ${all_results[0]['net_pnl']:,.2f}")
        print(f"  Best Sharpe:  {all_results[0]['sharpe']:.2f}")
        print(f"  Results: {path}")
        print(f"{'='*70}\n")
    else:
        params = BacktestParams(
            tp_ticks=args.tp_ticks, sl_ticks=args.sl_ticks,
            threshold=args.threshold, cooldown_bars=args.cooldown,
            max_position=args.max_position, slippage_ticks=args.slippage,
        )
        result, trades = run_backtest(features_df, model, calibrator, params)
        print_result(result)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = results_dir / f"backtest_{ts}.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        print(f"\n  Result saved -> {path}")


if __name__ == "__main__":
    main()
