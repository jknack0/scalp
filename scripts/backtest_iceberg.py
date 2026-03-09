#!/usr/bin/env python3
"""Backtest the L2 book strategy (iceberg + absorption) on raw MBP-10 data.

Replays every tick from DataBento .dbn.zst files, detecting icebergs and
absorption events, then trading with detected institutional S/R levels.

Usage:
    python scripts/backtest_iceberg.py
    python scripts/backtest_iceberg.py --target-ticks 12 --stop-ticks 6
    python scripts/backtest_iceberg.py --min-appearances 30 --consumed-threshold 100
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.l2_replay import L2ReplayConfig, L2ReplayEngine
from src.strategies.base import Direction
from src.strategies.iceberg_strategy import L2Strategy, L2StrategyConfig


def main():
    parser = argparse.ArgumentParser(
        description="Backtest L2 book strategy on MBP-10 data"
    )

    # Strategy params
    parser.add_argument(
        "--target-ticks", type=int, default=8,
        help="Target distance in ticks (default: 8 = $10)",
    )
    parser.add_argument(
        "--stop-ticks", type=int, default=8,
        help="Stop distance beyond level in ticks (default: 8 = $10)",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.3,
        help="Minimum iceberg confidence to trade (default: 0.3)",
    )
    parser.add_argument(
        "--min-appearances", type=int, default=10,
        help="Min L2 snapshots a level must persist (default: 10)",
    )
    parser.add_argument(
        "--consumed-threshold", type=int, default=50,
        help="Min contracts consumed at level for iceberg (default: 50)",
    )
    parser.add_argument(
        "--test-threshold", type=int, default=5,
        help="Min tests for absorption signal (default: 5)",
    )
    parser.add_argument(
        "--min-absorbed", type=int, default=30,
        help="Min contracts absorbed for absorption-only signal (default: 30)",
    )
    parser.add_argument(
        "--max-signals", type=int, default=5,
        help="Max signals per day (default: 5)",
    )
    parser.add_argument(
        "--expiry", type=int, default=300,
        help="Max hold time in seconds (default: 300)",
    )
    parser.add_argument(
        "--l2-sample-ms", type=int, default=100,
        help="L2 sampling rate for detector in ms (default: 100)",
    )

    # Engine params
    parser.add_argument(
        "--l2-dir", default="data/l2_rth",
        help="Directory containing L2 data (.parquet or .dbn.zst)",
    )
    parser.add_argument(
        "--dbn", action="store_true",
        help="Use raw DBN files instead of pre-filtered Parquet",
    )
    parser.add_argument(
        "--slippage", type=int, default=1,
        help="Slippage in ticks per side (default: 1)",
    )
    parser.add_argument(
        "--progress", type=int, default=5_000_000,
        help="Print progress every N messages",
    )

    args = parser.parse_args()

    # Build strategy
    strat_cfg = L2StrategyConfig(
        target_ticks=args.target_ticks,
        stop_ticks=args.stop_ticks,
        min_iceberg_confidence=args.min_confidence,
        min_appearances=args.min_appearances,
        consumed_threshold=args.consumed_threshold,
        test_threshold=args.test_threshold,
        min_absorbed_size=args.min_absorbed,
        max_signals_per_day=args.max_signals,
        expiry_seconds=args.expiry,
        l2_sample_ms=args.l2_sample_ms,
    )
    strategy = L2Strategy(strat_cfg)

    # Build engine config
    use_parquet = not args.dbn
    engine_cfg = L2ReplayConfig(
        strategy=strategy,
        dbn_dir=args.l2_dir if not use_parquet else args.l2_dir,
        use_parquet=use_parquet,
        slippage_ticks=args.slippage,
        progress_interval=args.progress,
    )

    print(f"\n{'=' * 70}")
    print(f"L2 BOOK STRATEGY BACKTEST (iceberg + absorption)")
    print(f"{'=' * 70}")
    print(f"  L2 data:          {args.l2_dir}")
    print(f"  Target:           {args.target_ticks} ticks (${args.target_ticks * 1.25:.2f})")
    print(f"  Stop:             {args.stop_ticks} ticks (${args.stop_ticks * 1.25:.2f})")
    print(f"  Min iceberg conf: {args.min_confidence}")
    print(f"  Min appearances:  {args.min_appearances}")
    print(f"  Consumed thresh:  {args.consumed_threshold}")
    print(f"  Test threshold:   {args.test_threshold}")
    print(f"  Min absorbed:     {args.min_absorbed}")
    print(f"  Max signals/day:  {args.max_signals}")
    print(f"  Expiry:           {args.expiry}s")
    print(f"  L2 sample rate:   {args.l2_sample_ms}ms")
    print(f"  Slippage:         {args.slippage} tick/side")
    print(f"{'=' * 70}\n")

    t0 = time.perf_counter()
    engine = L2ReplayEngine()
    result = engine.run(engine_cfg)
    elapsed = time.perf_counter() - t0

    # Print results
    m = result.metrics
    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"  Runtime:        {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Trades:         {m.total_trades}")
    print(f"  Win rate:       {m.win_rate:.1%}")
    print(f"  Net PnL:        ${m.net_pnl:,.2f}")
    print(f"  Gross PnL:      ${m.gross_pnl:,.2f}")
    print(f"  Sharpe:         {m.sharpe_ratio:.3f}")
    print(f"  Sortino:        {m.sortino_ratio:.3f}")
    print(f"  Profit factor:  {m.profit_factor:.2f}")
    print(f"  Max drawdown:   {m.max_drawdown_pct:.1f}%")
    print(f"  Avg win:        ${m.avg_win:.2f}")
    print(f"  Avg loss:       ${m.avg_loss:.2f}")
    print(f"  Best trade:     ${m.best_trade:.2f}")
    print(f"  Worst trade:    ${m.worst_trade:.2f}")
    print(f"  Commission:     ${m.total_commission:.2f}")

    # Exit reason breakdown
    if result.trades:
        reasons = Counter(t.exit_reason for t in result.trades)
        print(f"\n  Exit reasons:")
        for reason, count in reasons.most_common():
            pct = count / len(result.trades) * 100
            print(f"    {reason:15s}: {count:3d} ({pct:.0f}%)")

        # P&L by exit reason
        print(f"\n  PnL by exit reason:")
        for reason in reasons:
            reason_trades = [t for t in result.trades if t.exit_reason == reason]
            reason_pnl = sum(t.net_pnl for t in reason_trades)
            print(f"    {reason:15s}: ${reason_pnl:>9,.2f}")

        # Signal type breakdown
        types = Counter(t.metadata.get("signal_type", "unknown") for t in result.trades)
        print(f"\n  Signal types:")
        for stype, count in types.most_common():
            type_trades = [t for t in result.trades if t.metadata.get("signal_type") == stype]
            type_pnl = sum(t.net_pnl for t in type_trades)
            type_wr = sum(1 for t in type_trades if t.net_pnl > 0) / len(type_trades)
            print(f"    {stype:25s}: {count:3d} trades, WR={type_wr:.1%}, PnL=${type_pnl:>9,.2f}")

        # Direction breakdown
        longs = [t for t in result.trades if t.direction == Direction.LONG]
        shorts = [t for t in result.trades if t.direction == Direction.SHORT]
        print(f"\n  Direction breakdown:")
        if longs:
            long_pnl = sum(t.net_pnl for t in longs)
            long_wr = sum(1 for t in longs if t.net_pnl > 0) / len(longs)
            print(f"    LONG:  {len(longs):3d} trades, WR={long_wr:.1%}, PnL=${long_pnl:>9,.2f}")
        if shorts:
            short_pnl = sum(t.net_pnl for t in shorts)
            short_wr = sum(1 for t in shorts if t.net_pnl > 0) / len(shorts)
            print(f"    SHORT: {len(shorts):3d} trades, WR={short_wr:.1%}, PnL=${short_pnl:>9,.2f}")

    print(f"\n{'=' * 70}\n")

    # Save results
    out_dir = Path("results") / "l2_book"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now().strftime("%Y-%m-%d")
    out_path = out_dir / f"backtest_{run_date}.json"
    if out_path.exists():
        i = 2
        while (out_dir / f"backtest_{run_date}_{i}.json").exists():
            i += 1
        out_path = out_dir / f"backtest_{run_date}_{i}.json"

    output = {
        "strategy": "l2_book",
        "date": run_date,
        "runtime_seconds": round(elapsed, 1),
        "params": {
            "target_ticks": args.target_ticks,
            "stop_ticks": args.stop_ticks,
            "min_iceberg_confidence": args.min_confidence,
            "min_appearances": args.min_appearances,
            "consumed_threshold": args.consumed_threshold,
            "test_threshold": args.test_threshold,
            "min_absorbed_size": args.min_absorbed,
            "max_signals_per_day": args.max_signals,
            "expiry_seconds": args.expiry,
            "l2_sample_ms": args.l2_sample_ms,
            "slippage_ticks": args.slippage,
        },
        "metrics": {
            "trades": m.total_trades,
            "win_rate": m.win_rate,
            "net_pnl": m.net_pnl,
            "gross_pnl": m.gross_pnl,
            "sharpe": m.sharpe_ratio,
            "sortino": m.sortino_ratio,
            "profit_factor": m.profit_factor,
            "max_drawdown_pct": m.max_drawdown_pct,
            "avg_win": m.avg_win,
            "avg_loss": m.avg_loss,
        },
        "trades": [
            {
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "direction": t.direction.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "net_pnl": t.net_pnl,
                "exit_reason": t.exit_reason,
                "signal_type": t.metadata.get("signal_type", ""),
                "hold_seconds": t.metadata.get("hold_seconds", 0),
                "level_price": t.metadata.get("level_price", 0),
                "level_side": t.metadata.get("level_side", ""),
                "confidence": t.metadata.get("confidence", 0),
            }
            for t in result.trades
        ],
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
