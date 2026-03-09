#!/usr/bin/env python3
"""Diagnose VPIN with daily reset on enriched 1s bars."""

import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.filters.vpin_monitor import VPINConfig, VPINMonitor

start = date(2025, 1, 1)
end = date(2025, 12, 31)

engine = BacktestEngine()
config = BacktestConfig(
    strategies=[], start_date=start, end_date=end,
    parquet_dir="data/parquet_1s_enriched",
)
bars = engine._load_bars(config)
print(f"Loaded {len(bars)} bars\n")

# Test with daily reset
for bsize in [100, 500, 1000]:
    for nb in [10, 20, 50]:
        vm = VPINMonitor(config=VPINConfig(bucket_size=bsize, n_buckets=nb), persist=False)
        trending = 0
        mean_rev = 0
        undef = 0
        vpins = []

        for row in bars.iter_rows(named=True):
            buy_v = float(row.get("aggressive_buy_vol", 0) or 0)
            sell_v = float(row.get("aggressive_sell_vol", 0) or 0)
            ts = row["timestamp"]
            if hasattr(ts, 'replace') and getattr(ts, 'tzinfo', None):
                ts = ts.replace(tzinfo=None)

            if buy_v + sell_v > 0:
                vm.on_bar_l1(buy_vol=buy_v, sell_vol=sell_v, timestamp=ts)

            regime, vpin = vm.get_regime()
            if regime == "trending":
                trending += 1
            elif regime == "mean_reversion":
                mean_rev += 1
            else:
                undef += 1

            if len(vpins) < 10000 and regime != "undefined":
                vpins.append(vpin)

        total = trending + mean_rev + undef
        vpins.sort()
        n = len(vpins)
        p50 = vpins[n//2] if vpins else 0
        p95 = vpins[int(n*0.95)] if vpins else 0
        p05 = vpins[int(n*0.05)] if vpins else 0

        print(f"  bucket={bsize:5d} n={nb:2d}  trending={trending/total*100:5.1f}%  mean_rev={mean_rev/total*100:5.1f}%  undef={undef/total*100:5.1f}%  p05={p05:.3f} p50={p50:.3f} p95={p95:.3f}")
        sys.stdout.flush()
