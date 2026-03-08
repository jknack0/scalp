#!/bin/bash
# Run Phase 4 validation for all strategies
# 4 original strategies: 10 years (2015-2024) on standard parquet
# OBI: 1 year (2025) on L1-enriched parquet

set -e

COMMON="--capital 10000 --workers 4 --use-rth-bars -q"
SCRIPT="python scripts/generate_phase4_report.py"

echo "=== ORB (10yr) ==="
$SCRIPT --start 2015-01-01 --end 2024-12-31 $COMMON --strategy orb

echo ""
echo "=== VWAP (10yr) ==="
$SCRIPT --start 2015-01-01 --end 2024-12-31 $COMMON --strategy vwap

echo ""
echo "=== CVD (10yr) ==="
$SCRIPT --start 2015-01-01 --end 2024-12-31 $COMMON --strategy cvd

echo ""
echo "=== Vol Regime (10yr) ==="
$SCRIPT --start 2015-01-01 --end 2024-12-31 $COMMON --strategy vol_regime

echo ""
echo "=== OBI (1yr, L1 data) ==="
$SCRIPT --start 2025-01-01 --end 2025-12-31 $COMMON --rth-parquet-dir data/parquet_1m_l1 --strategy obi

echo ""
echo "=== ALL DONE ==="
