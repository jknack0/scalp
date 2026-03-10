#!/usr/bin/env python3
"""Generate Phase 4 validation report for all strategies.

Runs CPCV, DSR, and WFA for each strategy, then applies the DecisionEngine
to produce a markdown report and locked-params YAML.

Results are saved incrementally per strategy, so you can run one at a time
and the report will incorporate all previously completed strategies.

Usage:
    python scripts/report/phase4.py --start 2020-01-01 --end 2024-01-01
    python scripts/report/phase4.py --start 2020-01-01 --end 2024-01-01 --strategy orb
    python scripts/report/phase4.py --start 2020-01-01 --end 2024-01-01 --strategy vwap -q
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

# Suppress logs early — before any imports trigger structlog/logging setup
_quiet = "-q" in sys.argv or "--quiet" in sys.argv
if _quiet:
    os.environ["LOG_LEVEL"] = "WARNING"
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger().setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import copy

from src.backtesting.cpcv import CPCVConfig, CPCVValidator
from src.backtesting.decision_engine import DecisionConfig, DecisionEngine, ValidationSummary
from src.backtesting.dsr import DSRConfig, DeflatedSharpeCalculator
from src.backtesting.engine import BacktestConfig
from src.backtesting.wfa import WFAConfig, WFARunner
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy

# Force structlog configuration before any logger is created
from src.core.logging import configure_logging
configure_logging(log_level="WARNING" if _quiet else "INFO", log_file=None)


STRATEGY_MAP = {
    "orb": ORBStrategy,
    "vwap": VWAPReversionStrategy,
}

_ORB_DEFAULT_CONFIG = {
    "strategy": {"strategy_id": "orb", "max_signals_per_day": 1},
    "orb": {"target_multiplier": 0.5, "volume_multiplier": 1.5, "max_signal_time": "10:30",
            "expiry_minutes": 60, "require_vwap_alignment": True},
    "filter": {"min_score": 0, "require_direction_agreement": False, "signals": {}},
    "exit": {"time_stop_minutes": 60},
}

_VWAP_DEFAULT_CONFIG = {
    "strategy": {"strategy_id": "vwap_reversion", "max_signals_per_day": 4},
    "vwap": {"entry_sd_reversion": 1.5, "stop_sd": 2.0, "pullback_entry_sd": 0.5,
             "mode_cooldown_bars": 5, "expiry_minutes": 30},
    "filter": {"min_score": 0, "require_direction_agreement": False, "signals": {}},
    "exit": {"target": "vwap", "stop_ticks": 8, "time_stop_minutes": 30},
}

_DEFAULT_CONFIGS = {"orb": _ORB_DEFAULT_CONFIG, "vwap": _VWAP_DEFAULT_CONFIG}

from src.models.hmm_regime import RegimeState

# HMM state presets — curated combos per strategy style
_HMM_BREAKOUT_PRESETS = [
    [RegimeState.HIGH_VOL_UP, RegimeState.HIGH_VOL_DOWN, RegimeState.BREAKOUT],
    [RegimeState.BREAKOUT],
    [RegimeState.HIGH_VOL_UP, RegimeState.HIGH_VOL_DOWN],
]
_HMM_REVERSION_PRESETS = [
    [RegimeState.LOW_VOL_RANGE, RegimeState.MEAN_REVERSION],
    [RegimeState.LOW_VOL_RANGE],
]
_HMM_PULLBACK_PRESETS = [
    [RegimeState.HIGH_VOL_UP, RegimeState.HIGH_VOL_DOWN],
    [RegimeState.HIGH_VOL_UP, RegimeState.HIGH_VOL_DOWN, RegimeState.BREAKOUT],
]

# Grid sizes target ~50-80 combos per strategy to keep WFA runtime manageable.
# Each combo runs a full backtest (~30s), × 9 WFA cycles × 4 strategies.
PARAM_GRIDS = {
    # ORB: 4 × 2 × 2 × 2 × 3 = 96 combos
    "orb": {
        "target_multiplier": [0.4, 0.5, 0.6, 0.7],            # signal geometry
        "volume_multiplier": [1.5, 2.0],                        # entry filter
        "max_signal_time": ["10:30", "11:00"],                   # time filter
        "min_confidence": [0.5, 0.65],                           # confidence gate
        "require_hmm_states": _HMM_BREAKOUT_PRESETS,             # HMM regime (3)
    },
    # VWAP: 3 × 2 × 3 × 2 × 2 = 72 combos
    "vwap": {
        "entry_sd_reversion": [1.8, 2.0, 2.5],                  # entry threshold
        "flat_slope_threshold": [0.001, 0.003],                  # mode switching
        "stop_sd": [2.5, 3.0, 3.5],                             # stop sizing
        "min_session_age_minutes": [10, 20],                     # time filter
        "reversion_hmm_states": _HMM_REVERSION_PRESETS,          # HMM regime (2)
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Phase 4 validation report for all strategies"
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--strategy", default="all",
        choices=["all", *STRATEGY_MAP.keys()],
        help="Strategy to validate (default: all)",
    )
    parser.add_argument(
        "--capital", type=float, default=10_000.0, help="Initial capital (default 10000)"
    )
    parser.add_argument(
        "--output-dir", default="docs/phase4", help="Output directory (default docs/phase4)"
    )
    parser.add_argument(
        "--parquet-dir", default="data/parquet", help="Parquet data directory"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel worker processes (default: auto-detect)"
    )
    parser.add_argument(
        "--resample", default=None,
        help="Resample bars before replay (e.g. '5s', '15s', '1m'). Huge speedup.",
    )
    parser.add_argument(
        "--use-rth-bars", action="store_true",
        help="Use pre-built RTH bars (default: data/parquet_5s_rth)",
    )
    parser.add_argument(
        "--rth-parquet-dir", default="data/parquet_5s_rth",
        help="Directory for pre-built RTH bars (default: data/parquet_5s_rth)",
    )
    parser.add_argument(
        "--use-hmm", action="store_true",
        help="Enable HMM regime gating (requires trained model in models/hmm/v1)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress info logs, show only progress bars and summary"
    )
    return parser.parse_args()


def _make_strategy_factory(strategy_name, strategy_cls):
    """Create a strategy factory for CPCV."""
    def factory():
        return [strategy_cls(config=copy.deepcopy(_DEFAULT_CONFIGS[strategy_name]))]
    return factory


def _save_strategy_result(
    output_dir: Path,
    strategy_name: str,
    summary: ValidationSummary,
    locked_params: dict,
    daily_pnl: list[float],
    cpcv_result=None,
    dsr_result=None,
    wfa_result=None,
) -> None:
    """Save per-strategy results to JSON for incremental report building."""
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "strategy_id": summary.strategy_id,
        # ValidationSummary fields
        "summary": {
            "cpcv_pbo": summary.cpcv_pbo,
            "dsr": summary.dsr,
            "wfa_efficiency": summary.wfa_efficiency,
            "wfa_is_oos_correlation": summary.wfa_is_oos_correlation,
            "param_stability_score": summary.param_stability_score,
            "total_oos_trades": summary.total_oos_trades,
            "oos_sharpe": summary.oos_sharpe,
            "oos_win_rate": summary.oos_win_rate,
            "oos_profit_factor": summary.oos_profit_factor,
        },
        "locked_params": locked_params,
        "daily_pnl": daily_pnl,
    }

    # Full CPCV results
    if cpcv_result is not None:
        data["cpcv"] = {
            "pbo": cpcv_result.pbo,
            "n_paths": cpcv_result.n_paths,
            "oos_sharpes": cpcv_result.oos_sharpes,
            "is_sharpes": cpcv_result.is_sharpes,
            "oos_returns": list(cpcv_result.oos_returns),
            "is_returns": list(cpcv_result.is_returns),
            "avg_oos_sharpe": cpcv_result.avg_oos_sharpe,
            "avg_is_sharpe": cpcv_result.avg_is_sharpe,
            "sharpe_decay": cpcv_result.sharpe_decay,
            "verdict": cpcv_result.verdict,
        }

    # Full DSR results
    if dsr_result is not None:
        data["dsr_detail"] = {
            "observed_sharpe": dsr_result.observed_sharpe,
            "psr": dsr_result.psr,
            "expected_max_sharpe": dsr_result.expected_max_sharpe,
            "dsr": dsr_result.dsr,
            "n_trials": dsr_result.n_trials,
            "sample_size": dsr_result.sample_size,
            "skewness": dsr_result.skewness,
            "kurtosis": dsr_result.kurtosis,
            "sr_std_error": dsr_result.sr_std_error,
            "verdict": dsr_result.verdict,
        }

    # Full WFA results
    if wfa_result is not None:
        data["wfa"] = {
            "n_cycles": wfa_result.n_cycles,
            "efficiency_ratio": wfa_result.efficiency_ratio,
            "is_oos_correlation": wfa_result.is_oos_correlation,
            "avg_is_sharpe": wfa_result.avg_is_sharpe,
            "avg_oos_sharpe": wfa_result.avg_oos_sharpe,
            "param_drift": wfa_result.param_drift,
            "verdict": wfa_result.verdict,
            "cycles": [
                {
                    "cycle_id": c.cycle_id,
                    "train_start": str(c.train_start),
                    "train_end": str(c.train_end),
                    "test_start": str(c.test_start),
                    "test_end": str(c.test_end),
                    "best_params": c.best_params,
                    "is_sharpe": c.is_sharpe,
                    "oos_sharpe": c.oos_sharpe,
                    "is_trades": c.is_trades,
                    "oos_trades": c.oos_trades,
                    "oos_win_rate": c.oos_win_rate,
                    "oos_profit_factor": c.oos_profit_factor,
                    "grid_results": c.grid_results,
                }
                for c in wfa_result.cycles
            ],
        }

    path = results_dir / f"{strategy_name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"    Saved: {path}")


def _load_strategy_result(
    output_dir: Path, strategy_name: str
) -> tuple[ValidationSummary, dict, list[float]] | None:
    """Load a previously saved strategy result from JSON."""
    path = output_dir / "results" / f"{strategy_name}.json"
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    s = data["summary"]
    summary = ValidationSummary(
        strategy_id=data["strategy_id"],
        cpcv_pbo=s["cpcv_pbo"],
        dsr=s["dsr"],
        wfa_efficiency=s["wfa_efficiency"],
        wfa_is_oos_correlation=s["wfa_is_oos_correlation"],
        param_stability_score=s["param_stability_score"],
        total_oos_trades=s["total_oos_trades"],
        oos_sharpe=s["oos_sharpe"],
        oos_win_rate=s["oos_win_rate"],
        oos_profit_factor=s["oos_profit_factor"],
    )
    return summary, data["locked_params"], data["daily_pnl"]


def _run_strategy(
    strategy_name: str,
    strategy_cls,
    start: date,
    end: date,
    capital: float,
    parquet_dir: str,
    max_workers: int | None,
    n_trials: int,
    output_dir: Path,
    resample_freq: str | None = None,
    use_rth_bars: bool = False,
    rth_parquet_dir: str = "data/parquet_5s_rth",
    use_hmm: bool = False,
) -> tuple[ValidationSummary, dict, list[float]]:
    """Run CPCV + WFA + DSR for a single strategy and save results."""
    print(f"\n{'-' * 40}")
    print(f"Validating: {strategy_name}")
    print(f"{'-' * 40}")

    # Template config
    strat_config = copy.deepcopy(_DEFAULT_CONFIGS[strategy_name])

    # Load HMM / regime detector if requested
    regime_detector = None
    if use_hmm:
        from src.models.regime_detector import RegimeDetector
        try:
            regime_detector = RegimeDetector.load("models/hmm/v1")
            print("  HMM model loaded")
        except Exception as e:
            print(f"  HMM model not found, running without: {e}")

    template = BacktestConfig(
        strategies=[strategy_cls(config=strat_config, regime_detector=regime_detector)],
        start_date=start,
        end_date=end,
        initial_capital=capital,
        parquet_dir=parquet_dir,
        resample_freq=resample_freq,
        use_rth_bars=use_rth_bars,
        rth_parquet_dir=rth_parquet_dir,
        use_hmm=use_hmm,
    )

    # 1. CPCV
    print("  Running CPCV...")
    factory = _make_strategy_factory(strategy_name, strategy_cls)
    strat_path = f"{strategy_cls.__module__}.{strategy_cls.__name__}"
    cpcv_validator = CPCVValidator(
        factory, template,
        config_cls_path=strat_path,
        strategy_cls_path=strat_path,
    )
    cpcv_result = cpcv_validator.run(CPCVConfig(max_workers=max_workers))
    print(f"    PBO: {cpcv_result.pbo:.4f} ({cpcv_result.verdict})")

    # 2. WFA
    print("  Running WFA...")
    param_grid = dict(PARAM_GRIDS[strategy_name])
    # Strip HMM state params when not using HMM (they'd bias optimization)
    if not use_hmm:
        hmm_param_keys = [
            "require_hmm_states", "reversion_hmm_states", "pullback_hmm_states",
            "high_vol_hmm_states", "low_vol_hmm_states",
        ]
        for k in hmm_param_keys:
            param_grid.pop(k, None)
    combos = 1
    for v in param_grid.values():
        combos *= len(v)
    print(f"    Grid: {combos} combos ({', '.join(param_grid.keys())})")
    wfa_runner = WFARunner(strategy_name, param_grid, template)
    wfa_result = wfa_runner.run(WFAConfig(max_workers=max_workers))
    print(f"    Efficiency: {wfa_result.efficiency_ratio:.3f} ({wfa_result.verdict})")

    # 3. DSR (use actual daily PnL, not per-fold totals)
    print("  Computing DSR...")
    if cpcv_result.oos_daily_pnls:
        daily_returns = np.array(cpcv_result.oos_daily_pnls)
    elif cpcv_result.oos_returns:
        daily_returns = np.array(cpcv_result.oos_returns)
    else:
        daily_returns = np.array([0.0])
    dsr_result = DeflatedSharpeCalculator.compute(
        daily_returns, n_trials,
        strategy_id=strategy_name,
    )
    print(f"    DSR: {dsr_result.dsr:.4f} ({dsr_result.verdict})")

    # Build summary
    summary = DecisionEngine.from_results(cpcv_result, dsr_result, wfa_result)

    # Locked params = last WFA cycle's best params
    locked_params = wfa_result.cycles[-1].best_params if wfa_result.cycles else {}

    # Daily P&L (use actual daily PnL, not per-fold totals)
    daily_pnl = list(cpcv_result.oos_daily_pnls) if cpcv_result.oos_daily_pnls else list(cpcv_result.oos_returns)

    # Save immediately with full backtest results
    _save_strategy_result(
        output_dir, strategy_name, summary, locked_params, daily_pnl,
        cpcv_result=cpcv_result,
        dsr_result=dsr_result,
        wfa_result=wfa_result,
    )

    return summary, locked_params, daily_pnl


def _generate_combined_report(
    output_dir: Path,
    summaries: list[ValidationSummary],
    locked_params_map: dict[str, dict],
    daily_pnl_map: dict[str, np.ndarray],
) -> None:
    """Run DecisionEngine and write report + locked-params YAML."""
    engine = DecisionEngine()

    print(f"\n{'=' * 60}")
    print("Running DecisionEngine...")
    decisions = engine.evaluate_all(
        summaries=summaries,
        locked_params_map=locked_params_map,
        daily_pnl_map=daily_pnl_map if daily_pnl_map else None,
    )

    # Correlation matrix for report
    survivors = [d for d in decisions if d.decision == "PROCEED"]
    correlation_matrix = None
    if len(survivors) >= 2:
        survivor_pnl = {
            d.strategy_id: daily_pnl_map[d.strategy_id]
            for d in survivors
            if d.strategy_id in daily_pnl_map
        }
        if len(survivor_pnl) >= 2:
            correlation_matrix = DecisionEngine.compute_correlation(survivor_pnl)

    # Generate and write report
    report = engine.generate_report(decisions, summaries, correlation_matrix)
    report_path = output_dir / "validation-report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written: {report_path}")

    yaml_content = DecisionEngine.generate_locked_params_yaml(decisions)
    yaml_path = output_dir / "locked-params.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Locked params written: {yaml_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    for d in decisions:
        status = "PROCEED ✓" if d.decision == "PROCEED" else "RETIRE ✗"
        print(f"  {d.strategy_id:15s}  {status}")
    print(f"{'=' * 60}")

    n_survivors = sum(1 for d in decisions if d.decision == "PROCEED")
    sys.exit(0 if n_survivors > 0 else 1)


def main() -> None:
    args = parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.backtesting._parallel import resolve_max_workers

    workers = resolve_max_workers(args.workers)

    # Determine which strategies to run
    if args.strategy == "all":
        strategies_to_run = list(STRATEGY_MAP.keys())
    else:
        strategies_to_run = [args.strategy]

    n_trials = len(STRATEGY_MAP)  # always 4 for DSR correction

    print("=" * 60)
    print("Phase 4 Validation Report Generator")
    print(f"  Date range: {start} -> {end}")
    print(f"  Capital: ${args.capital:,.0f}")
    print(f"  Workers: {workers}")
    print(f"  Resample: {args.resample or 'off (1s)'}")
    print(f"  RTH bars: {args.use_rth_bars} ({args.rth_parquet_dir})")
    print(f"  HMM gating: {args.use_hmm}")
    print(f"  Strategies: {', '.join(strategies_to_run)}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Collect results: run requested strategies, load cached for the rest
    summaries: list[ValidationSummary] = []
    locked_params_map: dict[str, dict] = {}
    daily_pnl_map: dict[str, np.ndarray] = {}

    for strategy_name in STRATEGY_MAP:
        if strategy_name in strategies_to_run:
            strategy_cls = STRATEGY_MAP[strategy_name]
            summary, locked_params, daily_pnl = _run_strategy(
                strategy_name, strategy_cls,
                start, end, args.capital, args.parquet_dir,
                args.workers, n_trials, output_dir,
                resample_freq=args.resample,
                use_rth_bars=args.use_rth_bars,
                rth_parquet_dir=args.rth_parquet_dir,
                use_hmm=args.use_hmm,
            )
        else:
            # Try to load from previous run
            loaded = _load_strategy_result(output_dir, strategy_name)
            if loaded is None:
                print(f"\n  Skipping {strategy_name} (no cached results)")
                continue
            summary, locked_params, daily_pnl = loaded
            print(f"\n  Loaded cached: {strategy_name}")

        summaries.append(summary)
        locked_params_map[strategy_name] = locked_params
        if daily_pnl:
            daily_pnl_map[strategy_name] = np.array(daily_pnl)

    if not summaries:
        print("\nNo strategy results available. Run at least one strategy first.")
        sys.exit(1)

    _generate_combined_report(output_dir, summaries, locked_params_map, daily_pnl_map)


if __name__ == "__main__":
    main()
