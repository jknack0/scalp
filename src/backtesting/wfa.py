"""Walk-Forward Anchored (WFA) validation.

Sequential rolling train/test windows where the best parameters from each
training window are evaluated unchanged on the test window. Tests whether a
strategy's parameters remain stable over time or are just curve-fitting.

Key metric: efficiency ratio = mean(OOS Sharpe) / mean(IS Sharpe). Target >= 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from itertools import product

import numpy as np
import polars as pl

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.metrics import BacktestResult, MetricsCalculator
from src.core.logging import get_logger
from src.features.feature_hub import FeatureHub
from src.strategies.base import StrategyBase

logger = get_logger("wfa")

from zoneinfo import ZoneInfo

_ET = ZoneInfo("US/Eastern")
_SESSION_START = time(9, 30)
_SESSION_END = time(16, 0)

# Strategy factory map — imported lazily to avoid circular imports
STRATEGY_MAP = {
    "orb": ("src.strategies.orb_strategy", "ORBConfig", "ORBStrategy"),
    "vwap": ("src.strategies.vwap_strategy", "VWAPConfig", "VWAPStrategy"),
    "cvd": ("src.strategies.cvd_divergence_strategy", "CVDDivergenceConfig", "CVDDivergenceStrategy"),
    "vol_regime": ("src.strategies.vol_regime_strategy", "VolRegimeConfig", "VolRegimeStrategy"),
    "obi": ("src.strategies.obi_strategy", "OBIConfig", "OBIStrategy"),
}


@dataclass
class WFAConfig:
    """Configuration for Walk-Forward Analysis."""

    train_days: int = 63
    test_days: int = 21
    min_cycles: int = 6
    efficiency_threshold: float = 0.5
    max_workers: int | None = None  # None → auto-detect


@dataclass(frozen=True)
class WFACycle:
    """A single WFA train/test cycle result."""

    cycle_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    best_params: dict
    is_sharpe: float
    oos_sharpe: float
    is_trades: int
    oos_trades: int
    oos_win_rate: float
    oos_profit_factor: float
    grid_results: int


@dataclass(frozen=True)
class WFAResult:
    """Complete WFA validation output."""

    strategy_id: str
    cycles: list
    n_cycles: int
    efficiency_ratio: float
    is_oos_correlation: float
    avg_is_sharpe: float
    avg_oos_sharpe: float
    param_drift: dict
    verdict: str


class WFARunner:
    """Runs Walk-Forward Analysis with parameter grid search.

    For each rolling window: grid-search parameters on the training set,
    then evaluate the best combo unchanged on the test set.

    Args:
        strategy_name: Key in STRATEGY_MAP (e.g. "orb", "vwap").
        param_grid: {param_name: [values to try]}.
        backtest_config: Template config (dates/strategies overridden per window).
    """

    def __init__(
        self,
        strategy_name: str,
        param_grid: dict[str, list],
        backtest_config: BacktestConfig,
    ) -> None:
        self._strategy_name = strategy_name
        self._param_grid = param_grid
        self._config = backtest_config
        self._engine = BacktestEngine()

    def run(self, wfa_config: WFAConfig = WFAConfig()) -> WFAResult:
        """Execute Walk-Forward Analysis.

        1. Load bars, extract trading days
        2. Generate rolling windows
        3. For each cycle: grid search on train, evaluate best on test
        4. Compute efficiency ratio, IS/OOS correlation, param drift
        """
        bars_df = self._engine._load_bars(self._config)
        if bars_df.is_empty():
            logger.warning("wfa_no_bars")
            return WFAResult(
                strategy_id=self._strategy_name,
                cycles=[], n_cycles=0,
                efficiency_ratio=0.0, is_oos_correlation=0.0,
                avg_is_sharpe=0.0, avg_oos_sharpe=0.0,
                param_drift={}, verdict="FAIL",
            )

        trading_days = self._get_trading_days(bars_df)
        logger.info("wfa_days", total_days=len(trading_days))

        windows = self.generate_windows(trading_days, wfa_config)
        logger.info("wfa_windows", n_cycles=len(windows))

        from src.backtesting._parallel import BacktestConfigData, resolve_max_workers

        max_workers = resolve_max_workers(wfa_config.max_workers)

        if max_workers > 1:
            bt_data = BacktestConfigData.from_config(self._config)
            # Pre-load HMM in main process so workers don't call joblib.load()
            if bt_data.use_hmm and bt_data._hmm_classifier is None:
                from src.models.hmm_regime import HMMRegimeClassifier
                try:
                    bt_data._hmm_classifier = HMMRegimeClassifier.load(bt_data.hmm_model_path)
                except Exception:
                    pass
            cycles = self._run_cycles_parallel(
                windows, bars_df, bt_data, max_workers
            )
        else:
            cycles = self._run_cycles_sequential(windows, bars_df)

        # Compute summary statistics
        n_cycles = len(cycles)
        is_sharpes = [c.is_sharpe for c in cycles]
        oos_sharpes = [c.oos_sharpe for c in cycles]

        avg_is = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        avg_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        efficiency = avg_oos / avg_is if abs(avg_is) > 1e-10 else 0.0

        # IS/OOS correlation
        if n_cycles >= 3:
            corr_matrix = np.corrcoef(is_sharpes, oos_sharpes)
            correlation = float(corr_matrix[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # Parameter drift: {param_name: [value per cycle]}
        param_drift: dict[str, list] = {p: [] for p in self._param_grid}
        for cycle in cycles:
            for p in self._param_grid:
                param_drift[p].append(cycle.best_params.get(p))

        # Verdict
        verdict = (
            "PASS"
            if efficiency >= wfa_config.efficiency_threshold
            and n_cycles >= wfa_config.min_cycles
            else "FAIL"
        )

        logger.info(
            "wfa_complete",
            efficiency_ratio=round(efficiency, 3),
            is_oos_correlation=round(correlation, 3),
            avg_is_sharpe=round(avg_is, 3),
            avg_oos_sharpe=round(avg_oos, 3),
            verdict=verdict,
        )

        return WFAResult(
            strategy_id=self._strategy_name,
            cycles=cycles,
            n_cycles=n_cycles,
            efficiency_ratio=efficiency,
            is_oos_correlation=correlation,
            avg_is_sharpe=avg_is,
            avg_oos_sharpe=avg_oos,
            param_drift=param_drift,
            verdict=verdict,
        )

    @staticmethod
    def generate_windows(
        trading_days: list[date], config: WFAConfig
    ) -> list[tuple[list[date], list[date]]]:
        """Generate rolling (train_dates, test_dates) pairs.

        Cycle i:
          train = days[i*test_days : i*test_days + train_days]
          test  = days[i*test_days + train_days : i*test_days + train_days + test_days]

        Stops when the test window would exceed available days.
        """
        n = len(trading_days)
        windows: list[tuple[list[date], list[date]]] = []
        i = 0

        while True:
            train_start = i * config.test_days
            train_end = train_start + config.train_days
            test_start = train_end
            test_end = test_start + config.test_days

            if test_end > n:
                break

            train_dates = trading_days[train_start:train_end]
            test_dates = trading_days[test_start:test_end]
            windows.append((train_dates, test_dates))
            i += 1

        return windows

    def _run_cycles_parallel(
        self,
        windows: list[tuple[list[date], list[date]]],
        bars_df: pl.DataFrame,
        bt_data,
        max_workers: int,
    ) -> list[WFACycle]:
        """Run all WFA cycles in parallel using ProcessPoolExecutor."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from tqdm import tqdm

        cycles: list[WFACycle | None] = [None] * len(windows)

        logger.info("wfa_parallel_start", workers=max_workers, cycles=len(windows))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _wfa_cycle_worker,
                    i,
                    train_dates,
                    test_dates,
                    self._strategy_name,
                    self._param_grid,
                    bt_data,
                ): i
                for i, (train_dates, test_dates) in enumerate(windows)
            }

            with tqdm(total=len(windows), desc="WFA cycles", unit="cycle") as pbar:
                for future in as_completed(futures):
                    cycle_id = futures[future]
                    try:
                        cycle = future.result()
                        cycles[cycle.cycle_id] = cycle
                        logger.info(
                            "wfa_cycle_done", cycle_id=cycle.cycle_id,
                            is_sharpe=round(cycle.is_sharpe, 3),
                            oos_sharpe=round(cycle.oos_sharpe, 3),
                            best_params=cycle.best_params,
                        )
                    except Exception:
                        logger.exception("wfa_cycle_failed", cycle_id=cycle_id)
                    pbar.update(1)

        return [c for c in cycles if c is not None]

    def _run_cycles_sequential(
        self,
        windows: list[tuple[list[date], list[date]]],
        bars_df: pl.DataFrame,
    ) -> list[WFACycle]:
        """Run all WFA cycles sequentially (original behaviour)."""
        cycles: list[WFACycle] = []
        for i, (train_dates, test_dates) in enumerate(windows):
            cycle = self._run_cycle(train_dates, test_dates, bars_df, i)
            cycles.append(cycle)
            logger.info(
                "wfa_cycle_done", cycle_id=i,
                is_sharpe=round(cycle.is_sharpe, 3),
                oos_sharpe=round(cycle.oos_sharpe, 3),
                best_params=cycle.best_params,
            )
        return cycles

    def _run_cycle(
        self,
        train_dates: list[date],
        test_dates: list[date],
        bars_df: pl.DataFrame,
        cycle_id: int,
    ) -> WFACycle:
        """Run a single WFA cycle: grid search on train, evaluate best on test."""
        # Generate all param combinations
        param_names = list(self._param_grid.keys())
        param_values = list(self._param_grid.values())
        combos = list(product(*param_values))

        # Grid search on training data
        best_sharpe = float("-inf")
        best_params: dict = {}
        best_is_trades = 0

        for combo in combos:
            params = dict(zip(param_names, combo))
            strategies = self._make_strategy(params)
            result = self._run_on_dates(train_dates, bars_df, strategies)
            sharpe = result.metrics.sharpe_ratio

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
                best_is_trades = result.metrics.total_trades

        # Evaluate best params on test data (no refit)
        oos_strategies = self._make_strategy(best_params)
        oos_result = self._run_on_dates(test_dates, bars_df, oos_strategies)

        return WFACycle(
            cycle_id=cycle_id,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            test_start=test_dates[0],
            test_end=test_dates[-1],
            best_params=best_params,
            is_sharpe=best_sharpe,
            oos_sharpe=oos_result.metrics.sharpe_ratio,
            is_trades=best_is_trades,
            oos_trades=oos_result.metrics.total_trades,
            oos_win_rate=oos_result.metrics.win_rate,
            oos_profit_factor=oos_result.metrics.profit_factor,
            grid_results=len(combos),
        )

    def _make_strategy(self, params: dict) -> list[StrategyBase]:
        """Create a fresh strategy instance with param overrides via setattr."""
        import importlib

        module_path, config_cls_name, strategy_cls_name = STRATEGY_MAP[self._strategy_name]
        mod = importlib.import_module(module_path)
        config_cls = getattr(mod, config_cls_name)
        strategy_cls = getattr(mod, strategy_cls_name)

        config = config_cls()
        if not getattr(self._config, "use_hmm", False):
            config.require_hmm_states = []

        for param_name, value in params.items():
            setattr(config, param_name, value)

        # Pass HMM classifier if available on template strategies
        hmm_cls = None
        if self._config.strategies:
            hmm_cls = getattr(self._config.strategies[0], "hmm_classifier", None)

        hub = FeatureHub()
        return [strategy_cls(config, hub, hmm_classifier=hmm_cls)]

    def _run_on_dates(
        self,
        dates: list[date],
        bars_df: pl.DataFrame,
        strategies: list[StrategyBase],
    ) -> BacktestResult:
        """Run backtest on a specific set of dates by filtering bars.

        Same approach as CPCVValidator._run_on_dates().
        """
        if not dates:
            metrics, eq, dp = MetricsCalculator.from_trades(
                [], self._config.initial_capital
            )
            return BacktestResult(
                trades=[], equity_curve=eq, daily_pnl=dp,
                metrics=metrics, config_summary={},
            )

        date_set = set(dates)
        min_date = min(dates)
        max_date = max(dates)

        # Filter by date range using naive datetime (matches Parquet Datetime(us) schema)
        start_dt = datetime.combine(min_date, time(0, 0))
        end_dt = datetime.combine(max_date + timedelta(days=1), time(0, 0))

        # Broad filter, then fine-filter by date
        filtered = bars_df.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_dt)
        )

        if not filtered.is_empty():
            filtered = filtered.filter(
                pl.col("timestamp").dt.date().is_in(list(date_set))
            )

        if filtered.is_empty():
            metrics, eq, dp = MetricsCalculator.from_trades(
                [], self._config.initial_capital
            )
            return BacktestResult(
                trades=[], equity_curve=eq, daily_pnl=dp,
                metrics=metrics, config_summary={},
            )

        fold_config = BacktestConfig(
            strategies=strategies,
            start_date=min_date,
            end_date=max_date,
            initial_capital=self._config.initial_capital,
            commission_model=self._config.commission_model,
            slippage_model=self._config.slippage_model,
            bar_type=self._config.bar_type,
            symbol=self._config.symbol,
            session_start=self._config.session_start,
            session_end=self._config.session_end,
            max_position=self._config.max_position,
            parquet_dir=self._config.parquet_dir,
            use_rth_bars=self._config.use_rth_bars,
            rth_parquet_dir=self._config.rth_parquet_dir,
        )

        engine = _FoldEngine(filtered)
        return engine.run(fold_config)

    @staticmethod
    def _get_trading_days(bars_df: pl.DataFrame) -> list[date]:
        """Extract unique trading days from bars, filtered to RTH."""
        timestamps = bars_df["timestamp"].to_list()

        days: set[date] = set()
        for ts_dt in timestamps:
            dt = ts_dt.replace(tzinfo=timezone.utc).astimezone(_ET)
            t = dt.time()
            if _SESSION_START <= t < _SESSION_END:
                days.add(dt.date())

        return sorted(days)


# ---------------------------------------------------------------------------
# Module-level worker for multiprocessing (must be pickleable)
# ---------------------------------------------------------------------------

def _wfa_cycle_worker(
    cycle_id: int,
    train_dates: list[date],
    test_dates: list[date],
    strategy_name: str,
    param_grid: dict[str, list],
    bt_config_data,
) -> WFACycle:
    """Run a single WFA cycle in a worker process.

    Each worker loads its own bars from parquet to avoid pickling ~1 GB.
    Does a full grid search on train_dates, then evaluates the best params
    on test_dates.  Returns the completed ``WFACycle``.
    """
    import logging as _logging
    import os as _os
    _log_level = _os.environ.get("LOG_LEVEL", "INFO").upper()
    if _log_level in ("WARNING", "ERROR", "CRITICAL"):
        _logging.basicConfig(level=_logging.WARNING, force=True)
        _logging.getLogger().setLevel(_logging.WARNING)
    from src.core.logging import configure_logging
    configure_logging(log_level=_log_level, log_file=None)

    from src.backtesting._parallel import (
        load_bars_for_dates,
        make_strategies,
        run_on_dates,
    )

    module_path, config_cls_name, strategy_cls_name = STRATEGY_MAP[strategy_name]
    config_cls_path = f"{module_path}.{config_cls_name}"
    strategy_cls_path = f"{module_path}.{strategy_cls_name}"

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(product(*param_values))

    # Pre-load bars for this cycle's date range (train + test) once
    all_dates = train_dates + test_dates
    bars_df = load_bars_for_dates(
        bt_config_data.parquet_dir, all_dates,
        use_rth_bars=bt_config_data.use_rth_bars,
        rth_parquet_dir=bt_config_data.rth_parquet_dir,
    )

    best_sharpe = float("-inf")
    best_params: dict = {}
    best_is_trades = 0

    train_set = set(train_dates)

    # Filter bars to train dates for grid search
    train_bars = bars_df.filter(
        pl.col("timestamp").dt.date().is_in(list(train_set))
    )

    hmm_kwargs = dict(
        use_hmm=bt_config_data.use_hmm,
        hmm_model_path=bt_config_data.hmm_model_path,
        hmm_classifier=bt_config_data._hmm_classifier,
    )
    for combo in combos:
        params = dict(zip(param_names, combo))
        strategies = make_strategies(config_cls_path, strategy_cls_path, params, **hmm_kwargs)
        result = _run_on_bars(train_dates, train_bars, strategies, bt_config_data)
        sharpe = result.metrics.sharpe_ratio

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params
            best_is_trades = result.metrics.total_trades

    # OOS evaluation
    test_set = set(test_dates)
    test_bars = bars_df.filter(
        pl.col("timestamp").dt.date().is_in(list(test_set))
    )
    oos_strategies = make_strategies(config_cls_path, strategy_cls_path, best_params, **hmm_kwargs)
    oos_result = _run_on_bars(test_dates, test_bars, oos_strategies, bt_config_data)

    return WFACycle(
        cycle_id=cycle_id,
        train_start=train_dates[0],
        train_end=train_dates[-1],
        test_start=test_dates[0],
        test_end=test_dates[-1],
        best_params=best_params,
        is_sharpe=best_sharpe,
        oos_sharpe=oos_result.metrics.sharpe_ratio,
        is_trades=best_is_trades,
        oos_trades=oos_result.metrics.total_trades,
        oos_win_rate=oos_result.metrics.win_rate,
        oos_profit_factor=oos_result.metrics.profit_factor,
        grid_results=len(combos),
    )


def _run_on_bars(
    dates: list[date],
    bars_df: pl.DataFrame,
    strategies: list[StrategyBase],
    config_data,
) -> BacktestResult:
    """Run backtest on pre-filtered bars (used inside WFA cycle worker)."""
    if bars_df.is_empty():
        metrics, eq, dp = MetricsCalculator.from_trades(
            [], config_data.initial_capital
        )
        return BacktestResult(
            trades=[], equity_curve=eq, daily_pnl=dp,
            metrics=metrics, config_summary={},
        )

    from src.backtesting._parallel import FoldEngine

    min_date = min(dates)
    max_date = max(dates)
    fold_config = config_data.to_backtest_config(strategies)
    fold_config.start_date = min_date
    fold_config.end_date = max_date

    engine = FoldEngine(bars_df)
    return engine.run(fold_config)


class _FoldEngine(BacktestEngine):
    """BacktestEngine subclass that uses pre-filtered bars instead of loading from Parquet."""

    def __init__(self, bars_df: pl.DataFrame) -> None:
        self._bars_df = bars_df

    def _load_bars(self, config: BacktestConfig) -> pl.DataFrame:
        return self._bars_df
