"""Combinatorial Purged Cross-Validation (CPCV).

Implements the CPCV framework from Lopez de Prado, adapted for fixed-config
strategies. Measures IS-OOS consistency: do folds with above-median IS Sharpe
also show above-median OOS Sharpe?

PBO (Probability of Backtest Overfitting) near 0 = good. PBO >= 0.10 = FAIL.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from itertools import combinations

import numpy as np
import polars as pl

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.metrics import MetricsCalculator
from src.core.logging import get_logger
from src.strategies.base import StrategyBase

logger = get_logger("cpcv")

from zoneinfo import ZoneInfo

_ET = ZoneInfo("US/Eastern")
_SESSION_START = time(9, 30)
_SESSION_END = time(16, 0)


@dataclass
class CPCVConfig:
    """Configuration for CPCV validation."""

    n_groups: int = 8
    k_test: int = 2
    embargo_pct: float = 0.05
    purge_bars: int = 1
    max_workers: int | None = None  # None → auto-detect


@dataclass(frozen=True)
class CPCVFold:
    """A single train/test split in CPCV."""

    fold_id: int
    test_groups: tuple[int, ...]
    train_dates: list[date]
    test_dates: list[date]
    embargo_dates: list[date]
    purged_days: int


@dataclass(frozen=True)
class CPCVResult:
    """Complete CPCV validation output."""

    strategy_id: str
    pbo: float
    n_paths: int
    oos_sharpes: list[float]
    is_sharpes: list[float]
    oos_returns: list[float]
    is_returns: list[float]
    avg_oos_sharpe: float
    avg_is_sharpe: float
    sharpe_decay: float
    verdict: str
    oos_daily_pnls: list[float] = field(default_factory=list)


class CPCVValidator:
    """Runs CPCV validation on a strategy using bar-replay backtesting.

    Args:
        strategy_factory: Callable returning fresh strategy instances per fold.
        backtest_config: Template config (strategies/dates overridden per fold).
    """

    def __init__(
        self,
        strategy_factory: Callable[[], list[StrategyBase]],
        backtest_config: BacktestConfig,
        *,
        config_cls_path: str | None = None,
        strategy_cls_path: str | None = None,
    ) -> None:
        self._strategy_factory = strategy_factory
        self._config = backtest_config
        self._engine = BacktestEngine()
        self._config_cls_path = config_cls_path
        self._strategy_cls_path = strategy_cls_path

    def run(self, cpcv_config: CPCVConfig = CPCVConfig()) -> CPCVResult:
        """Execute CPCV validation.

        1. Load bars, extract trading days
        2. Generate C(n_groups, k_test) folds with purge + embargo
        3. Run IS + OOS backtest per fold
        4. Compute PBO from IS/OOS Sharpe vectors
        """
        bars_df = self._engine._load_bars(self._config)
        if bars_df.is_empty():
            logger.warning("cpcv_no_bars")
            return CPCVResult(
                strategy_id=self._get_strategy_id(),
                pbo=0.0, n_paths=0,
                oos_sharpes=[], is_sharpes=[],
                oos_returns=[], is_returns=[],
                avg_oos_sharpe=0.0, avg_is_sharpe=0.0,
                sharpe_decay=0.0, verdict="FAIL",
            )

        trading_days = self._get_trading_days(bars_df)
        logger.info("cpcv_days", total_days=len(trading_days))

        folds = self.generate_folds(trading_days, cpcv_config)
        logger.info("cpcv_folds", n_folds=len(folds))

        from src.backtesting._parallel import BacktestConfigData, resolve_max_workers

        max_workers = resolve_max_workers(cpcv_config.max_workers)
        can_parallel = (
            max_workers > 1
            and self._config_cls_path is not None
            and self._strategy_cls_path is not None
        )

        if can_parallel:
            bt_data = BacktestConfigData.from_config(self._config)
            # Pre-load HMM in main process so workers don't call joblib.load()
            if bt_data.use_hmm and bt_data._hmm_classifier is None:
                from src.models.hmm_regime import HMMRegimeClassifier
                try:
                    bt_data._hmm_classifier = HMMRegimeClassifier.load(bt_data.hmm_model_path)
                except Exception:
                    pass
            is_sharpes, oos_sharpes, is_returns, oos_returns, oos_daily_pnls = (
                self._run_folds_parallel(folds, bars_df, bt_data, max_workers)
            )
        else:
            is_sharpes, oos_sharpes, is_returns, oos_returns, oos_daily_pnls = (
                self._run_folds_sequential(folds, bars_df)
            )

        pbo = self._compute_pbo(is_sharpes, oos_sharpes)
        avg_is = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        avg_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        decay = avg_oos / avg_is if abs(avg_is) > 1e-10 else 0.0

        verdict = "PASS" if pbo < 0.10 else "FAIL"

        logger.info(
            "cpcv_complete",
            pbo=round(pbo, 4),
            avg_is_sharpe=round(avg_is, 3),
            avg_oos_sharpe=round(avg_oos, 3),
            sharpe_decay=round(decay, 3),
            verdict=verdict,
        )

        return CPCVResult(
            strategy_id=self._get_strategy_id(),
            pbo=pbo,
            n_paths=len(folds),
            oos_sharpes=oos_sharpes,
            is_sharpes=is_sharpes,
            oos_returns=oos_returns,
            is_returns=is_returns,
            avg_oos_sharpe=avg_oos,
            avg_is_sharpe=avg_is,
            sharpe_decay=decay,
            verdict=verdict,
            oos_daily_pnls=oos_daily_pnls,
        )

    @staticmethod
    def generate_folds(
        trading_days: list[date], config: CPCVConfig
    ) -> list[CPCVFold]:
        """Generate all CPCV folds with purging and embargo.

        Splits trading_days into n_groups contiguous blocks, then creates
        C(n_groups, k_test) folds. Each fold purges boundary days and
        embargoes days after test blocks from training.
        """
        n = len(trading_days)
        n_groups = config.n_groups

        # Split into n_groups contiguous blocks
        groups: list[list[date]] = []
        base_size = n // n_groups
        remainder = n % n_groups
        idx = 0
        for g in range(n_groups):
            size = base_size + (1 if g < remainder else 0)
            groups.append(trading_days[idx : idx + size])
            idx += size

        # Build a set of all days for quick lookup
        all_days_sorted = trading_days  # already sorted

        folds: list[CPCVFold] = []
        for fold_id, test_combo in enumerate(combinations(range(n_groups), config.k_test)):
            # Test dates = days in selected groups
            test_dates: list[date] = []
            for gi in test_combo:
                test_dates.extend(groups[gi])
            test_set = set(test_dates)

            # Candidate train dates = all other groups
            candidate_train = [d for d in all_days_sorted if d not in test_set]

            # Identify test block boundaries (contiguous runs in test_dates sorted)
            test_dates_sorted = sorted(test_dates)
            test_blocks = _find_contiguous_blocks(test_dates_sorted, all_days_sorted)

            # Purge: remove purge_bars trading days before and after each test block boundary
            purge_set: set[date] = set()
            for block_start, block_end in test_blocks:
                start_idx = all_days_sorted.index(block_start)
                end_idx = all_days_sorted.index(block_end)

                # Remove purge_bars days before block start from training
                for offset in range(1, config.purge_bars + 1):
                    pi = start_idx - offset
                    if 0 <= pi < len(all_days_sorted):
                        purge_set.add(all_days_sorted[pi])

                # Remove purge_bars days after block end from training
                for offset in range(1, config.purge_bars + 1):
                    pi = end_idx + offset
                    if 0 <= pi < len(all_days_sorted):
                        purge_set.add(all_days_sorted[pi])

            # Embargo: remove embargo_pct * len(candidate_train) days after each test block
            embargo_size = int(config.embargo_pct * len(candidate_train))
            embargo_set: set[date] = set()
            for _, block_end in test_blocks:
                end_idx = all_days_sorted.index(block_end)
                count = 0
                for offset in range(1, len(all_days_sorted)):
                    ei = end_idx + offset
                    if ei >= len(all_days_sorted):
                        break
                    d = all_days_sorted[ei]
                    if d not in test_set:
                        embargo_set.add(d)
                        count += 1
                        if count >= embargo_size:
                            break

            # Final train dates: candidate minus purged minus embargoed
            removed = purge_set | embargo_set
            train_dates = [d for d in candidate_train if d not in removed]

            # embargo_dates = days actually removed by embargo (not already in test)
            embargo_dates = sorted(embargo_set - purge_set)

            folds.append(CPCVFold(
                fold_id=fold_id,
                test_groups=test_combo,
                train_dates=train_dates,
                test_dates=sorted(test_dates),
                embargo_dates=embargo_dates,
                purged_days=len(purge_set - test_set),
            ))

        return folds

    def _run_folds_parallel(
        self,
        folds: list[CPCVFold],
        bars_df: pl.DataFrame,
        bt_data,
        max_workers: int,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """Run all folds in parallel using ProcessPoolExecutor."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from tqdm import tqdm

        is_sharpes = [0.0] * len(folds)
        oos_sharpes = [0.0] * len(folds)
        is_returns = [0.0] * len(folds)
        oos_returns = [0.0] * len(folds)
        all_oos_daily: list[float] = []

        logger.info("cpcv_parallel_start", workers=max_workers, folds=len(folds))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _cpcv_fold_worker,
                    fold,
                    self._config_cls_path,
                    self._strategy_cls_path,
                    bt_data,
                ): fold.fold_id
                for fold in folds
            }

            with tqdm(total=len(folds), desc="CPCV folds", unit="fold") as pbar:
                for future in as_completed(futures):
                    fold_id = futures[future]
                    try:
                        fid, is_s, oos_s, is_r, oos_r, oos_daily = future.result()
                        is_sharpes[fid] = is_s
                        oos_sharpes[fid] = oos_s
                        is_returns[fid] = is_r
                        oos_returns[fid] = oos_r
                        all_oos_daily.extend(oos_daily)
                        logger.info(
                            "cpcv_fold_done", fold_id=fid,
                            is_sharpe=round(is_s, 3), oos_sharpe=round(oos_s, 3),
                        )
                    except Exception:
                        logger.exception("cpcv_fold_failed", fold_id=fold_id)
                    pbar.update(1)

        return is_sharpes, oos_sharpes, is_returns, oos_returns, all_oos_daily

    def _run_folds_sequential(
        self,
        folds: list[CPCVFold],
        bars_df: pl.DataFrame,
    ) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        """Run all folds sequentially (original behaviour)."""
        is_sharpes: list[float] = []
        oos_sharpes: list[float] = []
        is_returns: list[float] = []
        oos_returns: list[float] = []
        all_oos_daily: list[float] = []

        for fold in folds:
            is_s, oos_s, is_r, oos_r, oos_daily = self._run_fold(fold, bars_df)
            is_sharpes.append(is_s)
            oos_sharpes.append(oos_s)
            is_returns.append(is_r)
            oos_returns.append(oos_r)
            all_oos_daily.extend(oos_daily)
            logger.info(
                "cpcv_fold_done", fold_id=fold.fold_id,
                test_groups=fold.test_groups,
                is_sharpe=round(is_s, 3), oos_sharpe=round(oos_s, 3),
            )

        return is_sharpes, oos_sharpes, is_returns, oos_returns, all_oos_daily

    def _run_fold(
        self, fold: CPCVFold, bars_df: pl.DataFrame
    ) -> tuple[float, float, float, float, list[float]]:
        """Run IS and OOS backtests for a single fold.

        Returns (is_sharpe, oos_sharpe, is_return, oos_return, oos_daily_pnls).
        """
        # IS backtest on train dates
        is_strategies = self._strategy_factory()
        is_result = self._run_on_dates(fold.train_dates, bars_df, is_strategies)
        is_sharpe = is_result.metrics.sharpe_ratio
        is_return = is_result.metrics.net_pnl

        # OOS backtest on test dates
        oos_strategies = self._strategy_factory()
        oos_result = self._run_on_dates(fold.test_dates, bars_df, oos_strategies)
        oos_sharpe = oos_result.metrics.sharpe_ratio
        oos_return = oos_result.metrics.net_pnl

        # Extract daily PnL from OOS result
        oos_daily: list[float] = []
        if not oos_result.daily_pnl.is_empty() and "pnl" in oos_result.daily_pnl.columns:
            oos_daily = oos_result.daily_pnl["pnl"].to_list()

        return is_sharpe, oos_sharpe, is_return, oos_return, oos_daily

    def _run_on_dates(
        self,
        dates: list[date],
        bars_df: pl.DataFrame,
        strategies: list[StrategyBase],
    ):
        """Run backtest on a specific set of dates by filtering bars."""
        from src.backtesting.metrics import BacktestResult

        if not dates:
            metrics, eq, dp = MetricsCalculator.from_trades(
                [], self._config.initial_capital
            )
            return BacktestResult(
                trades=[], equity_curve=eq, daily_pnl=dp,
                metrics=metrics, config_summary={},
            )

        # Filter bars to only include the specified dates
        date_set = set(dates)
        min_date = min(dates)
        max_date = max(dates)

        # Filter by date range using naive datetime (matches Parquet Datetime(us) schema)
        start_dt = datetime.combine(min_date, time(0, 0))
        end_dt = datetime.combine(max_date + timedelta(days=1), time(0, 0))

        # Broad filter first, then fine-filter by date
        filtered = bars_df.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_dt)
        )

        # Fine-filter: only keep bars whose date is in the date_set
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

        # Create a config with the filtered date range and fresh strategies
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
            l1_parquet_dir=self._config.l1_parquet_dir,
            l1_bar_seconds=self._config.l1_bar_seconds,
        )

        # Run engine but with pre-filtered bars — override _load_bars
        engine = _FoldEngine(filtered)
        return engine.run(fold_config)

    @staticmethod
    def _compute_pbo(is_sharpes: list[float], oos_sharpes: list[float]) -> float:
        """Compute PBO from IS/OOS Sharpe vectors (fixed-config variant).

        Measures IS-OOS rank concordance:
        - Concordant: both above or both below their respective medians
        - Discordant: one above, one below
        - PBO = discordant / total
        """
        n = len(is_sharpes)
        if n == 0:
            return 0.0

        is_arr = np.array(is_sharpes)
        oos_arr = np.array(oos_sharpes)

        # Edge case: all identical
        if np.std(is_arr) < 1e-10 and np.std(oos_arr) < 1e-10:
            return 0.0

        m_is = float(np.median(is_arr))
        m_oos = float(np.median(oos_arr))

        discordant = 0
        for i in range(n):
            is_above = is_arr[i] > m_is
            oos_above = oos_arr[i] > m_oos
            if is_above != oos_above:
                discordant += 1

        return discordant / n

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

    def _get_strategy_id(self) -> str:
        """Get strategy ID from factory output."""
        strategies = self._strategy_factory()
        if strategies:
            return strategies[0].config.strategy_id
        return "unknown"


class _FoldEngine(BacktestEngine):
    """BacktestEngine subclass that uses pre-filtered bars instead of loading from Parquet."""

    def __init__(self, bars_df: pl.DataFrame) -> None:
        self._bars_df = bars_df

    def _load_bars(self, config: BacktestConfig) -> pl.DataFrame:
        return self._bars_df


# ---------------------------------------------------------------------------
# Module-level worker for multiprocessing (must be pickleable)
# ---------------------------------------------------------------------------

def _cpcv_fold_worker(
    fold: CPCVFold,
    config_cls_path: str,
    strategy_cls_path: str,
    bt_config_data,
) -> tuple[int, float, float, float, float, list[float]]:
    """Run a single CPCV fold in a worker process.

    Each worker loads its own bars from parquet to avoid pickling ~1 GB.
    Returns ``(fold_id, is_sharpe, oos_sharpe, is_return, oos_return, oos_daily_pnls)``.
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
        FoldEngine,
        load_bars_for_dates,
        load_l1_bars_for_dates,
        make_strategies,
    )
    from src.backtesting.metrics import MetricsCalculator

    # Load bars for all dates in this fold (train + test) once
    all_dates = fold.train_dates + fold.test_dates
    if bt_config_data.l1_parquet_dir:
        bars_df = load_l1_bars_for_dates(
            bt_config_data.l1_parquet_dir, all_dates,
            bar_seconds=bt_config_data.l1_bar_seconds,
        )
    else:
        bars_df = load_bars_for_dates(
            bt_config_data.parquet_dir, all_dates,
            use_rth_bars=bt_config_data.use_rth_bars,
            rth_parquet_dir=bt_config_data.rth_parquet_dir,
        )

    def _run(dates, strategies):
        date_set = set(dates)
        filtered = bars_df.filter(
            pl.col("timestamp").dt.date().is_in(list(date_set))
        )
        if filtered.is_empty():
            metrics, eq, dp = MetricsCalculator.from_trades(
                [], bt_config_data.initial_capital
            )
            from src.backtesting.metrics import BacktestResult
            return BacktestResult(
                trades=[], equity_curve=eq, daily_pnl=dp,
                metrics=metrics, config_summary={},
            )
        fold_config = bt_config_data.to_backtest_config(strategies)
        fold_config.start_date = min(dates)
        fold_config.end_date = max(dates)
        return FoldEngine(filtered).run(fold_config)

    # IS — pass pre-loaded HMM classifier to avoid joblib.load() in workers
    hmm_kwargs = dict(
        use_hmm=bt_config_data.use_hmm,
        hmm_model_path=bt_config_data.hmm_model_path,
        hmm_classifier=bt_config_data._hmm_classifier,
    )
    is_strats = make_strategies(config_cls_path, strategy_cls_path, **hmm_kwargs)
    is_result = _run(fold.train_dates, is_strats)

    # OOS
    oos_strats = make_strategies(config_cls_path, strategy_cls_path, **hmm_kwargs)
    oos_result = _run(fold.test_dates, oos_strats)

    # Extract daily PnL from OOS result
    oos_daily = []
    if not oos_result.daily_pnl.is_empty() and "pnl" in oos_result.daily_pnl.columns:
        oos_daily = oos_result.daily_pnl["pnl"].to_list()

    return (
        fold.fold_id,
        is_result.metrics.sharpe_ratio,
        oos_result.metrics.sharpe_ratio,
        is_result.metrics.net_pnl,
        oos_result.metrics.net_pnl,
        oos_daily,
    )


def _find_contiguous_blocks(
    test_dates_sorted: list[date], all_days_sorted: list[date]
) -> list[tuple[date, date]]:
    """Find contiguous blocks of test dates within the full trading calendar.

    Returns list of (block_start, block_end) tuples.
    """
    if not test_dates_sorted:
        return []

    test_set = set(test_dates_sorted)
    day_to_idx = {d: i for i, d in enumerate(all_days_sorted)}

    blocks: list[tuple[date, date]] = []
    block_start = test_dates_sorted[0]

    for i in range(1, len(test_dates_sorted)):
        prev = test_dates_sorted[i - 1]
        curr = test_dates_sorted[i]

        prev_idx = day_to_idx[prev]
        curr_idx = day_to_idx[curr]

        # If not adjacent in the trading calendar, close this block
        if curr_idx != prev_idx + 1:
            blocks.append((block_start, prev))
            block_start = curr

    # Close final block
    blocks.append((block_start, test_dates_sorted[-1]))
    return blocks
