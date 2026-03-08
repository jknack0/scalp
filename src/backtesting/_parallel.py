"""Shared utilities for parallel backtest execution (CPCV / WFA).

Provides pickleable data carriers and module-level worker helpers so that
``concurrent.futures.ProcessPoolExecutor`` can dispatch independent
folds / cycles across CPU cores.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import polars as pl

from src.analysis.commission_model import CostModel
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.metrics import BacktestResult, MetricsCalculator
from src.backtesting.slippage import VolatilitySlippageModel
from src.strategies.base import StrategyBase


# ---------------------------------------------------------------------------
# Pickleable config carrier
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfigData:
    """Pickleable subset of ``BacktestConfig``.

    Strips live strategy instances so the object can be sent across process
    boundaries.  Use ``to_backtest_config(strategies)`` to reconstruct.
    """

    initial_capital: float
    broker_name: str
    broker_commission_per_side: float
    exchange_fee: float
    nfa_fee: float
    avg_slippage_ticks: float
    tick_value: float
    bar_type: str
    resample_freq: str | None
    symbol: str
    session_start: time
    session_end: time
    max_position: int
    parquet_dir: str
    use_rth_bars: bool = False
    rth_parquet_dir: str = "data/parquet_5s_rth"
    l1_parquet_dir: str | None = None
    l1_bar_seconds: int = 5
    use_hmm: bool = False
    hmm_model_path: str = "models/hmm/v1"
    _hmm_classifier: object = None  # Pre-loaded HMMRegimeClassifier (pickleable)

    @classmethod
    def from_config(cls, config: BacktestConfig) -> BacktestConfigData:
        cm = config.commission_model
        return cls(
            initial_capital=config.initial_capital,
            broker_name=cm.broker_name,
            broker_commission_per_side=cm.broker_commission_per_side,
            exchange_fee=cm.exchange_fee,
            nfa_fee=cm.nfa_fee,
            avg_slippage_ticks=cm.avg_slippage_ticks,
            tick_value=cm.tick_value,
            bar_type=config.bar_type,
            resample_freq=config.resample_freq,
            symbol=config.symbol,
            session_start=config.session_start,
            session_end=config.session_end,
            max_position=config.max_position,
            parquet_dir=config.parquet_dir,
            use_rth_bars=config.use_rth_bars,
            rth_parquet_dir=config.rth_parquet_dir,
            l1_parquet_dir=config.l1_parquet_dir,
            l1_bar_seconds=config.l1_bar_seconds,
            use_hmm=config.use_hmm,
            hmm_model_path=config.hmm_model_path,
        )

    def to_backtest_config(self, strategies: list[StrategyBase]) -> BacktestConfig:
        return BacktestConfig(
            strategies=strategies,
            start_date=date(2000, 1, 1),  # placeholder — overridden by run_on_dates
            end_date=date(2099, 12, 31),
            initial_capital=self.initial_capital,
            commission_model=CostModel(
                broker_name=self.broker_name,
                broker_commission_per_side=self.broker_commission_per_side,
                exchange_fee=self.exchange_fee,
                nfa_fee=self.nfa_fee,
                avg_slippage_ticks=self.avg_slippage_ticks,
                tick_value=self.tick_value,
            ),
            slippage_model=VolatilitySlippageModel(),
            bar_type=self.bar_type,
            resample_freq=self.resample_freq,
            symbol=self.symbol,
            session_start=self.session_start,
            session_end=self.session_end,
            max_position=self.max_position,
            parquet_dir=self.parquet_dir,
            use_rth_bars=self.use_rth_bars,
            rth_parquet_dir=self.rth_parquet_dir,
            l1_parquet_dir=self.l1_parquet_dir,
            l1_bar_seconds=self.l1_bar_seconds,
        )


# ---------------------------------------------------------------------------
# FoldEngine — avoids re-loading Parquet in workers
# ---------------------------------------------------------------------------

class FoldEngine(BacktestEngine):
    """BacktestEngine subclass that uses pre-filtered bars."""

    def __init__(self, bars_df: pl.DataFrame) -> None:
        self._bars_df = bars_df

    def _load_bars(self, config: BacktestConfig) -> pl.DataFrame:
        return self._bars_df


# ---------------------------------------------------------------------------
# Strategy reconstruction (pickleable)
# ---------------------------------------------------------------------------

def make_strategies(
    config_cls_path: str,
    strategy_cls_path: str,
    params: dict | None = None,
    *,
    use_hmm: bool = False,
    hmm_model_path: str = "models/hmm/v1",
    hmm_classifier: object = None,
) -> list[StrategyBase]:
    """Reconstruct strategy instances from dotted class paths.

    This is a module-level function so it can be pickled and sent to worker
    processes.
    """
    from src.features.feature_hub import FeatureHub

    mod_path_cfg, cls_name_cfg = config_cls_path.rsplit(".", 1)
    config_cls = getattr(importlib.import_module(mod_path_cfg), cls_name_cfg)

    mod_path_strat, cls_name_strat = strategy_cls_path.rsplit(".", 1)
    strategy_cls = getattr(importlib.import_module(mod_path_strat), cls_name_strat)

    cfg = config_cls()
    if not use_hmm:
        cfg.require_hmm_states = []
    if params:
        for k, v in params.items():
            setattr(cfg, k, v)

    # Use pre-loaded HMM classifier; fall back to loading from disk
    _hmm = hmm_classifier
    if use_hmm and _hmm is None:
        from src.models.hmm_regime import HMMRegimeClassifier
        try:
            _hmm = HMMRegimeClassifier.load(hmm_model_path)
        except Exception:
            pass

    hub = FeatureHub()
    return [strategy_cls(cfg, hub, hmm_classifier=_hmm)]


# ---------------------------------------------------------------------------
# Data loading for workers (avoids pickling large DataFrames)
# ---------------------------------------------------------------------------

def load_bars_for_dates(
    parquet_dir: str, dates: list[date],
    *,
    use_rth_bars: bool = False,
    rth_parquet_dir: str = "data/parquet_5s_rth",
) -> pl.DataFrame:
    """Load parquet partitions covering *dates* and filter to those dates.

    Each worker calls this instead of receiving a pre-loaded DataFrame,
    avoiding the MemoryError from pickling ~1 GB per worker on Windows.

    When *use_rth_bars* is True, reads from the pre-built RTH 5s directory
    instead, which is ~87% smaller.
    """
    effective_dir = rth_parquet_dir if use_rth_bars else parquet_dir
    if not dates:
        return pl.DataFrame()

    min_date = min(dates)
    max_date = max(dates)
    start_year = min_date.year
    end_year = max_date.year

    frames: list[pl.DataFrame] = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(effective_dir, f"year={year}", "data.parquet")
        if os.path.exists(path):
            frames.append(pl.read_parquet(path))

    if not frames:
        return pl.DataFrame()

    df = pl.concat(frames)

    # Broad datetime filter
    start_dt = datetime.combine(min_date, time(0, 0))
    end_dt = datetime.combine(max_date + timedelta(days=1), time(0, 0))
    df = df.filter(
        (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_dt)
    )

    # Fine filter to exact dates
    date_set = set(dates)
    if not df.is_empty():
        df = df.filter(
            pl.col("timestamp").dt.date().is_in(list(date_set))
        )

    return df


def load_l1_bars_for_dates(
    l1_dir: str, dates: list[date],
    bar_seconds: int = 5,
) -> pl.DataFrame:
    """Load L1 tick data for *dates* and aggregate into OBI-enriched bars."""
    if not dates:
        return pl.DataFrame()

    min_date = min(dates)
    max_date = max(dates)

    frames: list[pl.DataFrame] = []
    for year in range(min_date.year, max_date.year + 1):
        path = os.path.join(l1_dir, f"year={year}", "data.parquet")
        if os.path.exists(path):
            frames.append(pl.read_parquet(path))

    if not frames:
        return pl.DataFrame()

    df = pl.concat(frames)

    # Strip timezone if present
    ts_dtype = df["timestamp"].dtype
    if hasattr(ts_dtype, 'time_zone') and ts_dtype.time_zone is not None:
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    start_dt = datetime.combine(min_date, time(0, 0))
    end_dt = datetime.combine(max_date + timedelta(days=1), time(0, 0))
    df = df.filter(
        (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_dt)
    )

    date_set = set(dates)
    if not df.is_empty():
        df = df.filter(pl.col("timestamp").dt.date().is_in(list(date_set)))

    if df.is_empty():
        return pl.DataFrame()

    # Lee-Ready trade classification
    df = df.with_columns(
        ((pl.col("bid_price") + pl.col("ask_price")) / 2.0).alias("_mid"),
    )
    df = df.with_columns(
        pl.when(pl.col("price") > pl.col("_mid"))
        .then(pl.col("size")).otherwise(pl.lit(0)).alias("_agg_buy"),
        pl.when(pl.col("price") < pl.col("_mid"))
        .then(pl.col("size")).otherwise(pl.lit(0)).alias("_agg_sell"),
    )

    interval = f"{bar_seconds}s"
    bars = df.group_by_dynamic("timestamp", every=interval).agg(
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("size").sum().alias("volume"),
        pl.col("bid_size").mean().alias("avg_bid_size"),
        pl.col("ask_size").mean().alias("avg_ask_size"),
        pl.col("_agg_buy").sum().alias("aggressive_buy_vol"),
        pl.col("_agg_sell").sum().alias("aggressive_sell_vol"),
    )
    bars = bars.filter(pl.col("volume") > 0).sort("timestamp")
    return bars


# ---------------------------------------------------------------------------
# Shared run-on-dates logic
# ---------------------------------------------------------------------------

def run_on_dates(
    dates: list[date],
    strategies: list[StrategyBase],
    config_data: BacktestConfigData,
) -> BacktestResult:
    """Load bars from parquet and run a backtest on *dates*.

    Workers call this directly — no large DataFrame needs to be pickled.
    """
    if not dates:
        metrics, eq, dp = MetricsCalculator.from_trades(
            [], config_data.initial_capital
        )
        return BacktestResult(
            trades=[], equity_curve=eq, daily_pnl=dp,
            metrics=metrics, config_summary={},
        )

    filtered = load_bars_for_dates(
        config_data.parquet_dir, dates,
        use_rth_bars=config_data.use_rth_bars,
        rth_parquet_dir=config_data.rth_parquet_dir,
    )

    if filtered.is_empty():
        metrics, eq, dp = MetricsCalculator.from_trades(
            [], config_data.initial_capital
        )
        return BacktestResult(
            trades=[], equity_curve=eq, daily_pnl=dp,
            metrics=metrics, config_summary={},
        )

    min_date = min(dates)
    max_date = max(dates)

    fold_config = config_data.to_backtest_config(strategies)
    fold_config.start_date = min_date
    fold_config.end_date = max_date

    engine = FoldEngine(filtered)
    return engine.run(fold_config)


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------

def resolve_max_workers(max_workers: int | None) -> int:
    """Resolve *None* to ``min(cpu_count - 2, 12)``, minimum 1."""
    if max_workers is not None:
        return max(1, max_workers)
    cpu = os.cpu_count() or 4
    return max(1, min(cpu - 2, 12))
