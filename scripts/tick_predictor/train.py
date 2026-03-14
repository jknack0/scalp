"""Training pipeline for TickDirectionPredictor.

Usage:
    python -u scripts/tick_predictor/train.py --start 2024-01-01 --end 2025-03-01
    python -u scripts/tick_predictor/train.py --start 2024-01-01 --end 2025-03-01 --val-days 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np
import polars as pl

from src.core.events import BarEvent
from src.core.logging import configure_logging, get_logger
from src.signals.tick_predictor.features.feature_builder import (
    FEATURE_NAMES,
    FeatureBuilder,
)
from src.signals.tick_predictor.labels.triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierLabeler,
)
from src.signals.tick_predictor.model.calibrator import TemperatureCalibrator
from src.signals.tick_predictor.model.trainer import ModelTrainer, TrainerConfig

logger = get_logger("tick_predictor.train")

MODEL_DIR = Path("models/tick_predictor")
DATA_DIR = Path("data/tick_predictor")


@dataclass
class TrainingResult:
    cv_accuracy_mean: float
    cv_accuracy_std: float
    val_ece: float
    val_brier_score: float
    val_accuracy: float
    temperature: float
    top_features: dict[str, float]
    training_samples: int
    val_samples: int
    model_path: str
    calibrator_path: str
    success: bool
    error_message: str | None = None


class TrainingPipeline:
    """Orchestrates feature generation, labeling, training, and calibration."""

    def __init__(
        self,
        label_config: TripleBarrierConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        min_cv_accuracy: float = 0.52,
        rth_only: bool = True,
        drop_flat: bool = True,
    ) -> None:
        self.label_config = label_config or TripleBarrierConfig()
        self.trainer_config = trainer_config or TrainerConfig()
        self.min_cv_accuracy = min_cv_accuracy
        self.rth_only = rth_only
        self.drop_flat = drop_flat

    def _cache_suffix(self) -> str:
        """Config-specific suffix for cache filenames."""
        cfg = self.label_config
        rth = "_rth" if self.rth_only else ""
        # v6: causal regime + shifted multi-timeframe joins (no future leakage)
        return f"_v6_h{cfg.vertical_barrier_bars}_tp{cfg.tp_ticks}_sl{cfg.sl_ticks}{rth}"

    def run(
        self, start_date: str, end_date: str, val_days: int = 15
    ) -> TrainingResult:
        """Execute full training pipeline."""
        cfg = self.label_config

        print(f"\n{'='*70}")
        print(f"  TICK DIRECTION PREDICTOR — TRAINING PIPELINE")
        print(f"  Data range:  {start_date} to {end_date}")
        print(f"  Val days:    {val_days}")
        print(f"  Horizon:     {cfg.vertical_barrier_bars} bars ({cfg.vertical_barrier_bars}s)")
        print(f"  TP/SL:       {cfg.tp_ticks}/{cfg.sl_ticks} ticks "
              f"({cfg.tp_ticks * cfg.tick_size:.2f}/{cfg.sl_ticks * cfg.tick_size:.2f} pts)")
        print(f"  RTH only:    {self.rth_only}")
        print(f"  Drop FLAT:   {self.drop_flat}")
        print(f"{'='*70}\n")

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # ── STEP 1: Feature generation ──────────────────────────
        print("  STEP 1: Feature generation...")
        t0 = _time.perf_counter()

        suffix = self._cache_suffix()
        features_path = DATA_DIR / f"features_{start_date}_{end_date}{suffix}.parquet"
        bars_df = None  # reuse for labeling
        if features_path.exists():
            print(f"    [Cache HIT] {features_path}")
            features_df = pl.read_parquet(features_path)
        else:
            features_df, bars_df = self._generate_features(start_date, end_date)
            features_df.write_parquet(features_path)
            print(f"    Saved features -> {features_path}")

        print(f"    Features: {len(features_df):,} rows ({_time.perf_counter()-t0:.1f}s)")

        # ── STEP 2: Label generation ────────────────────────────
        print("\n  STEP 2: Label generation...")
        t0 = _time.perf_counter()

        labels_path = DATA_DIR / f"labels_{start_date}_{end_date}{suffix}.parquet"
        if labels_path.exists():
            print(f"    [Cache HIT] {labels_path}")
            labels_df = pl.read_parquet(labels_path)
        else:
            labeler = TripleBarrierLabeler(self.label_config)
            if bars_df is not None:
                labels_df = labeler.generate_labels_from_bars(bars_df)
            else:
                labels_df = labeler.generate_labels(start_date, end_date)
            labeler.save_labels(labels_df, start_date, end_date)

        dist = labels_df.group_by("label").len().sort("label")
        total = len(labels_df)
        for row in dist.iter_rows(named=True):
            pct = row["len"] / total * 100
            name = {-1: "DOWN", 0: "FLAT", 1: "UP"}.get(row["label"], "?")
            print(f"    {name}: {row['len']:,} ({pct:.1f}%)")
            if pct < 20:
                print(f"    WARNING: {name} class < 20%")
        print(f"    Labels: {total:,} rows ({_time.perf_counter()-t0:.1f}s)")

        # ── STEP 3: Join & validate ─────────────────────────────
        print("\n  STEP 3: Join & validate...")

        joined = features_df.join(labels_df, on="timestamp_ns", how="inner")
        joined = joined.drop_nulls(subset=FEATURE_NAMES)
        print(f"    Joined: {len(joined):,} rows (from {len(features_df):,} features + {len(labels_df):,} labels)")

        # Drop FLAT class (label == 0) — typically <2% of samples at long horizons
        if self.drop_flat:
            n_before = len(joined)
            joined = joined.filter(pl.col("label") != 0)
            n_dropped = n_before - len(joined)
            print(f"    Dropped FLAT: {n_dropped:,} rows ({n_dropped/n_before*100:.1f}%)")
            print(f"    After drop:   {len(joined):,} rows")

        # Temporal split
        import datetime as dt
        end_dt = dt.date.fromisoformat(end_date)
        val_start = end_dt - timedelta(days=val_days)
        val_start_ns = int(dt.datetime.combine(val_start, dt.time.min).timestamp() * 1e9)

        train_df = joined.filter(pl.col("timestamp_ns") < val_start_ns)
        val_df = joined.filter(pl.col("timestamp_ns") >= val_start_ns)

        print(f"    Train: {len(train_df):,} rows")
        print(f"    Val:   {len(val_df):,} rows (last {val_days} days)")

        if len(train_df) < 1000:
            return TrainingResult(
                cv_accuracy_mean=0, cv_accuracy_std=0, val_ece=1.0,
                val_brier_score=1.0, val_accuracy=0, temperature=1.0,
                top_features={}, training_samples=0, val_samples=0,
                model_path="", calibrator_path="", success=False,
                error_message="Insufficient training data",
            )

        # Convert to numpy
        X_train = train_df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
        y_train_raw = train_df["label"].to_numpy()
        w_train = train_df["sample_weight"].to_numpy().astype(np.float32)

        X_val = val_df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
        y_val_raw = val_df["label"].to_numpy()

        # Encode labels: -1->0, 0->1, 1->2
        label_map = {-1: 0, 0: 1, 1: 2}
        y_train = np.vectorize(label_map.get)(y_train_raw).astype(np.int32)
        y_val = np.vectorize(label_map.get)(y_val_raw).astype(np.int32)

        # ── STEP 4: Walk-forward CV ─────────────────────────────
        print("\n  STEP 4: Walk-forward CV...")
        t0 = _time.perf_counter()

        trainer = ModelTrainer(self.trainer_config)
        cv_results = trainer.purged_walkforward_cv(X_train, y_train, w_train)

        mean_acc = float(np.mean([r.accuracy for r in cv_results]))
        std_acc = float(np.std([r.accuracy for r in cv_results]))

        print(f"\n    CV Results ({len(cv_results)} folds):")
        print(f"    {'Fold':>4} {'Acc':>7} {'F1':>7} {'Brier':>7} {'LogLoss':>8}")
        for r in cv_results:
            print(f"    {r.fold:>4} {r.accuracy:>7.4f} {r.f1_macro:>7.4f} "
                  f"{r.brier_score:>7.4f} {r.log_loss:>8.4f}")
        print(f"    Mean accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"    ({_time.perf_counter()-t0:.1f}s)")

        if mean_acc < self.min_cv_accuracy:
            print(f"\n    ABORT: Mean CV accuracy {mean_acc:.4f} < {self.min_cv_accuracy}")
            return TrainingResult(
                cv_accuracy_mean=mean_acc, cv_accuracy_std=std_acc,
                val_ece=1.0, val_brier_score=1.0, val_accuracy=0,
                temperature=1.0, top_features={}, training_samples=len(X_train),
                val_samples=len(X_val), model_path="", calibrator_path="",
                success=False, error_message=f"CV accuracy {mean_acc:.4f} < {self.min_cv_accuracy}",
            )

        # ── STEP 5: Final model training ────────────────────────
        print("\n  STEP 5: Final model training...")
        t0 = _time.perf_counter()

        booster = trainer.train_final_model(X_train, y_train, w_train)
        model_path = str(MODEL_DIR / "lgbm_latest.txt")
        print(f"    Model saved ({_time.perf_counter()-t0:.1f}s)")

        # Feature importances
        imp = booster.feature_importance(importance_type="gain")
        top_features = {
            FEATURE_NAMES[i]: float(imp[i])
            for i in np.argsort(imp)[::-1][:10]
        }
        print(f"    Top features: {list(top_features.keys())[:5]}")

        # ── STEP 6: Calibration ─────────────────────────────────
        print("\n  STEP 6: Temperature calibration...")

        raw_proba_val = booster.predict(X_val)
        if raw_proba_val.ndim == 1:
            raw_proba_val = raw_proba_val.reshape(-1, 3)

        calibrator = TemperatureCalibrator()
        calibrator.fit(raw_proba_val, y_val)

        cal_proba = calibrator.predict_proba_calibrated(raw_proba_val)
        val_ece = calibrator.compute_ece(cal_proba, y_val)
        val_brier = calibrator.compute_brier_score(cal_proba, y_val)
        val_accuracy = float(np.mean(np.argmax(cal_proba, axis=1) == y_val))

        calibrator_path = str(MODEL_DIR / "calibrator_latest.pkl")
        calibrator.save(calibrator_path)

        # Reliability diagram
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        calibrator.plot_reliability_diagram(
            cal_proba, y_val, class_idx=2,
            save_path=str(MODEL_DIR / f"reliability_{ts}.png"),
        )

        print(f"    Temperature: {calibrator.temperature:.4f}")
        print(f"    Val ECE:     {val_ece:.4f}")
        print(f"    Val Brier:   {val_brier:.4f}")
        print(f"    Val Acc:     {val_accuracy:.4f}")

        # ── STEP 7: Summary ─────────────────────────────────────
        result = TrainingResult(
            cv_accuracy_mean=mean_acc,
            cv_accuracy_std=std_acc,
            val_ece=val_ece,
            val_brier_score=val_brier,
            val_accuracy=val_accuracy,
            temperature=calibrator.temperature,
            top_features=top_features,
            training_samples=len(X_train),
            val_samples=len(X_val),
            model_path=model_path,
            calibrator_path=calibrator_path,
            success=True,
        )

        print(f"\n{'='*70}")
        print(f"  TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"  CV accuracy:    {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"  Val accuracy:   {val_accuracy:.4f}")
        print(f"  Val ECE:        {val_ece:.4f}")
        print(f"  Val Brier:      {val_brier:.4f}")
        print(f"  Temperature:    {calibrator.temperature:.4f}")
        print(f"  Train samples:  {len(X_train):,}")
        print(f"  Val samples:    {len(X_val):,}")
        print(f"  Model:          {model_path}")
        print(f"  Calibrator:     {calibrator_path}")
        print()

        # Save result JSON
        result_path = MODEL_DIR / f"training_result_{ts}.json"
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        print(f"  Result saved -> {result_path}")

        return result

    def _generate_features(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Generate features vectorized via Polars (fast).

        Uses L1 tick data when available (real bid/ask sizes), otherwise
        falls back to 1s OHLCV bars with approximated order flow.
        """
        import datetime as dt

        start = dt.date.fromisoformat(start_date)
        end = dt.date.fromisoformat(end_date)

        # Try L1 data first
        l1_paths = [
            f"data/l1/year={y}/data.parquet"
            for y in range(start.year, end.year + 1)
            if Path(f"data/l1/year={y}/data.parquet").exists()
        ]

        if l1_paths:
            bars_df = self._aggregate_l1_to_bars(l1_paths, start, end)
            print(f"    Using L1 data (real bid/ask sizes)")
        else:
            bars_df = self._load_ohlcv_bars(start, end)
            print(f"    Using OHLCV data (approximated order flow)")

        print(f"    Loaded {len(bars_df):,} 1s bars (all hours)")

        # RTH filter: 9:30 AM - 4:00 PM ET
        if self.rth_only:
            from datetime import time as dt_time
            ts_dtype = bars_df.schema["timestamp"]
            # Handle timezone-aware vs naive timestamps
            ts_col = pl.col("timestamp")
            if hasattr(ts_dtype, "time_zone") and ts_dtype.time_zone:
                et_col = ts_col.dt.convert_time_zone("US/Eastern")
            else:
                et_col = ts_col.dt.replace_time_zone("UTC").dt.convert_time_zone("US/Eastern")
            bars_df = bars_df.with_columns(et_col.alias("_et_ts"))
            bars_df = bars_df.filter(
                (pl.col("_et_ts").dt.time() >= dt_time(9, 30))
                & (pl.col("_et_ts").dt.time() < dt_time(16, 0))
            ).drop("_et_ts")
            print(f"    After RTH filter: {len(bars_df):,} bars")

        # Compute regime features (join_asof from 5m regime to 1s bars)
        bars_df = self._add_regime_features(bars_df)

        features = self._vectorized_features(bars_df)
        return features, bars_df

    def _vectorized_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute all 26 features using Polars column operations."""
        import time as _t
        t0 = _t.perf_counter()

        MES_TICK = 0.25

        has_l1 = "avg_bid_size" in df.columns and df["avg_bid_size"].sum() > 0

        if has_l1:
            # Real L1 order flow
            df = df.with_columns([
                pl.col("avg_bid_size").fill_null(0.0).alias("bid_size"),
                pl.col("avg_ask_size").fill_null(0.0).alias("ask_size"),
                (pl.col("aggressive_buy_vol").fill_null(0.0)
                 - pl.col("aggressive_sell_vol").fill_null(0.0)).alias("cvd_delta"),
            ])
        else:
            # OHLCV approximations
            hl_denom = (pl.col("high") - pl.col("low")) + 1e-9
            ratio = (pl.col("close") - pl.col("low")) / hl_denom
            df = df.with_columns([
                (pl.col("volume").cast(pl.Float64) * (1.0 - ratio)).alias("bid_size"),
                (pl.col("volume").cast(pl.Float64) * ratio).alias("ask_size"),
                (pl.col("volume").cast(pl.Float64) * (2.0 * ratio - 1.0)).alias("cvd_delta"),
            ])

        # Spread in ticks
        df = df.with_columns(
            ((pl.col("high") - pl.col("low")) / MES_TICK).alias("spread_ticks"),
        )

        # Log returns
        df = df.with_columns([
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_ret_1"),
            (pl.col("close") / pl.col("close").shift(5)).log().alias("log_ret_5"),
            (pl.col("close") / pl.col("close").shift(10)).log().alias("log_ret_10"),
        ])

        print(f"    Computing order flow features...")

        # ── Tier 1: Order Flow ──────────────────────────────────
        # OFI: sum of (bid_diff - ask_diff) over window
        df = df.with_columns([
            (pl.col("bid_size").diff().fill_null(0.0)
             - pl.col("ask_size").diff().fill_null(0.0)).alias("_ofi_delta"),
        ])
        df = df.with_columns([
            pl.col("_ofi_delta").rolling_sum(10).alias("ofi_10_raw"),
            pl.col("_ofi_delta").rolling_sum(30).alias("ofi_30_raw"),
        ])

        # OBI: (bid - ask) / (bid + ask)
        total_sz = pl.col("bid_size") + pl.col("ask_size")
        obi_expr = pl.when(total_sz > 0).then(
            (pl.col("bid_size") - pl.col("ask_size")) / total_sz
        ).otherwise(0.0)
        df = df.with_columns([
            obi_expr.alias("obi_1_raw"),
            obi_expr.rolling_mean(5).alias("obi_5_raw"),
        ])

        # Microprice
        half_spread = pl.lit(MES_TICK)
        bid_px = pl.col("close") - half_spread
        ask_px = pl.col("close") + half_spread
        mp = pl.when(total_sz > 0).then(
            (pl.col("bid_size") * ask_px + pl.col("ask_size") * bid_px) / total_sz
        ).otherwise(pl.col("close"))
        df = df.with_columns([
            mp.alias("microprice_raw"),
            (mp - pl.col("close")).alias("microprice_vs_mid_raw"),
        ])

        print(f"    Computing CVD & volume features...")

        # ── Tier 2: CVD & Volume ────────────────────────────────
        df = df.with_columns([
            pl.col("cvd_delta").rolling_sum(10).alias("cvd_delta_10_raw"),
            pl.col("cvd_delta").rolling_sum(30).alias("cvd_delta_30_raw"),
        ])

        # CVD slope: rolling linear regression over 20 bars on cumulative CVD
        # Vectorized closed-form: slope = (n*Σ(x*y) - Σx*Σy) / (n*Σx² - (Σx)²)
        # For fixed x = 0..n-1, Σx and Σx² are constants. We only need
        # rolling Σy and rolling Σ(x*y). Σ(x*y) = Σ(i * y_{t-n+1+i}) for i=0..n-1
        # which equals (n-1)*rolling_sum(y) - rolling_sum of cumulative-shifted y.
        # Simpler: use the identity slope = 12/(n(n²-1)) * Σ((i-(n-1)/2)*y_i)
        # The weights w_i = i - (n-1)/2 for i=0..n-1 are symmetric: -9.5,-8.5,...,9.5
        # Σ(w_i * y_i) = Σ(i*y_i) - (n-1)/2 * Σy_i
        # We can compute Σ(i*y_i) via: Σ(i*y_{t-n+1+i}) = (n-1)*y_t + (n-2)*y_{t-1}+...
        # But easier: just use numpy stride_tricks for zero-copy rolling windows
        df = df.with_columns(pl.col("cvd_delta").cum_sum().alias("_cvd_cumsum"))
        n_w = 20
        cvd_arr = df["_cvd_cumsum"].to_numpy()
        x = np.arange(n_w, dtype=np.float64)
        x_mean = x.mean()
        x_var = np.sum((x - x_mean) ** 2)
        # Stride-trick: create [N-n_w+1, n_w] view without copying
        from numpy.lib.stride_tricks import sliding_window_view
        if len(cvd_arr) >= n_w:
            windows = sliding_window_view(cvd_arr, n_w)  # [N-19, 20]
            y_means = windows.mean(axis=1)
            # slope = Σ((x_i - x_mean)(y_i - y_mean)) / Σ((x_i - x_mean)²)
            slopes_valid = np.einsum("j,ij->i", x - x_mean, windows - y_means[:, None]) / x_var
            slopes = np.full(len(cvd_arr), np.nan, dtype=np.float64)
            slopes[n_w - 1 :] = slopes_valid
        else:
            slopes = np.full(len(cvd_arr), np.nan, dtype=np.float64)
        df = df.with_columns(pl.Series("cvd_slope_raw", slopes))

        # Volume imbalance
        buy_vol = pl.col("cvd_delta").clip(lower_bound=0.0)
        sell_vol = (-pl.col("cvd_delta")).clip(lower_bound=0.0)
        vol_sum_10 = pl.col("volume").cast(pl.Float64).rolling_sum(10)
        df = df.with_columns([
            pl.when(vol_sum_10 > 0).then(
                (buy_vol.rolling_sum(10) - sell_vol.rolling_sum(10)) / vol_sum_10
            ).otherwise(0.0).alias("volume_imbalance_10_raw"),
        ])

        # Volume z-score
        vol_f = pl.col("volume").cast(pl.Float64)
        df = df.with_columns([
            pl.when(vol_f.rolling_std(20) > 0).then(
                (vol_f - vol_f.rolling_mean(20)) / vol_f.rolling_std(20)
            ).otherwise(0.0).alias("volume_zscore_20_raw"),
        ])

        print(f"    Computing price action features...")

        # ── Tier 3: Price Action & Volatility ───────────────────
        df = df.with_columns([
            pl.col("log_ret_1").alias("return_1_raw"),
            pl.col("log_ret_5").alias("return_5_raw"),
            pl.col("log_ret_10").alias("return_10_raw"),
            pl.col("log_ret_1").rolling_std(20).alias("realized_vol_20_raw"),
        ])

        # Return autocorrelation (lag-1 over 10 bars) — vectorized via stride tricks
        # autocorr = cov(r[:-1], r[1:]) / var(r) within each 10-bar window
        log_rets = df["log_ret_1"].to_numpy().copy()
        log_rets = np.where(np.isfinite(log_rets), log_rets, 0.0)
        n_ac = 10
        autocorr = np.full(len(log_rets), np.nan, dtype=np.float64)
        if len(log_rets) >= n_ac:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(log_rets, n_ac)  # [N-9, 10]
            w_mean = windows.mean(axis=1, keepdims=True)
            dm = windows - w_mean
            var = np.sum(dm ** 2, axis=1) / n_ac
            cov = np.sum(dm[:, :-1] * dm[:, 1:], axis=1) / (n_ac - 1)
            ac = np.where(var > 1e-18, cov / var, 0.0)
            autocorr[n_ac - 1 :] = ac
        df = df.with_columns(pl.Series("return_autocorr_10_raw", autocorr))

        # Spread features
        df = df.with_columns([
            pl.col("spread_ticks").alias("spread_raw"),
            pl.when(pl.col("spread_ticks").rolling_std(50) > 0).then(
                (pl.col("spread_ticks") - pl.col("spread_ticks").rolling_mean(50))
                / pl.col("spread_ticks").rolling_std(50)
            ).otherwise(0.0).alias("spread_zscore_50_raw"),
        ])

        print(f"    Computing multi-timeframe technicals...")

        # ── Tier 4: Regime features (already joined from _add_regime_features) ──
        df = df.with_columns([
            pl.col("regime_p_trending").alias("regime_p_trending_raw"),
            pl.col("regime_p_ranging").alias("regime_p_ranging_raw"),
            pl.col("regime_p_highvol").alias("regime_p_highvol_raw"),
            pl.col("regime_bars_in").alias("regime_bars_in_raw"),
        ])

        # ── Tier 5: 1m Tactical (Bollinger %B, RSI, ATR ratio, OBV slope) ──
        from src.data.bars import resample_bars

        bars_1m = resample_bars(
            df.select(["timestamp", "open", "high", "low", "close", "volume"]), "1m"
        )
        c_1m = pl.col("close")

        # Bollinger %B (20, 2)
        bb_mid = c_1m.rolling_mean(20)
        bb_std = c_1m.rolling_std(20)
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std
        bb_range = bb_upper - bb_lower
        bars_1m = bars_1m.with_columns(
            pl.when(bb_range > 1e-12)
            .then((c_1m - bb_lower) / bb_range)
            .otherwise(0.5)
            .alias("bb_pctb_1m_raw")
        )

        # RSI-14
        pc = c_1m.diff()
        gain = pl.when(pc > 0).then(pc).otherwise(0.0)
        loss = pl.when(pc < 0).then(-pc).otherwise(0.0)
        bars_1m = bars_1m.with_columns(
            pl.when(loss.rolling_mean(14) > 1e-12)
            .then(100.0 - 100.0 / (1.0 + gain.rolling_mean(14) / loss.rolling_mean(14)))
            .otherwise(100.0)
            .alias("rsi_14_1m_raw")
        )

        # ATR ratio: ATR(5) / ATR(20)
        tr_1m = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - c_1m.shift(1)).abs(),
            (pl.col("low") - c_1m.shift(1)).abs(),
        )
        bars_1m = bars_1m.with_columns(tr_1m.alias("_tr"))
        atr5 = pl.col("_tr").rolling_mean(5)
        atr20 = pl.col("_tr").rolling_mean(20)
        bars_1m = bars_1m.with_columns(
            pl.when(atr20 > 1e-12).then(atr5 / atr20).otherwise(1.0)
            .alias("atr_ratio_1m_raw")
        )

        # OBV slope (10-bar linear regression on cumulative OBV)
        signed_vol = pl.when(c_1m.diff() > 0).then(pl.col("volume").cast(pl.Float64))  \
            .when(c_1m.diff() < 0).then(-pl.col("volume").cast(pl.Float64))  \
            .otherwise(0.0)
        bars_1m = bars_1m.with_columns(signed_vol.cum_sum().alias("_obv"))
        obv_arr = bars_1m["_obv"].to_numpy()
        n_1m = len(obv_arr)
        obv_slope = np.full(n_1m, np.nan, dtype=np.float64)
        if n_1m >= 10:
            from numpy.lib.stride_tricks import sliding_window_view
            x = np.arange(10, dtype=np.float64)
            x_mean = x.mean()
            x_var = np.sum((x - x_mean) ** 2)
            wins = sliding_window_view(obv_arr, 10)
            y_means = wins.mean(axis=1)
            slopes = np.einsum("j,ij->i", x - x_mean, wins - y_means[:, None]) / x_var
            obv_slope[9:] = slopes
        bars_1m = bars_1m.with_columns(pl.Series("obv_slope_1m_raw", obv_slope))

        # Join 1m -> 1s (shift timestamps +1m so 1s bars only see PREVIOUS completed 1m bar)
        tac_cols = ["bb_pctb_1m_raw", "rsi_14_1m_raw", "atr_ratio_1m_raw", "obv_slope_1m_raw"]
        bars_1m = bars_1m.with_columns(
            (pl.col("timestamp") + pl.duration(minutes=1)).dt.epoch("ns").alias("_tf_ts_ns")
        ).select(["_tf_ts_ns"] + tac_cols).sort("_tf_ts_ns")
        df = df.sort("timestamp_ns").join_asof(
            bars_1m, left_on="timestamp_ns", right_on="_tf_ts_ns", strategy="backward",
        )
        for col in tac_cols:
            df = df.with_columns(pl.col(col).fill_null(strategy="forward"))
        print(f"    1m tactical: {len(bars_1m):,} bars")

        # ── Tier 6: 5m Strategic (Donchian pos, Stoch K, BB width, EMA cross) ──
        bars_5m = resample_bars(
            df.select(["timestamp", "open", "high", "low", "close", "volume"]), "5m"
        )
        c_5m = pl.col("close")

        # Donchian position: (close - low20) / (high20 - low20)
        h20 = pl.col("high").rolling_max(20)
        l20 = pl.col("low").rolling_min(20)
        don_range = h20 - l20
        bars_5m = bars_5m.with_columns(
            pl.when(don_range > 1e-12)
            .then((c_5m - l20) / don_range)
            .otherwise(0.5)
            .alias("donchian_pos_5m_raw")
        )

        # Stochastic K-14
        h14 = pl.col("high").rolling_max(14)
        l14 = pl.col("low").rolling_min(14)
        stoch_range = h14 - l14
        bars_5m = bars_5m.with_columns(
            pl.when(stoch_range > 1e-12)
            .then((c_5m - l14) / stoch_range * 100.0)
            .otherwise(50.0)
            .alias("stoch_k_14_5m_raw")
        )

        # Bollinger bandwidth: (upper - lower) / middle
        bb5_mid = c_5m.rolling_mean(20)
        bb5_std = c_5m.rolling_std(20)
        bars_5m = bars_5m.with_columns(
            pl.when(bb5_mid > 1e-12)
            .then(4.0 * bb5_std / bb5_mid)
            .otherwise(0.0)
            .alias("bb_width_5m_raw")
        )

        # EMA cross: (EMA20 - EMA50) / ATR
        ema20 = c_5m.ewm_mean(span=20, adjust=False)
        ema50 = c_5m.ewm_mean(span=50, adjust=False)
        tr_5m = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - c_5m.shift(1)).abs(),
            (pl.col("low") - c_5m.shift(1)).abs(),
        )
        bars_5m = bars_5m.with_columns(tr_5m.rolling_mean(14).alias("_atr_5m"))
        bars_5m = bars_5m.with_columns(
            pl.when(pl.col("_atr_5m") > 1e-12)
            .then((ema20 - ema50) / pl.col("_atr_5m"))
            .otherwise(0.0)
            .alias("ema_cross_5m_raw")
        )

        # Join 5m -> 1s (shift timestamps +5m so 1s bars only see PREVIOUS completed 5m bar)
        strat_cols = ["donchian_pos_5m_raw", "stoch_k_14_5m_raw",
                      "bb_width_5m_raw", "ema_cross_5m_raw"]
        bars_5m = bars_5m.with_columns(
            (pl.col("timestamp") + pl.duration(minutes=5)).dt.epoch("ns").alias("_tf_ts_ns")
        ).select(["_tf_ts_ns"] + strat_cols).sort("_tf_ts_ns")
        df = df.sort("timestamp_ns").join_asof(
            bars_5m, left_on="timestamp_ns", right_on="_tf_ts_ns", strategy="backward",
        )
        for col in strat_cols:
            df = df.with_columns(pl.col(col).fill_null(strategy="forward"))
        print(f"    5m strategic: {len(bars_5m):,} bars")

        print(f"    Z-score normalizing all features...")

        # ── Z-score normalize all features (rolling 100-bar) ────
        raw_cols = [
            "ofi_10_raw", "ofi_30_raw", "obi_1_raw", "obi_5_raw",
            "microprice_raw", "microprice_vs_mid_raw",
            "cvd_delta_10_raw", "cvd_delta_30_raw", "cvd_slope_raw",
            "volume_imbalance_10_raw", "volume_zscore_20_raw",
            "return_1_raw", "return_5_raw", "return_10_raw",
            "realized_vol_20_raw", "return_autocorr_10_raw",
            "spread_raw", "spread_zscore_50_raw",
            "regime_p_trending_raw", "regime_p_ranging_raw",
            "regime_p_highvol_raw", "regime_bars_in_raw",
            "bb_pctb_1m_raw", "rsi_14_1m_raw", "atr_ratio_1m_raw", "obv_slope_1m_raw",
            "donchian_pos_5m_raw", "stoch_k_14_5m_raw", "bb_width_5m_raw", "ema_cross_5m_raw",
        ]
        norm_exprs = []
        for raw_col, feat_name in zip(raw_cols, FEATURE_NAMES):
            col = pl.col(raw_col)
            roll_mean = col.rolling_mean(100)
            roll_std = col.rolling_std(100)
            norm_exprs.append(
                pl.when(roll_std > 1e-12)
                .then((col - roll_mean) / roll_std)
                .otherwise(0.0)
                .alias(feat_name)
            )
        df = df.with_columns(norm_exprs)

        # Drop warmup rows (first 100 bars have incomplete rolling stats)
        df = df.slice(100)

        # Drop rows with any null in feature columns
        df = df.drop_nulls(subset=FEATURE_NAMES)

        elapsed = _t.perf_counter() - t0
        print(f"    Vectorized features: {len(df):,} rows in {elapsed:.1f}s")

        return df.select(["timestamp_ns"] + FEATURE_NAMES)

    def _add_regime_features(self, bars_df: pl.DataFrame) -> pl.DataFrame:
        """Add regime_v2 features via join_asof from 5m regime predictions."""
        from pathlib import Path as _Path
        regime_model_path = _Path("models/regime_v2")
        if not regime_model_path.exists():
            print("    [SKIP] Regime model not found at models/regime_v2 — filling with zeros")
            return bars_df.with_columns([
                pl.lit(0.0).alias("regime_p_trending"),
                pl.lit(0.0).alias("regime_p_ranging"),
                pl.lit(0.0).alias("regime_p_highvol"),
                pl.lit(0.0).alias("regime_bars_in"),
            ])

        print("    Computing regime_v2 features...")
        t0_r = _time.perf_counter()

        from src.data.bars import resample_bars
        from src.models.regime_detector_v2 import (
            RegimeDetectorV2,
            build_features_v2,
        )

        # Resample to 5m bars for regime model
        bars_5m = resample_bars(bars_df, "5m")
        print(f"    Resampled to {len(bars_5m):,} 5m bars")

        # Load trained regime model and run inference
        detector = RegimeDetectorV2.load(str(regime_model_path))
        features_5m, ts_5m = build_features_v2(bars_5m, detector.config)
        probas = detector.predict_proba_sequence(features_5m, causal=True)

        # Build regime DataFrame aligned to 5m timestamps
        regime_records = []
        for i, proba in enumerate(probas):
            regime_records.append({
                "timestamp_ns": int(ts_5m[i]),
                "regime_p_trending": float(proba.probabilities[0]),
                "regime_p_ranging": float(proba.probabilities[1]),
                "regime_p_highvol": float(proba.probabilities[2]),
                "regime_bars_in": float(proba.bars_in_regime),
            })
        regime_df = pl.DataFrame(regime_records).sort("timestamp_ns")
        print(f"    Regime predictions: {len(regime_df):,} rows")

        # join_asof: shift regime timestamps +5m so 1s bars only see PREVIOUS completed 5m regime
        regime_df = regime_df.with_columns(
            (pl.col("timestamp_ns") + 5 * 60 * 1_000_000_000).alias("timestamp_ns")
        )
        bars_df = bars_df.sort("timestamp_ns")
        bars_df = bars_df.join_asof(
            regime_df,
            on="timestamp_ns",
            strategy="backward",
        )
        # Fill NaN from warmup period with zeros
        for col in ["regime_p_trending", "regime_p_ranging", "regime_p_highvol", "regime_bars_in"]:
            bars_df = bars_df.with_columns(pl.col(col).fill_null(0.0))

        elapsed_r = _time.perf_counter() - t0_r
        print(f"    Regime features added ({elapsed_r:.1f}s)")
        return bars_df

    def _aggregate_l1_to_bars(
        self, paths: list[str], start, end
    ) -> pl.DataFrame:
        """Aggregate L1 ticks into 1s bars with real bid/ask fields."""
        import datetime as dt

        start_dt = dt.datetime.combine(start, dt.time.min, tzinfo=dt.timezone.utc)
        end_dt = dt.datetime.combine(end, dt.time.min, tzinfo=dt.timezone.utc)

        print(f"    Loading L1 ticks...")
        ticks = (
            pl.scan_parquet(paths)
            .filter(
                (pl.col("timestamp") >= start_dt)
                & (pl.col("timestamp") < end_dt)
            )
            .sort("timestamp")
            .collect()
        )
        print(f"    {len(ticks):,} L1 ticks loaded")

        # Truncate to 1-second bars
        bars = (
            ticks
            .with_columns(
                pl.col("timestamp").dt.truncate("1s").alias("bar_ts"),
            )
            .group_by("bar_ts")
            .agg([
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("size").sum().alias("volume"),
                # Real bid/ask sizes (time-weighted average within bar)
                pl.col("bid_size").mean().cast(pl.Float64).alias("avg_bid_size"),
                pl.col("ask_size").mean().cast(pl.Float64).alias("avg_ask_size"),
                # Aggressive buy/sell volume
                pl.col("size").filter(pl.col("side") == "B").sum().alias("aggressive_buy_vol"),
                pl.col("size").filter(pl.col("side") == "S").sum().alias("aggressive_sell_vol"),
            ])
            .sort("bar_ts")
            .with_columns(
                pl.col("bar_ts").dt.epoch("ns").alias("timestamp_ns"),
                pl.col("bar_ts").alias("timestamp"),
            )
        )

        # Fill nulls for buy/sell vol (bars with only one side)
        bars = bars.with_columns([
            pl.col("aggressive_buy_vol").fill_null(0).cast(pl.Float64),
            pl.col("aggressive_sell_vol").fill_null(0).cast(pl.Float64),
        ])

        return bars

    def _load_ohlcv_bars(self, start, end) -> pl.DataFrame:
        """Load pre-built 1s OHLCV bars (fallback when no L1 data)."""
        import datetime as dt

        years = range(start.year, end.year + 1)
        paths = [
            f"data/parquet/year={y}/data.parquet"
            for y in years
            if Path(f"data/parquet/year={y}/data.parquet").exists()
        ]
        if not paths:
            raise FileNotFoundError(f"No parquet files for {start} to {end}")

        start_dt = dt.datetime.combine(start, dt.time.min)
        end_dt = dt.datetime.combine(end, dt.time.min)

        bars_df = (
            pl.scan_parquet(paths)
            .filter(
                (pl.col("timestamp") >= start_dt)
                & (pl.col("timestamp") < end_dt)
            )
            .select(["timestamp", "open", "high", "low", "close", "volume"])
            .sort("timestamp")
            .collect()
        )
        bars_df = bars_df.with_columns(
            (pl.col("timestamp").dt.epoch("ns")).alias("timestamp_ns")
        )
        return bars_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TickDirectionPredictor")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--val-days", type=int, default=15, help="Validation days (default: 15)")
    parser.add_argument("--min-cv-accuracy", type=float, default=0.52)
    parser.add_argument("--horizon", type=int, default=60,
                        help="Prediction horizon in 1s bars (default: 60 = 1 min)")
    parser.add_argument("--tp-ticks", type=int, default=8,
                        help="Take-profit barrier in MES ticks (default: 8 = 2 pts)")
    parser.add_argument("--sl-ticks", type=int, default=6,
                        help="Stop-loss barrier in MES ticks (default: 6 = 1.5 pts)")
    parser.add_argument("--rth-only", action="store_true", default=True,
                        help="Filter to RTH bars only (9:30-16:00 ET, default: true)")
    parser.add_argument("--no-rth-filter", dest="rth_only", action="store_false",
                        help="Include all hours (ETH + RTH)")
    parser.add_argument("--drop-flat", action="store_true", default=True,
                        help="Drop FLAT labels (default: true)")
    parser.add_argument("--keep-flat", dest="drop_flat", action="store_false",
                        help="Keep FLAT labels")
    parser.add_argument("--sweep", action="store_true", default=False,
                        help="Run hyperparameter sweep")
    args = parser.parse_args()

    configure_logging(log_level="INFO")

    label_config = TripleBarrierConfig(
        vertical_barrier_bars=args.horizon,
        tp_ticks=args.tp_ticks,
        sl_ticks=args.sl_ticks,
    )

    if args.sweep:
        _run_sweep(args, label_config)
    else:
        pipeline = TrainingPipeline(
            label_config=label_config,
            min_cv_accuracy=args.min_cv_accuracy,
            rth_only=args.rth_only,
            drop_flat=args.drop_flat,
        )
        result = pipeline.run(args.start, args.end, val_days=args.val_days)

        if not result.success:
            print(f"\n  TRAINING FAILED: {result.error_message}")
            sys.exit(1)


def _run_sweep(args, label_config: TripleBarrierConfig) -> None:
    """Grid search over hyperparameters with walk-forward CV."""
    import itertools

    sweep_grid = {
        "learning_rate": [0.005, 0.01],
        "n_estimators": [1000, 2000],
        "num_leaves": [31, 127],
        "early_stopping_rounds": [50, 100],
        "min_child_samples": [100, 200],
    }

    # Generate all combinations
    keys = list(sweep_grid.keys())
    values = list(sweep_grid.values())
    combos = list(itertools.product(*values))

    print(f"\n{'='*70}")
    print(f"  HYPERPARAMETER SWEEP — {len(combos)} configurations")
    print(f"{'='*70}\n")

    results_dir = Path("results/tick_predictor")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for idx, combo in enumerate(combos):
        config_dict = dict(zip(keys, combo))
        print(f"\n  [{idx+1}/{len(combos)}] {config_dict}")

        trainer_config = TrainerConfig(**config_dict)
        pipeline = TrainingPipeline(
            label_config=label_config,
            trainer_config=trainer_config,
            min_cv_accuracy=0.0,  # Don't abort during sweep
            rth_only=args.rth_only,
            drop_flat=args.drop_flat,
        )
        result = pipeline.run(args.start, args.end, val_days=args.val_days)

        entry = {
            "config": config_dict,
            "cv_accuracy_mean": result.cv_accuracy_mean,
            "cv_accuracy_std": result.cv_accuracy_std,
            "val_accuracy": result.val_accuracy,
            "val_ece": result.val_ece,
            "val_brier_score": result.val_brier_score,
            "success": result.success,
        }
        all_results.append(entry)

        print(f"    CV: {result.cv_accuracy_mean:.4f} +/- {result.cv_accuracy_std:.4f}  "
              f"Val: {result.val_accuracy:.4f}")

    # Sort by CV accuracy
    all_results.sort(key=lambda x: x["cv_accuracy_mean"], reverse=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_path = results_dir / f"sweep_{ts}.json"
    with open(sweep_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE — {len(all_results)} configs")
    print(f"  Best config: {all_results[0]['config']}")
    print(f"  Best CV acc: {all_results[0]['cv_accuracy_mean']:.4f}")
    print(f"  Best Val acc: {all_results[0]['val_accuracy']:.4f}")
    print(f"  Results: {sweep_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
