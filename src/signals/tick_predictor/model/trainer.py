"""LightGBM trainer with purged walk-forward cross-validation.

Trains a 3-class (DOWN/FLAT/UP) direction predictor on features from
FeatureBuilder and labels from TripleBarrierLabeler.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.core.logging import get_logger
from src.signals.tick_predictor.features.feature_builder import FEATURE_NAMES

logger = get_logger("tick_predictor.trainer")

MODEL_DIR = Path("models/tick_predictor")

# Label encoding: -1 -> 0 (DOWN), 0 -> 1 (FLAT), 1 -> 2 (UP)
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}


@dataclass
class CVResult:
    fold: int
    train_size: int
    test_size: int
    accuracy: float
    f1_macro: float
    brier_score: float
    log_loss: float
    feature_importances: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainerConfig:
    num_leaves: int = 63
    max_depth: int = 7
    learning_rate: float = 0.03
    n_estimators: int = 300
    subsample: float = 0.7
    colsample_bytree: float = 0.8
    min_child_samples: int = 50
    early_stopping_rounds: int = 30
    n_cv_splits: int = 5
    embargo_bars: int = 20


class ModelTrainer:
    """Train LightGBM direction predictor with purged walk-forward CV."""

    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()

    def load_training_data(
        self, features_path: str, labels_path: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and join feature/label Parquet files.

        Returns:
            (X [N, 18], y [N,] encoded 0/1/2, weights [N,])
        """
        import polars as pl

        features_df = pl.read_parquet(features_path)
        labels_df = pl.read_parquet(labels_path)

        # Inner join on timestamp_ns
        df = features_df.join(labels_df, on="timestamp_ns", how="inner")

        # Drop rows with nulls in feature columns
        feat_cols = [c for c in df.columns if c in FEATURE_NAMES]
        df = df.drop_nulls(subset=feat_cols)

        total = len(df)
        logger.info("training_data_loaded",
                     total_rows=total,
                     features=len(feat_cols))

        # Label distribution
        dist = df.group_by("label").len().sort("label")
        for row in dist.iter_rows(named=True):
            logger.info("label_dist", label=int(row["label"]), count=row["len"],
                        pct=f"{row['len']/total*100:.1f}%")

        X = df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
        y_raw = df["label"].to_numpy()
        # Encode: -1->0, 0->1, 1->2
        y = np.vectorize(LABEL_MAP.get)(y_raw).astype(np.int32)
        weights = df["sample_weight"].to_numpy().astype(np.float32)

        return X, y, weights

    def purged_walkforward_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        n_splits: int | None = None,
        embargo_bars: int | None = None,
    ) -> list[CVResult]:
        """Purged walk-forward cross-validation (temporal, no shuffle)."""
        from sklearn.metrics import accuracy_score, f1_score, log_loss

        n_splits = n_splits or self.config.n_cv_splits
        embargo = embargo_bars or self.config.embargo_bars
        n = len(X)
        # Expanding window: first 50% is minimum train, remaining 50% split into test folds
        min_train_frac = 0.5
        min_train = int(n * min_train_frac)
        test_total = n - min_train
        fold_size = test_total // n_splits

        results: list[CVResult] = []

        for fold in range(n_splits):
            test_start = min_train + fold * fold_size
            test_end = min_train + (fold + 1) * fold_size if fold < n_splits - 1 else n
            train_end_purged = max(0, test_start - embargo)

            if test_end <= test_start or train_end_purged < 100:
                continue

            X_train = X[:train_end_purged]
            y_train = y[:train_end_purged]
            w_train = weights[:train_end_purged]

            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            if len(X_test) < 100:
                continue

            # Internal validation for early stopping (last 10% of train)
            val_split = max(1, int(len(X_train) * 0.1))
            X_tr, X_val = X_train[:-val_split], X_train[-val_split:]
            y_tr, y_val = y_train[:-val_split], y_train[-val_split:]
            w_tr = w_train[:-val_split]

            params = self._lgb_params()
            train_set = lgb.Dataset(X_tr, y_tr, weight=w_tr,
                                    feature_name=FEATURE_NAMES, free_raw_data=False)
            val_set = lgb.Dataset(X_val, y_val, reference=train_set, free_raw_data=False)

            callbacks = [
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]
            booster = lgb.train(
                params, train_set,
                num_boost_round=self.config.n_estimators,
                valid_sets=[val_set],
                callbacks=callbacks,
            )

            raw_proba = booster.predict(X_test)
            y_pred = np.argmax(raw_proba, axis=1)

            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            ll = float(log_loss(y_test, raw_proba, labels=[0, 1, 2]))

            # Multi-class Brier
            n_test = len(y_test)
            one_hot = np.zeros((n_test, 3), dtype=np.float64)
            one_hot[np.arange(n_test), y_test] = 1.0
            brier = float(np.mean(np.sum((raw_proba - one_hot) ** 2, axis=1)))

            # Feature importances (gain)
            imp = booster.feature_importance(importance_type="gain")
            imp_dict = {
                FEATURE_NAMES[i]: float(imp[i])
                for i in np.argsort(imp)[::-1][:10]
            }

            cv_result = CVResult(
                fold=fold, train_size=len(X_tr), test_size=len(X_test),
                accuracy=acc, f1_macro=f1, brier_score=brier,
                log_loss=ll, feature_importances=imp_dict,
            )
            results.append(cv_result)

            logger.info("cv_fold_complete",
                        fold=fold, accuracy=f"{acc:.4f}", f1=f"{f1:.4f}",
                        brier=f"{brier:.4f}", log_loss=f"{ll:.4f}",
                        train_size=len(X_tr), test_size=len(X_test))

        # Summary
        if results:
            mean_acc = float(np.mean([r.accuracy for r in results]))
            std_acc = float(np.std([r.accuracy for r in results]))
            logger.info("cv_summary",
                        mean_accuracy=f"{mean_acc:.4f}",
                        std_accuracy=f"{std_acc:.4f}",
                        n_folds=len(results))
            if mean_acc < 0.52:
                logger.warning("cv_low_accuracy",
                               mean_accuracy=mean_acc,
                               message="Mean CV accuracy < 0.52 — model may not be predictive")

        return results

    def train_final_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> lgb.Booster:
        """Train on full dataset, save model + importances."""
        # Use last 10% as early-stopping val
        val_split = max(1, int(len(X) * 0.1))
        X_tr, X_val = X[:-val_split], X[-val_split:]
        y_tr, y_val = y[:-val_split], y[-val_split:]
        w_tr = weights[:-val_split]

        params = self._lgb_params()
        train_set = lgb.Dataset(X_tr, y_tr, weight=w_tr,
                                feature_name=FEATURE_NAMES, free_raw_data=False)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        booster = lgb.train(
            params, train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=[val_set],
            callbacks=callbacks,
        )

        # Save
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = MODEL_DIR / f"lgbm_{ts}.txt"
        booster.save_model(str(model_path))

        # Feature importances
        imp = booster.feature_importance(importance_type="gain")
        imp_dict = {
            FEATURE_NAMES[i]: float(imp[i])
            for i in np.argsort(imp)[::-1]
        }
        imp_path = MODEL_DIR / f"importance_{ts}.json"
        with open(imp_path, "w") as f:
            json.dump(imp_dict, f, indent=2)

        # Atomic symlink swap
        latest_path = MODEL_DIR / "lgbm_latest.txt"
        tmp_link = MODEL_DIR / f".lgbm_latest_{ts}.tmp"
        try:
            os.symlink(model_path.name, str(tmp_link))
            os.replace(str(tmp_link), str(latest_path))
        except OSError:
            # Windows fallback: just copy
            import shutil
            shutil.copy2(str(model_path), str(latest_path))

        logger.info("model_saved",
                     path=str(model_path),
                     latest=str(latest_path),
                     n_trees=booster.num_trees(),
                     top_features=dict(list(imp_dict.items())[:10]))

        return booster

    def compute_shap_values(
        self, model: lgb.Booster, X_sample: np.ndarray
    ) -> np.ndarray:
        """Compute SHAP values and save summary plot."""
        try:
            import shap
        except ImportError:
            logger.warning("shap_not_installed", message="pip install shap for SHAP analysis")
            return np.zeros((len(X_sample), len(FEATURE_NAMES), 3))

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # shap_values is list of 3 arrays [N, 18] for each class
        if isinstance(shap_values, list) and len(shap_values) == 3:
            shap_arr = np.stack(shap_values, axis=-1)  # [N, 18, 3]

            # Log mean |SHAP| for UP class (index 2)
            mean_shap = np.mean(np.abs(shap_values[2]), axis=0)
            shap_imp = {
                FEATURE_NAMES[i]: float(mean_shap[i])
                for i in np.argsort(mean_shap)[::-1][:10]
            }
            logger.info("shap_importance_up_class", **shap_imp)

            # Save plot
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fig, ax = plt.subplots(figsize=(10, 6))
                sorted_idx = np.argsort(mean_shap)[::-1]
                ax.barh(
                    [FEATURE_NAMES[i] for i in sorted_idx],
                    [mean_shap[i] for i in sorted_idx],
                )
                ax.set_xlabel("Mean |SHAP|")
                ax.set_title("SHAP Feature Importance (UP class)")
                ax.invert_yaxis()
                plt.tight_layout()
                save_path = MODEL_DIR / f"shap_{ts}.png"
                fig.savefig(str(save_path), dpi=100)
                plt.close(fig)
                logger.info("shap_plot_saved", path=str(save_path))
            except Exception as e:
                logger.warning("shap_plot_failed", error=str(e))

            return shap_arr

        return np.zeros((len(X_sample), len(FEATURE_NAMES), 3))

    # ── internals ───────────────────────────────────────────────

    def _lgb_params(self) -> dict:
        cfg = self.config
        return {
            "objective": "multiclass",
            "num_class": 3,
            "num_leaves": cfg.num_leaves,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "min_child_samples": cfg.min_child_samples,
            "is_unbalance": True,
            "device": "gpu",
            "verbose": -1,
            "seed": 42,
        }
