"""Evaluation metrics for TickDirectionPredictor.

ECE, Brier score, reliability diagrams, threshold sweep, and DSR.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from src.core.logging import get_logger

logger = get_logger("tick_predictor.evaluation")

MODEL_DIR = Path("models/tick_predictor")
MES_TICK_SIZE = 0.25


class PredictorEvaluator:
    """Comprehensive evaluation suite for the tick direction predictor."""

    def compute_ece(
        self, proba: np.ndarray, y_true: np.ndarray, n_bins: int = 15
    ) -> float:
        """Expected Calibration Error with equal-width bins."""
        confidences = np.max(proba, axis=1)
        predictions = np.argmax(proba, axis=1)
        accuracies = (predictions == y_true).astype(np.float64)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_true)

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (
                confidences <= bin_boundaries[i + 1]
            )
            count = int(np.sum(mask))
            if count > 0:
                avg_conf = float(np.mean(confidences[mask]))
                avg_acc = float(np.mean(accuracies[mask]))
                ece += (count / n) * abs(avg_acc - avg_conf)

        return ece

    def plot_reliability_diagram(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
        class_idx: int = 2,
        save_path: str | None = None,
    ) -> None:
        """Reliability diagram: fraction of positives vs mean predicted probability."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        class_proba = proba[:, class_idx]
        is_class = (y_true == class_idx).astype(np.float64)

        n_bins = 15
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accs = []
        bin_counts = []

        for i in range(n_bins):
            mask = (class_proba > bin_boundaries[i]) & (
                class_proba <= bin_boundaries[i + 1]
            )
            count = int(np.sum(mask))
            if count > 0:
                bin_centers.append(float(np.mean(class_proba[mask])))
                bin_accs.append(float(np.mean(is_class[mask])))
                bin_counts.append(count)

        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax1.plot(bin_centers, bin_accs, "o-", color="tab:blue", label="Model")
        ax1.set_xlabel("Mean predicted probability")
        ax1.set_ylabel("Fraction of positives")
        ax1.set_title(
            f"Reliability Diagram (class={['DOWN','FLAT','UP'][class_idx]})"
        )
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.bar(
            bin_centers,
            bin_counts,
            width=1.0 / n_bins * 0.8,
            alpha=0.3,
            color="tab:orange",
        )
        ax2.set_ylabel("Sample count")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100)
            logger.info("reliability_diagram_saved", path=save_path)
        plt.close(fig)

    def compute_brier_score(self, proba: np.ndarray, y_true: np.ndarray) -> float:
        """Multi-class Brier score."""
        n = len(y_true)
        one_hot = np.zeros((n, 3), dtype=np.float64)
        one_hot[np.arange(n), y_true] = 1.0
        return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))

    def compute_deflated_sharpe(
        self, returns: np.ndarray, n_trials: int
    ) -> float:
        """Deflated Sharpe Ratio via existing bot implementation."""
        try:
            from src.backtesting.dsr import DeflatedSharpeCalculator

            calc = DeflatedSharpeCalculator()
            result = calc.compute(
                strategy_returns=returns,
                n_trials=n_trials,
            )
            return result.dsr
        except Exception as e:
            logger.warning("dsr_computation_failed", error=str(e))
            return 0.0

    def compute_threshold_sweep(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
        cost_per_trade_points: float = 0.15,
        tp_ticks: int = 4,
        sl_ticks: int = 3,
    ) -> pl.DataFrame:
        """Sweep confidence thresholds to find optimal operating point.

        Args:
            proba: [N, 3] calibrated probabilities
            y_true: [N,] true labels (0=DOWN, 1=FLAT, 2=UP)
            cost_per_trade_points: round-trip cost in points
            tp_ticks: take-profit in ticks for PnL estimation
            sl_ticks: stop-loss in ticks for PnL estimation
        """
        avg_tp = tp_ticks * MES_TICK_SIZE  # points won on correct
        avg_sl = sl_ticks * MES_TICK_SIZE  # points lost on incorrect

        thresholds = np.arange(0.50, 0.91, 0.02)
        rows = []

        for thresh in thresholds:
            # For each sample, check if max class probability exceeds threshold
            max_proba = np.max(proba, axis=1)
            mask = max_proba >= thresh
            n_trade = int(np.sum(mask))
            trade_pct = n_trade / len(proba) if len(proba) > 0 else 0

            if n_trade == 0:
                rows.append({
                    "threshold": float(thresh),
                    "trade_pct": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "est_pnl_per_trade": 0.0,
                    "sharpe_proxy": 0.0,
                })
                continue

            preds = np.argmax(proba[mask], axis=1)
            actuals = y_true[mask]

            # Precision: fraction where prediction == actual
            correct = (preds == actuals).astype(np.float64)
            precision = float(np.mean(correct))

            # Recall: of all directional signals, how many caught?
            directional_mask = y_true != 1  # not FLAT
            n_directional = int(np.sum(directional_mask))
            if n_directional > 0:
                directional_caught = int(
                    np.sum((max_proba >= thresh) & directional_mask & (np.argmax(proba, axis=1) == y_true))
                )
                recall = directional_caught / n_directional
            else:
                recall = 0.0

            # Estimated PnL per trade
            est_pnl = precision * avg_tp - (1 - precision) * avg_sl - cost_per_trade_points

            # Sharpe proxy: mean / std of per-trade returns
            per_trade_returns = np.where(correct, avg_tp, -avg_sl) - cost_per_trade_points
            if len(per_trade_returns) > 1:
                std_ret = float(np.std(per_trade_returns, ddof=1))
                sharpe_proxy = float(np.mean(per_trade_returns)) / std_ret if std_ret > 0 else 0.0
            else:
                sharpe_proxy = 0.0

            rows.append({
                "threshold": float(thresh),
                "trade_pct": trade_pct,
                "precision": precision,
                "recall": recall,
                "est_pnl_per_trade": est_pnl,
                "sharpe_proxy": sharpe_proxy,
            })

        return pl.DataFrame(rows)

    def generate_full_report(
        self,
        model,
        calibrator,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_dir: str = "models/tick_predictor/",
    ) -> dict:
        """Run all metrics, generate plots, return summary dict."""
        from sklearn.metrics import accuracy_score

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Predictions
        raw_proba = model.predict(X_test)
        if raw_proba.ndim == 1:
            raw_proba = raw_proba.reshape(-1, 3)

        cal_proba = calibrator.predict_proba_calibrated(raw_proba)
        y_pred = np.argmax(cal_proba, axis=1)
        accuracy = float(accuracy_score(y_test, y_pred))

        # Metrics
        ece = self.compute_ece(cal_proba, y_test)
        brier = self.compute_brier_score(cal_proba, y_test)

        # Reliability diagram
        rel_path = str(save_path / f"reliability_{ts}.png")
        self.plot_reliability_diagram(cal_proba, y_test, class_idx=2, save_path=rel_path)

        # Threshold sweep
        sweep_df = self.compute_threshold_sweep(cal_proba, y_test)

        logger.info("evaluation_complete",
                     accuracy=f"{accuracy:.4f}",
                     ece=f"{ece:.4f}",
                     brier=f"{brier:.4f}")

        # Print threshold sweep
        print("\n  THRESHOLD SWEEP:")
        print(f"  {'Thresh':>7} {'Trade%':>7} {'Prec':>6} {'Recall':>7} {'PnL/trade':>10} {'Sharpe':>7}")
        for row in sweep_df.iter_rows(named=True):
            print(f"  {row['threshold']:>7.2f} {row['trade_pct']:>7.1%} "
                  f"{row['precision']:>6.1%} {row['recall']:>7.1%} "
                  f"${row['est_pnl_per_trade']:>9.3f} {row['sharpe_proxy']:>7.2f}")

        report = {
            "ece": ece,
            "brier_score": brier,
            "accuracy": accuracy,
            "threshold_sweep": sweep_df.to_dicts(),
            "reliability_path": rel_path,
            "timestamp": ts,
        }

        # Save report JSON
        report_path = save_path / f"eval_{ts}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("report_saved", path=str(report_path))

        return report
