"""Temperature scaling calibrator for LightGBM probability outputs.

Divides logits by a scalar T > 0, then re-applies softmax.  This recalibrates
confidence levels without changing prediction rankings (argmax is preserved).
"""

from __future__ import annotations

import os
import pickle
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from src.core.logging import get_logger

logger = get_logger("tick_predictor.calibrator")

MODEL_DIR = Path("models/tick_predictor")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class TemperatureCalibrator:
    """Temperature scaling for multi-class probability calibration."""

    def __init__(
        self,
        recal_window: int = 1000,
        ece_alert_threshold: float = 0.08,
        n_bins: int = 15,
    ) -> None:
        self.temperature: float = 1.0
        self.n_bins = n_bins
        self.ece_alert_threshold = ece_alert_threshold
        self._recal_deque: deque[tuple[np.ndarray, int]] = deque(maxlen=recal_window)
        self._fit_timestamp: str | None = None

    def fit(self, raw_proba: np.ndarray, y_val: np.ndarray) -> None:
        """Find optimal temperature on validation set.

        Args:
            raw_proba: shape [N, 3] — raw softmax from LightGBM
            y_val: shape [N,] — true labels encoded as 0/1/2
        """
        logits = np.log(np.clip(raw_proba, 1e-9, 1.0))
        n = len(y_val)

        def nll(T: float) -> float:
            scaled = _softmax(logits / T)
            probs = scaled[np.arange(n), y_val]
            return -float(np.mean(np.log(np.clip(probs, 1e-9, 1.0))))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        self._fit_timestamp = datetime.now().isoformat()

        ece = self.compute_ece(self.predict_proba_calibrated(raw_proba), y_val)
        logger.info("calibrator_fit", temperature=f"{self.temperature:.4f}", ece=f"{ece:.4f}")

    def predict_proba_calibrated(self, raw_proba: np.ndarray) -> np.ndarray:
        """Apply temperature scaling. Hot path — must be <20us for single row.

        Args:
            raw_proba: shape [1, 3] or [N, 3]

        Returns:
            Calibrated probabilities, same shape.
        """
        logits = np.log(np.clip(raw_proba, 1e-9, 1.0))
        scaled = logits / self.temperature
        return _softmax(scaled)

    def update(
        self, recent_raw_proba: np.ndarray, recent_y_true: np.ndarray
    ) -> float:
        """Rolling recalibration from recent predictions.

        Args:
            recent_raw_proba: shape [N, 3]
            recent_y_true: shape [N,] encoded 0/1/2

        Returns:
            New ECE after refit.
        """
        for i in range(len(recent_y_true)):
            self._recal_deque.append((recent_raw_proba[i], int(recent_y_true[i])))

        if len(self._recal_deque) < 50:
            return 1.0

        all_proba = np.array([p for p, _ in self._recal_deque])
        all_y = np.array([y for _, y in self._recal_deque], dtype=np.int32)

        self.fit(all_proba, all_y)
        cal_proba = self.predict_proba_calibrated(all_proba)
        new_ece = self.compute_ece(cal_proba, all_y)

        if new_ece > self.ece_alert_threshold:
            logger.warning("calibrator_ece_alert",
                           ece=f"{new_ece:.4f}",
                           temperature=f"{self.temperature:.4f}")

        # Atomic save
        self.save(str(MODEL_DIR / "calibrator_latest.pkl"))

        return new_ece

    def compute_ece(
        self, proba: np.ndarray, y_true: np.ndarray, n_bins: int | None = None
    ) -> float:
        """Expected Calibration Error with equal-width bins."""
        n_bins = n_bins or self.n_bins
        confidences = np.max(proba, axis=1)
        predictions = np.argmax(proba, axis=1)
        accuracies = (predictions == y_true).astype(np.float64)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_true)

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            count = int(np.sum(mask))
            if count > 0:
                avg_conf = float(np.mean(confidences[mask]))
                avg_acc = float(np.mean(accuracies[mask]))
                ece += (count / n) * abs(avg_acc - avg_conf)

        return ece

    def compute_brier_score(self, proba: np.ndarray, y_true: np.ndarray) -> float:
        """Multi-class Brier score."""
        n = len(y_true)
        one_hot = np.zeros((n, 3), dtype=np.float64)
        one_hot[np.arange(n), y_true] = 1.0
        return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))

    def plot_reliability_diagram(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
        class_idx: int = 2,
        save_path: str | None = None,
    ) -> None:
        """Generate reliability diagram for a specific class."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        class_proba = proba[:, class_idx]
        is_class = (y_true == class_idx).astype(np.float64)

        n_bins = self.n_bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accs = []
        bin_counts = []

        for i in range(n_bins):
            mask = (class_proba > bin_boundaries[i]) & (class_proba <= bin_boundaries[i + 1])
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
        ax1.set_title(f"Reliability Diagram (class={['DOWN','FLAT','UP'][class_idx]})")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.bar(bin_centers, bin_counts, width=1.0 / n_bins * 0.8,
                alpha=0.3, color="tab:orange", label="Count")
        ax2.set_ylabel("Sample count")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=100)
            logger.info("reliability_diagram_saved", path=save_path)
        plt.close(fig)

    def save(self, path: str) -> None:
        """Persist calibrator state via atomic write."""
        data = {
            "temperature": self.temperature,
            "recal_deque": list(self._recal_deque),
            "fit_timestamp": self._fit_timestamp,
            "n_bins": self.n_bins,
            "ece_alert_threshold": self.ece_alert_threshold,
        }
        dir_path = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=dir_path, delete=False, suffix=".tmp"
        ) as f:
            pickle.dump(data, f)
            tmp_path = f.name
        os.replace(tmp_path, path)

    def load(self, path: str) -> None:
        """Restore calibrator from pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.temperature = data["temperature"]
        self._recal_deque = deque(data.get("recal_deque", []),
                                  maxlen=self._recal_deque.maxlen)
        self._fit_timestamp = data.get("fit_timestamp")
        self.n_bins = data.get("n_bins", self.n_bins)
        self.ece_alert_threshold = data.get("ece_alert_threshold", self.ece_alert_threshold)
        logger.info("calibrator_loaded", path=path, temperature=f"{self.temperature:.4f}")
