"""TickPredictorSignal — SignalBase integration for direction prediction.

Registered as "tick_predictor" in SignalRegistry.  Returns a signed float
encoding direction + confidence: positive = UP, negative = DOWN, ~0 = FLAT.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.events import BarEvent
from src.core.logging import get_logger
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry
from src.signals.tick_predictor.features.feature_builder import (
    FEATURE_NAMES,
    NUM_FEATURES,
    FeatureBuilder,
)

logger = get_logger("tick_predictor.signal")

DIRECTION_NAMES = ["DOWN", "FLAT", "UP"]

# Label encoding: model output index -> direction
# 0 = DOWN, 1 = FLAT, 2 = UP


@dataclass
class PredictionRecord:
    timestamp_ns: int
    raw_proba: np.ndarray  # shape (3,)
    direction: str
    confidence: float


@SignalRegistry.register
class TickPredictorSignal(SignalBase):
    """Predict tick direction 10-20 bars ahead from OHLCV-1s features."""

    name = "tick_predictor"

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self._horizon_ticks = config.get("horizon_ticks", 15)
        self._min_confidence = config.get("min_confidence", 0.60)
        model_path = config.get("model_path", "models/tick_predictor/lgbm_latest.txt")
        calibrator_path = config.get(
            "calibrator_path", "models/tick_predictor/calibrator_latest.pkl"
        )

        self._feature_builder = FeatureBuilder(
            capacity=config.get("ring_buffer_capacity", 200)
        )
        self._feature_buf = np.zeros((1, NUM_FEATURES), dtype=np.float32)

        # Optional regime signal for HMM features
        self._regime_signal = None

        # Load model (lazy — only if file exists)
        self._model = None
        self._calibrator = None
        self._model_loaded = False

        if Path(model_path).exists():
            try:
                import lightgbm as lgb
                self._model = lgb.Booster(model_file=model_path)
                # Validate feature count matches current FEATURE_NAMES
                model_n_features = self._model.num_feature()
                if model_n_features != NUM_FEATURES:
                    logger.warning(
                        "tick_predictor_model_feature_mismatch",
                        model_features=model_n_features,
                        expected_features=NUM_FEATURES,
                        message="Model needs retraining with new features",
                    )
                    self._model = None
                else:
                    logger.info("tick_predictor_model_loaded", path=model_path)
            except Exception as e:
                logger.warning("tick_predictor_model_load_failed", error=str(e))

        if Path(calibrator_path).exists():
            try:
                from src.signals.tick_predictor.model.calibrator import (
                    TemperatureCalibrator,
                )
                self._calibrator = TemperatureCalibrator()
                self._calibrator.load(calibrator_path)
            except Exception as e:
                logger.warning("tick_predictor_calibrator_load_failed", error=str(e))

        self._model_loaded = self._model is not None
        self._is_warm = False
        self._last_prediction: PredictionRecord | None = None

        # Latency tracking
        self._latency_deque: deque[float] = deque(maxlen=1000)

        # Prediction/outcome history for nightly recalibration
        self._prediction_history: deque[PredictionRecord] = deque(maxlen=2000)
        self._outcome_history: deque[int] = deque(maxlen=2000)
        self._pending_outcomes: deque[tuple[int, float]] = deque(maxlen=2000)

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        """Compute direction prediction from latest bar.

        Called by SignalEngine on every bar.
        """
        if not bars:
            return self._neutral("no_bars")

        latest_bar = bars[-1]

        # Inject regime probabilities if available
        if self._regime_signal is not None:
            self._update_regime_features()

        # Feed bar into feature builder
        feature_vector = self._feature_builder.on_bar(latest_bar)

        # Track outcomes for past predictions
        self._track_outcomes(latest_bar)

        # Not warm yet — features contain NaN
        if not self._feature_builder.is_warm:
            return self._neutral("warming_up")

        # No model loaded
        if not self._model_loaded:
            return self._neutral("no_model")

        # Check for NaN in features
        if np.any(np.isnan(feature_vector)):
            return self._neutral("nan_features")

        # Inference
        self._feature_buf[0, :] = feature_vector
        t0 = time.perf_counter_ns()

        raw_proba = self._model.predict(self._feature_buf)  # [1, 3]
        if raw_proba.ndim == 1:
            raw_proba = raw_proba.reshape(1, -1)

        if self._calibrator is not None:
            cal_proba = self._calibrator.predict_proba_calibrated(raw_proba)
        else:
            cal_proba = raw_proba

        latency_us = (time.perf_counter_ns() - t0) / 1000.0
        self._latency_deque.append(latency_us)

        if latency_us > 500:
            logger.warning("tick_predictor_slow_inference", latency_us=f"{latency_us:.0f}")

        direction_idx = int(np.argmax(cal_proba[0]))
        direction = DIRECTION_NAMES[direction_idx]
        confidence = float(cal_proba[0, direction_idx])

        self._is_warm = True

        # Encode direction + confidence as signed float
        if direction == "UP":
            value = confidence
        elif direction == "DOWN":
            value = -confidence
        else:
            value = 0.0

        # Record prediction for recalibration
        record = PredictionRecord(
            timestamp_ns=latest_bar.timestamp_ns,
            raw_proba=raw_proba[0].copy(),
            direction=direction,
            confidence=confidence,
        )
        self._last_prediction = record
        self._prediction_history.append(record)
        self._pending_outcomes.append((latest_bar.timestamp_ns, latest_bar.close))

        return SignalResult(
            value=value,
            passes=True,
            direction="long" if direction == "UP" else ("short" if direction == "DOWN" else "none"),
            metadata={
                "direction": direction,
                "confidence": confidence,
                "up_prob": float(cal_proba[0, 2]),
                "down_prob": float(cal_proba[0, 0]),
                "flat_prob": float(cal_proba[0, 1]),
                "is_warm": self._is_warm,
                "horizon_ticks": self._horizon_ticks,
                "latency_us": latency_us,
            },
        )

    def reset(self) -> None:
        """Clear state on session close."""
        self._feature_builder.reset()
        self._is_warm = False
        self._last_prediction = None
        self._latency_deque.clear()
        self._pending_outcomes.clear()
        logger.info("tick_predictor_reset")

    async def on_session_close(self, session_close_time_ns: int) -> None:
        """Nightly recalibration from day's predictions."""
        if self._calibrator is None:
            return

        if len(self._prediction_history) < 50:
            logger.warning("tick_predictor_skip_recal",
                           reason="insufficient_samples",
                           count=len(self._prediction_history))
            return

        # Build arrays from matched prediction/outcome pairs
        n = min(len(self._prediction_history), len(self._outcome_history))
        if n < 50:
            logger.warning("tick_predictor_skip_recal",
                           reason="insufficient_outcomes", count=n)
            return

        raw_proba = np.array([
            p.raw_proba for p in list(self._prediction_history)[:n]
        ])
        y_true = np.array(list(self._outcome_history)[:n], dtype=np.int32)

        new_ece = self._calibrator.update(raw_proba, y_true)
        logger.info("tick_predictor_recalibrated",
                     ece=f"{new_ece:.4f}",
                     temperature=f"{self._calibrator.temperature:.4f}",
                     n_samples=n)

    @classmethod
    def from_yaml(cls, config: dict) -> TickPredictorSignal:
        """Construct from YAML config dict."""
        return cls(config=config)

    def set_regime_signal(self, regime_signal) -> None:
        """Wire up a RegimeV2Signal instance for HMM features."""
        self._regime_signal = regime_signal

    # ── internals ───────────────────────────────────────────────

    def _update_regime_features(self) -> None:
        """Read latest regime probabilities and push into feature builder."""
        result = getattr(self._regime_signal, "_last_result", None)
        if result is None:
            return
        meta = result.metadata if hasattr(result, "metadata") else {}
        probas = meta.get("probabilities")
        if probas and isinstance(probas, dict):
            self._feature_builder.set_regime_proba(
                p_trending=float(probas.get("TRENDING", 0.0)),
                p_ranging=float(probas.get("RANGING", 0.0)),
                p_highvol=float(probas.get("HIGH_VOL", 0.0)),
                bars_in=int(meta.get("bars_in_regime", 0)),
            )

    def _neutral(self, reason: str) -> SignalResult:
        return SignalResult(
            value=0.0,
            passes=True,
            direction="none",
            metadata={"is_warm": False, "reason": reason},
        )

    def _track_outcomes(self, current_bar: BarEvent) -> None:
        """Check pending predictions that have reached their horizon."""
        current_close = current_bar.close
        current_ts = current_bar.timestamp_ns
        horizon_ns = self._horizon_ticks * 1_000_000_000  # 1s bars

        while self._pending_outcomes:
            pred_ts, pred_close = self._pending_outcomes[0]
            if current_ts - pred_ts >= horizon_ns:
                self._pending_outcomes.popleft()
                move = current_close - pred_close
                if move > 0:
                    label = 2  # UP
                elif move < 0:
                    label = 0  # DOWN
                else:
                    label = 1  # FLAT
                self._outcome_history.append(label)
            else:
                break
