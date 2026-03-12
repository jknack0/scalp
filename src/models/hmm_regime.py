"""HMM Intraday Regime Classifier — 2-state, 3-feature, online-capable.

3 features per bar: realized_vol, vpin_approx, return_autocorr.
All z-score normalized with a 500-bar rolling window, clipped to [-3, 3].

Usage:
    # Training (offline)
    features, timestamps = build_feature_matrix(df_bars, config)
    clf = HMMRegimeClassifier(config)
    clf.fit(features)
    clf.save("models/hmm/v4")

    # Batch prediction (backtest)
    states = clf.predict_sequence(features)

    # Online prediction (live)
    state = clf.predict(bar_dict)          # returns int (0/1)
    proba = clf.regime_proba(bar_dict)     # returns np.ndarray (2,)
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

N_FEATURES = 3


class NotReadyError(Exception):
    """Raised when predict is called before warm-up buffer is saturated."""


class RegimeState(IntEnum):
    """Intraday regime states (2 or 3 state models)."""

    RANGE_BOUND = 0   # Low vol, mean-reverting — good for VWAP reversion
    VOLATILE = 1      # High vol / trending / crisis — avoid mean reversion
    TRENDING = 2      # Moderate vol, directional — used by 3-state models


DEFAULT_FEATURE_COLUMNS: list[str] = [
    "realized_vol",
    "vpin_approx",
    "return_autocorr",
]


@dataclass(frozen=True)
class HMMRegimeConfig:
    """Configuration for the HMM regime classifier."""

    n_states: int = 2
    n_iter: int = 200
    covariance_type: str = "full"
    random_state: int = 42
    zscore_window: int = 500
    warmup_bars: int = 500
    rvol_window: int = 50
    vpin_window: int = 50
    autocorr_window: int = 30
    predict_window: int = 100  # sliding window for online HMM predict
    feature_columns: list[str] = field(
        default_factory=lambda: list(DEFAULT_FEATURE_COLUMNS)
    )


# ── Vectorized helpers (batch mode) ─────────────────────────────────


def _rolling_autocorr(arr: np.ndarray, window: int = 30) -> np.ndarray:
    """Vectorized rolling lag-1 Pearson autocorrelation of returns."""
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window + 1:
        return result

    from numpy.lib.stride_tricks import sliding_window_view

    # curr[i] = arr[i-window+1 : i+1], prev[i] = arr[i-window : i]
    curr = sliding_window_view(arr[1:], window)    # (n-1-window+1, window)
    prev = sliding_window_view(arr[:-1], window)   # same shape

    cm = curr - curr.mean(axis=1, keepdims=True)
    pm = prev - prev.mean(axis=1, keepdims=True)

    num = np.sum(cm * pm, axis=1)
    denom = np.sqrt(np.sum(cm**2, axis=1) * np.sum(pm**2, axis=1))
    denom = np.where(denom < 1e-10, 1.0, denom)

    # First valid index: window (need window+1 data points incl. lag)
    result[window:] = num / denom
    return result


def _rolling_zscore(raw: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score normalization, clipped to [-3, 3]."""
    n, k = raw.shape
    normed = np.full_like(raw, np.nan)

    for col in range(k):
        series = pl.Series("v", raw[:, col])
        r_mean = series.rolling_mean(window_size=window, min_samples=2).to_numpy()
        r_std = series.rolling_std(window_size=window, min_samples=2).to_numpy()
        r_std = np.where((r_std < 1e-10) | np.isnan(r_std), 1.0, r_std)
        normed[:, col] = (raw[:, col] - r_mean) / r_std

    return np.clip(normed, -3.0, 3.0)


# ── Batch feature extraction ────────────────────────────────────────


def build_feature_matrix(
    df: pl.DataFrame,
    config: HMMRegimeConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the HMM feature matrix from bar data using vectorized operations.

    Computes 3 features: realized_vol, vpin_approx, return_autocorr.
    All z-score normalized with rolling window, clipped to [-3, 3].

    Returns:
        (features, timestamps) where features is (N, 3) float64 and
        timestamps is (N,) int64 nanosecond timestamps. Warm-up rows dropped.
    """
    cfg = config or HMMRegimeConfig()

    close = df["close"].to_numpy().astype(np.float64)
    open_ = df["open"].to_numpy().astype(np.float64)
    volume = df["volume"].to_numpy().astype(np.float64)
    n = len(close)

    # Timestamps
    ts_col = df["timestamp"]
    if ts_col.dtype == pl.Datetime or str(ts_col.dtype).startswith("Datetime"):
        ts_arr = ts_col.dt.epoch("ns").to_numpy().astype(np.int64)
    else:
        ts_arr = ts_col.to_numpy().astype(np.int64)

    # Log returns
    log_ret = np.zeros(n, dtype=np.float64)
    log_ret[1:] = np.log(
        np.maximum(close[1:], 1e-10) / np.maximum(close[:-1], 1e-10)
    )

    # ── 1. Realized vol — rolling std of log returns ─────────────────
    realized_vol = (
        pl.Series("lr", log_ret)
        .rolling_std(window_size=cfg.rvol_window, min_samples=2)
        .to_numpy()
    )

    # ── 2. VPIN approx — tick rule over rolling window ───────────────
    buy_mask = close > open_
    buy_vol = np.where(buy_mask, volume, 0.0)
    sell_vol = np.where(~buy_mask, volume, 0.0)

    buy_sum = (
        pl.Series("bv", buy_vol)
        .rolling_sum(window_size=cfg.vpin_window, min_samples=1)
        .to_numpy()
    )
    sell_sum = (
        pl.Series("sv", sell_vol)
        .rolling_sum(window_size=cfg.vpin_window, min_samples=1)
        .to_numpy()
    )
    total_sum = (
        pl.Series("tv", volume)
        .rolling_sum(window_size=cfg.vpin_window, min_samples=1)
        .to_numpy()
    )
    total_safe = np.where(total_sum < 1, 1.0, total_sum)
    vpin_approx = np.abs(buy_sum - sell_sum) / total_safe

    # ── 3. Return autocorrelation — rolling lag-1 Pearson ────────────
    return_autocorr = _rolling_autocorr(log_ret, window=cfg.autocorr_window)

    # ── Stack raw features (N, 3) ────────────────────────────────────
    raw = np.column_stack([realized_vol, vpin_approx, return_autocorr])

    # ── Rolling z-score normalization, clip [-3, 3] ──────────────────
    normed = _rolling_zscore(raw, window=cfg.zscore_window)

    # Drop warm-up rows
    start = cfg.zscore_window
    valid = ~np.any(np.isnan(normed[start:]), axis=1) & ~np.any(
        np.isinf(normed[start:]), axis=1
    )

    return normed[start:][valid], ts_arr[start:][valid]


# ── HMM Classifier ──────────────────────────────────────────────────


class HMMRegimeClassifier:
    """2-state Gaussian HMM regime classifier with online + batch prediction.

    Stateful: maintains rolling windows internally for online bar-by-bar
    prediction. Thread-safe via Lock around state mutations.
    """

    def __init__(self, config: HMMRegimeConfig | None = None) -> None:
        self.config = config or HMMRegimeConfig()
        self.model: GaussianHMM | None = None
        self.state_map: dict[int, RegimeState] | None = None
        self._lock = threading.Lock()
        self._reset_online_state()

    def _reset_online_state(self) -> None:
        """Reset all rolling windows for online prediction."""
        cfg = self.config
        self._prev_close: float | None = None
        self._log_returns: deque[float] = deque(maxlen=cfg.zscore_window + 1)
        self._buy_vol_window: deque[float] = deque(maxlen=cfg.vpin_window)
        self._sell_vol_window: deque[float] = deque(maxlen=cfg.vpin_window)
        self._raw_features: deque[np.ndarray] = deque(maxlen=cfg.zscore_window)
        self._zscored_window: deque[np.ndarray] = deque(maxlen=cfg.predict_window)
        self._bar_count: int = 0
        self._current_regime: RegimeState | None = None
        self._last_proba: np.ndarray | None = None

    def reset(self) -> None:
        """Reset online state (call on session boundaries)."""
        with self._lock:
            self._reset_online_state()

    # ── Training ─────────────────────────────────────────────────────

    def fit(self, data: np.ndarray | list[dict]) -> None:
        """Fit HMM on a (N, 3) feature matrix or list of bar dicts.

        Uses KMeans initialization for stable convergence, then relabels
        HMM states to semantic RegimeState values via Hungarian algorithm.
        """
        if isinstance(data, list):
            features = self._features_from_dicts(data)
        else:
            features = data

        cfg = self.config

        # KMeans seeding for stable init
        km = KMeans(
            n_clusters=cfg.n_states,
            random_state=cfg.random_state,
            n_init=10,
        )
        km.fit(features)

        self.model = GaussianHMM(
            n_components=cfg.n_states,
            covariance_type=cfg.covariance_type,
            n_iter=cfg.n_iter,
            random_state=cfg.random_state,
            init_params="stc",  # skip 'm' — use KMeans means
        )
        self.model.means_ = km.cluster_centers_
        self.model.fit(features)
        self._label_states()
        logger.info("HMM fitted: %d states, %d samples", cfg.n_states, len(features))

    def _features_from_dicts(self, bars: list[dict]) -> np.ndarray:
        """Build feature matrix from list of bar dicts (for fit(bars) API)."""
        df = pl.DataFrame({
            "timestamp": list(range(len(bars))),
            "open": [float(b["open"]) for b in bars],
            "high": [float(b["high"]) for b in bars],
            "low": [float(b["low"]) for b in bars],
            "close": [float(b["close"]) for b in bars],
            "volume": [float(b["volume"]) for b in bars],
        })
        features, _ = build_feature_matrix(df, self.config)
        return features

    def _label_states(self) -> None:
        """Map HMM components to semantic RegimeState via Hungarian algorithm.

        Scoring heuristic (3 features: realized_vol, vpin_approx, return_autocorr):
        - RANGE_BOUND:  low realized_vol, low vpin, negative return_autocorr
        - VOLATILE:     high realized_vol, high vpin, positive autocorr
        - TRENDING:     moderate vol, moderate vpin, positive autocorr (3-state only)
        """
        assert self.model is not None
        means = self.model.means_  # (n_states, 3)

        n_states = means.shape[0]
        # Use only the regimes that match the model's state count
        n_regimes = min(n_states, len(RegimeState))
        cost = np.zeros((n_states, n_regimes), dtype=np.float64)

        for s in range(n_states):
            rvol, vpin, ret_ac = means[s]

            # RANGE_BOUND: low vol, low informed flow, mean-reverting (negative autocorr)
            cost[s, RegimeState.RANGE_BOUND] = -(-rvol - vpin - ret_ac)

            # VOLATILE: high vol, high informed flow, trending
            cost[s, RegimeState.VOLATILE] = -(rvol + vpin + ret_ac)

            # TRENDING (3-state): moderate vol, positive autocorr, low vpin
            if n_regimes > 2:
                cost[s, RegimeState.TRENDING] = -(0.5 * rvol + ret_ac - vpin)

        row_ind, col_ind = linear_sum_assignment(cost)
        self.state_map = {
            int(r): RegimeState(c) for r, c in zip(row_ind, col_ind)
        }

    # ── Online prediction ────────────────────────────────────────────

    def _compute_online_features(self, bar: dict) -> np.ndarray | None:
        """Ingest a bar and compute raw (un-normalized) feature vector.

        Returns None if insufficient history for all features.
        """
        close = float(bar["close"])
        open_ = float(bar["open"])
        volume = float(bar["volume"])
        cfg = self.config

        # Log return
        if self._prev_close is not None:
            lr = np.log(max(close, 1e-10) / max(self._prev_close, 1e-10))
        else:
            lr = 0.0
        self._prev_close = close
        self._log_returns.append(lr)

        # VPIN tracking
        buy = volume if close > open_ else 0.0
        sell = volume if close <= open_ else 0.0
        self._buy_vol_window.append(buy)
        self._sell_vol_window.append(sell)

        self._bar_count += 1

        # Need enough bars for all rolling windows
        min_needed = max(cfg.rvol_window, cfg.autocorr_window + 1, 2)
        if self._bar_count < min_needed:
            return None

        # 1. Realized vol — std of last rvol_window log returns
        rets = list(self._log_returns)
        recent = rets[-cfg.rvol_window:]
        realized_vol = float(np.std(recent, ddof=1)) if len(recent) >= 2 else 0.0

        # 2. VPIN approx
        buy_sum = sum(self._buy_vol_window)
        sell_sum = sum(self._sell_vol_window)
        total = buy_sum + sell_sum
        vpin = abs(buy_sum - sell_sum) / max(total, 1.0)

        # 3. Return autocorrelation — lag-1 Pearson over autocorr_window
        rets_arr = np.array(rets, dtype=np.float64)
        if len(rets_arr) >= cfg.autocorr_window + 1:
            curr = rets_arr[-cfg.autocorr_window:]
            prev = rets_arr[-cfg.autocorr_window - 1 : -1]
            cm = curr - curr.mean()
            pm = prev - prev.mean()
            denom = np.sqrt(float(np.sum(cm**2) * np.sum(pm**2)))
            ret_autocorr = float(np.sum(cm * pm) / max(denom, 1e-10))
        else:
            ret_autocorr = 0.0

        return np.array(
            [realized_vol, vpin, ret_autocorr], dtype=np.float64
        )

    def _zscore_and_clip(self, raw: np.ndarray) -> np.ndarray | None:
        """Z-score normalize against rolling history, clip to [-3, 3]."""
        self._raw_features.append(raw)

        if len(self._raw_features) < 2:
            return None

        history = np.array(self._raw_features)
        means = history.mean(axis=0)
        stds = history.std(axis=0, ddof=1)
        stds = np.where(stds < 1e-10, 1.0, stds)

        normed = (raw - means) / stds
        return np.clip(normed, -3.0, 3.0)

    def predict(self, bar: dict) -> int:
        """Online: ingest one bar, update rolling state, return regime (0/1).

        Raises NotReadyError if warm-up buffer (500 bars) not saturated.
        """
        with self._lock:
            if self.model is None:
                raise NotReadyError("Model not fitted")

            raw = self._compute_online_features(bar)
            if raw is None or self._bar_count < self.config.warmup_bars:
                raise NotReadyError(
                    f"Warm-up: {self._bar_count}/{self.config.warmup_bars} bars"
                )

            normed = self._zscore_and_clip(raw)
            if normed is None:
                raise NotReadyError("Insufficient z-score history")

            self._zscored_window.append(normed)
            window = np.array(self._zscored_window)  # (W, 4)

            # Forward algorithm posteriors over sliding window
            proba_matrix = self.model.predict_proba(window)
            last_proba_raw = proba_matrix[-1]  # (n_components,)

            # Map to RegimeState order
            mapped_proba = np.zeros(len(RegimeState), dtype=np.float64)
            for hmm_idx, regime in self.state_map.items():
                mapped_proba[regime.value] = last_proba_raw[hmm_idx]

            state = RegimeState(int(np.argmax(mapped_proba)))

            # Log transitions
            if self._current_regime is not None and state != self._current_regime:
                logger.info(
                    "regime_transition: %s -> %s",
                    self._current_regime.name,
                    state.name,
                )
            self._current_regime = state
            self._last_proba = mapped_proba

            return int(state)

    def regime_proba(self, bar: dict) -> np.ndarray:
        """Online: ingest one bar, return posterior probability vector (2,).

        Raises NotReadyError if warm-up not complete.
        """
        self.predict(bar)  # updates internal state
        return self._last_proba.copy()

    @property
    def last_proba(self) -> np.ndarray | None:
        """Last computed posterior probabilities (or None if no prediction yet)."""
        return self._last_proba

    # ── Batch prediction (backtest) ──────────────────────────────────

    def predict_sequence(self, features: np.ndarray) -> list[RegimeState]:
        """Batch Viterbi decoding on pre-built (N, 3) feature matrix.

        Use with build_feature_matrix() for backtest pipelines.
        """
        assert self.model is not None and self.state_map is not None
        raw_states = self.model.predict(features)
        return [self.state_map[int(s)] for s in raw_states]

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist model, state map, and config to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "state_map": self.state_map,
                "config": self.config,
            },
            path / "hmm_model.joblib",
        )
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> HMMRegimeClassifier:
        """Load a persisted HMM classifier."""
        path = Path(path)
        data = joblib.load(path / "hmm_model.joblib")
        obj = cls(config=data["config"])
        obj.model = data["model"]
        obj.state_map = data["state_map"]
        return obj


# ── Evaluation helpers ───────────────────────────────────────────────


def compute_persistence_accuracy(
    states: list[RegimeState], horizon: int = 5
) -> float:
    """Fraction of time the regime at t is the same as at t + horizon."""
    if len(states) <= horizon:
        return 0.0
    matches = sum(
        1 for i in range(len(states) - horizon) if states[i] == states[i + horizon]
    )
    return matches / (len(states) - horizon)


def compute_transition_matrix(states: list[RegimeState]) -> np.ndarray:
    """Empirical transition matrix (n_states x n_states), row-normalized."""
    n = len(RegimeState)
    counts = np.zeros((n, n), dtype=np.float64)
    for i in range(len(states) - 1):
        counts[states[i].value, states[i + 1].value] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return counts / row_sums


@dataclass(frozen=True)
class HMMValidationReport:
    """Summary of HMM model quality metrics."""

    n_samples: int
    persistence_5bar: float
    state_distribution: dict[str, float]
    transition_matrix: np.ndarray
    log_likelihood: float


def validate_model(
    classifier: HMMRegimeClassifier,
    features: np.ndarray,
) -> HMMValidationReport:
    """Run full validation suite on a fitted classifier."""
    assert classifier.model is not None
    states = classifier.predict_sequence(features)

    n = len(states)
    dist = {}
    for regime in RegimeState:
        count = sum(1 for s in states if s == regime)
        dist[regime.name] = count / n if n > 0 else 0.0

    return HMMValidationReport(
        n_samples=n,
        persistence_5bar=compute_persistence_accuracy(states, horizon=5),
        state_distribution=dist,
        transition_matrix=compute_transition_matrix(states),
        log_likelihood=classifier.model.score(features),
    )
