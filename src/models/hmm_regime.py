"""HMM Intraday Regime Classifier.

Fits a 5-state Gaussian Hidden Markov Model on a subset of FeatureVector
fields to classify intraday market regimes. States are semantically labeled
via the Hungarian algorithm matching emission means to regime archetypes.

Usage:
    # Training (offline)
    matrix, timestamps = build_feature_matrix(df_1s, config)
    clf = HMMRegimeClassifier(config)
    clf.fit(matrix)

    # Prediction (backtest)
    states = clf.predict(matrix)

    # Online (live)
    state, proba = clf.predict_proba(last_50_bars_matrix)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class RegimeState(IntEnum):
    """Five intraday regime states."""

    HIGH_VOL_UP = 0
    HIGH_VOL_DOWN = 1
    LOW_VOL_RANGE = 2
    BREAKOUT = 3
    MEAN_REVERSION = 4


# Feature column names used by the HMM (6 of 15+).
DEFAULT_FEATURE_COLUMNS: list[str] = [
    "atr_ticks",
    "vwap_dev_sd",
    "cvd_slope",
    "poc_distance_ticks",
    "realized_vol",
    "return_20bar",
]


@dataclass(frozen=True)
class HMMRegimeConfig:
    """Configuration for the HMM regime classifier."""

    n_states: int = 5
    n_iter: int = 100
    covariance_type: str = "full"
    random_state: int = 42
    zscore_window: int = 250
    feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_COLUMNS))


# ── Feature matrix construction ──────────────────────────────────────


def _rolling_linreg_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling linear regression slope using cumulative sums."""
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result

    # Precompute x constants for fixed window
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)
    if x_var == 0:
        return result

    # Sliding window via stride tricks
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(arr, window)  # (n - window + 1, window)
    y_means = windows.mean(axis=1)
    # slope = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
    slopes = np.sum((x[np.newaxis, :] - x_mean) * (windows - y_means[:, np.newaxis]), axis=1) / x_var
    result[window - 1 :] = slopes
    return result


def _rolling_zscore(raw: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling z-score normalization (no lookahead).

    Uses a fixed rolling window once enough data is available.
    """
    n, k = raw.shape
    normed = np.full_like(raw, np.nan)

    # Use Polars for fast rolling stats (operates column-wise)
    for col in range(k):
        series = pl.Series("v", raw[:, col])
        r_mean = series.rolling_mean(window_size=window, min_samples=2).to_numpy()
        r_std = series.rolling_std(window_size=window, min_samples=2).to_numpy()
        r_std = np.where((r_std < 1e-10) | np.isnan(r_std), 1.0, r_std)
        normed[:, col] = (raw[:, col] - r_mean) / r_std

    return normed


def build_feature_matrix(
    df_1s: pl.DataFrame,
    config: HMMRegimeConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the HMM feature matrix from 1s bars using vectorized operations.

    Computes 6 features: atr_ticks, vwap_dev_sd, cvd_slope, poc_distance_ticks,
    realized_vol, return_20bar. All computed via Polars/NumPy — no row-by-row
    Python loop, so it handles millions of bars efficiently.

    Returns:
        (features, timestamps) where features is (N, 6) float64 and
        timestamps is (N,) int64 nanosecond timestamps. Warm-up rows are dropped.
    """
    cfg = config or HMMRegimeConfig()
    tick_size = 0.25
    atr_period = 14

    # Extract columns as numpy arrays
    close = df_1s["close"].to_numpy().astype(np.float64)
    high = df_1s["high"].to_numpy().astype(np.float64)
    low = df_1s["low"].to_numpy().astype(np.float64)
    open_ = df_1s["open"].to_numpy().astype(np.float64)
    volume = df_1s["volume"].to_numpy().astype(np.float64)
    n = len(close)

    # Timestamps — handle both datetime and int columns
    ts_col = df_1s["timestamp"]
    if ts_col.dtype == pl.Datetime or str(ts_col.dtype).startswith("Datetime"):
        ts_arr = ts_col.dt.epoch("ns").to_numpy().astype(np.int64)
    else:
        ts_arr = ts_col.to_numpy().astype(np.int64)

    # ── 1. ATR in ticks (Wilder EWM of True Range) ──────────────────
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    # Wilder smoothing via Polars ewm_mean (alpha = 1/period, adjust=False)
    tr_series = pl.Series("tr", tr)
    atr = tr_series.ewm_mean(alpha=1.0 / atr_period, adjust=False).to_numpy()
    atr_ticks = atr / tick_size

    # ── 2. VWAP deviation in standard deviations ────────────────────
    typical = (high + low + close) / 3.0
    cum_pv = np.cumsum(typical * volume)
    cum_vol = np.cumsum(volume)
    cum_vol_safe = np.where(cum_vol == 0, 1, cum_vol)
    vwap = cum_pv / cum_vol_safe

    # Rolling SD of (typical - vwap) using Polars
    dev = typical - vwap
    dev_series = pl.Series("dev", dev)
    rolling_sd = dev_series.rolling_std(window_size=20, min_samples=2).to_numpy()
    rolling_sd = np.where((rolling_sd < 1e-10) | np.isnan(rolling_sd), 1.0, rolling_sd)
    vwap_dev_sd = dev / rolling_sd

    # ── 3. CVD slope (20-bar linreg slope of bar deltas) ────────────
    bar_delta = np.where(close > open_, volume, np.where(close < open_, -volume, 0.0))
    cvd_slope = _rolling_linreg_slope(bar_delta, window=20)

    # ── 4. POC distance in ticks (developing session POC) ───────────
    # Approximate: rolling mode of rounded close prices over 390 bars (~1 session)
    # Use volume-weighted price bucket to find local POC
    poc_window = 390
    poc_dist = np.zeros(n, dtype=np.float64)
    # Vectorized approach: rolling volume-weighted mean as POC proxy
    cum_cv = np.cumsum(close * volume)
    poc_approx = np.where(cum_vol_safe > 0, cum_cv / cum_vol_safe, close)
    poc_dist = np.abs(close - poc_approx) / tick_size

    # ── 5. Realized vol (20-bar rolling std of log returns) ─────────
    log_ret = np.zeros(n, dtype=np.float64)
    log_ret[1:] = np.log(np.maximum(close[1:], 1e-10) / np.maximum(close[:-1], 1e-10))
    rvol_series = pl.Series("lr", log_ret)
    realized_vol = rvol_series.rolling_std(window_size=20, min_samples=20).to_numpy()

    # ── 6. 20-bar log return ────────────────────────────────────────
    return_20bar = np.full(n, np.nan, dtype=np.float64)
    return_20bar[20:] = np.log(
        np.maximum(close[20:], 1e-10) / np.maximum(close[:-20], 1e-10)
    )

    # ── Stack raw features (N, 6) ───────────────────────────────────
    raw_matrix = np.column_stack([
        atr_ticks,
        vwap_dev_sd,
        cvd_slope,
        poc_dist,
        realized_vol,
        return_20bar,
    ])

    # ── Rolling z-score normalization ────────────────────────────────
    normed = _rolling_zscore(raw_matrix, window=cfg.zscore_window)

    # Drop warm-up rows
    start = max(cfg.zscore_window, 20)
    valid = ~np.any(np.isnan(normed[start:]), axis=1) & ~np.any(np.isinf(normed[start:]), axis=1)

    features = normed[start:][valid]
    ts_out = ts_arr[start:][valid]

    return features, ts_out


# ── HMM Classifier ──────────────────────────────────────────────────


class HMMRegimeClassifier:
    """Gaussian HMM regime classifier with semantic state labeling."""

    def __init__(self, config: HMMRegimeConfig | None = None) -> None:
        self.config = config or HMMRegimeConfig()
        self.model: GaussianHMM | None = None
        self.state_map: dict[int, RegimeState] | None = None
        self._norm_means: np.ndarray | None = None
        self._norm_stds: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> None:
        """Fit the HMM on a (N, 6) feature matrix.

        Uses KMeans initialization for stable convergence, then relabels
        HMM states to semantic RegimeState values via Hungarian algorithm.
        """
        cfg = self.config
        self.model = GaussianHMM(
            n_components=cfg.n_states,
            covariance_type=cfg.covariance_type,
            n_iter=cfg.n_iter,
            random_state=cfg.random_state,
            init_params="stmc",
        )
        self.model.fit(features)
        self._label_states()
        logger.info("HMM fitted: %d states, %d samples", cfg.n_states, len(features))

    def _label_states(self) -> None:
        """Map HMM component indices to semantic RegimeState via Hungarian algorithm.

        Scoring heuristic based on emission means:
        - HIGH_VOL_UP:    high ATR + positive return + positive CVD slope
        - HIGH_VOL_DOWN:  high ATR + negative return + negative CVD slope
        - LOW_VOL_RANGE:  low ATR + low VWAP deviation + low POC distance
        - BREAKOUT:       moderate ATR + high POC distance + high realized vol
        - MEAN_REVERSION: moderate ATR + high VWAP deviation (abs) + negative CVD×return
        """
        assert self.model is not None
        means = self.model.means_  # (n_states, n_features)

        # Feature indices: atr_ticks=0, vwap_dev_sd=1, cvd_slope=2,
        #                  poc_distance_ticks=3, realized_vol=4, return_20bar=5
        n_states = means.shape[0]
        n_regimes = len(RegimeState)

        # Build cost matrix (negative score = better fit)
        cost = np.zeros((n_states, n_regimes), dtype=np.float64)

        for s in range(n_states):
            m = means[s]
            atr, vwap_dev, cvd_sl, poc_dist, rvol, ret20 = m

            # HIGH_VOL_UP: high ATR, positive return, positive CVD slope
            cost[s, RegimeState.HIGH_VOL_UP] = -(atr + ret20 + cvd_sl)

            # HIGH_VOL_DOWN: high ATR, negative return, negative CVD slope
            cost[s, RegimeState.HIGH_VOL_DOWN] = -(atr - ret20 - cvd_sl)

            # LOW_VOL_RANGE: low ATR, small VWAP dev, small POC distance
            cost[s, RegimeState.LOW_VOL_RANGE] = -(-atr - abs(vwap_dev) - poc_dist)

            # BREAKOUT: high POC distance, high realized vol
            cost[s, RegimeState.BREAKOUT] = -(poc_dist + rvol + abs(ret20))

            # MEAN_REVERSION: high |VWAP dev|, CVD diverges from return
            cost[s, RegimeState.MEAN_REVERSION] = -(abs(vwap_dev) - cvd_sl * ret20)

        # Hungarian assignment (minimize cost = maximize score)
        row_ind, col_ind = linear_sum_assignment(cost)
        self.state_map = {int(row): RegimeState(col) for row, col in zip(row_ind, col_ind)}

    def predict(self, features: np.ndarray) -> list[RegimeState]:
        """Viterbi decoding: predict most likely regime sequence (offline/backtest)."""
        assert self.model is not None and self.state_map is not None
        raw_states = self.model.predict(features)
        return [self.state_map[int(s)] for s in raw_states]

    def predict_proba(self, feature_window: np.ndarray) -> tuple[RegimeState, np.ndarray]:
        """Online prediction: pass a window of recent bars, return final row's state + probs.

        Args:
            feature_window: (W, 6) matrix of recent bars (e.g., last 50).

        Returns:
            (state, probabilities) where probabilities is (n_states,) mapped to RegimeState order.
        """
        assert self.model is not None and self.state_map is not None
        posteriors = self.model.predict_proba(feature_window)
        last_probs_raw = posteriors[-1]  # (n_components,)

        # Reorder probabilities to RegimeState order
        probs = np.zeros(len(RegimeState), dtype=np.float64)
        for hmm_idx, regime in self.state_map.items():
            probs[regime.value] = last_probs_raw[hmm_idx]

        state = RegimeState(int(np.argmax(probs)))
        return state, probs

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


def compute_persistence_accuracy(states: list[RegimeState], horizon: int = 5) -> float:
    """Fraction of time the regime at t is the same as at t + horizon."""
    if len(states) <= horizon:
        return 0.0
    matches = sum(1 for i in range(len(states) - horizon) if states[i] == states[i + horizon])
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
    states = classifier.predict(features)

    # State distribution
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
