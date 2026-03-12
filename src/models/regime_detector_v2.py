"""Phase 7: Production Regime Detector V2 — 3-state Student-t HMM, forward-only.

Step 2 upgrade from MVP:
- pomegranate Student-t emissions (heavier tails for financial returns)
- 4 features: [log_return, garman_klass_vol, hurst_dfa, autocorr_sum]
- Forward-only inference: predict_proba() only (NOT Viterbi — non-causal)
- Anti-whipsaw: hysteresis, min bars in regime, flip halt
- Position sizing from confidence thresholds

Usage:
    # Training (offline)
    detector = RegimeDetectorV2(config)
    detector.fit(features)             # (N, 4) matrix
    detector.save("models/regime_v2")

    # Batch (backtest) — forward-only over full sequence
    probas = detector.predict_proba_sequence(features)  # list[RegimeProba]

    # Online (live) — bar-by-bar
    proba = detector.update(bar_dict)  # RegimeProba | None
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import polars as pl
import torch
from hmmlearn.hmm import GaussianHMM
from pomegranate.distributions import StudentT
from pomegranate.hmm import DenseHMM
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

N_FEATURES = 4  # log_return, gk_vol, hurst_dfa, autocorr_sum


# ── Regime labels ────────────────────────────────────────────────────

class RegimeLabel(IntEnum):
    """3-state regime labels, ordered by index."""
    TRENDING = 0
    RANGING = 1
    HIGH_VOL = 2


PositionSize = Literal["full", "half", "flat"]


# ── Output dataclass ─────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeProba:
    """Regime detector output per bar.

    Attributes:
        probabilities: [P(trending), P(ranging), P(high_vol)]
        regime: Most likely regime label (with hysteresis applied)
        confidence: Max probability across states
        transition_signal: True if regime changed this bar
        position_size: "full" / "half" / "flat" based on confidence thresholds
        bars_in_regime: Consecutive bars in current regime
        whipsaw_halt: True if flip count exceeded threshold
    """
    probabilities: tuple[float, float, float]
    regime: RegimeLabel
    confidence: float
    transition_signal: bool
    position_size: PositionSize
    bars_in_regime: int
    whipsaw_halt: bool


# ── Config ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeDetectorV2Config:
    """Configuration for the Phase 7 regime detector (Step 2: Student-t)."""

    # HMM
    n_states: int = 3
    n_iter: int = 200
    tol: float = 0.1
    random_state: int = 42
    covariance_type: str = "diag"   # pomegranate StudentT only supports diag
    emission_type: str = "gaussian"  # "gaussian" (hmmlearn) or "studentt" (pomegranate)

    # Student-t
    studentt_dof: int = 5        # Degrees of freedom (fixed, not learned by EM)

    # Features
    gk_vol_window: int = 20          # Garman-Klass vol rolling window
    hurst_window: int = 250          # DFA rolling window (bars)
    hurst_stride: int = 5            # Compute DFA every N bars (batch), ffill between
    autocorr_window: int = 100       # Rolling window for autocorrelation
    autocorr_max_lag: int = 10       # Max lag for autocorrelation sum

    # Online normalization
    zscore_window: int = 500         # Rolling z-score window
    warmup_bars: int = 300           # Min bars before prediction (>= hurst_window)
    predict_window: int = 100        # Sliding window for forward algo

    # Position sizing thresholds
    confidence_full: float = 0.65    # >0.65 = full size
    confidence_half: float = 0.45    # 0.45-0.65 = half size
    # <0.45 = flat

    # Hysteresis
    enter_threshold: float = 0.65    # Enter regime at P > this
    exit_threshold: float = 0.45     # Exit regime at P < this

    # Anti-whipsaw
    min_bars_in_regime: int = 5      # Min consecutive bars before acting
    flip_halt_count: int = 2         # Max flips in window before halt
    flip_halt_window: int = 20       # Window (bars) for flip counting
    ambiguity_threshold: float = 0.55  # Max P < this → no-regime


# ── Feature extraction (batch) ───────────────────────────────────────

def _garman_klass_vol(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """Garman-Klass volatility estimator, rolling window.

    GK = 0.5 * ln(H/L)^2 - (2ln2-1) * ln(C/O)^2
    Then take sqrt of rolling mean.
    """
    hl = np.log(np.maximum(high, 1e-10) / np.maximum(low, 1e-10))
    co = np.log(np.maximum(close, 1e-10) / np.maximum(open_, 1e-10))
    gk_var = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2

    gk_series = pl.Series("gk", gk_var)
    gk_mean = gk_series.rolling_mean(window_size=window, min_samples=2).to_numpy()
    return np.sqrt(np.maximum(gk_mean, 0.0))


def _dfa_hurst_single(returns: np.ndarray) -> float:
    """Detrended Fluctuation Analysis Hurst exponent over a window of returns."""
    n = len(returns)
    if n < 16:
        return 0.5

    min_scale = 4
    max_scale = max(n // 4, min_scale + 1)
    scales = np.unique(
        np.logspace(np.log10(min_scale), np.log10(max_scale), num=10).astype(int)
    )

    # Profile: cumulative sum of demeaned series
    profile = np.cumsum(returns - np.mean(returns))

    fluctuations = []
    valid_scales = []

    for scale in scales:
        n_seg = len(profile) // scale
        if n_seg < 1:
            continue

        # Precompute x constants for this scale (linear detrend)
        x = np.arange(scale, dtype=np.float64)
        x_mean = x.mean()
        x_var = np.sum((x - x_mean) ** 2)

        rms_sum = 0.0
        for seg in range(n_seg):
            segment = profile[seg * scale : (seg + 1) * scale]
            y_mean = segment.mean()
            if x_var > 0:
                slope = np.dot(x - x_mean, segment - y_mean) / x_var
            else:
                slope = 0.0
            trend = slope * (x - x_mean) + y_mean
            rms_sum += np.mean((segment - trend) ** 2)

        rms = np.sqrt(rms_sum / n_seg)
        if rms > 1e-15:
            fluctuations.append(rms)
            valid_scales.append(scale)

    if len(fluctuations) < 2:
        return 0.5

    log_s = np.log(np.array(valid_scales, dtype=np.float64))
    log_f = np.log(np.array(fluctuations))
    s_mean = log_s.mean()
    f_mean = log_f.mean()
    denom = np.sum((log_s - s_mean) ** 2)
    if denom < 1e-15:
        return 0.5

    hurst = np.sum((log_s - s_mean) * (log_f - f_mean)) / denom
    return float(np.clip(hurst, 0.0, 1.5))


def _rolling_hurst_dfa(
    log_returns: np.ndarray, window: int = 250, stride: int = 5
) -> np.ndarray:
    """Rolling DFA Hurst exponent with stride-and-forward-fill for performance."""
    n = len(log_returns)
    result = np.full(n, np.nan)

    for i in range(window, n, stride):
        result[i] = _dfa_hurst_single(log_returns[i - window : i])

    # Forward fill
    last = np.nan
    for i in range(n):
        if not np.isnan(result[i]):
            last = result[i]
        else:
            result[i] = last

    return result


def _autocorr_sum_single(returns: np.ndarray, max_lag: int = 10) -> float:
    """Sum of autocorrelations at lags 1..max_lag over a single window."""
    var = np.var(returns)
    if var < 1e-20:
        return 0.0
    mean = np.mean(returns)
    centered = returns - mean
    n = len(returns)
    ac_sum = 0.0
    for lag in range(1, min(max_lag + 1, n)):
        ac = np.dot(centered[lag:], centered[:-lag]) / ((n - lag) * var)
        ac_sum += ac
    return float(ac_sum)


def _rolling_autocorr_sum(
    log_returns: np.ndarray, window: int = 100, max_lag: int = 10
) -> np.ndarray:
    """Rolling sum of autocorrelations at lags 1..max_lag."""
    n = len(log_returns)
    result = np.full(n, np.nan)

    for i in range(window, n):
        result[i] = _autocorr_sum_single(
            log_returns[i - window : i], max_lag=max_lag
        )

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


def build_features_v2(
    df: pl.DataFrame,
    config: RegimeDetectorV2Config | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 4-feature matrix for V2 regime detector.

    Features: [log_return, garman_klass_vol, hurst_dfa, autocorr_sum]

    Args:
        df: Bar DataFrame with columns: timestamp, open, high, low, close, volume
        config: Detector config (for window sizes)

    Returns:
        (features, timestamps) — features is (N, 4), timestamps is (N,) int64 ns.
        Warm-up rows dropped.
    """
    cfg = config or RegimeDetectorV2Config()

    close = df["close"].to_numpy().astype(np.float64)
    open_ = df["open"].to_numpy().astype(np.float64)
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    n = len(close)

    # Timestamps
    ts_col = df["timestamp"]
    if ts_col.dtype == pl.Datetime or str(ts_col.dtype).startswith("Datetime"):
        ts_arr = ts_col.dt.epoch("ns").to_numpy().astype(np.int64)
    else:
        ts_arr = ts_col.to_numpy().astype(np.int64)

    # Feature 1: Log returns
    log_ret = np.zeros(n, dtype=np.float64)
    log_ret[1:] = np.log(
        np.maximum(close[1:], 1e-10) / np.maximum(close[:-1], 1e-10)
    )

    # Feature 2: Garman-Klass volatility
    gk_vol = _garman_klass_vol(open_, high, low, close, window=cfg.gk_vol_window)

    # Feature 3: Hurst exponent (DFA, rolling with stride)
    hurst = _rolling_hurst_dfa(
        log_ret, window=cfg.hurst_window, stride=cfg.hurst_stride
    )

    # Feature 4: Autocorrelation sum (lags 1..max_lag)
    autocorr = _rolling_autocorr_sum(
        log_ret, window=cfg.autocorr_window, max_lag=cfg.autocorr_max_lag
    )

    # Stack raw features (N, 4)
    raw = np.column_stack([log_ret, gk_vol, hurst, autocorr])

    # Rolling z-score normalization
    normed = _rolling_zscore(raw, window=cfg.zscore_window)

    # Drop warm-up
    start = cfg.zscore_window
    valid = ~np.any(np.isnan(normed[start:]), axis=1) & ~np.any(
        np.isinf(normed[start:]), axis=1
    )

    return normed[start:][valid], ts_arr[start:][valid]


# ── Detector class ───────────────────────────────────────────────────

class RegimeDetectorV2:
    """Phase 7 production regime detector — 3-state Student-t HMM, forward-only.

    Step 2 upgrade:
    - pomegranate DenseHMM with StudentT emissions (heavy tails)
    - 4 features: log_return, gk_vol, hurst_dfa, autocorr_sum
    - Forward-only inference: predict_proba() never Viterbi
    - Anti-whipsaw: hysteresis, min bars, flip halt
    - Position sizing from confidence thresholds
    """

    def __init__(self, config: RegimeDetectorV2Config | None = None) -> None:
        self.config = config or RegimeDetectorV2Config()
        self.model: DenseHMM | None = None
        self.state_map: dict[int, RegimeLabel] | None = None
        self._lock = threading.Lock()
        self._reset_online_state()

    def _reset_online_state(self) -> None:
        """Reset all rolling windows and anti-whipsaw state."""
        cfg = self.config
        self._prev_close: float | None = None
        self._raw_features: deque[np.ndarray] = deque(maxlen=cfg.zscore_window)
        self._zscored_window: deque[np.ndarray] = deque(maxlen=cfg.predict_window)
        self._bar_count: int = 0

        # Garman-Klass rolling buffer
        self._gk_buffer: deque[float] = deque(maxlen=cfg.gk_vol_window)

        # Returns buffer for Hurst + autocorrelation
        self._returns_buffer: deque[float] = deque(
            maxlen=max(cfg.hurst_window, cfg.autocorr_window)
        )

        # Anti-whipsaw state
        self._current_regime: RegimeLabel = RegimeLabel.RANGING
        self._bars_in_regime: int = 0
        self._recent_flips: deque[int] = deque(maxlen=cfg.flip_halt_window)
        self._whipsaw_halt: bool = False
        self._last_proba: RegimeProba | None = None

    def reset(self) -> None:
        """Reset online state (call on session boundaries)."""
        with self._lock:
            self._reset_online_state()

    # ── Training ─────────────────────────────────────────────────────

    def fit(self, features: np.ndarray) -> None:
        """Fit 3-state HMM on (N, 4) feature matrix.

        Uses KMeans initialization for stable convergence.
        Supports both pomegranate StudentT and hmmlearn Gaussian backends.
        """
        cfg = self.config
        n_features = features.shape[1]

        # Reproducibility
        torch.manual_seed(cfg.random_state)
        np.random.seed(cfg.random_state)

        # KMeans seeding for emission means
        km = KMeans(
            n_clusters=cfg.n_states,
            random_state=cfg.random_state,
            n_init=10,
        )
        km.fit(features)
        centers = km.cluster_centers_  # (n_states, n_features)

        if cfg.emission_type == "gaussian":
            self._fit_gaussian(features, centers, cfg)
        else:
            self._fit_studentt(features, centers, cfg, n_features)

        self._label_states()
        logger.info(
            "RegimeDetectorV2 fitted (%s): %d states, %d features, %d samples",
            cfg.emission_type, cfg.n_states, n_features, len(features),
        )

    def _fit_studentt(
        self, features: np.ndarray, centers: np.ndarray,
        cfg: RegimeDetectorV2Config, n_features: int,
    ) -> None:
        dists = []
        for i in range(cfg.n_states):
            d = StudentT(
                dofs=cfg.studentt_dof,
                means=centers[i].tolist(),
                covs=[1.0] * n_features,
                covariance_type="diag",
            )
            dists.append(d)

        self.model = DenseHMM(
            distributions=dists,
            max_iter=cfg.n_iter,
            tol=cfg.tol,
            verbose=False,
        )
        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        self.model.fit(X)

    def _fit_gaussian(
        self, features: np.ndarray, centers: np.ndarray,
        cfg: RegimeDetectorV2Config,
    ) -> None:
        self.model = GaussianHMM(
            n_components=cfg.n_states,
            covariance_type="full",
            n_iter=cfg.n_iter,
            random_state=cfg.random_state,
            init_params="stc",
        )
        self.model.means_ = centers
        self.model.fit(features)

    def _label_states(self) -> None:
        """Map HMM components to semantic RegimeLabel via Hungarian algorithm.

        Heuristic for 4 features [log_return, gk_vol, hurst, autocorr]:
        - TRENDING:  high Hurst (persistence), positive autocorr, moderate vol
        - RANGING:   low vol, near-zero return, low Hurst
        - HIGH_VOL:  high vol, high return variance
        """
        assert self.model is not None
        n_states = self.config.n_states
        is_gaussian = self.config.emission_type == "gaussian"
        cost = np.zeros((n_states, 3), dtype=np.float64)

        for s in range(n_states):
            if is_gaussian:
                means = self.model.means_[s]
                covars = self.model.covars_
                if self.model.covariance_type == "full":
                    lr_var = covars[s][0, 0]
                elif self.model.covariance_type == "diag":
                    lr_var = covars[s][0]
                else:
                    lr_var = float(covars[s])
            else:
                dist = self.model.distributions[s]
                means = dist.means.data.numpy().flatten()
                covs_raw = dist.covs.data.numpy()
                lr_var = covs_raw[0] if covs_raw.ndim == 1 else float(covs_raw.flat[0])

            lr_mean = means[0]
            gk_mean = means[1]
            hurst_mean = means[2] if len(means) > 2 else 0.0
            ac_mean = means[3] if len(means) > 3 else 0.0

            # TRENDING: directional, persistent, moderate vol
            cost[s, RegimeLabel.TRENDING] = -(
                abs(lr_mean) + 0.5 * hurst_mean + 0.3 * ac_mean - 0.2 * gk_mean
            )

            # RANGING: low vol, near-zero return, low persistence
            cost[s, RegimeLabel.RANGING] = -(
                -gk_mean - abs(lr_mean) - 0.3 * hurst_mean
            )

            # HIGH_VOL: high vol, high return variance
            cost[s, RegimeLabel.HIGH_VOL] = -(gk_mean + lr_var)

        row_ind, col_ind = linear_sum_assignment(cost)
        self.state_map = {
            int(r): RegimeLabel(c) for r, c in zip(row_ind, col_ind)
        }
        logger.info("State map: %s", self.state_map)

    # ── Batch prediction (backtest) ──────────────────────────────────

    def predict_proba_sequence(
        self, features: np.ndarray
    ) -> list[RegimeProba]:
        """Forward-only batch inference over a feature sequence.

        Returns one RegimeProba per row, with anti-whipsaw applied.
        Uses predict_proba() (forward-backward), NOT predict() (Viterbi).
        """
        assert self.model is not None and self.state_map is not None

        cfg = self.config
        n = len(features)

        # Get posteriors from the appropriate backend
        if cfg.emission_type == "gaussian":
            proba_matrix = self.model.predict_proba(features)  # (T, n_states)
        else:
            X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            proba_tensor = self.model.predict_proba(X)
            proba_matrix = proba_tensor[0].detach().numpy()  # (T, n_states)

        # Map HMM indices to RegimeLabel order
        mapped = np.zeros((n, 3), dtype=np.float64)
        for hmm_idx, label in self.state_map.items():
            mapped[:, label.value] = proba_matrix[:, hmm_idx]

        # Apply anti-whipsaw rules sequentially
        results: list[RegimeProba] = []
        current_regime = RegimeLabel.RANGING
        bars_in_regime = 0
        recent_flips: deque[int] = deque(maxlen=cfg.flip_halt_window)
        whipsaw_halt = False

        for i in range(n):
            p_trending, p_ranging, p_high_vol = mapped[i]
            proba_tuple = (float(p_trending), float(p_ranging), float(p_high_vol))
            max_p = max(proba_tuple)

            # Ambiguity check: max P < threshold → hold current
            if max_p < cfg.ambiguity_threshold:
                new_regime = current_regime
            else:
                new_regime = self._apply_hysteresis(
                    current_regime, mapped[i], cfg
                )

            # Track transitions
            transition = new_regime != current_regime

            if transition:
                recent_flips.append(1)
                bars_in_regime = 1
                current_regime = new_regime
            else:
                recent_flips.append(0)
                bars_in_regime += 1

            # Flip halt check
            flip_count = sum(recent_flips)
            if flip_count > cfg.flip_halt_count:
                whipsaw_halt = True
            elif flip_count <= 1:
                whipsaw_halt = False

            # Position sizing
            position_size = self._compute_position_size(max_p, cfg)

            # Override to flat if whipsaw halt or not enough bars
            if whipsaw_halt or bars_in_regime < cfg.min_bars_in_regime:
                position_size = "flat"

            results.append(RegimeProba(
                probabilities=proba_tuple,
                regime=current_regime,
                confidence=float(max_p),
                transition_signal=transition,
                position_size=position_size,
                bars_in_regime=bars_in_regime,
                whipsaw_halt=whipsaw_halt,
            ))

        return results

    @staticmethod
    def _apply_hysteresis(
        current: RegimeLabel,
        probas: np.ndarray,
        cfg: RegimeDetectorV2Config,
    ) -> RegimeLabel:
        """Apply enter/exit hysteresis thresholds."""
        p_current = probas[current.value]

        if p_current >= cfg.exit_threshold:
            return current

        for label in RegimeLabel:
            if label == current:
                continue
            if probas[label.value] > cfg.enter_threshold:
                return label

        return current

    @staticmethod
    def _compute_position_size(
        confidence: float,
        cfg: RegimeDetectorV2Config,
    ) -> PositionSize:
        if confidence > cfg.confidence_full:
            return "full"
        elif confidence >= cfg.confidence_half:
            return "half"
        else:
            return "flat"

    # ── Online prediction ────────────────────────────────────────────

    def update(self, bar: dict) -> RegimeProba | None:
        """Online: ingest one bar, return RegimeProba or None if warming up.

        Thread-safe. Uses forward algorithm over sliding window.
        """
        with self._lock:
            if self.model is None:
                return None

            raw = self._compute_online_features(bar)
            if raw is None:
                return None

            if self._bar_count < self.config.warmup_bars:
                return None

            normed = self._zscore_and_clip(raw)
            if normed is None:
                return None

            self._zscored_window.append(normed)

            # pomegranate forward-backward needs at least 2 observations
            if self.config.emission_type != "gaussian" and len(self._zscored_window) < 2:
                return None

            window = np.array(self._zscored_window)

            # Forward algorithm over sliding window
            if self.config.emission_type == "gaussian":
                proba_matrix = self.model.predict_proba(window)
                last_proba_raw = proba_matrix[-1]
            else:
                X = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                proba_tensor = self.model.predict_proba(X)
                last_proba_raw = proba_tensor[0, -1].detach().numpy()

            # Map to RegimeLabel order
            mapped = np.zeros(3, dtype=np.float64)
            for hmm_idx, label in self.state_map.items():
                mapped[label.value] = last_proba_raw[hmm_idx]

            p_tuple = (float(mapped[0]), float(mapped[1]), float(mapped[2]))
            max_p = max(p_tuple)
            cfg = self.config

            # Ambiguity
            if max_p < cfg.ambiguity_threshold:
                new_regime = self._current_regime
            else:
                new_regime = self._apply_hysteresis(
                    self._current_regime, mapped, cfg
                )

            # Track transition
            transition = new_regime != self._current_regime

            if transition:
                self._recent_flips.append(1)
                self._bars_in_regime = 1
                self._current_regime = new_regime
            else:
                self._recent_flips.append(0)
                self._bars_in_regime += 1

            # Flip halt
            flip_count = sum(self._recent_flips)
            if flip_count > cfg.flip_halt_count:
                self._whipsaw_halt = True
            elif flip_count <= 1:
                self._whipsaw_halt = False

            # Position sizing
            position_size = self._compute_position_size(max_p, cfg)
            if self._whipsaw_halt or self._bars_in_regime < cfg.min_bars_in_regime:
                position_size = "flat"

            proba = RegimeProba(
                probabilities=p_tuple,
                regime=self._current_regime,
                confidence=float(max_p),
                transition_signal=transition,
                position_size=position_size,
                bars_in_regime=self._bars_in_regime,
                whipsaw_halt=self._whipsaw_halt,
            )

            if transition:
                logger.info(
                    "regime_transition_v2: %s -> %s (conf=%.3f)",
                    self._last_proba.regime.name if self._last_proba else "INIT",
                    new_regime.name,
                    max_p,
                )

            self._last_proba = proba
            return proba

    def _compute_online_features(self, bar: dict) -> np.ndarray | None:
        """Compute raw [log_return, gk_vol, hurst, autocorr] from a single bar."""
        close = float(bar["close"])
        open_ = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])

        # Log return
        if self._prev_close is not None:
            lr = np.log(max(close, 1e-10) / max(self._prev_close, 1e-10))
        else:
            lr = 0.0
        self._prev_close = close
        self._returns_buffer.append(lr)

        # Garman-Klass per-bar variance
        hl = np.log(max(high, 1e-10) / max(low, 1e-10))
        co = np.log(max(close, 1e-10) / max(open_, 1e-10))
        gk_var = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
        self._gk_buffer.append(gk_var)

        self._bar_count += 1

        if len(self._gk_buffer) < 2:
            return None

        # Rolling GK vol
        gk_vol = np.sqrt(max(np.mean(self._gk_buffer), 0.0))

        # Hurst (DFA) — need hurst_window returns
        cfg = self.config
        if len(self._returns_buffer) < cfg.hurst_window:
            return None
        returns_arr = np.array(self._returns_buffer)
        hurst = _dfa_hurst_single(returns_arr[-cfg.hurst_window :])

        # Autocorrelation sum
        if len(self._returns_buffer) < cfg.autocorr_window:
            return None
        autocorr = _autocorr_sum_single(
            returns_arr[-cfg.autocorr_window :], max_lag=cfg.autocorr_max_lag
        )

        return np.array([lr, gk_vol, hurst, autocorr], dtype=np.float64)

    def _zscore_and_clip(self, raw: np.ndarray) -> np.ndarray | None:
        """Z-score against rolling history, clip to [-3, 3]."""
        self._raw_features.append(raw)

        if len(self._raw_features) < 2:
            return None

        history = np.array(self._raw_features)
        means = history.mean(axis=0)
        stds = history.std(axis=0, ddof=1)
        stds = np.where(stds < 1e-10, 1.0, stds)

        normed = (raw - means) / stds
        return np.clip(normed, -3.0, 3.0)

    # ── Protocol compatibility ───────────────────────────────────────

    def current_regime(self) -> str:
        """RegimeDetector protocol: return current regime label string."""
        return self._current_regime.name.lower()

    # ── Properties ───────────────────────────────────────────────────

    @property
    def last_proba(self) -> RegimeProba | None:
        return self._last_proba

    @property
    def is_halted(self) -> bool:
        return self._whipsaw_halt

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist model, state map, and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.config.emission_type == "gaussian":
            # hmmlearn model goes in joblib with metadata
            joblib.dump(
                {
                    "model": self.model,
                    "state_map": self.state_map,
                    "config": self.config,
                    "version": 3,
                    "n_features": N_FEATURES,
                },
                path / "regime_v2.joblib",
            )
        else:
            # pomegranate model saved via torch
            torch.save(self.model, path / "hmm_model.pt")
            joblib.dump(
                {
                    "state_map": self.state_map,
                    "config": self.config,
                    "version": 3,
                    "n_features": N_FEATURES,
                },
                path / "regime_v2.joblib",
            )
        logger.info("RegimeDetectorV2 (%s) saved to %s", self.config.emission_type, path)

    @classmethod
    def load(cls, path: str | Path) -> RegimeDetectorV2:
        """Load a persisted V2 detector."""
        path = Path(path)
        data = joblib.load(path / "regime_v2.joblib")
        obj = cls(config=data["config"])
        obj.state_map = data["state_map"]

        if data["config"].emission_type == "gaussian":
            obj.model = data["model"]
        else:
            model_path = path / "hmm_model.pt"
            if model_path.exists():
                obj.model = torch.load(model_path, weights_only=False)
            else:
                # Legacy fallback
                obj.model = data.get("model")

        return obj


# ── Evaluation helpers ───────────────────────────────────────────────

def compute_regime_stats(probas: list[RegimeProba]) -> dict:
    """Compute summary statistics from a sequence of RegimeProba outputs."""
    n = len(probas)
    if n == 0:
        return {}

    # State distribution
    dist = {label.name: 0 for label in RegimeLabel}
    for p in probas:
        dist[p.regime.name] += 1
    dist = {k: v / n for k, v in dist.items()}

    # Position size distribution
    size_dist = {"full": 0, "half": 0, "flat": 0}
    for p in probas:
        size_dist[p.position_size] += 1
    size_dist = {k: v / n for k, v in size_dist.items()}

    # Transitions
    transitions = sum(1 for p in probas if p.transition_signal)

    # Halt fraction
    halted = sum(1 for p in probas if p.whipsaw_halt)

    # Average confidence
    avg_conf = np.mean([p.confidence for p in probas])

    # Regime stint lengths
    stints: list[int] = []
    current_stint = 0
    for p in probas:
        if p.transition_signal and current_stint > 0:
            stints.append(current_stint)
            current_stint = 1
        else:
            current_stint += 1
    if current_stint > 0:
        stints.append(current_stint)

    return {
        "n_bars": n,
        "state_distribution": dist,
        "position_size_distribution": size_dist,
        "transitions": transitions,
        "halt_fraction": halted / n,
        "avg_confidence": float(avg_conf),
        "avg_stint_length": float(np.mean(stints)) if stints else 0.0,
        "median_stint_length": float(np.median(stints)) if stints else 0.0,
    }
