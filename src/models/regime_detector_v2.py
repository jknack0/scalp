"""Phase 7: Production Regime Detector V2 — 3-state HMM, forward-only.

Step 4 upgrade: T2 Features + PCA Decorrelation
- T2 features: ADX (trend strength), vol-of-vol, VPIN (informed trading)
- PCA whitening: decorrelates features before HMM, retains 95%+ variance
- Dynamic feature count: T1 (4 features) or T2 (7 features)

Previous steps:
- Step 1: 3-state GaussianHMM, 2 features, forward-only
- Step 2: 4 features (+ Hurst DFA, autocorr), dual-backend (Gaussian default, Student-t optional)
- Step 3: Transition layer (BOCPD + CUSUM + vol z-score, 2-of-3 agreement)
- Anti-whipsaw: hysteresis, min bars in regime, flip halt
- Position sizing from confidence thresholds

Usage:
    # Training (offline)
    detector = RegimeDetectorV2(config)
    detector.fit(features)             # (N, 7) matrix for T2
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
from scipy.special import gammaln
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

N_FEATURES_T1 = 4  # log_return, gk_vol, hurst_dfa, autocorr_sum
N_FEATURES_T2 = 7  # T1 + adx, vol_of_vol, vpin


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

    # Feature tier: "t1" = 4 features, "t2" = 7 features (+ ADX, vol-of-vol, VPIN)
    feature_tier: str = "t2"

    # Features (T1)
    gk_vol_window: int = 20          # Garman-Klass vol rolling window
    hurst_window: int = 250          # DFA rolling window (bars)
    hurst_stride: int = 5            # Compute DFA every N bars (batch), ffill between
    autocorr_window: int = 100       # Rolling window for autocorrelation
    autocorr_max_lag: int = 10       # Max lag for autocorrelation sum

    # Features (T2)
    adx_period: int = 14             # ADX smoothing period
    vol_of_vol_window: int = 20      # Rolling std of GK vol window
    vpin_bucket_size: int = 100      # VPIN volume per bucket
    vpin_n_buckets: int = 50         # VPIN rolling bucket count

    # PCA decorrelation
    pca_enabled: bool = True         # Apply PCA whitening before HMM
    pca_min_variance: float = 0.95   # Retain components explaining this fraction

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

    # Transition layer (Step 3: BOCPD + CUSUM + vol z-score)
    transition_agreement: int = 2         # N-of-3 detectors must agree
    transition_lookback: int = 5          # Bars to look back for agreement

    # BOCPD
    bocpd_hazard_lambda: int = 60         # Expected run length (bars between changepoints)
    bocpd_threshold: float = 0.3          # P(r=0) threshold for changepoint signal
    bocpd_max_run: int = 300              # Truncation length for run-length distribution

    # CUSUM
    cusum_k: float = 0.5                  # Slack parameter (z-score units)
    cusum_h: float = 4.0                  # Decision threshold

    # Vol z-score changepoint
    vol_zscore_threshold: float = 2.0     # GK vol z-score threshold for changepoint
    vol_zscore_window: int = 50           # Rolling window for vol z-score baseline


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


def _rolling_adx(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Rolling ADX (Average Directional Index) via Wilder smoothing.

    Returns ADX values (0-100 scale) for each bar. First `period*2` bars are NaN.
    """
    n = len(close)
    adx_out = np.full(n, np.nan)
    if n < period * 2 + 1:
        return adx_out

    # True Range, +DM, -DM (starting from bar 1)
    tr = np.empty(n - 1)
    plus_dm = np.empty(n - 1)
    minus_dm = np.empty(n - 1)
    for i in range(1, n):
        tr[i - 1] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i - 1] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i - 1] = down_move if (down_move > up_move and down_move > 0) else 0.0

    # Wilder smoothing
    def _wilder(arr: np.ndarray, p: int) -> np.ndarray:
        out = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = out[i - 1] + (arr[i] - out[i - 1]) / p
        return out

    atr = _wilder(tr, period)
    plus_di_s = _wilder(plus_dm, period)
    minus_di_s = _wilder(minus_dm, period)

    # DI values
    safe_atr = np.where(atr < 1e-10, 1.0, atr)
    plus_di = 100.0 * plus_di_s / safe_atr
    minus_di = 100.0 * minus_di_s / safe_atr

    # DX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    safe_sum = np.where(di_sum < 1e-10, 1.0, di_sum)
    dx = 100.0 * di_diff / safe_sum

    # ADX = Wilder smooth of DX
    adx_raw = _wilder(dx, period)

    # Map back: adx_raw[i] corresponds to bar i+1 (since TR starts at bar 1)
    adx_out[1:] = adx_raw
    return adx_out


def _rolling_vol_of_vol(gk_vol: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling standard deviation of GK volatility (vol-of-vol)."""
    series = pl.Series("v", gk_vol)
    result = series.rolling_std(window_size=window, min_samples=2).to_numpy()
    return result


def _rolling_vpin(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    bucket_size: int = 100,
    n_buckets: int = 50,
) -> np.ndarray:
    """Rolling VPIN (Volume-Synchronized Probability of Informed Trading).

    Uses BVC (Bulk Volume Classification) approximation.
    Returns VPIN value (0-1) per bar based on trailing bucket window.
    """
    n = len(close)
    vpin_out = np.full(n, np.nan)

    bucket_imbalances: deque[float] = deque(maxlen=n_buckets)
    bucket_buy = 0.0
    bucket_sell = 0.0
    bucket_remaining = float(bucket_size)

    for i in range(n):
        vol = float(volume[i])
        if vol <= 0:
            if len(bucket_imbalances) > 0:
                vpin_out[i] = float(np.mean(list(bucket_imbalances)))
            continue

        # BVC classification
        hl_range = high[i] - low[i]
        buy_pct = (close[i] - low[i]) / hl_range if hl_range > 1e-10 else 0.5
        bar_buy = vol * buy_pct
        bar_sell = vol - bar_buy

        remaining_buy = bar_buy
        remaining_sell = bar_sell
        remaining_vol = vol

        while remaining_vol > 1e-9:
            fill = min(remaining_vol, bucket_remaining)
            if remaining_vol > 0:
                proportion = fill / remaining_vol
            else:
                break

            bucket_buy += remaining_buy * proportion
            bucket_sell += remaining_sell * proportion
            remaining_buy -= remaining_buy * proportion
            remaining_sell -= remaining_sell * proportion
            remaining_vol -= fill
            bucket_remaining -= fill

            if bucket_remaining < 1e-9:
                imbalance = abs(bucket_buy - bucket_sell) / bucket_size
                bucket_imbalances.append(imbalance)
                bucket_buy = 0.0
                bucket_sell = 0.0
                bucket_remaining = float(bucket_size)

        if len(bucket_imbalances) > 0:
            vpin_out[i] = float(np.mean(list(bucket_imbalances)))

    return vpin_out


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
    """Build the feature matrix for V2 regime detector.

    T1 (4 features): [log_return, garman_klass_vol, hurst_dfa, autocorr_sum]
    T2 (7 features): T1 + [adx, vol_of_vol, vpin]

    Args:
        df: Bar DataFrame with columns: timestamp, open, high, low, close, volume
        config: Detector config (for window sizes and feature tier)

    Returns:
        (features, timestamps) — features is (N, n_features), timestamps is (N,) int64 ns.
        Warm-up rows dropped.
    """
    cfg = config or RegimeDetectorV2Config()

    close = df["close"].to_numpy().astype(np.float64)
    open_ = df["open"].to_numpy().astype(np.float64)
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    volume = df["volume"].to_numpy().astype(np.float64)
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

    columns = [log_ret, gk_vol, hurst, autocorr]

    # T2 features
    if cfg.feature_tier == "t2":
        # Feature 5: ADX (trend strength)
        adx = _rolling_adx(open_, high, low, close, period=cfg.adx_period)
        # Scale ADX from 0-100 to roughly 0-1 before z-scoring
        adx_scaled = adx / 100.0

        # Feature 6: Vol-of-vol (rolling std of GK vol)
        vov = _rolling_vol_of_vol(gk_vol, window=cfg.vol_of_vol_window)

        # Feature 7: VPIN
        vpin = _rolling_vpin(
            close, high, low, volume,
            bucket_size=cfg.vpin_bucket_size,
            n_buckets=cfg.vpin_n_buckets,
        )

        columns.extend([adx_scaled, vov, vpin])

    # Stack raw features (N, n_features)
    raw = np.column_stack(columns)

    # Rolling z-score normalization
    normed = _rolling_zscore(raw, window=cfg.zscore_window)

    # Drop warm-up
    start = cfg.zscore_window
    valid = ~np.any(np.isnan(normed[start:]), axis=1) & ~np.any(
        np.isinf(normed[start:]), axis=1
    )

    return normed[start:][valid], ts_arr[start:][valid]


# ── Changepoint detectors (Step 3) ──────────────────────────────────


class BOCPDDetector:
    """Bayesian Online Changepoint Detection (truncated).

    Maintains a run-length distribution P(r_t | x_{1:t}) with constant
    hazard rate H = 1/lambda. Signals a changepoint when P(r_t=0) exceeds
    a threshold.

    Reference: Adams & MacKay (2007) "Bayesian Online Changepoint Detection"
    """

    def __init__(
        self,
        hazard_lambda: int = 60,
        threshold: float = 0.3,
        max_run: int = 300,
    ) -> None:
        self.hazard_lambda = hazard_lambda
        self.threshold = threshold
        self.max_run = max_run
        self.reset()

    def reset(self) -> None:
        # Run-length distribution: P(r_t = k) for k in [0, max_run]
        self._run_lengths = np.zeros(self.max_run + 1)
        self._run_lengths[0] = 1.0  # Start with r=0
        # Per-run-length sufficient statistics (Gaussian conjugate)
        # mu0=0, kappa0=1, alpha0=1, beta0=0.01 (Normal-Inverse-Gamma prior)
        self._mu0 = 0.0
        self._kappa0 = 1.0
        self._alpha0 = 1.0
        self._beta0 = 0.01
        # Sufficient stats arrays: one per possible run length
        self._sum_x = np.zeros(self.max_run + 1)
        self._sum_x2 = np.zeros(self.max_run + 1)
        self._counts = np.zeros(self.max_run + 1)
        self._changepoint_prob = 0.0
        self._prev_map_rl = 0    # Previous MAP run length
        self._n_obs = 0

    def _predictive_probs_vectorized(self, x: float) -> np.ndarray:
        """Vectorized Student-t predictive probability for all run lengths.

        Uses Normal-Inverse-Gamma conjugate posterior. Returns array of
        predictive densities, one per run length.
        """
        counts = self._counts
        kappa_n = self._kappa0 + counts
        alpha_n = self._alpha0 + counts / 2.0

        # Mean and beta for each run length
        mean_n = np.where(
            counts > 0,
            (self._kappa0 * self._mu0 + self._sum_x) / kappa_n,
            self._mu0,
        )
        safe_counts = np.maximum(counts, 1.0)
        sample_mean = self._sum_x / safe_counts
        s = self._sum_x2 - self._sum_x ** 2 / safe_counts
        beta_n = np.where(
            counts > 0,
            self._beta0 + 0.5 * s + (
                self._kappa0 * counts * (sample_mean - self._mu0) ** 2
            ) / (2.0 * kappa_n),
            self._beta0,
        )

        # Student-t predictive parameters
        dof = 2.0 * alpha_n
        scale2 = beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n)
        scale = np.sqrt(np.maximum(scale2, 1e-20))

        z = (x - mean_n) / scale
        log_prob = (
            gammaln((dof + 1) / 2) - gammaln(dof / 2)
            - 0.5 * np.log(dof * np.pi) - np.log(scale)
            - (dof + 1) / 2 * np.log1p(z**2 / dof)
        )
        return np.exp(np.clip(log_prob, -500, 500))

    def update(self, x: float) -> bool:
        """Ingest one observation, return True if changepoint detected."""
        H = 1.0 / self.hazard_lambda
        self._n_obs += 1

        # Vectorized predictive probabilities for all run lengths
        # Only compute for run lengths with nonzero probability
        active = self._run_lengths > 1e-300
        pred_probs = np.zeros(self.max_run + 1)
        if np.any(active):
            all_preds = self._predictive_probs_vectorized(x)
            pred_probs[active] = all_preds[active]

        # Growth: extend existing run lengths
        new_rl = np.zeros(self.max_run + 1)
        new_rl[1 : self.max_run + 1] = (
            self._run_lengths[: self.max_run] * pred_probs[: self.max_run] * (1 - H)
        )

        # Changepoint: all mass collapses to r=0
        new_rl[0] = np.sum(self._run_lengths * pred_probs * H)

        # Normalize
        total = new_rl.sum()
        if total > 1e-300:
            new_rl *= (1.0 / total)
        else:
            new_rl[0] = 1.0

        # Update sufficient statistics (shift and add new observation)
        new_sum_x = np.empty(self.max_run + 1)
        new_sum_x[0] = x
        new_sum_x[1 : self.max_run + 1] = self._sum_x[: self.max_run] + x

        new_sum_x2 = np.empty(self.max_run + 1)
        new_sum_x2[0] = x * x
        new_sum_x2[1 : self.max_run + 1] = self._sum_x2[: self.max_run] + x * x

        new_counts = np.empty(self.max_run + 1)
        new_counts[0] = 1
        new_counts[1 : self.max_run + 1] = self._counts[: self.max_run] + 1

        self._run_lengths = new_rl
        self._sum_x = new_sum_x
        self._sum_x2 = new_sum_x2
        self._counts = new_counts

        # Changepoint detection: MAP run length drops sharply
        map_rl = int(np.argmax(new_rl))
        if self._prev_map_rl > 0:
            drop_frac = (self._prev_map_rl - map_rl) / self._prev_map_rl
        else:
            drop_frac = 0.0
        self._changepoint_prob = max(0.0, drop_frac)
        detected = (
            drop_frac > self.threshold
            and self._prev_map_rl > 5
        )
        self._prev_map_rl = map_rl

        return detected

    @property
    def changepoint_prob(self) -> float:
        return self._changepoint_prob

    def update_batch(self, values: np.ndarray) -> np.ndarray:
        """Process a batch of values, return boolean array of changepoint signals."""
        signals = np.zeros(len(values), dtype=bool)
        for i, x in enumerate(values):
            signals[i] = self.update(x)
        return signals


class CUSUMDetector:
    """Two-sided CUSUM detector for mean shifts.

    Tracks cumulative sums in both directions. Signals when either
    the positive or negative CUSUM exceeds threshold h.
    Resets after detection.
    """

    def __init__(self, k: float = 0.5, h: float = 4.0) -> None:
        self.k = k    # Slack (allowance) parameter
        self.h = h    # Decision threshold
        self.reset()

    def reset(self) -> None:
        self._s_pos = 0.0  # Positive CUSUM
        self._s_neg = 0.0  # Negative CUSUM
        self._mean = 0.0
        self._n = 0

    def update(self, x: float) -> bool:
        """Ingest one observation, return True if mean shift detected."""
        self._n += 1

        # Online mean estimate (target)
        alpha = min(1.0 / self._n, 0.02)  # Slow-adapting mean
        self._mean += alpha * (x - self._mean)

        deviation = x - self._mean

        # Two-sided CUSUM
        self._s_pos = max(0.0, self._s_pos + deviation - self.k)
        self._s_neg = max(0.0, self._s_neg - deviation - self.k)

        if self._s_pos > self.h or self._s_neg > self.h:
            # Reset after detection
            self._s_pos = 0.0
            self._s_neg = 0.0
            return True

        return False

    def update_batch(self, values: np.ndarray) -> np.ndarray:
        """Process a batch of values, return boolean array of shift signals."""
        signals = np.zeros(len(values), dtype=bool)
        for i, x in enumerate(values):
            signals[i] = self.update(x)
        return signals


class VolZScoreDetector:
    """Volatility z-score changepoint detector.

    Monitors GK volatility z-score against a rolling baseline.
    Signals when |z-score| exceeds threshold, indicating a vol regime change.
    """

    def __init__(self, threshold: float = 2.0, window: int = 50) -> None:
        self.threshold = threshold
        self.window = window
        self.reset()

    def reset(self) -> None:
        self._buffer: deque[float] = deque(maxlen=self.window)
        self._zscore = 0.0

    def update(self, gk_vol: float) -> bool:
        """Ingest one GK vol value, return True if vol regime change detected."""
        self._buffer.append(gk_vol)

        if len(self._buffer) < 10:
            self._zscore = 0.0
            return False

        arr = np.array(self._buffer)
        mean = arr[:-1].mean()  # Baseline excludes current
        std = arr[:-1].std(ddof=1)

        if std < 1e-10:
            self._zscore = 0.0
            return False

        self._zscore = (gk_vol - mean) / std
        return abs(self._zscore) > self.threshold

    @property
    def zscore(self) -> float:
        return self._zscore

    def update_batch(self, values: np.ndarray) -> np.ndarray:
        """Process a batch of values, return boolean array of vol change signals."""
        signals = np.zeros(len(values), dtype=bool)
        for i, x in enumerate(values):
            signals[i] = self.update(x)
        return signals


class TransitionLayer:
    """2-of-3 agreement filter for regime transitions.

    Combines BOCPD, CUSUM, and vol z-score changepoint detectors.
    A transition is only confirmed when N-of-3 detectors signal
    within a lookback window.
    """

    def __init__(self, config: RegimeDetectorV2Config) -> None:
        self.config = config
        self.bocpd = BOCPDDetector(
            hazard_lambda=config.bocpd_hazard_lambda,
            threshold=config.bocpd_threshold,
            max_run=config.bocpd_max_run,
        )
        self.cusum = CUSUMDetector(
            k=config.cusum_k,
            h=config.cusum_h,
        )
        self.vol_zscore = VolZScoreDetector(
            threshold=config.vol_zscore_threshold,
            window=config.vol_zscore_window,
        )
        self._bocpd_signals: deque[bool] = deque(maxlen=config.transition_lookback)
        self._cusum_signals: deque[bool] = deque(maxlen=config.transition_lookback)
        self._vol_signals: deque[bool] = deque(maxlen=config.transition_lookback)

    def reset(self) -> None:
        self.bocpd.reset()
        self.cusum.reset()
        self.vol_zscore.reset()
        self._bocpd_signals.clear()
        self._cusum_signals.clear()
        self._vol_signals.clear()

    def update(self, log_return: float, gk_vol: float) -> bool:
        """Update all 3 detectors, return True if agreement threshold met."""
        b = self.bocpd.update(log_return)
        c = self.cusum.update(log_return)
        v = self.vol_zscore.update(gk_vol)

        self._bocpd_signals.append(b)
        self._cusum_signals.append(c)
        self._vol_signals.append(v)

        return self._check_agreement()

    def _check_agreement(self) -> bool:
        """Check if N-of-3 detectors fired within the lookback window."""
        votes = 0
        if any(self._bocpd_signals):
            votes += 1
        if any(self._cusum_signals):
            votes += 1
        if any(self._vol_signals):
            votes += 1
        return votes >= self.config.transition_agreement

    def update_batch(
        self, log_returns: np.ndarray, gk_vols: np.ndarray
    ) -> np.ndarray:
        """Process batch, return boolean array of confirmed transitions."""
        n = len(log_returns)
        confirmed = np.zeros(n, dtype=bool)
        for i in range(n):
            confirmed[i] = self.update(log_returns[i], gk_vols[i])
        return confirmed

    @property
    def detail(self) -> dict[str, bool]:
        """Current state of each detector (for debugging)."""
        return {
            "bocpd": bool(any(self._bocpd_signals)) if self._bocpd_signals else False,
            "cusum": bool(any(self._cusum_signals)) if self._cusum_signals else False,
            "vol_zscore": bool(any(self._vol_signals)) if self._vol_signals else False,
        }


# ── Detector class ───────────────────────────────────────────────────

class RegimeDetectorV2:
    """Phase 7 production regime detector — 3-state HMM, forward-only.

    Step 4 upgrade:
    - T2 features: ADX, vol-of-vol, VPIN (7 total)
    - PCA decorrelation before HMM (removes cross-feature correlations)
    - Dynamic feature count based on tier config
    - Dual-backend: hmmlearn Gaussian (default) or pomegranate Student-t
    - Forward-only inference, anti-whipsaw, transition layer
    """

    def __init__(self, config: RegimeDetectorV2Config | None = None) -> None:
        self.config = config or RegimeDetectorV2Config()
        self.model: DenseHMM | None = None
        self.state_map: dict[int, RegimeLabel] | None = None
        self._pca_components: np.ndarray | None = None  # (n_kept, n_features)
        self._pca_mean: np.ndarray | None = None         # (n_features,)
        self._n_features: int = (
            N_FEATURES_T2 if self.config.feature_tier == "t2" else N_FEATURES_T1
        )
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

        # T2: ADX online state
        self._adx_highs: deque[float] = deque(maxlen=cfg.adx_period * 3)
        self._adx_lows: deque[float] = deque(maxlen=cfg.adx_period * 3)
        self._adx_closes: deque[float] = deque(maxlen=cfg.adx_period * 3)
        self._adx_atr: float = 0.0
        self._adx_plus_di_s: float = 0.0
        self._adx_minus_di_s: float = 0.0
        self._adx_value: float = 0.0
        self._adx_warmup: int = 0

        # T2: Vol-of-vol online state
        self._gk_vol_buffer: deque[float] = deque(maxlen=cfg.vol_of_vol_window)

        # T2: VPIN online state
        self._vpin_bucket_imbalances: deque[float] = deque(maxlen=cfg.vpin_n_buckets)
        self._vpin_bucket_buy: float = 0.0
        self._vpin_bucket_sell: float = 0.0
        self._vpin_bucket_remaining: float = float(cfg.vpin_bucket_size)

        # Anti-whipsaw state
        self._current_regime: RegimeLabel = RegimeLabel.RANGING
        self._bars_in_regime: int = 0
        self._recent_flips: deque[int] = deque(maxlen=cfg.flip_halt_window)
        self._whipsaw_halt: bool = False
        self._last_proba: RegimeProba | None = None

        # Transition layer (Step 3)
        self._transition_layer = TransitionLayer(cfg)

    def reset(self) -> None:
        """Reset online state (call on session boundaries)."""
        with self._lock:
            self._reset_online_state()

    # ── Training ─────────────────────────────────────────────────────

    def fit(self, features: np.ndarray) -> None:
        """Fit 3-state HMM on (N, n_features) feature matrix.

        Steps:
        1. Optional PCA decorrelation (retain pca_min_variance of explained variance)
        2. KMeans initialization for stable convergence
        3. Fit HMM (Gaussian or Student-t backend)
        4. Label states via Hungarian algorithm
        """
        cfg = self.config
        self._n_features = features.shape[1]

        # Reproducibility
        torch.manual_seed(cfg.random_state)
        np.random.seed(cfg.random_state)

        # PCA decorrelation (fit and transform)
        if cfg.pca_enabled:
            features = self._fit_pca(features, cfg.pca_min_variance)
            logger.info(
                "PCA: %d -> %d components (%.1f%% variance retained)",
                self._n_features, features.shape[1],
                cfg.pca_min_variance * 100,
            )

        n_features_post = features.shape[1]

        # KMeans seeding for emission means
        km = KMeans(
            n_clusters=cfg.n_states,
            random_state=cfg.random_state,
            n_init=10,
        )
        km.fit(features)
        centers = km.cluster_centers_  # (n_states, n_features_post)

        if cfg.emission_type == "gaussian":
            self._fit_gaussian(features, centers, cfg)
        else:
            self._fit_studentt(features, centers, cfg, n_features_post)

        self._label_states()
        logger.info(
            "RegimeDetectorV2 fitted (%s): %d states, %d raw features "
            "-> %d HMM dims, %d samples",
            cfg.emission_type, cfg.n_states, self._n_features,
            n_features_post, len(features),
        )

    def _fit_pca(self, features: np.ndarray, min_variance: float) -> np.ndarray:
        """Fit PCA on features and transform. Stores components for inference."""
        self._pca_mean = features.mean(axis=0)
        centered = features - self._pca_mean

        # SVD (more numerically stable than eigendecomposition)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        explained_var = s ** 2 / (len(features) - 1)
        cumulative = np.cumsum(explained_var) / explained_var.sum()

        # Keep enough components for min_variance
        n_keep = int(np.searchsorted(cumulative, min_variance) + 1)
        n_keep = min(n_keep, len(s))  # Can't keep more than we have

        self._pca_components = vt[:n_keep]  # (n_keep, n_features)
        return centered @ self._pca_components.T

    def _apply_pca(self, features: np.ndarray) -> np.ndarray:
        """Apply pre-fitted PCA transform."""
        if self._pca_components is None or self._pca_mean is None:
            return features
        centered = features - self._pca_mean
        return centered @ self._pca_components.T

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

        If PCA is active, emission means are projected back to original feature
        space for interpretable labeling.

        Heuristic (original feature indices):
        - [0] log_return, [1] gk_vol, [2] hurst, [3] autocorr
        - T2 adds: [4] adx, [5] vol_of_vol, [6] vpin
        - TRENDING:  high Hurst, positive autocorr, high ADX, moderate vol
        - RANGING:   low vol, near-zero return, low Hurst, low ADX
        - HIGH_VOL:  high vol, high return variance, high vol-of-vol
        """
        assert self.model is not None
        n_states = self.config.n_states
        is_gaussian = self.config.emission_type == "gaussian"
        cost = np.zeros((n_states, 3), dtype=np.float64)

        for s in range(n_states):
            if is_gaussian:
                raw_means = self.model.means_[s]
                covars = self.model.covars_
                if self.model.covariance_type == "full":
                    lr_var_pca = covars[s][0, 0]
                elif self.model.covariance_type == "diag":
                    lr_var_pca = covars[s][0]
                else:
                    lr_var_pca = float(covars[s])
            else:
                dist = self.model.distributions[s]
                raw_means = dist.means.data.numpy().flatten()
                covs_raw = dist.covs.data.numpy()
                lr_var_pca = covs_raw[0] if covs_raw.ndim == 1 else float(covs_raw.flat[0])

            # Project back to original feature space if PCA active
            if self._pca_components is not None and self._pca_mean is not None:
                means = raw_means @ self._pca_components + self._pca_mean
            else:
                means = np.asarray(raw_means)

            lr_mean = means[0]
            gk_mean = means[1]
            hurst_mean = means[2] if len(means) > 2 else 0.0
            ac_mean = means[3] if len(means) > 3 else 0.0
            adx_mean = means[4] if len(means) > 4 else 0.0
            vov_mean = means[5] if len(means) > 5 else 0.0

            # TRENDING: directional, persistent, high trend strength
            cost[s, RegimeLabel.TRENDING] = -(
                abs(lr_mean) + 0.5 * hurst_mean + 0.3 * ac_mean
                - 0.2 * gk_mean + 0.4 * adx_mean
            )

            # RANGING: low vol, near-zero return, low persistence, low ADX
            cost[s, RegimeLabel.RANGING] = -(
                -gk_mean - abs(lr_mean) - 0.3 * hurst_mean - 0.3 * adx_mean
            )

            # HIGH_VOL: high vol, high return variance, high vol-of-vol
            cost[s, RegimeLabel.HIGH_VOL] = -(
                gk_mean + lr_var_pca + 0.3 * vov_mean
            )

        row_ind, col_ind = linear_sum_assignment(cost)
        self.state_map = {
            int(r): RegimeLabel(c) for r, c in zip(row_ind, col_ind)
        }
        logger.info("State map: %s", self.state_map)

    # ── Batch prediction (backtest) ──────────────────────────────────

    def _forward_only_proba(self, features: np.ndarray) -> np.ndarray:
        """Compute causal forward-only filtering probabilities.

        Returns P(state_t | obs_1..t) using only the forward pass,
        NOT the smoothed P(state_t | obs_1..T) from forward-backward.
        This prevents future information leakage in backtests.
        """
        from hmmlearn import _hmmc

        log_frameprob = self.model._compute_log_likelihood(features)
        _, fwdlattice = _hmmc.forward_log(
            self.model.startprob_, self.model.transmat_, log_frameprob
        )
        # fwdlattice is log P(obs_1..t, state_t) — normalize to get P(state_t | obs_1..t)
        from scipy.special import logsumexp
        log_norm = logsumexp(fwdlattice, axis=1, keepdims=True)
        log_posteriors = fwdlattice - log_norm
        return np.exp(log_posteriors)

    def predict_proba_sequence(
        self, features: np.ndarray, *, causal: bool = False,
    ) -> list[RegimeProba]:
        """Batch inference over a feature sequence.

        Returns one RegimeProba per row, with anti-whipsaw applied.

        Args:
            causal: If True, use forward-only filtering (no future leakage).
                    If False (default), use forward-backward smoothing.
        """
        assert self.model is not None and self.state_map is not None

        cfg = self.config

        # Apply PCA if fitted
        if cfg.pca_enabled and self._pca_components is not None:
            features = self._apply_pca(features)

        n = len(features)

        # Get posteriors from the appropriate backend
        if causal and cfg.emission_type == "gaussian":
            proba_matrix = self._forward_only_proba(features)  # (T, n_states)
        elif cfg.emission_type == "gaussian":
            proba_matrix = self.model.predict_proba(features)  # (T, n_states)
        else:
            X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            proba_tensor = self.model.predict_proba(X)
            proba_matrix = proba_tensor[0].detach().numpy()  # (T, n_states)

        # Map HMM indices to RegimeLabel order
        mapped = np.zeros((n, 3), dtype=np.float64)
        for hmm_idx, label in self.state_map.items():
            mapped[:, label.value] = proba_matrix[:, hmm_idx]

        # Run transition layer detectors over raw features
        # Use un-normalized log returns and GK vol for changepoint detection
        transition_layer = TransitionLayer(cfg)
        transition_confirmed = transition_layer.update_batch(
            features[:, 0],  # log returns (z-scored, but direction preserved)
            features[:, 1],  # GK vol (z-scored)
        )

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

            # HMM proposes transition — require transition layer confirmation
            hmm_wants_transition = new_regime != current_regime

            if cfg.transition_agreement <= 0:
                # No transition layer: accept all HMM transitions
                transition = hmm_wants_transition
            elif hmm_wants_transition and transition_confirmed[i]:
                # Confirmed transition
                transition = True
            elif hmm_wants_transition:
                # HMM wants to switch but changepoint detectors disagree — suppress
                new_regime = current_regime
                transition = False
            else:
                transition = False

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

            # Apply PCA if fitted
            if self.config.pca_enabled and self._pca_components is not None:
                window = self._apply_pca(window)

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

            # Update transition layer with raw features
            raw_lr = raw[0] if raw is not None else 0.0
            raw_gk = raw[1] if raw is not None else 0.0
            changepoint_confirmed = self._transition_layer.update(raw_lr, raw_gk)

            # Ambiguity
            if max_p < cfg.ambiguity_threshold:
                new_regime = self._current_regime
            else:
                new_regime = self._apply_hysteresis(
                    self._current_regime, mapped, cfg
                )

            # HMM proposes transition — require transition layer confirmation
            hmm_wants_transition = new_regime != self._current_regime

            if cfg.transition_agreement <= 0:
                transition = hmm_wants_transition
            elif hmm_wants_transition and changepoint_confirmed:
                transition = True
            elif hmm_wants_transition:
                new_regime = self._current_regime
                transition = False
            else:
                transition = False

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
        """Compute raw features from a single bar.

        T1: [log_return, gk_vol, hurst, autocorr]
        T2: T1 + [adx, vol_of_vol, vpin]
        """
        close = float(bar["close"])
        open_ = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        volume = float(bar.get("volume", 0))

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

        feats = [lr, gk_vol, hurst, autocorr]

        # T2 features
        if cfg.feature_tier == "t2":
            # ADX (online Wilder smoothing)
            self._adx_highs.append(high)
            self._adx_lows.append(low)
            self._adx_closes.append(close)
            adx_val = self._compute_online_adx(high, low, close, cfg.adx_period)
            adx_scaled = adx_val / 100.0

            # Vol-of-vol
            self._gk_vol_buffer.append(gk_vol)
            if len(self._gk_vol_buffer) >= 2:
                vov = float(np.std(list(self._gk_vol_buffer), ddof=1))
            else:
                vov = 0.0

            # VPIN (online bucket fill)
            vpin_val = self._compute_online_vpin(close, high, low, volume, cfg)

            feats.extend([adx_scaled, vov, vpin_val])

        return np.array(feats, dtype=np.float64)

    def _compute_online_adx(
        self, high: float, low: float, close: float, period: int
    ) -> float:
        """Online ADX computation with Wilder smoothing."""
        self._adx_warmup += 1
        if self._adx_warmup < 2:
            return 0.0

        highs = self._adx_highs
        lows = self._adx_lows
        closes = self._adx_closes

        if len(closes) < 2:
            return 0.0

        # Current bar TR, +DM, -DM
        prev_close = closes[-2]
        prev_high = highs[-2]
        prev_low = lows[-2]

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        # Wilder smoothing (online)
        alpha = 1.0 / period
        if self._adx_warmup == 2:
            self._adx_atr = tr
            self._adx_plus_di_s = plus_dm
            self._adx_minus_di_s = minus_dm
            self._adx_value = 0.0
        else:
            self._adx_atr += alpha * (tr - self._adx_atr)
            self._adx_plus_di_s += alpha * (plus_dm - self._adx_plus_di_s)
            self._adx_minus_di_s += alpha * (minus_dm - self._adx_minus_di_s)

        safe_atr = max(self._adx_atr, 1e-10)
        plus_di = 100.0 * self._adx_plus_di_s / safe_atr
        minus_di = 100.0 * self._adx_minus_di_s / safe_atr

        di_sum = plus_di + minus_di
        if di_sum < 1e-10:
            dx = 0.0
        else:
            dx = 100.0 * abs(plus_di - minus_di) / di_sum

        self._adx_value += alpha * (dx - self._adx_value)
        return self._adx_value

    def _compute_online_vpin(
        self, close: float, high: float, low: float,
        volume: float, cfg: RegimeDetectorV2Config,
    ) -> float:
        """Online VPIN computation using BVC bucket filling."""
        if volume <= 0:
            if len(self._vpin_bucket_imbalances) > 0:
                return float(np.mean(list(self._vpin_bucket_imbalances)))
            return 0.0

        hl_range = high - low
        buy_pct = (close - low) / hl_range if hl_range > 1e-10 else 0.5
        bar_buy = volume * buy_pct
        bar_sell = volume - bar_buy

        remaining_buy = bar_buy
        remaining_sell = bar_sell
        remaining_vol = volume

        while remaining_vol > 1e-9:
            fill = min(remaining_vol, self._vpin_bucket_remaining)
            if remaining_vol > 0:
                proportion = fill / remaining_vol
            else:
                break

            self._vpin_bucket_buy += remaining_buy * proportion
            self._vpin_bucket_sell += remaining_sell * proportion
            remaining_buy -= remaining_buy * proportion
            remaining_sell -= remaining_sell * proportion
            remaining_vol -= fill
            self._vpin_bucket_remaining -= fill

            if self._vpin_bucket_remaining < 1e-9:
                imbalance = abs(
                    self._vpin_bucket_buy - self._vpin_bucket_sell
                ) / cfg.vpin_bucket_size
                self._vpin_bucket_imbalances.append(imbalance)
                self._vpin_bucket_buy = 0.0
                self._vpin_bucket_sell = 0.0
                self._vpin_bucket_remaining = float(cfg.vpin_bucket_size)

        if len(self._vpin_bucket_imbalances) > 0:
            return float(np.mean(list(self._vpin_bucket_imbalances)))
        return 0.0

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
                    "version": 4,
                    "n_features": self._n_features,
                    "pca_components": self._pca_components,
                    "pca_mean": self._pca_mean,
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
                    "version": 4,
                    "n_features": self._n_features,
                    "pca_components": self._pca_components,
                    "pca_mean": self._pca_mean,
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
        obj._n_features = data.get("n_features", N_FEATURES_T1)
        obj._pca_components = data.get("pca_components")
        obj._pca_mean = data.get("pca_mean")

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
