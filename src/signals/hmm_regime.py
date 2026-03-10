"""HMM regime classification signal.

Wraps the existing HMMRegimeClassifier to produce a SignalResult.
Computes the 6-feature matrix from raw bars, z-score normalizes,
and predicts the current regime state.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from src.core.events import BarEvent
from src.models.hmm_regime import HMMRegimeClassifier, RegimeState
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class HMMRegimeConfig:
    pass_states: list[RegimeState] = field(default_factory=list)
    model_path: str = "models/hmm/v1"

    def __post_init__(self):
        # Convert string state names to RegimeState enums
        if self.pass_states and isinstance(self.pass_states[0], str):
            converted = [RegimeState[s] for s in self.pass_states]
            object.__setattr__(self, "pass_states", converted)


def _ols_slope(y: np.ndarray) -> float:
    """OLS slope of y against integer index [0..n-1]."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def _wilder_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Compute Wilder ATR over the last `period` bars using EMA smoothing.

    Returns ATR in price units for the final bar.
    """
    n = len(closes)
    if n < 2:
        return 0.0

    # True range for each bar (skip bar 0 which has no prev close)
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = closes[0]
    prev_close[1:] = closes[:-1]
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
    )

    # Wilder EMA: alpha = 1/period
    alpha = 1.0 / period
    atr = tr[0]
    for i in range(1, n):
        atr = alpha * tr[i] + (1.0 - alpha) * atr
    return float(atr)


def _session_vwap_dev_sd(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray
) -> float:
    """Compute VWAP deviation of the last close in rolling standard deviations.

    VWAP is cumulative over all bars (session VWAP). The deviation SD is
    the rolling 20-bar std of (typical - vwap).
    """
    n = len(closes)
    typical = (highs + lows + closes) / 3.0
    cum_pv = np.cumsum(typical * volumes)
    cum_vol = np.cumsum(volumes)
    cum_vol_safe = np.where(cum_vol == 0, 1.0, cum_vol)
    vwap = cum_pv / cum_vol_safe

    dev = typical - vwap

    # Rolling 20-bar std of deviation
    window = min(20, n)
    if window < 2:
        return 0.0
    recent_dev = dev[-window:]
    sd = float(np.std(recent_dev, ddof=1))
    if sd < 1e-10:
        return 0.0
    return float(dev[-1] / sd)


def _build_feature_matrix(bars: list[BarEvent]) -> np.ndarray:
    """Build (N, 6) raw feature matrix from bars.

    Features:
        0: atr_ticks       — Wilder ATR(14) / 0.25
        1: vwap_dev_sd     — session VWAP deviation in rolling SDs
        2: cvd_slope        — OLS slope of per-bar CVD over 20 bars
        3: poc_distance_ticks — |close - session POC| / 0.25
        4: realized_vol     — 20-bar rolling std of log returns
        5: return_20bar     — 20-bar log return
    """
    tick_size = 0.25
    n = len(bars)

    highs = np.array([b.high for b in bars], dtype=np.float64)
    lows = np.array([b.low for b in bars], dtype=np.float64)
    closes = np.array([b.close for b in bars], dtype=np.float64)
    opens = np.array([b.open for b in bars], dtype=np.float64)
    volumes = np.array([b.volume for b in bars], dtype=np.float64)

    # Pre-compute log returns
    log_ret = np.zeros(n, dtype=np.float64)
    log_ret[1:] = np.log(np.maximum(closes[1:], 1e-10) / np.maximum(closes[:-1], 1e-10))

    # Per-bar CVD (bar approximation: up bar -> +volume, down -> -volume)
    bar_delta = np.where(
        closes > opens, volumes, np.where(closes < opens, -volumes, 0.0)
    )

    # Volume profile for POC (session cumulative)
    profile: dict[float, float] = defaultdict(float)

    rows = np.empty((n, 6), dtype=np.float64)

    for i in range(n):
        # --- ATR ticks (use bars up to i) ---
        start_atr = max(0, i - 13)  # at least 14 bars if available
        atr_val = _wilder_atr(
            highs[start_atr : i + 1],
            lows[start_atr : i + 1],
            closes[start_atr : i + 1],
            period=14,
        )
        rows[i, 0] = atr_val / tick_size

        # --- VWAP dev SD (cumulative session) ---
        rows[i, 1] = _session_vwap_dev_sd(
            highs[: i + 1], lows[: i + 1], closes[: i + 1], volumes[: i + 1]
        )

        # --- CVD slope (20-bar OLS) ---
        cvd_start = max(0, i - 19)
        rows[i, 2] = _ols_slope(bar_delta[cvd_start : i + 1])

        # --- POC distance ticks ---
        level = round(closes[i] / tick_size) * tick_size
        profile[level] += volumes[i]
        poc_price = max(profile, key=profile.__getitem__)
        rows[i, 3] = abs(closes[i] - poc_price) / tick_size

        # --- Realized vol (20-bar rolling std of log returns) ---
        rv_start = max(0, i - 19)
        window_rets = log_ret[rv_start : i + 1]
        if len(window_rets) >= 2:
            rows[i, 4] = float(np.std(window_rets, ddof=1))
        else:
            rows[i, 4] = 0.0

        # --- 20-bar log return ---
        if i >= 20:
            rows[i, 5] = math.log(max(closes[i], 1e-10) / max(closes[i - 20], 1e-10))
        else:
            rows[i, 5] = 0.0

    return rows


def _zscore_normalize(matrix: np.ndarray) -> np.ndarray:
    """Column-wise z-score normalization. Returns a copy."""
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=1)
    stds = np.where(stds < 1e-10, 1.0, stds)
    return (matrix - means) / stds


@SignalRegistry.register
class HMMRegimeSignal(SignalBase):
    """HMM regime classification signal.

    Requires at least 70 bars (50 for the HMM window + 20 for rolling
    feature warm-up). Builds a 6-feature matrix, z-score normalizes,
    takes the last 50 rows, and predicts the regime via the classifier.

    If no classifier is provided, passes is always True and value is 0.
    """

    name = "hmm_regime"

    def __init__(
        self,
        config: HMMRegimeConfig | None = None,
        classifier: HMMRegimeClassifier | None = None,
    ) -> None:
        self.config = config or HMMRegimeConfig()
        self._classifier = classifier

        # Auto-load classifier from model_path if not injected
        if self._classifier is None and self.config.model_path:
            from pathlib import Path
            model_dir = Path(self.config.model_path)
            if (model_dir / "hmm_model.joblib").exists():
                self._classifier = HMMRegimeClassifier.load(model_dir)
                import logging
                logging.getLogger(__name__).info(
                    "HMM classifier auto-loaded from %s", self.config.model_path
                )

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        min_bars = 70  # 50 for HMM window + 20 for rolling features

        # Not enough data
        if len(bars) < min_bars:
            return SignalResult(
                value=0.0,
                passes=True,
                direction="none",
                metadata={"reason": "insufficient_bars", "bar_count": len(bars)},
            )

        # No classifier loaded — pass-through
        if self._classifier is None or self._classifier.model is None:
            return SignalResult(
                value=0.0,
                passes=True,
                direction="none",
                metadata={"reason": "no_classifier"},
            )

        # Build raw feature matrix from all bars
        raw = _build_feature_matrix(bars)  # (N, 6)

        # Z-score normalize
        normed = _zscore_normalize(raw)

        # Take last 50 rows for HMM prediction window
        window = normed[-50:]

        # Replace any NaN/Inf with 0 for safety
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict regime
        state, probs = self._classifier.predict_proba(window)

        # Check pass condition
        pass_states = self.config.pass_states
        if pass_states:
            passes = state in pass_states
        else:
            passes = True  # empty list = all states pass

        # Direction heuristic based on regime
        if state == RegimeState.HIGH_VOL_UP:
            direction = "long"
        elif state == RegimeState.HIGH_VOL_DOWN:
            direction = "short"
        else:
            direction = "none"

        # Build probability dict for metadata
        prob_dict = {rs.name: float(probs[rs.value]) for rs in RegimeState}

        return SignalResult(
            value=float(state.value),
            passes=passes,
            direction=direction,
            metadata={
                "regime": state.name,
                "regime_value": state.value,
                "probabilities": prob_dict,
                "pass_states": [s.name for s in pass_states],
            },
        )
