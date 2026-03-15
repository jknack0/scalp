"""Streaming feature builder for TickDirectionPredictor (26 features).

Computes a fixed-length feature vector from a sliding window of BarEvents.
Maintains RingBuffers for close, open, high, low, volume.

All features use ONLY data available at time t (no lookahead).
Features that cannot be computed yet (buffer not full enough) return np.nan.

Feature layout (26 total):
  SET A [0-17]  Pure OHLCV (price action, volume, CVD, candle structure)
  SET B [18-25] Estimated L1 (approximated from OHLCV bar shape)
"""

from __future__ import annotations

import numpy as np

from src.core.events import BarEvent
from src.core.logging import get_logger
from src.signals.tick_predictor.features.ring_buffer import RingBuffer

logger = get_logger("tick_predictor.feature_builder")

MES_TICK_SIZE = 0.25
MES_TICK_VALUE = 1.25
MES_POINT_VALUE = 5.00

# ── Feature names ────────────────────────────────────────────────────

SET_A_NAMES: list[str] = [
    # Price action (6)
    "return_1",
    "return_5",
    "return_15",
    "realized_vol_20",
    "hl_range",
    "hl_range_zscore",
    # Volume (5)
    "volume_zscore_20",
    "volume_zscore_50",
    "volume_ratio_5",
    "volume_accel",
    "up_vol_ratio_10",
    # OHLCV CVD (4)
    "cvd_delta",
    "cvd_cumsum_10",
    "cvd_cumsum_30",
    "cvd_slope_10",
    # Candle structure (3)
    "close_position",
    "close_pos_zscore",
    "body_ratio",
]

# SET B — estimated from OHLCV, replace with real L1 values when available
SET_B_NAMES: list[str] = [
    "obi_est",
    "obi_est_5",
    "microprice_est",
    "microprice_vs_mid",
    "ofi_est_10",
    "ofi_est_30",
    "spread_proxy",
    "spread_zscore",
]

FEATURE_NAMES: list[str] = SET_A_NAMES + SET_B_NAMES
NUM_FEATURES = len(FEATURE_NAMES)

# Features that are already z-scores/ratios — skip outer normalization
_SKIP_NORM = {
    "volume_zscore_20", "volume_zscore_50",
    "close_pos_zscore", "hl_range_zscore",
    "obi_est_5", "spread_zscore",
}
_SKIP_NORM_IDX = frozenset(i for i, n in enumerate(FEATURE_NAMES) if n in _SKIP_NORM)

_MIN_BARS_WARMUP = 51  # need >=50 for zscore windows


class _NormBuffer:
    """Rolling z-score normalizer using a configurable window."""

    __slots__ = ("_buf", "_capacity", "_head", "_count")

    def __init__(self, capacity: int = 100) -> None:
        self._buf = np.zeros(capacity, dtype=np.float64)
        self._capacity = capacity
        self._head = 0
        self._count = 0

    def update_and_normalize(self, value: float) -> float:
        if np.isnan(value):
            return np.nan
        self._buf[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1
        if self._count < 5:
            return 0.0
        data = self._buf[:self._count] if self._count < self._capacity else self._buf
        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1))
        if std < 1e-9:
            return 0.0
        return (value - mean) / std

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._head = 0
        self._count = 0


class FeatureBuilder:
    """Computes 26-feature vector from streaming BarEvents."""

    def __init__(self, capacity: int = 200) -> None:
        self._capacity = capacity
        self._close = RingBuffer(capacity, "close")
        self._open = RingBuffer(capacity, "open")
        self._high = RingBuffer(capacity, "high")
        self._low = RingBuffer(capacity, "low")
        self._volume = RingBuffer(capacity, "volume")

        # Derived series maintained incrementally
        self._cvd_delta = RingBuffer(capacity, "cvd_delta")
        self._close_position = RingBuffer(capacity, "close_position")
        self._obi_est = RingBuffer(capacity, "obi_est")
        self._bid_est = RingBuffer(capacity, "bid_est")
        self._ask_est = RingBuffer(capacity, "ask_est")
        self._ofi_raw = RingBuffer(capacity, "ofi_raw")
        self._hl_range_ticks = RingBuffer(capacity, "hl_range_ticks")
        self._vol_zscore_20 = RingBuffer(capacity, "vol_zscore_20")

        self._bar_count = 0
        self._prev_bid_est = 0.0
        self._prev_ask_est = 0.0
        self._prev_close = 0.0

        # Per-feature rolling normalizers
        self._norms = [_NormBuffer(100) for _ in range(NUM_FEATURES)]

        # Cache last raw vector for get_feature_dict
        self._last_raw = np.full(NUM_FEATURES, np.nan, dtype=np.float64)

    def on_bar(self, bar: BarEvent) -> np.ndarray:
        """Append bar data and return feature vector shape (26,)."""
        o, h, l, c = bar.open, bar.high, bar.low, bar.close
        vol = float(bar.volume)
        hl = h - l
        hl_denom = hl + 1e-9

        # ── Derived values ───────────────────────────────────
        cvd_d = vol * (2.0 * (c - l) / hl_denom - 1.0)
        close_pos = (c - l) / hl_denom
        body_r = abs(c - o) / hl_denom
        hl_ticks = hl / MES_TICK_SIZE

        # SET B — estimated from OHLCV, replace with real L1 values when available
        ask_est = vol * (c - l) / hl_denom   # buyers lifting ask
        bid_est = vol * (h - c) / hl_denom   # sellers hitting bid
        total_est = bid_est + ask_est + 1e-9
        obi = (bid_est - ask_est) / total_est

        # OFI: only accumulate when price unchanged
        if self._bar_count > 0:
            ofi_raw = (bid_est - self._prev_bid_est) - (ask_est - self._prev_ask_est)
        else:
            ofi_raw = 0.0

        # ── Append to buffers ────────────────────────────────
        self._close.append(c)
        self._open.append(o)
        self._high.append(h)
        self._low.append(l)
        self._volume.append(vol)
        self._cvd_delta.append(cvd_d)
        self._close_position.append(close_pos)
        self._obi_est.append(obi)
        self._bid_est_buf_append(bid_est)
        self._ask_est_buf_append(ask_est)
        self._ofi_raw.append(ofi_raw)
        self._hl_range_ticks.append(hl_ticks)

        self._prev_bid_est = bid_est
        self._prev_ask_est = ask_est
        self._prev_close = c
        self._bar_count += 1

        # ── Compute features ────────────────────────────────
        raw = self._compute_raw(c, vol, cvd_d, close_pos, body_r,
                                hl_ticks, obi, bid_est, ask_est)
        self._last_raw = raw

        # Z-score normalize (skip already-normalized features)
        out = np.empty(NUM_FEATURES, dtype=np.float64)
        for i in range(NUM_FEATURES):
            if i in _SKIP_NORM_IDX:
                out[i] = raw[i]  # already scaled
            else:
                out[i] = self._norms[i].update_and_normalize(raw[i])
        return out

    def _bid_est_buf_append(self, v: float) -> None:
        self._bid_est.append(v)

    def _ask_est_buf_append(self, v: float) -> None:
        self._ask_est.append(v)

    def is_warm(self) -> bool:
        return self._bar_count >= _MIN_BARS_WARMUP

    def reset(self) -> None:
        """Clear all buffers (called on session close)."""
        for buf in (self._close, self._open, self._high, self._low,
                    self._volume, self._cvd_delta, self._close_position,
                    self._obi_est, self._bid_est, self._ask_est,
                    self._ofi_raw, self._hl_range_ticks, self._vol_zscore_20):
            buf.reset()
        self._bar_count = 0
        self._prev_bid_est = 0.0
        self._prev_ask_est = 0.0
        self._prev_close = 0.0
        for n in self._norms:
            n.reset()
        self._last_raw = np.full(NUM_FEATURES, np.nan, dtype=np.float64)

    def get_feature_dict(self) -> dict[str, float]:
        """Last feature vector as named dict (for logging/debug)."""
        return {name: float(self._last_raw[i])
                for i, name in enumerate(FEATURE_NAMES)}

    # ── Internal computation ─────────────────────────────────────────

    def _compute_raw(self, c: float, vol: float, cvd_d: float,
                     close_pos: float, body_r: float, hl_ticks: float,
                     obi: float, bid_est: float, ask_est: float) -> np.ndarray:
        out = np.full(NUM_FEATURES, np.nan, dtype=np.float64)
        n = self._bar_count

        if n < 2:
            return out

        close_arr = self._close.get_array()
        vol_arr = self._volume.get_array()

        # ══════════════════════════════════════════════════════════════
        # SET A — Pure OHLCV (18 features)
        # ══════════════════════════════════════════════════════════════

        # ── Price action (6) ─────────────────────────────────────────
        # 0: return_1
        out[0] = float(np.log(close_arr[-1] / close_arr[-2]))

        # 1: return_5
        if n >= 6:
            out[1] = float(np.log(close_arr[-1] / close_arr[-6]))

        # 2: return_15
        if n >= 16:
            out[2] = float(np.log(close_arr[-1] / close_arr[-16]))

        # 3: realized_vol_20
        if n >= 21:
            log_rets = np.diff(np.log(close_arr[-21:]))
            out[3] = float(np.std(log_rets, ddof=1))

        # 4: hl_range (in ticks)
        out[4] = hl_ticks

        # 5: hl_range_zscore
        if n >= 50:
            hl_arr = self._hl_range_ticks.get_array()
            hl_50 = hl_arr[-50:]
            mean_hl = float(np.mean(hl_50))
            std_hl = float(np.std(hl_50, ddof=1))
            out[5] = (hl_ticks - mean_hl) / (std_hl + 1e-9)

        # ── Volume (5) ──────────────────────────────────────────────
        # 6: volume_zscore_20
        if n >= 20:
            v20 = vol_arr[-20:]
            mean_v = float(np.mean(v20))
            std_v = float(np.std(v20, ddof=1))
            vz20 = (vol - mean_v) / (std_v + 1e-9)
            out[6] = vz20
            self._vol_zscore_20.append(vz20)
        else:
            self._vol_zscore_20.append(0.0)

        # 7: volume_zscore_50
        if n >= 50:
            v50 = vol_arr[-50:]
            mean_v = float(np.mean(v50))
            std_v = float(np.std(v50, ddof=1))
            out[7] = (vol - mean_v) / (std_v + 1e-9)

        # 8: volume_ratio_5
        if n >= 5:
            out[8] = vol / (float(np.mean(vol_arr[-5:])) + 1e-9)

        # 9: volume_accel = vol_zscore_20[t] - vol_zscore_20[t-5]
        if n >= 25:
            vz_arr = self._vol_zscore_20.get_array()
            if len(vz_arr) >= 6:
                out[9] = float(vz_arr[-1] - vz_arr[-6])

        # 10: up_vol_ratio_10
        if n >= 10:
            close_10 = close_arr[-10:]
            open_10 = self._open.get_array()[-10:]
            vol_10 = vol_arr[-10:]
            up_mask = close_10 > open_10
            total_v = float(np.sum(vol_10))
            if total_v > 0:
                out[10] = float(np.sum(vol_10[up_mask])) / total_v
            else:
                out[10] = 0.5

        # ── OHLCV CVD (4) ───────────────────────────────────────────
        # 11: cvd_delta (current bar)
        out[11] = cvd_d

        cvd_arr = self._cvd_delta.get_array()

        # 12: cvd_cumsum_10
        if n >= 10:
            out[12] = float(np.sum(cvd_arr[-10:]))

        # 13: cvd_cumsum_30
        if n >= 30:
            out[13] = float(np.sum(cvd_arr[-30:]))

        # 14: cvd_slope_10
        if n >= 11:
            cumsum_now = float(np.sum(cvd_arr[-10:]))
            # Need cumsum from 10 bars ago: sum of cvd_arr[-(10+10):-10]
            if n >= 20:
                cumsum_10ago = float(np.sum(cvd_arr[-20:-10]))
            else:
                cumsum_10ago = 0.0
            out[14] = (cumsum_now - cumsum_10ago) / 10.0

        # ── Candle structure (3) ─────────────────────────────────────
        # 15: close_position
        out[15] = close_pos

        # 16: close_pos_zscore
        if n >= 20:
            cp_arr = self._close_position.get_array()
            cp20 = cp_arr[-20:]
            mean_cp = float(np.mean(cp20))
            std_cp = float(np.std(cp20, ddof=1))
            out[16] = (close_pos - mean_cp) / (std_cp + 1e-9)

        # 17: body_ratio
        out[17] = body_r

        # ══════════════════════════════════════════════════════════════
        # SET B — estimated from OHLCV, replace with real L1 values when available
        # ══════════════════════════════════════════════════════════════

        # 18: obi_est
        out[18] = obi

        # 19: obi_est_5
        if n >= 5:
            obi_arr = self._obi_est.get_array()
            out[19] = float(np.mean(obi_arr[-5:]))

        # 20: microprice_est
        hl_range_pts = hl_ticks * MES_TICK_SIZE
        half_spread = hl_range_pts / 2.0
        total_est = bid_est + ask_est + 1e-9
        mp = (bid_est * (c + half_spread) + ask_est * (c - half_spread)) / total_est
        out[20] = mp

        # 21: microprice_vs_mid
        out[21] = mp - c

        # 22: ofi_est_10
        if n >= 10:
            ofi_arr = self._ofi_raw.get_array()
            out[22] = float(np.sum(ofi_arr[-10:]))

        # 23: ofi_est_30
        if n >= 30:
            ofi_arr = self._ofi_raw.get_array()
            out[23] = float(np.sum(ofi_arr[-30:]))

        # 24: spread_proxy (hl_range in ticks)
        out[24] = hl_ticks

        # 25: spread_zscore
        if n >= 50:
            hl_arr = self._hl_range_ticks.get_array()
            s50 = hl_arr[-50:]
            mean_s = float(np.mean(s50))
            std_s = float(np.std(s50, ddof=1))
            out[25] = (hl_ticks - mean_s) / (std_s + 1e-9)

        return out
