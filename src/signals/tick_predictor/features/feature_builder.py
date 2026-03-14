"""Feature vector builder for TickDirectionPredictor.

Computes a fixed-length feature vector from a sliding window of BarEvents.
Maintains RingBuffers for close, volume, bid_size_approx, ask_size_approx,
and cvd_delta_approx.

All features use **only** data available at time *t* (no look-ahead).
Features that cannot be computed yet (buffer not full enough) return np.nan.
All features are z-score normalized via a rolling 100-bar window.

Feature layout (30 total):
  [0-17]  1s microstructure (order flow, CVD, price action)
  [18-21] HMM regime_v2 (trending/ranging/highvol proba + bars_in)
  [22-25] 1m tactical (Bollinger %B, RSI-14, ATR ratio, OBV slope)
  [26-29] 5m strategic (Donchian position, Stochastic K-14, BB width, EMA cross)
"""

from __future__ import annotations

import numpy as np

from src.core.events import BarEvent
from src.signals.tick_predictor.features.ring_buffer import RingBuffer

MES_TICK_SIZE = 0.25

FEATURE_NAMES: list[str] = [
    # Tier 1 — Order Flow (1s)
    "ofi_10",
    "ofi_30",
    "obi_1",
    "obi_5",
    "microprice",
    "microprice_vs_mid",
    # Tier 2 — CVD & Volume (1s)
    "cvd_delta_10",
    "cvd_delta_30",
    "cvd_slope",
    "volume_imbalance_10",
    "volume_zscore_20",
    # Tier 3 — Price Action & Volatility (1s)
    "return_1",
    "return_5",
    "return_10",
    "realized_vol_20",
    "return_autocorr_10",
    "spread",
    "spread_zscore_50",
    # Tier 4 — HMM Regime (5m)
    "regime_p_trending",
    "regime_p_ranging",
    "regime_p_highvol",
    "regime_bars_in",
    # Tier 5 — Tactical (1m)
    "bb_pctb_1m",        # Channel: Bollinger %B — where is price within bands
    "rsi_14_1m",         # Oscillator: RSI — overbought/oversold
    "atr_ratio_1m",      # Volatility: ATR(5)/ATR(20) — expansion/contraction
    "obv_slope_1m",      # Volume: OBV regression slope — conviction
    # Tier 6 — Strategic (5m)
    "donchian_pos_5m",   # Channel: Donchian position — where in 20-bar range
    "stoch_k_14_5m",     # Oscillator: Stochastic K — range position
    "bb_width_5m",       # Volatility: Bollinger bandwidth — squeeze/expansion
    "ema_cross_5m",      # Trend: (EMA20-EMA50)/ATR — normalized trend strength
]

NUM_FEATURES = len(FEATURE_NAMES)
_MIN_BARS_WARMUP = 51  # Need >=50 bars for spread_zscore_50


class _NormBuffer:
    """Rolling z-score normalizer using a 100-bar window."""

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
        data = self._buf[: self._count] if self._count < self._capacity else self._buf
        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1))
        if std < 1e-12:
            return 0.0
        return (value - mean) / std

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._head = 0
        self._count = 0


class _TacticalIndicators:
    """1m tactical: Bollinger %B, RSI-14, ATR ratio, OBV slope."""

    __slots__ = (
        "_close", "_high", "_low", "_volume", "_bar_count",
        "_obv", "_obv_buf",
        "bb_pctb", "rsi", "atr_ratio", "obv_slope",
    )

    def __init__(self, capacity: int = 50) -> None:
        self._close = RingBuffer(capacity, "tac_close")
        self._high = RingBuffer(capacity, "tac_high")
        self._low = RingBuffer(capacity, "tac_low")
        self._volume = RingBuffer(capacity, "tac_volume")
        self._bar_count = 0
        self._obv = 0.0
        self._obv_buf = RingBuffer(capacity, "tac_obv")
        self.bb_pctb = np.nan
        self.rsi = np.nan
        self.atr_ratio = np.nan
        self.obv_slope = np.nan

    def update(self, open_: float, high: float, low: float, close: float,
               volume: float = 0.0) -> None:
        prev_close = float(self._close.get_array()[-1]) if self._bar_count > 0 else close
        self._close.append(close)
        self._high.append(high)
        self._low.append(low)
        self._volume.append(volume)
        self._bar_count += 1
        n = self._bar_count

        c_arr = self._close.get_array()
        h_arr = self._high.get_array()
        l_arr = self._low.get_array()

        # Bollinger %B (20, 2)
        if n >= 20:
            c20 = c_arr[-20:]
            mid = float(np.mean(c20))
            std = float(np.std(c20, ddof=1))
            if std > 1e-12:
                upper = mid + 2.0 * std
                lower = mid - 2.0 * std
                self.bb_pctb = (close - lower) / (upper - lower)
            else:
                self.bb_pctb = 0.5

        # RSI-14
        if n >= 15:
            rets = np.diff(c_arr[-15:])
            avg_g = float(np.mean(np.maximum(rets, 0.0)))
            avg_l = float(np.mean(np.maximum(-rets, 0.0)))
            self.rsi = (100.0 - 100.0 / (1.0 + avg_g / avg_l)) if avg_l > 1e-12 else 100.0

        # ATR ratio: ATR(5) / ATR(20)
        if n >= 21:
            # True range
            tr_vals = np.maximum(
                h_arr[-21:] - l_arr[-21:],
                np.maximum(
                    np.abs(h_arr[-21:] - np.append(c_arr[-22:-21] if n >= 22 else c_arr[-21:-20], c_arr[-21:-1])),
                    np.abs(l_arr[-21:] - np.append(c_arr[-22:-21] if n >= 22 else c_arr[-21:-20], c_arr[-21:-1])),
                ),
            )
            atr5 = float(np.mean(tr_vals[-5:]))
            atr20 = float(np.mean(tr_vals[-20:]))
            self.atr_ratio = atr5 / atr20 if atr20 > 1e-12 else 1.0
        elif n >= 6:
            # Simplified TR for short warmup
            tr_vals = h_arr[-6:] - l_arr[-6:]
            atr5 = float(np.mean(tr_vals[-5:]))
            self.atr_ratio = 1.0  # not enough data for ratio

        # OBV slope (10-bar regression)
        if close > prev_close:
            self._obv += volume
        elif close < prev_close:
            self._obv -= volume
        self._obv_buf.append(self._obv)

        if n >= 10:
            obv_arr = self._obv_buf.get_array()[-10:]
            x = np.arange(10, dtype=np.float64)
            self.obv_slope = float(np.polyfit(x, obv_arr, 1)[0])

    def reset(self) -> None:
        self._close.reset()
        self._high.reset()
        self._low.reset()
        self._volume.reset()
        self._bar_count = 0
        self._obv = 0.0
        self._obv_buf.reset()
        self.bb_pctb = self.rsi = self.atr_ratio = self.obv_slope = np.nan


class _StrategicIndicators:
    """5m strategic: Donchian position, Stochastic K-14, BB width, EMA cross."""

    __slots__ = (
        "_close", "_high", "_low", "_bar_count",
        "_ema20", "_ema50",
        "donchian_pos", "stoch_k", "bb_width", "ema_cross",
    )

    def __init__(self, capacity: int = 60) -> None:
        self._close = RingBuffer(capacity, "str_close")
        self._high = RingBuffer(capacity, "str_high")
        self._low = RingBuffer(capacity, "str_low")
        self._bar_count = 0
        self._ema20: float | None = None
        self._ema50: float | None = None
        self.donchian_pos = np.nan
        self.stoch_k = np.nan
        self.bb_width = np.nan
        self.ema_cross = np.nan

    def update(self, open_: float, high: float, low: float, close: float,
               volume: float = 0.0) -> None:
        self._close.append(close)
        self._high.append(high)
        self._low.append(low)
        self._bar_count += 1
        n = self._bar_count

        c_arr = self._close.get_array()
        h_arr = self._high.get_array()
        l_arr = self._low.get_array()

        # Donchian position: (close - low20) / (high20 - low20)
        if n >= 20:
            h20 = float(np.max(h_arr[-20:]))
            l20 = float(np.min(l_arr[-20:]))
            d = h20 - l20
            self.donchian_pos = (close - l20) / d if d > 1e-12 else 0.5

        # Stochastic K-14
        if n >= 14:
            h14 = float(np.max(h_arr[-14:]))
            l14 = float(np.min(l_arr[-14:]))
            d = h14 - l14
            self.stoch_k = ((close - l14) / d * 100.0) if d > 1e-12 else 50.0

        # Bollinger bandwidth: (upper - lower) / middle
        if n >= 20:
            c20 = c_arr[-20:]
            mid = float(np.mean(c20))
            std = float(np.std(c20, ddof=1))
            if mid > 1e-12:
                self.bb_width = (4.0 * std) / mid  # 2*std*2 / mid
            else:
                self.bb_width = 0.0

        # EMA cross: (EMA20 - EMA50) / ATR
        if self._ema20 is None:
            self._ema20 = close
            self._ema50 = close
        else:
            self._ema20 = 2.0 / 21.0 * close + (1.0 - 2.0 / 21.0) * self._ema20
            self._ema50 = 2.0 / 51.0 * close + (1.0 - 2.0 / 51.0) * self._ema50

        if n >= 14:
            # ATR for normalization
            tr_vals = np.maximum(
                h_arr[-14:] - l_arr[-14:],
                np.maximum(
                    np.abs(h_arr[-14:] - np.append(c_arr[-15:-14] if n >= 15 else c_arr[-14:-13], c_arr[-14:-1])),
                    np.abs(l_arr[-14:] - np.append(c_arr[-15:-14] if n >= 15 else c_arr[-14:-13], c_arr[-14:-1])),
                ),
            )
            atr = float(np.mean(tr_vals))
            if atr > 1e-12 and self._ema20 is not None:
                self.ema_cross = (self._ema20 - self._ema50) / atr

    def reset(self) -> None:
        self._close.reset()
        self._high.reset()
        self._low.reset()
        self._bar_count = 0
        self._ema20 = self._ema50 = None
        self.donchian_pos = self.stoch_k = self.bb_width = self.ema_cross = np.nan


class _BarAggregator:
    """Aggregates 1s bars into coarser timeframe bars (1m, 5m, etc.)."""

    __slots__ = ("_period", "_count", "_open", "_high", "_low", "_close", "_volume",
                 "_indicators")

    def __init__(self, period_seconds: int,
                 indicators: _TacticalIndicators | _StrategicIndicators) -> None:
        self._period = period_seconds
        self._indicators = indicators
        self._count = 0
        self._open = 0.0
        self._high = -1e18
        self._low = 1e18
        self._close = 0.0
        self._volume = 0.0

    def on_bar(self, open_: float, high: float, low: float, close: float,
               volume: float = 0.0) -> None:
        """Feed a 1s bar. When period_seconds bars accumulate, emit to indicators."""
        if self._count == 0:
            self._open = open_
        self._high = max(self._high, high)
        self._low = min(self._low, low)
        self._close = close
        self._volume += volume
        self._count += 1

        if self._count >= self._period:
            self._indicators.update(self._open, self._high, self._low, self._close,
                                    self._volume)
            self._count = 0
            self._high = -1e18
            self._low = 1e18
            self._volume = 0.0

    def reset(self) -> None:
        self._count = 0
        self._open = 0.0
        self._high = -1e18
        self._low = 1e18
        self._close = 0.0
        self._volume = 0.0
        self._indicators.reset()


class FeatureBuilder:
    """Computes 30-feature vector from streaming BarEvents."""

    def __init__(self, capacity: int = 200) -> None:
        self._capacity = capacity
        self._close = RingBuffer(capacity, "close")
        self._high = RingBuffer(capacity, "high")
        self._low = RingBuffer(capacity, "low")
        self._volume = RingBuffer(capacity, "volume")
        self._bid_size = RingBuffer(capacity, "bid_size")
        self._ask_size = RingBuffer(capacity, "ask_size")
        self._cvd_delta = RingBuffer(capacity, "cvd_delta")
        self._spread_raw = RingBuffer(capacity, "spread_raw")
        self._bar_count = 0
        # Per-feature rolling normalizers
        self._norms = [_NormBuffer(100) for _ in range(NUM_FEATURES)]

        # Regime state (set externally via set_regime_proba)
        self._regime_p_trending = 0.0
        self._regime_p_ranging = 0.0
        self._regime_p_highvol = 0.0
        self._regime_bars_in = 0.0

        # Multi-timeframe indicators
        self._tactical_1m = _TacticalIndicators(capacity=50)
        self._agg_1m = _BarAggregator(60, self._tactical_1m)
        self._strategic_5m = _StrategicIndicators(capacity=60)
        self._agg_5m = _BarAggregator(300, self._strategic_5m)

    # ── public API ──────────────────────────────────────────────

    def set_regime_proba(
        self,
        p_trending: float,
        p_ranging: float,
        p_highvol: float,
        bars_in: int,
    ) -> None:
        """Set latest regime probabilities from RegimeV2Signal."""
        self._regime_p_trending = p_trending
        self._regime_p_ranging = p_ranging
        self._regime_p_highvol = p_highvol
        self._regime_bars_in = float(bars_in)

    def on_bar(self, bar: BarEvent) -> np.ndarray:
        """Append bar data and return feature vector shape (30,).

        Uses real L1 bid/ask sizes when available (avg_bid_size > 0),
        otherwise falls back to OHLCV approximations.
        """
        close = bar.close
        volume = float(bar.volume)
        hl_range = bar.high - bar.low
        hl_denom = hl_range + 1e-9

        # Use real L1 data when available, else approximate from OHLCV
        if bar.avg_bid_size > 0 or bar.avg_ask_size > 0:
            bid_size = float(bar.avg_bid_size)
            ask_size = float(bar.avg_ask_size)
            cvd_delta = float(bar.aggressive_buy_vol - bar.aggressive_sell_vol)
        else:
            ratio = (close - bar.low) / hl_denom
            bid_size = volume * (1.0 - ratio)
            ask_size = volume * ratio
            cvd_delta = volume * (2.0 * ratio - 1.0)

        self._close.append(close)
        self._high.append(bar.high)
        self._low.append(bar.low)
        self._volume.append(volume)
        self._bid_size.append(bid_size)
        self._ask_size.append(ask_size)
        self._cvd_delta.append(cvd_delta)
        self._spread_raw.append(hl_range / MES_TICK_SIZE)
        self._bar_count += 1

        # Feed multi-timeframe aggregators
        self._agg_1m.on_bar(bar.open, bar.high, bar.low, close, volume)
        self._agg_5m.on_bar(bar.open, bar.high, bar.low, close, volume)

        raw = self._compute_raw()
        # Z-score normalize each feature independently
        out = np.empty(NUM_FEATURES, dtype=np.float64)
        for i in range(NUM_FEATURES):
            out[i] = self._norms[i].update_and_normalize(raw[i])
        return out

    def reset(self) -> None:
        """Clear all buffers (called on session close)."""
        self._close.reset()
        self._high.reset()
        self._low.reset()
        self._volume.reset()
        self._bid_size.reset()
        self._ask_size.reset()
        self._cvd_delta.reset()
        self._spread_raw.reset()
        self._bar_count = 0
        for n in self._norms:
            n.reset()
        self._regime_p_trending = 0.0
        self._regime_p_ranging = 0.0
        self._regime_p_highvol = 0.0
        self._regime_bars_in = 0.0
        self._agg_1m.reset()
        self._agg_5m.reset()

    @property
    def is_warm(self) -> bool:
        return self._bar_count >= _MIN_BARS_WARMUP

    # ── internals ───────────────────────────────────────────────

    def _compute_raw(self) -> np.ndarray:
        """Compute raw (un-normalized) feature vector."""
        out = np.full(NUM_FEATURES, np.nan, dtype=np.float64)
        n = self._bar_count

        close = self._close.get_array()
        vol = self._volume.get_array()
        bid = self._bid_size.get_array()
        ask = self._ask_size.get_array()
        cvd = self._cvd_delta.get_array()
        spread = self._spread_raw.get_array()

        # ── Tier 1: Order Flow ──────────────────────────────────
        if n >= 11:
            # OFI: net bid/ask size change accumulated over window
            bid_diff = np.diff(bid[-11:])
            ask_diff = np.diff(ask[-11:])
            out[0] = float(np.sum(bid_diff - ask_diff))  # ofi_10

        if n >= 31:
            bid_diff = np.diff(bid[-31:])
            ask_diff = np.diff(ask[-31:])
            out[1] = float(np.sum(bid_diff - ask_diff))  # ofi_30

        if n >= 1:
            b, a = float(bid[-1]), float(ask[-1])
            total = b + a
            out[2] = (b - a) / total if total > 0 else 0.0  # obi_1

        if n >= 5:
            b5 = float(np.mean(bid[-5:]))
            a5 = float(np.mean(ask[-5:]))
            t5 = b5 + a5
            out[3] = (b5 - a5) / t5 if t5 > 0 else 0.0  # obi_5

        if n >= 1:
            # Microprice approximation
            c = float(close[-1])
            b_s = float(bid[-1])
            a_s = float(ask[-1])
            total = b_s + a_s
            if total > 0:
                half_spread = MES_TICK_SIZE  # approx half-spread
                bid_px = c - half_spread
                ask_px = c + half_spread
                mp = (b_s * ask_px + a_s * bid_px) / total
                out[4] = mp           # microprice
                out[5] = mp - c       # microprice_vs_mid
            else:
                out[4] = float(close[-1])
                out[5] = 0.0

        # ── Tier 2: CVD & Volume ────────────────────────────────
        if n >= 10:
            out[6] = float(np.sum(cvd[-10:]))  # cvd_delta_10

        if n >= 30:
            out[7] = float(np.sum(cvd[-30:]))  # cvd_delta_30

        if n >= 20:
            # CVD slope: linear regression on cumulative CVD
            cvd_window = cvd[-20:]
            cumsum = np.cumsum(cvd_window)
            x = np.arange(20, dtype=np.float64)
            out[8] = float(np.polyfit(x, cumsum, 1)[0])  # cvd_slope

        if n >= 10:
            buy = np.maximum(cvd[-10:], 0.0)
            sell = np.maximum(-cvd[-10:], 0.0)
            total_v = float(np.sum(vol[-10:]))
            if total_v > 0:
                out[9] = float(np.sum(buy) - np.sum(sell)) / total_v  # volume_imbalance_10
            else:
                out[9] = 0.0

        if n >= 20:
            v20 = vol[-20:]
            mean_v = float(np.mean(v20))
            std_v = float(np.std(v20, ddof=1))
            if std_v > 0:
                out[10] = (float(vol[-1]) - mean_v) / std_v  # volume_zscore_20
            else:
                out[10] = 0.0

        # ── Tier 3: Price Action & Volatility ───────────────────
        if n >= 2:
            out[11] = float(np.log(close[-1] / close[-2]))  # return_1

        if n >= 6:
            out[12] = float(np.log(close[-1] / close[-6]))  # return_5

        if n >= 11:
            out[13] = float(np.log(close[-1] / close[-11]))  # return_10

        if n >= 21:
            log_rets = np.diff(np.log(close[-21:]))
            out[14] = float(np.std(log_rets, ddof=1))  # realized_vol_20

        if n >= 11:
            log_rets = np.diff(np.log(close[-11:]))
            if len(log_rets) >= 2:
                mean_r = np.mean(log_rets)
                demeaned = log_rets - mean_r
                var_r = float(np.dot(demeaned, demeaned)) / len(demeaned)
                if var_r > 1e-18:
                    cov = float(np.dot(demeaned[:-1], demeaned[1:])) / (len(demeaned) - 1)
                    out[15] = cov / var_r  # return_autocorr_10
                else:
                    out[15] = 0.0

        if n >= 1:
            out[16] = float(spread[-1])  # spread (in ticks)

        if n >= 51:
            s50 = spread[-50:]
            mean_s = float(np.mean(s50))
            std_s = float(np.std(s50, ddof=1))
            if std_s > 0:
                out[17] = (float(spread[-1]) - mean_s) / std_s  # spread_zscore_50
            else:
                out[17] = 0.0

        # ── Tier 4: HMM Regime ──────────────────────────────────
        out[18] = self._regime_p_trending   # regime_p_trending
        out[19] = self._regime_p_ranging    # regime_p_ranging
        out[20] = self._regime_p_highvol    # regime_p_highvol
        out[21] = self._regime_bars_in      # regime_bars_in

        # ── Tier 5: Tactical (1m) ────────────────────────────────
        out[22] = self._tactical_1m.bb_pctb      # Bollinger %B
        out[23] = self._tactical_1m.rsi           # RSI-14
        out[24] = self._tactical_1m.atr_ratio     # ATR(5)/ATR(20)
        out[25] = self._tactical_1m.obv_slope     # OBV slope

        # ── Tier 6: Strategic (5m) ───────────────────────────────
        out[26] = self._strategic_5m.donchian_pos  # Donchian position
        out[27] = self._strategic_5m.stoch_k       # Stochastic K-14
        out[28] = self._strategic_5m.bb_width      # Bollinger bandwidth
        out[29] = self._strategic_5m.ema_cross     # EMA cross / ATR

        return out
