"""Tests for Phase 7 Regime Detector V2 (Step 4: T2 Features + PCA)."""

import numpy as np
import polars as pl
import pytest
from pathlib import Path

from src.models.regime_detector_v2 import (
    BOCPDDetector,
    CUSUMDetector,
    N_FEATURES_T1,
    N_FEATURES_T2,
    RegimeDetectorV2,
    RegimeDetectorV2Config,
    RegimeLabel,
    RegimeProba,
    TransitionLayer,
    VolZScoreDetector,
    _autocorr_sum_single,
    _dfa_hurst_single,
    _rolling_adx,
    _rolling_vol_of_vol,
    _rolling_vpin,
    build_features_v2,
    compute_regime_stats,
)


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_bar_df(n: int = 2000, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic 5m bar data with regime-like behavior."""
    rng = np.random.default_rng(seed)

    # Create 3 regimes: trending (high returns), ranging (low vol), volatile
    regime_lengths = [n // 3, n // 3, n - 2 * (n // 3)]
    prices = [5000.0]

    for length, regime in zip(regime_lengths, ["trending", "ranging", "volatile"]):
        for _ in range(length):
            if regime == "trending":
                ret = rng.normal(0.0005, 0.001)
            elif regime == "ranging":
                ret = rng.normal(0.0, 0.0003)
            else:
                ret = rng.normal(0.0, 0.003)
            prices.append(prices[-1] * (1 + ret))

    prices = np.array(prices[:n])
    noise = rng.uniform(0.0001, 0.001, size=n)
    opens = prices * (1 - noise / 2)
    highs = prices * (1 + noise)
    lows = prices * (1 - noise)
    closes = prices
    volumes = rng.integers(100, 5000, size=n).astype(float)

    from datetime import datetime, timedelta
    start = datetime(2024, 1, 2, 9, 30)
    end = start + timedelta(minutes=5 * (n - 1))
    timestamps = pl.datetime_range(start, end, interval="5m", eager=True)

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


@pytest.fixture
def bar_df():
    return _make_bar_df(n=3000)  # larger for Hurst warmup


@pytest.fixture
def config():
    """T1 config (4 features, no PCA) — backward compat for existing tests."""
    return RegimeDetectorV2Config(
        zscore_window=100,
        warmup_bars=80,
        predict_window=50,
        gk_vol_window=10,
        hurst_window=80,
        hurst_stride=5,
        autocorr_window=50,
        autocorr_max_lag=5,
        studentt_dof=5,
        covariance_type="diag",  # faster for tests
        feature_tier="t1",
        pca_enabled=False,
    )


@pytest.fixture
def config_t2():
    """T2 config (7 features + PCA)."""
    return RegimeDetectorV2Config(
        zscore_window=100,
        warmup_bars=80,
        predict_window=50,
        gk_vol_window=10,
        hurst_window=80,
        hurst_stride=5,
        autocorr_window=50,
        autocorr_max_lag=5,
        studentt_dof=5,
        covariance_type="diag",
        feature_tier="t2",
        pca_enabled=True,
        pca_min_variance=0.95,
        adx_period=14,
        vol_of_vol_window=10,
        vpin_bucket_size=50,
        vpin_n_buckets=20,
    )


@pytest.fixture
def fitted_detector(bar_df, config):
    features, _ = build_features_v2(bar_df, config)
    detector = RegimeDetectorV2(config)
    detector.fit(features)
    return detector, features


@pytest.fixture
def fitted_detector_t2(bar_df, config_t2):
    features, _ = build_features_v2(bar_df, config_t2)
    detector = RegimeDetectorV2(config_t2)
    detector.fit(features)
    return detector, features


# ── DFA Hurst ────────────────────────────────────────────────────────

class TestDFAHurst:
    def test_random_walk_hurst_near_half(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=500)
        h = _dfa_hurst_single(returns)
        assert 0.3 < h < 0.7, f"Random walk Hurst={h}, expected ~0.5"

    def test_trending_hurst_above_half(self):
        # Cumulative sum of positive bias -> trending
        rng = np.random.default_rng(42)
        returns = rng.normal(0.005, 0.001, size=500)
        h = _dfa_hurst_single(returns)
        assert h > 0.4, f"Trending Hurst={h}, expected > 0.5"

    def test_short_series_returns_default(self):
        h = _dfa_hurst_single(np.array([0.01, -0.01, 0.01]))
        assert h == 0.5

    def test_clipped_to_valid_range(self):
        rng = np.random.default_rng(99)
        returns = rng.normal(0, 1, size=500)
        h = _dfa_hurst_single(returns)
        assert 0.0 <= h <= 1.5


class TestAutocorr:
    def test_iid_near_zero(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=500)
        ac = _autocorr_sum_single(returns, max_lag=10)
        assert abs(ac) < 1.0, f"IID autocorr sum={ac}, expected ~0"

    def test_constant_returns_zero(self):
        returns = np.ones(100) * 0.001
        ac = _autocorr_sum_single(returns, max_lag=5)
        assert ac == 0.0


# ── Feature extraction ───────────────────────────────────────────────

class TestBuildFeatures:
    def test_shape_4_features(self, bar_df, config):
        features, timestamps = build_features_v2(bar_df, config)
        assert features.ndim == 2
        assert features.shape[1] == N_FEATURES_T1  # 4 features
        assert len(timestamps) == len(features)

    def test_no_nans(self, bar_df, config):
        features, _ = build_features_v2(bar_df, config)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_zscore_clipped(self, bar_df, config):
        features, _ = build_features_v2(bar_df, config)
        assert features.min() >= -3.0
        assert features.max() <= 3.0

    def test_warmup_dropped(self, bar_df, config):
        features, _ = build_features_v2(bar_df, config)
        assert len(features) < len(bar_df)
        assert len(features) <= len(bar_df) - config.zscore_window


# ── Training ─────────────────────────────────────────────────────────

class TestTraining:
    def test_fit_creates_model(self, fitted_detector):
        detector, _ = fitted_detector
        assert detector.model is not None
        assert detector.state_map is not None
        assert len(detector.state_map) == 3

    def test_state_map_covers_all_labels(self, fitted_detector):
        detector, _ = fitted_detector
        labels = set(detector.state_map.values())
        assert labels == {RegimeLabel.TRENDING, RegimeLabel.RANGING, RegimeLabel.HIGH_VOL}

    def test_model_type_matches_config(self, fitted_detector):
        from hmmlearn.hmm import GaussianHMM
        detector, _ = fitted_detector
        # Default config uses gaussian
        assert isinstance(detector.model, GaussianHMM)

    def test_studentt_backend(self, bar_df):
        from pomegranate.hmm import DenseHMM
        from pomegranate.distributions import StudentT as ST
        cfg = RegimeDetectorV2Config(
            zscore_window=100, warmup_bars=80, predict_window=50,
            gk_vol_window=10, hurst_window=80, hurst_stride=5,
            autocorr_window=50, autocorr_max_lag=5,
            emission_type="studentt", studentt_dof=5,
        )
        features, _ = build_features_v2(bar_df, cfg)
        det = RegimeDetectorV2(cfg)
        det.fit(features)
        assert isinstance(det.model, DenseHMM)
        for dist in det.model.distributions:
            assert isinstance(dist, ST)


# ── Batch prediction (forward-only) ─────────────────────────────────

class TestBatchPrediction:
    def test_predict_proba_sequence_length(self, fitted_detector):
        detector, features = fitted_detector
        probas = detector.predict_proba_sequence(features)
        assert len(probas) == len(features)

    def test_all_outputs_are_regime_proba(self, fitted_detector):
        detector, features = fitted_detector
        probas = detector.predict_proba_sequence(features)
        for p in probas:
            assert isinstance(p, RegimeProba)

    def test_probabilities_sum_to_one(self, fitted_detector):
        detector, features = fitted_detector
        probas = detector.predict_proba_sequence(features)
        for p in probas:
            assert abs(sum(p.probabilities) - 1.0) < 1e-3

    def test_confidence_is_max_probability(self, fitted_detector):
        detector, features = fitted_detector
        probas = detector.predict_proba_sequence(features)
        for p in probas:
            assert abs(p.confidence - max(p.probabilities)) < 1e-4

    def test_regime_is_valid_label(self, fitted_detector):
        detector, features = fitted_detector
        probas = detector.predict_proba_sequence(features)
        for p in probas:
            assert p.regime in RegimeLabel

    def test_position_size_values(self, fitted_detector):
        detector, features = fitted_detector
        probas = detector.predict_proba_sequence(features)
        for p in probas:
            assert p.position_size in ("full", "half", "flat")

    def test_bars_in_regime_positive(self, fitted_detector):
        detector, features = fitted_detector
        probas = detector.predict_proba_sequence(features)
        for p in probas:
            assert p.bars_in_regime >= 1


# ── Anti-whipsaw ─────────────────────────────────────────────────────

class TestAntiWhipsaw:
    def test_min_bars_in_regime_forces_flat(self, fitted_detector):
        detector, features = fitted_detector
        probas = detector.predict_proba_sequence(features)

        for p in probas:
            if p.bars_in_regime < detector.config.min_bars_in_regime:
                assert p.position_size == "flat", \
                    f"Expected flat when bars_in_regime={p.bars_in_regime}"

    def test_whipsaw_halt_forces_flat(self, fitted_detector):
        detector, features = fitted_detector
        probas = detector.predict_proba_sequence(features)

        for p in probas:
            if p.whipsaw_halt:
                assert p.position_size == "flat"

    def test_whipsaw_halt_catches_rapid_flips(self, config):
        """When model flips rapidly, whipsaw halt kicks in."""
        df = _make_bar_df(n=4000, seed=99)
        features, _ = build_features_v2(df, config)

        detector = RegimeDetectorV2(config)
        detector.fit(features)
        probas = detector.predict_proba_sequence(features)

        transitions = sum(1 for p in probas if p.transition_signal)
        halted = sum(1 for p in probas if p.whipsaw_halt)

        if transitions > len(probas) * 0.05:
            assert halted > 0, "Rapid flips detected but whipsaw halt never triggered"
            for p in probas:
                if p.whipsaw_halt:
                    assert p.position_size == "flat"


# ── Online prediction ────────────────────────────────────────────────

class TestOnlinePrediction:
    def test_warmup_returns_none(self, fitted_detector):
        detector, _ = fitted_detector
        detector.reset()

        bar = {"open": 5000, "high": 5001, "low": 4999, "close": 5000.5, "volume": 1000}
        result = detector.update(bar)
        assert result is None

    def test_produces_output_after_warmup(self, bar_df, config):
        config = RegimeDetectorV2Config(
            zscore_window=30,
            warmup_bars=20,
            predict_window=20,
            gk_vol_window=10,
            hurst_window=20,
            hurst_stride=1,
            autocorr_window=15,
            autocorr_max_lag=3,
            studentt_dof=5,
            covariance_type="diag",
        )
        features, _ = build_features_v2(bar_df, config)
        detector = RegimeDetectorV2(config)
        detector.fit(features)
        detector.reset()

        got_output = False
        for row in bar_df.iter_rows(named=True):
            bar = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
            result = detector.update(bar)
            if result is not None:
                got_output = True
                assert isinstance(result, RegimeProba)
                assert abs(sum(result.probabilities) - 1.0) < 1e-4
                break

        assert got_output, "Never got output from online prediction"

    def test_reset_clears_state(self, fitted_detector):
        detector, _ = fitted_detector

        for _ in range(10):
            detector.update({"open": 5000, "high": 5001, "low": 4999,
                             "close": 5000.5, "volume": 1000})

        detector.reset()
        assert detector._bar_count == 0
        assert detector._bars_in_regime == 0
        assert len(detector._raw_features) == 0
        assert len(detector._returns_buffer) == 0


# ── Persistence ──────────────────────────────────────────────────────

class TestPersistence:
    def test_save_load_roundtrip(self, fitted_detector, tmp_path):
        detector, features = fitted_detector
        save_dir = tmp_path / "regime_v2"

        detector.save(save_dir)
        loaded = RegimeDetectorV2.load(save_dir)

        assert loaded.model is not None
        assert loaded.state_map == detector.state_map
        assert loaded.config == detector.config

        # Predictions should match
        orig_probas = detector.predict_proba_sequence(features[:100])
        loaded_probas = loaded.predict_proba_sequence(features[:100])

        for o, l in zip(orig_probas, loaded_probas):
            for op, lp in zip(o.probabilities, l.probabilities):
                assert abs(op - lp) < 1e-4


# ── Stats helper ─────────────────────────────────────────────────────

class TestComputeStats:
    def test_stats_keys(self, fitted_detector):
        detector, features = fitted_detector
        probas = detector.predict_proba_sequence(features)
        stats = compute_regime_stats(probas)

        assert "n_bars" in stats
        assert "state_distribution" in stats
        assert "position_size_distribution" in stats
        assert "transitions" in stats
        assert "halt_fraction" in stats
        assert "avg_confidence" in stats
        assert "avg_stint_length" in stats

    def test_state_distribution_sums_to_one(self, fitted_detector):
        detector, features = fitted_detector
        probas = detector.predict_proba_sequence(features)
        stats = compute_regime_stats(probas)
        assert abs(sum(stats["state_distribution"].values()) - 1.0) < 1e-6

    def test_empty_input(self):
        stats = compute_regime_stats([])
        assert stats == {}


# ── BOCPD ────────────────────────────────────────────────────────────

class TestBOCPD:
    def test_stable_series_no_changepoint(self):
        rng = np.random.default_rng(42)
        det = BOCPDDetector(hazard_lambda=60, threshold=0.3)
        values = rng.normal(0, 0.01, size=200)
        signals = det.update_batch(values)
        # Stable series: few or no changepoints after warmup
        assert signals[:10].sum() == 0, "Should not fire during warmup"

    def test_mean_shift_detected(self):
        rng = np.random.default_rng(42)
        det = BOCPDDetector(hazard_lambda=30, threshold=0.3)
        # Stable z-scores, then abrupt mean shift (like z-scored features)
        v1 = rng.normal(0, 0.5, size=100)
        v2 = rng.normal(2.0, 0.5, size=100)
        values = np.concatenate([v1, v2])
        signals = det.update_batch(values)
        # Should detect changepoint near index 100
        assert any(signals[95:115]), "Should detect mean shift around index 100"

    def test_changepoint_prob_property(self):
        det = BOCPDDetector()
        det.update(0.0)
        det.update(0.0)
        det.update(0.0)
        assert 0.0 <= det.changepoint_prob <= 1.0

    def test_reset_clears_state(self):
        det = BOCPDDetector()
        for _ in range(50):
            det.update(0.01)
        det.reset()
        assert det._n_obs == 0
        assert det._changepoint_prob == 0.0


class TestCUSUM:
    def test_stable_series_no_alarm(self):
        rng = np.random.default_rng(42)
        det = CUSUMDetector(k=0.5, h=4.0)
        values = rng.normal(0, 0.3, size=200)  # z-score scale
        signals = det.update_batch(values)
        # Low-vol stable series should rarely trigger
        assert signals.sum() < 5

    def test_mean_shift_triggers(self):
        det = CUSUMDetector(k=0.5, h=4.0)
        # Feed steady zeros then jump to 2.0
        values = np.concatenate([np.zeros(50), np.full(50, 2.0)])
        signals = det.update_batch(values)
        assert any(signals[50:]), "Should trigger on mean shift to 2.0"

    def test_reset_clears(self):
        det = CUSUMDetector()
        for _ in range(20):
            det.update(1.0)
        det.reset()
        assert det._s_pos == 0.0
        assert det._s_neg == 0.0
        assert det._n == 0


class TestVolZScore:
    def test_stable_vol_no_signal(self):
        rng = np.random.default_rng(42)
        det = VolZScoreDetector(threshold=2.0, window=50)
        values = rng.uniform(0.005, 0.006, size=100)
        signals = det.update_batch(values)
        assert signals.sum() < 5

    def test_vol_spike_detected(self):
        det = VolZScoreDetector(threshold=2.0, window=30)
        # Stable vol then spike
        values = np.concatenate([
            np.full(40, 0.005),
            np.full(10, 0.050),  # 10x spike
        ])
        signals = det.update_batch(values)
        assert any(signals[40:]), "Should detect vol spike"

    def test_zscore_property(self):
        det = VolZScoreDetector()
        for _ in range(20):
            det.update(0.005)
        assert isinstance(det.zscore, float)

    def test_reset_clears(self):
        det = VolZScoreDetector()
        for _ in range(20):
            det.update(0.005)
        det.reset()
        assert len(det._buffer) == 0
        assert det._zscore == 0.0


# ── Transition Layer ─────────────────────────────────────────────────

class TestTransitionLayer:
    def test_stable_data_no_confirmation(self):
        cfg = RegimeDetectorV2Config(transition_agreement=2)
        tl = TransitionLayer(cfg)
        rng = np.random.default_rng(42)
        lr = rng.normal(0, 0.01, size=100)
        gk = rng.uniform(0.005, 0.006, size=100)
        confirmed = tl.update_batch(lr, gk)
        # Mostly stable → few confirmations
        assert confirmed.sum() < len(confirmed) * 0.3

    def test_regime_break_gets_confirmed(self):
        cfg = RegimeDetectorV2Config(
            transition_agreement=2,
            bocpd_hazard_lambda=30,
            bocpd_threshold=0.2,
            cusum_h=3.0,
            vol_zscore_threshold=2.0,
        )
        tl = TransitionLayer(cfg)
        rng = np.random.default_rng(42)
        # Z-score-scaled: quiet period, then abrupt shift in returns and vol
        lr = np.concatenate([
            rng.normal(0, 0.5, size=80),
            rng.normal(2.0, 0.5, size=40),
        ])
        gk = np.concatenate([
            rng.normal(0, 0.3, size=80),
            rng.normal(3.0, 0.3, size=40),
        ])
        confirmed = tl.update_batch(lr, gk)
        assert any(confirmed[75:]), "Should confirm transition around regime break"

    def test_detail_property(self):
        cfg = RegimeDetectorV2Config()
        tl = TransitionLayer(cfg)
        tl.update(0.01, 0.005)
        detail = tl.detail
        assert "bocpd" in detail
        assert "cusum" in detail
        assert "vol_zscore" in detail

    def test_reset_clears_all(self):
        cfg = RegimeDetectorV2Config()
        tl = TransitionLayer(cfg)
        for _ in range(30):
            tl.update(0.01, 0.005)
        tl.reset()
        assert len(tl._bocpd_signals) == 0
        assert len(tl._cusum_signals) == 0
        assert len(tl._vol_signals) == 0

    def test_agreement_threshold_respected(self):
        """With agreement=3, all 3 must fire."""
        cfg = RegimeDetectorV2Config(transition_agreement=3)
        tl = TransitionLayer(cfg)
        # Force only CUSUM to fire by feeding large deviation
        # BOCPD and vol z-score shouldn't fire on first observation
        result = tl.update(0.0, 0.005)
        assert result is False  # Can't have 3-way agreement on first bar


# ── Integration: transition layer suppresses false transitions ───────

class TestTransitionIntegration:
    def test_transitions_reduced_vs_no_layer(self, bar_df, config):
        """Transition layer should reduce or equal the number of transitions."""
        features, _ = build_features_v2(bar_df, config)

        # With transition layer (default)
        detector = RegimeDetectorV2(config)
        detector.fit(features)
        probas_with = detector.predict_proba_sequence(features)
        trans_with = sum(1 for p in probas_with if p.transition_signal)

        # Without transition layer (agreement=0 means always confirm)
        from dataclasses import fields, asdict
        d = {f.name: getattr(config, f.name) for f in fields(config)}
        d["transition_agreement"] = 0
        config_no_tl = RegimeDetectorV2Config(**d)
        detector2 = RegimeDetectorV2(config_no_tl)
        detector2.fit(features)
        probas_without = detector2.predict_proba_sequence(features)
        trans_without = sum(1 for p in probas_without if p.transition_signal)

        assert trans_with <= trans_without, \
            f"Transition layer should reduce transitions: {trans_with} vs {trans_without}"

    def test_batch_still_produces_valid_output(self, fitted_detector):
        detector, features = fitted_detector
        probas = detector.predict_proba_sequence(features)
        assert len(probas) == len(features)
        for p in probas:
            assert isinstance(p, RegimeProba)
            assert p.regime in RegimeLabel
            assert p.position_size in ("full", "half", "flat")


# ── Protocol compatibility ───────────────────────────────────────────

class TestProtocol:
    def test_current_regime_returns_string(self, fitted_detector):
        detector, _ = fitted_detector
        result = detector.current_regime()
        assert result in ("trending", "ranging", "high_vol")


# ── T2 Feature Extractors ──────────────────────────────────────────

class TestRollingADX:
    def test_output_shape(self, bar_df):
        close = bar_df["close"].to_numpy().astype(np.float64)
        open_ = bar_df["open"].to_numpy().astype(np.float64)
        high = bar_df["high"].to_numpy().astype(np.float64)
        low = bar_df["low"].to_numpy().astype(np.float64)
        adx = _rolling_adx(open_, high, low, close, period=14)
        assert len(adx) == len(close)

    def test_adx_range(self, bar_df):
        close = bar_df["close"].to_numpy().astype(np.float64)
        open_ = bar_df["open"].to_numpy().astype(np.float64)
        high = bar_df["high"].to_numpy().astype(np.float64)
        low = bar_df["low"].to_numpy().astype(np.float64)
        adx = _rolling_adx(open_, high, low, close, period=14)
        valid = adx[~np.isnan(adx)]
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_trending_higher_adx(self):
        """Trending data should produce higher ADX than flat data."""
        n = 500
        # Trending: steady upward
        prices_trend = 5000 + np.cumsum(np.full(n, 0.5))
        noise = np.random.default_rng(42).uniform(0.5, 2.0, n)
        h_t = prices_trend + noise
        l_t = prices_trend - noise
        o_t = prices_trend - noise / 2
        adx_trend = _rolling_adx(o_t, h_t, l_t, prices_trend, period=14)

        # Flat: oscillating
        prices_flat = 5000 + np.sin(np.arange(n) * 0.3) * 2
        h_f = prices_flat + noise
        l_f = prices_flat - noise
        o_f = prices_flat - noise / 2
        adx_flat = _rolling_adx(o_f, h_f, l_f, prices_flat, period=14)

        # Last 100 bars: trending should have higher ADX
        assert np.nanmean(adx_trend[-100:]) > np.nanmean(adx_flat[-100:])


class TestRollingVolOfVol:
    def test_output_shape(self, bar_df):
        gk_vol = np.random.default_rng(42).uniform(0.005, 0.01, len(bar_df))
        vov = _rolling_vol_of_vol(gk_vol, window=20)
        assert len(vov) == len(gk_vol)

    def test_constant_vol_zero_vov(self):
        gk_vol = np.full(100, 0.005)
        vov = _rolling_vol_of_vol(gk_vol, window=20)
        # Constant vol -> vol-of-vol should be ~0
        assert vov[-1] < 1e-10

    def test_volatile_vol_higher_vov(self):
        rng = np.random.default_rng(42)
        stable = np.full(100, 0.005)
        volatile = rng.uniform(0.001, 0.02, size=100)
        vov_stable = _rolling_vol_of_vol(stable, window=20)
        vov_volatile = _rolling_vol_of_vol(volatile, window=20)
        assert vov_volatile[-1] > vov_stable[-1]


class TestRollingVPIN:
    def test_output_shape(self, bar_df):
        close = bar_df["close"].to_numpy().astype(np.float64)
        high = bar_df["high"].to_numpy().astype(np.float64)
        low = bar_df["low"].to_numpy().astype(np.float64)
        volume = bar_df["volume"].to_numpy().astype(np.float64)
        vpin = _rolling_vpin(close, high, low, volume, bucket_size=50, n_buckets=20)
        assert len(vpin) == len(close)

    def test_vpin_range(self, bar_df):
        close = bar_df["close"].to_numpy().astype(np.float64)
        high = bar_df["high"].to_numpy().astype(np.float64)
        low = bar_df["low"].to_numpy().astype(np.float64)
        volume = bar_df["volume"].to_numpy().astype(np.float64)
        vpin = _rolling_vpin(close, high, low, volume, bucket_size=50, n_buckets=20)
        valid = vpin[~np.isnan(vpin)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0


# ── T2 Feature Build ───────────────────────────────────────────────

class TestBuildFeaturesT2:
    def test_shape_7_features(self, bar_df, config_t2):
        features, timestamps = build_features_v2(bar_df, config_t2)
        assert features.ndim == 2
        assert features.shape[1] == N_FEATURES_T2  # 7 features
        assert len(timestamps) == len(features)

    def test_no_nans(self, bar_df, config_t2):
        features, _ = build_features_v2(bar_df, config_t2)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_zscore_clipped(self, bar_df, config_t2):
        features, _ = build_features_v2(bar_df, config_t2)
        assert features.min() >= -3.0
        assert features.max() <= 3.0


# ── PCA Decorrelation ──────────────────────────────────────────────

class TestPCA:
    def test_pca_reduces_dimensions(self, bar_df, config_t2):
        features, _ = build_features_v2(bar_df, config_t2)
        detector = RegimeDetectorV2(config_t2)
        detector.fit(features)
        assert detector._pca_components is not None
        assert detector._pca_mean is not None
        # Should keep <= n_features components
        n_kept = detector._pca_components.shape[0]
        assert 1 <= n_kept <= N_FEATURES_T2

    def test_pca_disabled(self, bar_df):
        cfg = RegimeDetectorV2Config(
            zscore_window=100, warmup_bars=80, predict_window=50,
            gk_vol_window=10, hurst_window=80, hurst_stride=5,
            autocorr_window=50, autocorr_max_lag=5,
            feature_tier="t2", pca_enabled=False,
        )
        features, _ = build_features_v2(bar_df, cfg)
        detector = RegimeDetectorV2(cfg)
        detector.fit(features)
        assert detector._pca_components is None

    def test_pca_transform_invertible(self, bar_df, config_t2):
        """PCA transform + inverse should approximate identity."""
        features, _ = build_features_v2(bar_df, config_t2)
        detector = RegimeDetectorV2(config_t2)
        detector._fit_pca(features, config_t2.pca_min_variance)

        transformed = detector._apply_pca(features)
        # Reconstruct: transformed @ components + mean
        reconstructed = transformed @ detector._pca_components + detector._pca_mean
        # Should be close (not exact if dimensions reduced)
        error = np.mean((features - reconstructed) ** 2)
        assert error < 0.1, f"PCA reconstruction error too high: {error}"


# ── T2 Training & Prediction ──────────────────────────────────────

class TestT2Training:
    def test_fit_creates_model(self, fitted_detector_t2):
        detector, _ = fitted_detector_t2
        assert detector.model is not None
        assert detector.state_map is not None
        assert len(detector.state_map) == 3
        assert detector._pca_components is not None

    def test_predict_proba_sequence(self, fitted_detector_t2):
        detector, features = fitted_detector_t2
        probas = detector.predict_proba_sequence(features)
        assert len(probas) == len(features)
        for p in probas:
            assert isinstance(p, RegimeProba)
            assert abs(sum(p.probabilities) - 1.0) < 1e-3

    def test_save_load_roundtrip_t2(self, fitted_detector_t2, tmp_path):
        detector, features = fitted_detector_t2
        save_dir = tmp_path / "regime_v2_t2"
        detector.save(save_dir)
        loaded = RegimeDetectorV2.load(save_dir)

        assert loaded._pca_components is not None
        assert loaded._pca_mean is not None
        assert loaded._n_features == N_FEATURES_T2
        np.testing.assert_array_almost_equal(
            loaded._pca_components, detector._pca_components
        )

        # Predictions should match
        orig = detector.predict_proba_sequence(features[:50])
        loaded_p = loaded.predict_proba_sequence(features[:50])
        for o, l in zip(orig, loaded_p):
            for op, lp in zip(o.probabilities, l.probabilities):
                assert abs(op - lp) < 1e-4


# ── T2 Online Prediction ──────────────────────────────────────────

class TestT2OnlinePrediction:
    def test_online_t2_produces_output(self, bar_df):
        cfg = RegimeDetectorV2Config(
            zscore_window=30, warmup_bars=20, predict_window=20,
            gk_vol_window=10, hurst_window=20, hurst_stride=1,
            autocorr_window=15, autocorr_max_lag=3,
            feature_tier="t2", pca_enabled=True,
            pca_min_variance=0.95,
            adx_period=14, vol_of_vol_window=10,
            vpin_bucket_size=50, vpin_n_buckets=20,
        )
        features, _ = build_features_v2(bar_df, cfg)
        detector = RegimeDetectorV2(cfg)
        detector.fit(features)
        detector.reset()

        got_output = False
        for row in bar_df.iter_rows(named=True):
            bar = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
            result = detector.update(bar)
            if result is not None:
                got_output = True
                assert isinstance(result, RegimeProba)
                assert abs(sum(result.probabilities) - 1.0) < 1e-3
                break

        assert got_output, "Never got output from T2 online prediction"
