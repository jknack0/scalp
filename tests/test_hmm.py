"""Tests for the HMM intraday regime classifier (src/models/hmm_regime.py).

Uses synthetic clustered data with 2 distinct centers (3 features) and
sequential blocks to provide temporal structure for the HMM.
"""

import numpy as np
import polars as pl
import pytest

from src.models.hmm_regime import (
    HMMRegimeClassifier,
    HMMRegimeConfig,
    HMMValidationReport,
    NotReadyError,
    RegimeState,
    build_feature_matrix,
    compute_persistence_accuracy,
    compute_transition_matrix,
    validate_model,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_clustered_data(
    n_per_cluster: int = 200,
    n_features: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic data with 2 well-separated sequential clusters.

    Features: realized_vol, vpin_approx, return_autocorr
    """
    rng = np.random.default_rng(seed)
    centers = np.array([
        [-1.0, -0.5, -0.8],   # RANGE_BOUND: low vol, low vpin, negative autocorr
        [1.5, 1.0, 0.5],      # VOLATILE: high vol, high vpin, positive autocorr
    ])

    blocks = []
    for center in centers:
        block = rng.normal(loc=center, scale=0.3, size=(n_per_cluster, n_features))
        blocks.append(block)

    return np.vstack(blocks)


@pytest.fixture
def clustered_data() -> np.ndarray:
    """400-row synthetic clustered dataset (3 features)."""
    return _make_clustered_data()


@pytest.fixture
def fitted_classifier(clustered_data: np.ndarray) -> HMMRegimeClassifier:
    """Pre-fitted HMM classifier on clustered data."""
    config = HMMRegimeConfig(n_states=2, n_iter=50)
    clf = HMMRegimeClassifier(config)
    clf.fit(clustered_data)
    return clf


# ── Tests ────────────────────────────────────────────────────────────


class TestRegimeState:
    def test_enum_values(self):
        """RegimeState has 3 members with values 0-2."""
        assert len(RegimeState) == 3
        assert RegimeState.RANGE_BOUND == 0
        assert RegimeState.VOLATILE == 1
        assert RegimeState.TRENDING == 2


class TestHMMClassifier:
    def test_fit_without_error(self, clustered_data: np.ndarray):
        """Model trains on synthetic data without raising."""
        config = HMMRegimeConfig(n_states=2, n_iter=50)
        clf = HMMRegimeClassifier(config)
        clf.fit(clustered_data)
        assert clf.model is not None
        assert clf.state_map is not None

    def test_kmeans_initialization(self, clustered_data: np.ndarray):
        """KMeans seeding produces distinct initial means (not collapsed)."""
        config = HMMRegimeConfig(n_states=2, n_iter=50)
        clf = HMMRegimeClassifier(config)
        clf.fit(clustered_data)
        # After EM, means may shift but should not all collapse to same point
        means = clf.model.means_
        dist = np.linalg.norm(means[0] - means[1])
        assert dist > 0.01, "State means collapsed to same point"

    def test_label_states_assigns_all_model_states(
        self, fitted_classifier: HMMRegimeClassifier
    ):
        """All HMM states are mapped to a RegimeState."""
        n_states = fitted_classifier.config.n_states
        assert len(fitted_classifier.state_map) == n_states
        assert all(isinstance(v, RegimeState) for v in fitted_classifier.state_map.values())

    def test_predict_sequence_returns_correct_length(
        self,
        fitted_classifier: HMMRegimeClassifier,
        clustered_data: np.ndarray,
    ):
        """predict_sequence() returns one RegimeState per input row."""
        states = fitted_classifier.predict_sequence(clustered_data)
        assert len(states) == len(clustered_data)
        assert all(isinstance(s, RegimeState) for s in states)

    def test_save_load_roundtrip(
        self,
        fitted_classifier: HMMRegimeClassifier,
        clustered_data: np.ndarray,
        tmp_path,
    ):
        """Predictions are identical after save/load cycle."""
        save_dir = tmp_path / "hmm_test"
        fitted_classifier.save(save_dir)

        loaded = HMMRegimeClassifier.load(save_dir)
        original_states = fitted_classifier.predict_sequence(clustered_data)
        loaded_states = loaded.predict_sequence(clustered_data)

        assert original_states == loaded_states

    def test_fit_from_bar_dicts(self):
        """fit() accepts list of bar dicts."""
        rng = np.random.default_rng(42)
        n = 800
        base = 5000.0
        closes = base + np.cumsum(rng.normal(0, 0.5, n))
        bars = []
        for i in range(n):
            bars.append({
                "open": closes[i] - rng.uniform(0, 0.5),
                "high": closes[i] + rng.uniform(0, 1.0),
                "low": closes[i] - rng.uniform(0, 1.0),
                "close": closes[i],
                "volume": int(rng.integers(50, 500)),
            })
        config = HMMRegimeConfig(n_states=2, n_iter=20, zscore_window=50, warmup_bars=50)
        clf = HMMRegimeClassifier(config)
        clf.fit(bars)
        assert clf.model is not None


class TestOnlinePredict:
    def test_not_ready_before_warmup(self):
        """predict() raises NotReadyError before warmup_bars are ingested."""
        config = HMMRegimeConfig(n_states=2, n_iter=20, warmup_bars=10)
        clf = HMMRegimeClassifier(config)

        # Fit on synthetic data first
        data = _make_clustered_data()
        clf.fit(data)

        bar = {"open": 5000.0, "high": 5001.0, "low": 4999.0, "close": 5000.5, "volume": 100}
        with pytest.raises(NotReadyError):
            clf.predict(bar)

    def test_predict_after_warmup(self):
        """predict() returns valid regime after warmup."""
        config = HMMRegimeConfig(
            n_states=3, n_iter=20, warmup_bars=60,
            rvol_window=10, vpin_window=10,
            autocorr_window=10, zscore_window=50, predict_window=20,
        )
        clf = HMMRegimeClassifier(config)
        clf.fit(_make_clustered_data())

        rng = np.random.default_rng(99)
        base = 5000.0
        for i in range(70):
            base += rng.normal(0, 0.5)
            bar = {
                "open": base - 0.25,
                "high": base + rng.uniform(0, 1),
                "low": base - rng.uniform(0, 1),
                "close": base,
                "volume": int(rng.integers(50, 500)),
            }
            try:
                state = clf.predict(bar)
                assert state in (0, 1, 2)
            except NotReadyError:
                continue

        # Should have a valid state by now
        assert clf._current_regime is not None

    def test_regime_proba_returns_valid_distribution(self):
        """regime_proba() returns 2 probabilities summing to ~1."""
        config = HMMRegimeConfig(
            n_states=3, n_iter=20, warmup_bars=60,
            rvol_window=10, vpin_window=10,
            autocorr_window=10, zscore_window=50, predict_window=20,
        )
        clf = HMMRegimeClassifier(config)
        clf.fit(_make_clustered_data())

        rng = np.random.default_rng(99)
        base = 5000.0
        proba = None
        for i in range(70):
            base += rng.normal(0, 0.5)
            bar = {
                "open": base - 0.25,
                "high": base + rng.uniform(0, 1),
                "low": base - rng.uniform(0, 1),
                "close": base,
                "volume": int(rng.integers(50, 500)),
            }
            try:
                proba = clf.regime_proba(bar)
            except NotReadyError:
                continue

        assert proba is not None
        assert proba.shape == (3,)
        assert proba.sum() == pytest.approx(1.0, abs=1e-6)
        assert all(p >= 0 for p in proba)

    def test_reset_clears_state(self):
        """reset() clears all online state."""
        config = HMMRegimeConfig(n_states=2, n_iter=20, warmup_bars=5)
        clf = HMMRegimeClassifier(config)
        clf.fit(_make_clustered_data())

        clf._bar_count = 100
        clf._current_regime = RegimeState.VOLATILE
        clf.reset()

        assert clf._bar_count == 0
        assert clf._current_regime is None

    def test_not_fitted_raises(self):
        """predict() raises NotReadyError on unfitted classifier."""
        clf = HMMRegimeClassifier()
        bar = {"open": 5000.0, "high": 5001.0, "low": 4999.0, "close": 5000.5, "volume": 100}
        with pytest.raises(NotReadyError, match="not fitted"):
            clf.predict(bar)


class TestEvaluation:
    def test_persistence_accuracy_on_clustered_data(
        self,
        fitted_classifier: HMMRegimeClassifier,
        clustered_data: np.ndarray,
    ):
        """Clustered data with 200-bar blocks should have reasonable persistence."""
        states = fitted_classifier.predict_sequence(clustered_data)
        pers = compute_persistence_accuracy(states, horizon=5)
        assert pers > 0.30, f"Persistence {pers:.1%} below 30% threshold"

    def test_transition_matrix_shape_and_rows(
        self,
        fitted_classifier: HMMRegimeClassifier,
        clustered_data: np.ndarray,
    ):
        """Transition matrix is (3,3) with non-empty rows summing to ~1."""
        states = fitted_classifier.predict_sequence(clustered_data)
        tm = compute_transition_matrix(states)
        assert tm.shape == (3, 3)
        for row_sum in tm.sum(axis=1):
            # Rows with no transitions sum to 0 (unused states); active rows sum to 1
            assert row_sum == pytest.approx(1.0, abs=1e-6) or row_sum == pytest.approx(0.0, abs=1e-6)

    def test_validate_model_structure(
        self,
        fitted_classifier: HMMRegimeClassifier,
        clustered_data: np.ndarray,
    ):
        """validate_model returns an HMMValidationReport with expected fields."""
        report = validate_model(fitted_classifier, clustered_data)

        assert isinstance(report, HMMValidationReport)
        assert report.n_samples == len(clustered_data)
        assert 0.0 <= report.persistence_5bar <= 1.0
        assert len(report.state_distribution) == 3
        assert report.transition_matrix.shape == (3, 3)
        assert isinstance(report.log_likelihood, float)


class TestFeatureMatrix:
    def test_feature_matrix_shape(self):
        """build_feature_matrix produces (N, 3) with no NaN/inf after warm-up."""
        rng = np.random.default_rng(42)
        n = 800
        base = 5000.0
        closes = base + np.cumsum(rng.normal(0, 0.5, n))
        timestamps = np.arange(n, dtype=np.int64) * 1_000_000_000

        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": closes - rng.uniform(0, 0.5, n),
            "high": closes + rng.uniform(0, 1.0, n),
            "low": closes - rng.uniform(0, 1.0, n),
            "close": closes,
            "volume": rng.integers(10, 200, n),
        })

        config = HMMRegimeConfig(zscore_window=50)
        features, ts = build_feature_matrix(df, config)

        assert features.ndim == 2
        assert features.shape[1] == 3
        assert features.shape[0] > 0
        assert len(ts) == features.shape[0]
        assert not np.any(np.isnan(features)), "NaN found in feature matrix"
        assert not np.any(np.isinf(features)), "Inf found in feature matrix"

    def test_features_clipped_to_bounds(self):
        """Z-scored features should be clipped to [-3, 3]."""
        rng = np.random.default_rng(42)
        n = 800
        base = 5000.0
        closes = base + np.cumsum(rng.normal(0, 0.5, n))
        timestamps = np.arange(n, dtype=np.int64) * 1_000_000_000

        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": closes - rng.uniform(0, 0.5, n),
            "high": closes + rng.uniform(0, 1.0, n),
            "low": closes - rng.uniform(0, 1.0, n),
            "close": closes,
            "volume": rng.integers(10, 200, n),
        })

        config = HMMRegimeConfig(zscore_window=50)
        features, _ = build_feature_matrix(df, config)

        assert features.min() >= -3.0
        assert features.max() <= 3.0
