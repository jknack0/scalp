"""Tests for the HMM intraday regime classifier (src/models/hmm_regime.py).

Uses synthetic clustered data with 5 distinct centers and sequential
blocks to provide temporal structure for the HMM.
"""

import numpy as np
import polars as pl
import pytest

from src.models.hmm_regime import (
    HMMRegimeClassifier,
    HMMRegimeConfig,
    HMMValidationReport,
    RegimeState,
    build_feature_matrix,
    compute_persistence_accuracy,
    compute_transition_matrix,
    validate_model,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_clustered_data(
    n_per_cluster: int = 200,
    n_features: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic data with 5 well-separated sequential clusters.

    Each cluster is a block of ``n_per_cluster`` rows drawn from a Gaussian
    centered at a distinct point. The blocks are concatenated in order to
    give the HMM temporal structure to learn from.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([
        [2.0, 0.5, 1.0, 0.3, 1.5, 0.8],    # HIGH_VOL_UP-like
        [2.0, -0.5, -1.0, 0.3, 1.5, -0.8],  # HIGH_VOL_DOWN-like
        [-1.0, 0.0, 0.0, -0.5, -1.0, 0.0],  # LOW_VOL_RANGE-like
        [0.5, 0.0, 0.3, 2.0, 1.0, 0.5],     # BREAKOUT-like
        [0.0, 2.0, -0.3, 0.0, 0.0, -0.2],   # MEAN_REVERSION-like
    ])

    blocks = []
    for center in centers:
        block = rng.normal(loc=center, scale=0.3, size=(n_per_cluster, n_features))
        blocks.append(block)

    return np.vstack(blocks)


@pytest.fixture
def clustered_data() -> np.ndarray:
    """1000-row synthetic clustered dataset."""
    return _make_clustered_data()


@pytest.fixture
def fitted_classifier(clustered_data: np.ndarray) -> HMMRegimeClassifier:
    """Pre-fitted HMM classifier on clustered data."""
    config = HMMRegimeConfig(n_states=5, n_iter=50)
    clf = HMMRegimeClassifier(config)
    clf.fit(clustered_data)
    return clf


# ── Tests ────────────────────────────────────────────────────────────


class TestRegimeState:
    def test_enum_values(self):
        """RegimeState has exactly 5 members with values 0-4."""
        assert len(RegimeState) == 5
        assert RegimeState.HIGH_VOL_UP == 0
        assert RegimeState.HIGH_VOL_DOWN == 1
        assert RegimeState.LOW_VOL_RANGE == 2
        assert RegimeState.BREAKOUT == 3
        assert RegimeState.MEAN_REVERSION == 4


class TestHMMClassifier:
    def test_fit_without_error(self, clustered_data: np.ndarray):
        """Model trains on synthetic data without raising."""
        config = HMMRegimeConfig(n_states=5, n_iter=50)
        clf = HMMRegimeClassifier(config)
        clf.fit(clustered_data)
        assert clf.model is not None
        assert clf.state_map is not None

    def test_label_states_assigns_all_enum_values(
        self, fitted_classifier: HMMRegimeClassifier
    ):
        """All 5 RegimeState values are present in state_map."""
        assigned = set(fitted_classifier.state_map.values())
        assert assigned == set(RegimeState)

    def test_predict_returns_correct_length(
        self,
        fitted_classifier: HMMRegimeClassifier,
        clustered_data: np.ndarray,
    ):
        """predict() returns one RegimeState per input row."""
        states = fitted_classifier.predict(clustered_data)
        assert len(states) == len(clustered_data)
        assert all(isinstance(s, RegimeState) for s in states)

    def test_predict_proba_shape_and_sum(
        self,
        fitted_classifier: HMMRegimeClassifier,
        clustered_data: np.ndarray,
    ):
        """predict_proba returns 5 probabilities summing to ~1.0."""
        # Use last 50 rows as window
        window = clustered_data[-50:]
        state, probs = fitted_classifier.predict_proba(window)

        assert isinstance(state, RegimeState)
        assert probs.shape == (5,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
        assert all(p >= 0 for p in probs)

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
        original_states = fitted_classifier.predict(clustered_data)
        loaded_states = loaded.predict(clustered_data)

        assert original_states == loaded_states


class TestEvaluation:
    def test_persistence_accuracy_on_clustered_data(
        self,
        fitted_classifier: HMMRegimeClassifier,
        clustered_data: np.ndarray,
    ):
        """Clustered data with 200-bar blocks should have >65% persistence at horizon=5."""
        states = fitted_classifier.predict(clustered_data)
        pers = compute_persistence_accuracy(states, horizon=5)
        assert pers > 0.65, f"Persistence {pers:.1%} below 65% threshold"

    def test_transition_matrix_shape_and_rows(
        self,
        fitted_classifier: HMMRegimeClassifier,
        clustered_data: np.ndarray,
    ):
        """Transition matrix is (5,5) with rows summing to ~1."""
        states = fitted_classifier.predict(clustered_data)
        tm = compute_transition_matrix(states)
        assert tm.shape == (5, 5)
        for row_sum in tm.sum(axis=1):
            assert row_sum == pytest.approx(1.0, abs=1e-6)

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
        assert len(report.state_distribution) == 5
        assert report.transition_matrix.shape == (5, 5)
        assert isinstance(report.log_likelihood, float)


class TestFeatureMatrix:
    def test_feature_matrix_shape(self):
        """build_feature_matrix produces (N, 6) with no NaN/inf after warm-up."""
        # Create minimal synthetic 1s bar data
        rng = np.random.default_rng(42)
        n = 500
        base = 5000.0
        closes = base + np.cumsum(rng.normal(0, 0.5, n))
        timestamps = np.arange(n, dtype=np.int64) * 1_000_000_000  # 1s intervals

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
        assert features.shape[1] == 6
        assert features.shape[0] > 0
        assert len(ts) == features.shape[0]
        assert not np.any(np.isnan(features)), "NaN found in feature matrix"
        assert not np.any(np.isinf(features)), "Inf found in feature matrix"
