"""
Tests for mlscratch.unsupervised.pca.PCA
"""

import numpy as np
import pytest
from mlscratch.unsupervised.pca import PCA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_2d():
    """10 points arranged along a near-diagonal in 2-D."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 30)
    X = np.column_stack([t, t + rng.normal(0, 0.01, 30)])
    return X


@pytest.fixture
def random_4d():
    rng = np.random.default_rng(1)
    return rng.standard_normal((80, 4))


# ---------------------------------------------------------------------------
# Basic API tests
# ---------------------------------------------------------------------------

class TestPCABasic:
    def test_fit_returns_self(self, simple_2d):
        pca = PCA(n_components=1)
        assert pca.fit(simple_2d) is pca

    def test_components_shape(self, random_4d):
        pca = PCA(n_components=2).fit(random_4d)
        assert pca.components_.shape == (2, 4)

    def test_mean_shape(self, random_4d):
        pca = PCA(n_components=2).fit(random_4d)
        assert pca.mean_.shape == (4,)

    def test_explained_variance_positive(self, random_4d):
        pca = PCA(n_components=3).fit(random_4d)
        assert np.all(pca.explained_variance_ >= 0)

    def test_explained_variance_ratio_sums_to_at_most_one(self, random_4d):
        pca = PCA(n_components=4).fit(random_4d)
        assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-9

    def test_fit_transform_shape(self, random_4d):
        result = PCA(n_components=2).fit_transform(random_4d)
        assert result.shape == (80, 2)

    def test_none_components_keeps_all(self, random_4d):
        pca = PCA(n_components=None).fit(random_4d)
        assert pca.components_.shape[0] == 4


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

class TestPCACorrectness:
    def test_transform_zero_mean(self, random_4d):
        """Projected data should be zero-centred."""
        X_proj = PCA(n_components=2).fit_transform(random_4d)
        np.testing.assert_allclose(X_proj.mean(axis=0), 0.0, atol=1e-10)

    def test_first_pc_captures_most_variance(self, simple_2d):
        pca = PCA(n_components=2).fit(simple_2d)
        # First component should explain ≥ 95 % of variance
        assert pca.explained_variance_ratio_[0] >= 0.95

    def test_components_are_orthonormal(self, random_4d):
        pca = PCA(n_components=3).fit(random_4d)
        gram = pca.components_ @ pca.components_.T
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)

    def test_inverse_transform_reconstruction(self, random_4d):
        """Reconstruction with all components should nearly recover the input."""
        pca = PCA(n_components=4).fit(random_4d)
        X_rec = pca.inverse_transform(pca.transform(random_4d))
        np.testing.assert_allclose(X_rec, random_4d, atol=1e-8)

    def test_partial_reconstruction_lower_error_than_noise(self, simple_2d):
        """1-component reconstruction of near-1D data has small MSE."""
        pca = PCA(n_components=1).fit(simple_2d)
        X_rec = pca.inverse_transform(pca.transform(simple_2d))
        mse = np.mean((simple_2d - X_rec) ** 2)
        assert mse < 1e-3

    def test_variance_ordered_descending(self, random_4d):
        pca = PCA(n_components=4).fit(random_4d)
        ev = pca.explained_variance_
        assert all(ev[i] >= ev[i + 1] for i in range(len(ev) - 1))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPCAEdgeCases:
    def test_single_feature(self):
        X = np.arange(20).reshape(-1, 1).astype(float)
        pca = PCA(n_components=1).fit(X)
        assert pca.components_.shape == (1, 1)

    def test_two_samples(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        pca = PCA(n_components=1).fit(X)
        assert pca.embedding_ if hasattr(pca, "embedding_") else True  # no error
