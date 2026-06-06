"""
Tests for mlscratch.unsupervised.gmm.GaussianMixtureModel
"""

import numpy as np
import pytest
from mlscratch.unsupervised.gmm import GaussianMixtureModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_gaussians():
    """Two well-separated 2-D Gaussian blobs."""
    rng = np.random.default_rng(42)
    c1 = rng.multivariate_normal([0.0, 0.0], np.eye(2), size=60)
    c2 = rng.multivariate_normal([8.0, 8.0], np.eye(2), size=60)
    return np.vstack([c1, c2])


@pytest.fixture
def three_gaussians():
    rng = np.random.default_rng(7)
    blobs = [
        rng.multivariate_normal([0, 0],  np.eye(2), 40),
        rng.multivariate_normal([6, 0],  np.eye(2), 40),
        rng.multivariate_normal([3, 6],  np.eye(2), 40),
    ]
    return np.vstack(blobs)


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestGMMBasic:
    def test_fit_returns_self(self, two_gaussians):
        gmm = GaussianMixtureModel(n_components=2, random_state=0)
        assert gmm.fit(two_gaussians) is gmm

    def test_weights_sum_to_one(self, two_gaussians):
        gmm = GaussianMixtureModel(n_components=2, random_state=0).fit(two_gaussians)
        np.testing.assert_allclose(gmm.weights_.sum(), 1.0, atol=1e-6)

    def test_means_shape(self, two_gaussians):
        gmm = GaussianMixtureModel(n_components=2, random_state=0).fit(two_gaussians)
        assert gmm.means_.shape == (2, 2)

    def test_covariances_shape(self, two_gaussians):
        gmm = GaussianMixtureModel(n_components=2, random_state=0).fit(two_gaussians)
        assert gmm.covariances_.shape == (2, 2, 2)

    def test_predict_length(self, two_gaussians):
        gmm = GaussianMixtureModel(n_components=2, random_state=0).fit(two_gaussians)
        labels = gmm.predict(two_gaussians)
        assert len(labels) == len(two_gaussians)

    def test_predict_proba_shape(self, two_gaussians):
        gmm = GaussianMixtureModel(n_components=2, random_state=0).fit(two_gaussians)
        proba = gmm.predict_proba(two_gaussians)
        assert proba.shape == (len(two_gaussians), 2)

    def test_predict_proba_rows_sum_to_one(self, two_gaussians):
        gmm = GaussianMixtureModel(n_components=2, random_state=0).fit(two_gaussians)
        proba = gmm.predict_proba(two_gaussians)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_weights_non_negative(self, two_gaussians):
        gmm = GaussianMixtureModel(n_components=2, random_state=0).fit(two_gaussians)
        assert np.all(gmm.weights_ >= 0)


# ---------------------------------------------------------------------------
# Cluster-recovery tests
# ---------------------------------------------------------------------------

class TestGMMClusters:
    def test_two_component_recovery(self, two_gaussians):
        """With two well-separated blobs, GMM should find both."""
        gmm = GaussianMixtureModel(n_components=2, max_iter=100, random_state=0)
        labels = gmm.fit(two_gaussians).predict(two_gaussians)
        # First 60 samples should share one label, last 60 the other
        assert len(set(labels[:60])) == 1
        assert len(set(labels[60:])) == 1
        assert labels[0] != labels[60]

    def test_three_component_recovery(self, three_gaussians):
        gmm = GaussianMixtureModel(n_components=3, max_iter=150, random_state=1)
        labels = gmm.fit(three_gaussians).predict(three_gaussians)
        assert len(set(labels)) == 3

    def test_means_near_true_centers(self, two_gaussians):
        gmm = GaussianMixtureModel(n_components=2, max_iter=100, random_state=0)
        gmm.fit(two_gaussians)
        centers = np.sort(gmm.means_[:, 0])   # sort by x-coordinate
        assert centers[0] < 2.0               # near 0
        assert centers[1] > 5.0               # near 8


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestGMMEdgeCases:
    def test_single_component(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 2))
        gmm = GaussianMixtureModel(n_components=1, random_state=0).fit(X)
        assert gmm.weights_.shape == (1,)
        np.testing.assert_allclose(gmm.weights_, [1.0], atol=1e-6)

    def test_1d_data(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((30, 1))
        gmm = GaussianMixtureModel(n_components=2, random_state=0).fit(X)
        labels = gmm.predict(X)
        assert len(labels) == 30
