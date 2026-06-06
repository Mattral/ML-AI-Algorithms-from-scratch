"""
Tests for mlscratch.unsupervised.tsne.TSNE

Note: t-SNE is stochastic and computationally heavy; tests focus on
output shape, reproducibility, and basic structural properties rather
than exact embedding values.  Use small datasets and few iterations.
"""

import numpy as np
import pytest
from mlscratch.unsupervised.tsne import TSNE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_clusters():
    """Three clearly separated 4-D clusters (10 pts each)."""
    rng = np.random.default_rng(42)
    a = rng.normal([0,  0,  0,  0], 0.3, (10, 4))
    b = rng.normal([10, 0,  0,  0], 0.3, (10, 4))
    c = rng.normal([5,  10, 0,  0], 0.3, (10, 4))
    return np.vstack([a, b, c])


@pytest.fixture
def tiny_2d():
    """Very small 2-D dataset for speed."""
    rng = np.random.default_rng(7)
    return rng.standard_normal((12, 2))


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestTSNEBasic:
    def test_fit_transform_returns_ndarray(self, tiny_2d):
        model = TSNE(n_components=2, n_iter=50, random_state=0)
        Y = model.fit_transform(tiny_2d)
        assert isinstance(Y, np.ndarray)

    def test_output_shape_2d(self, tiny_2d):
        Y = TSNE(n_components=2, n_iter=50, random_state=0).fit_transform(tiny_2d)
        assert Y.shape == (len(tiny_2d), 2)

    def test_output_shape_1d(self, tiny_2d):
        Y = TSNE(n_components=1, n_iter=50, random_state=0).fit_transform(tiny_2d)
        assert Y.shape == (len(tiny_2d), 1)

    def test_embedding_stored_after_fit(self, tiny_2d):
        model = TSNE(n_components=2, n_iter=50, random_state=0)
        model.fit(tiny_2d)
        assert model.embedding_ is not None
        assert model.embedding_.shape == (len(tiny_2d), 2)

    def test_fit_and_fit_transform_agree(self, tiny_2d):
        m1 = TSNE(n_components=2, n_iter=50, random_state=0)
        Y1 = m1.fit_transform(tiny_2d)
        m2 = TSNE(n_components=2, n_iter=50, random_state=0)
        m2.fit(tiny_2d)
        np.testing.assert_allclose(Y1, m2.embedding_, atol=1e-10)

    def test_reproducible_with_same_seed(self, tiny_2d):
        Y1 = TSNE(n_components=2, n_iter=50, random_state=5).fit_transform(tiny_2d)
        Y2 = TSNE(n_components=2, n_iter=50, random_state=5).fit_transform(tiny_2d)
        np.testing.assert_allclose(Y1, Y2, atol=1e-10)

    def test_different_seeds_give_different_results(self, tiny_2d):
        Y1 = TSNE(n_components=2, n_iter=100, random_state=0).fit_transform(tiny_2d)
        Y2 = TSNE(n_components=2, n_iter=100, random_state=99).fit_transform(tiny_2d)
        assert not np.allclose(Y1, Y2)


# ---------------------------------------------------------------------------
# Output properties
# ---------------------------------------------------------------------------

class TestTSNEProperties:
    def test_embedding_zero_mean(self, tiny_2d):
        """t-SNE centres the embedding at each iteration."""
        Y = TSNE(n_components=2, n_iter=100, random_state=0).fit_transform(tiny_2d)
        np.testing.assert_allclose(Y.mean(axis=0), 0.0, atol=1e-6)

    def test_no_nan_in_output(self, tiny_2d):
        Y = TSNE(n_components=2, n_iter=50, random_state=0).fit_transform(tiny_2d)
        assert not np.any(np.isnan(Y))

    def test_no_inf_in_output(self, tiny_2d):
        Y = TSNE(n_components=2, n_iter=50, random_state=0).fit_transform(tiny_2d)
        assert not np.any(np.isinf(Y))

    def test_cluster_separation_preserved(self, small_clusters):
        """Clusters far apart in high-dim should remain separable in 2-D."""
        Y = TSNE(
            n_components=2, perplexity=5, n_iter=300, random_state=0
        ).fit_transform(small_clusters)
        # Compute centroids in embedding for the three groups
        c1, c2, c3 = Y[:10].mean(0), Y[10:20].mean(0), Y[20:].mean(0)
        d12 = np.linalg.norm(c1 - c2)
        d13 = np.linalg.norm(c1 - c3)
        d23 = np.linalg.norm(c2 - c3)
        # Within-cluster spread
        spread = np.mean([
            np.mean(np.linalg.norm(Y[:10] - c1, axis=1)),
            np.mean(np.linalg.norm(Y[10:20] - c2, axis=1)),
            np.mean(np.linalg.norm(Y[20:] - c3, axis=1)),
        ])
        # Centroids should be further apart than the average within-cluster spread
        assert min(d12, d13, d23) > spread


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestTSNEEdgeCases:
    def test_single_sample(self):
        X = np.array([[1.0, 2.0, 3.0]])
        Y = TSNE(n_components=2, n_iter=10, random_state=0).fit_transform(X)
        assert Y.shape == (1, 2)

    def test_two_samples(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        Y = TSNE(n_components=2, n_iter=20, random_state=0).fit_transform(X)
        assert Y.shape == (2, 2)
        assert not np.any(np.isnan(Y))
