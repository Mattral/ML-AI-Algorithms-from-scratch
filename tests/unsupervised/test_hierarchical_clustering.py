"""
Tests for mlscratch.unsupervised.hierarchical_clustering.AgglomerativeClustering
"""

import numpy as np
import pytest
from mlscratch.unsupervised.hierarchical_clustering import AgglomerativeClustering


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_blobs():
    """Three small, well-separated clusters."""
    rng = np.random.default_rng(99)
    a = rng.normal([0, 0],   0.1, (10, 2))
    b = rng.normal([5, 0],   0.1, (10, 2))
    c = rng.normal([2.5, 4], 0.1, (10, 2))
    return np.vstack([a, b, c])


@pytest.fixture
def tiny_4pt():
    """4 points arranged as 2 obvious pairs."""
    return np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [10.0, 0.0],
        [10.1, 0.0],
    ])


# ---------------------------------------------------------------------------
# Instantiation / parameter validation
# ---------------------------------------------------------------------------

class TestHACInit:
    def test_invalid_linkage_raises(self):
        with pytest.raises(ValueError):
            AgglomerativeClustering(n_clusters=2, linkage="bad_linkage")

    @pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
    def test_valid_linkages_instantiate(self, linkage):
        model = AgglomerativeClustering(n_clusters=2, linkage=linkage)
        assert model.linkage == linkage


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestHACBasic:
    def test_fit_returns_self(self, three_blobs):
        model = AgglomerativeClustering(n_clusters=3)
        assert model.fit(three_blobs) is model

    def test_labels_length(self, three_blobs):
        model = AgglomerativeClustering(n_clusters=3).fit(three_blobs)
        assert len(model.labels_) == len(three_blobs)

    def test_fit_predict_consistent(self, three_blobs):
        m1 = AgglomerativeClustering(n_clusters=3)
        labels_fp = m1.fit_predict(three_blobs)
        m2 = AgglomerativeClustering(n_clusters=3)
        m2.fit(three_blobs)
        np.testing.assert_array_equal(labels_fp, m2.labels_)

    def test_n_unique_labels_equals_n_clusters(self, three_blobs):
        labels = AgglomerativeClustering(n_clusters=3).fit_predict(three_blobs)
        assert len(set(labels)) == 3

    def test_labels_are_integers(self, three_blobs):
        labels = AgglomerativeClustering(n_clusters=2).fit_predict(three_blobs)
        for lbl in labels:
            assert isinstance(int(lbl), int)


# ---------------------------------------------------------------------------
# Correctness across linkages
# ---------------------------------------------------------------------------

class TestHACLinkages:
    @pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
    def test_obvious_two_cluster_recovery(self, tiny_4pt, linkage):
        """All linkages must separate the two obvious pairs."""
        labels = AgglomerativeClustering(
            n_clusters=2, linkage=linkage
        ).fit_predict(tiny_4pt)
        # Points 0,1 should share a label; points 2,3 should share a label
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    @pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
    def test_three_blob_recovery(self, three_blobs, linkage):
        labels = AgglomerativeClustering(
            n_clusters=3, linkage=linkage
        ).fit_predict(three_blobs)
        # Each original group of 10 should share the same label
        assert len(set(labels[:10])) == 1
        assert len(set(labels[10:20])) == 1
        assert len(set(labels[20:])) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestHACEdgeCases:
    def test_n_clusters_equals_n_samples(self):
        X = np.eye(5)
        labels = AgglomerativeClustering(n_clusters=5).fit_predict(X)
        assert len(set(labels)) == 5

    def test_n_clusters_1(self):
        X = np.random.default_rng(0).standard_normal((10, 2))
        labels = AgglomerativeClustering(n_clusters=1).fit_predict(X)
        assert len(set(labels)) == 1
