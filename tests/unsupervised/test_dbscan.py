"""
Tests for mlscratch.unsupervised.dbscan.DBSCAN
"""

import numpy as np
import pytest
from mlscratch.unsupervised.dbscan import DBSCAN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_dense_blobs():
    """Two tight, well-separated blobs that DBSCAN should find cleanly."""
    rng = np.random.default_rng(42)
    blob_a = rng.normal(loc=[0.0, 0.0], scale=0.2, size=(30, 2))
    blob_b = rng.normal(loc=[5.0, 5.0], scale=0.2, size=(30, 2))
    return np.vstack([blob_a, blob_b])


@pytest.fixture
def data_with_noise():
    """Two blobs plus isolated noise points."""
    rng = np.random.default_rng(7)
    blob_a = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(20, 2))
    blob_b = rng.normal(loc=[6.0, 0.0], scale=0.3, size=(20, 2))
    noise = rng.uniform(low=-10, high=10, size=(5, 2))
    return np.vstack([blob_a, blob_b, noise])


# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------

class TestDBSCANBasic:
    def test_fit_returns_self(self, two_dense_blobs):
        model = DBSCAN(eps=0.5, min_samples=3)
        result = model.fit(two_dense_blobs)
        assert result is model

    def test_labels_assigned_after_fit(self, two_dense_blobs):
        model = DBSCAN(eps=0.5, min_samples=3).fit(two_dense_blobs)
        assert model.labels_ is not None
        assert len(model.labels_) == len(two_dense_blobs)

    def test_fit_predict_same_as_fit_then_labels(self, two_dense_blobs):
        m1 = DBSCAN(eps=0.5, min_samples=3)
        labels_fp = m1.fit_predict(two_dense_blobs)
        m2 = DBSCAN(eps=0.5, min_samples=3)
        m2.fit(two_dense_blobs)
        np.testing.assert_array_equal(labels_fp, m2.labels_)

    def test_core_sample_indices_subset(self, two_dense_blobs):
        model = DBSCAN(eps=0.5, min_samples=3).fit(two_dense_blobs)
        assert model.core_sample_indices_ is not None
        n = len(two_dense_blobs)
        assert all(0 <= i < n for i in model.core_sample_indices_)


# ---------------------------------------------------------------------------
# Cluster-count tests
# ---------------------------------------------------------------------------

class TestDBSCANClusters:
    def test_finds_two_clusters(self, two_dense_blobs):
        labels = DBSCAN(eps=0.6, min_samples=3).fit_predict(two_dense_blobs)
        unique_clusters = set(labels) - {-1}
        assert len(unique_clusters) == 2

    def test_cluster_ids_are_positive_integers(self, two_dense_blobs):
        labels = DBSCAN(eps=0.6, min_samples=3).fit_predict(two_dense_blobs)
        for lbl in labels:
            assert lbl == -1 or lbl >= 1

    def test_noise_label_is_minus_one(self, data_with_noise):
        labels = DBSCAN(eps=0.5, min_samples=4).fit_predict(data_with_noise)
        # At least one noise point expected (the scattered uniform samples)
        assert -1 in labels

    def test_tight_blobs_no_noise(self, two_dense_blobs):
        labels = DBSCAN(eps=1.0, min_samples=2).fit_predict(two_dense_blobs)
        # With generous eps every blob point should be in a cluster
        assert -1 not in labels


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestDBSCANEdgeCases:
    def test_all_noise_when_eps_too_small(self):
        X = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        labels = DBSCAN(eps=0.1, min_samples=2).fit_predict(X)
        assert all(l == -1 for l in labels)

    def test_single_cluster_when_eps_very_large(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 2))
        labels = DBSCAN(eps=1000.0, min_samples=2).fit_predict(X)
        unique_clusters = set(labels) - {-1}
        assert len(unique_clusters) == 1

    def test_single_point_dataset(self):
        X = np.array([[1.0, 2.0]])
        labels = DBSCAN(eps=0.5, min_samples=1).fit_predict(X)
        assert len(labels) == 1

    def test_output_dtype_is_int(self, two_dense_blobs):
        labels = DBSCAN(eps=0.6, min_samples=3).fit_predict(two_dense_blobs)
        assert labels.dtype in (np.int32, np.int64, int)
