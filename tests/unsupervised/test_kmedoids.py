"""
Tests for mlscratch.unsupervised.kmedoids.KMedoids
"""

import numpy as np
import pytest
from mlscratch.unsupervised.kmedoids import KMedoids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_clusters():
    """Two tight, well-separated blobs."""
    rng = np.random.default_rng(17)
    a = rng.normal([0, 0], 0.2, (20, 2))
    b = rng.normal([8, 8], 0.2, (20, 2))
    return np.vstack([a, b])


@pytest.fixture
def three_clusters():
    rng = np.random.default_rng(5)
    a = rng.normal([0, 0],  0.3, (15, 2))
    b = rng.normal([7, 0],  0.3, (15, 2))
    c = rng.normal([3, 6],  0.3, (15, 2))
    return np.vstack([a, b, c])


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestKMedoidsBasic:
    def test_fit_returns_self(self, two_clusters):
        model = KMedoids(n_clusters=2, random_state=0)
        assert model.fit(two_clusters) is model

    def test_labels_length(self, two_clusters):
        model = KMedoids(n_clusters=2, random_state=0).fit(two_clusters)
        assert len(model.labels_) == len(two_clusters)

    def test_n_unique_labels(self, two_clusters):
        labels = KMedoids(n_clusters=2, random_state=0).fit_predict(two_clusters)
        assert len(set(labels)) == 2

    def test_medoid_indices_within_bounds(self, two_clusters):
        model = KMedoids(n_clusters=2, random_state=0).fit(two_clusters)
        n = len(two_clusters)
        assert all(0 <= i < n for i in model.medoid_indices_)

    def test_medoid_indices_are_unique(self, two_clusters):
        model = KMedoids(n_clusters=2, random_state=0).fit(two_clusters)
        assert len(set(model.medoid_indices_)) == 2

    def test_inertia_positive(self, two_clusters):
        model = KMedoids(n_clusters=2, random_state=0).fit(two_clusters)
        assert model.inertia_ >= 0.0

    def test_fit_predict_consistent(self, two_clusters):
        m1 = KMedoids(n_clusters=2, random_state=0)
        labels_fp = m1.fit_predict(two_clusters)
        m2 = KMedoids(n_clusters=2, random_state=0)
        m2.fit(two_clusters)
        np.testing.assert_array_equal(labels_fp, m2.labels_)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestKMedoidsCorrectness:
    def test_two_cluster_separation(self, two_clusters):
        labels = KMedoids(n_clusters=2, random_state=0).fit_predict(two_clusters)
        # First 20 should share one label; last 20 the other
        assert len(set(labels[:20])) == 1
        assert len(set(labels[20:])) == 1
        assert labels[0] != labels[20]

    def test_three_cluster_separation(self, three_clusters):
        labels = KMedoids(n_clusters=3, random_state=0).fit_predict(three_clusters)
        assert len(set(labels[:15])) == 1
        assert len(set(labels[15:30])) == 1
        assert len(set(labels[30:])) == 1

    def test_medoids_are_actual_data_points(self, two_clusters):
        model = KMedoids(n_clusters=2, random_state=0).fit(two_clusters)
        for idx in model.medoid_indices_:
            assert any(
                np.allclose(two_clusters[idx], two_clusters[j])
                for j in range(len(two_clusters))
            )

    def test_inertia_decreases_or_stays_with_more_clusters(self, three_clusters):
        i2 = KMedoids(n_clusters=2, random_state=0).fit(three_clusters).inertia_
        i3 = KMedoids(n_clusters=3, random_state=0).fit(three_clusters).inertia_
        assert i3 <= i2 + 1e-6   # more clusters ⇒ lower or equal inertia


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestKMedoidsEdgeCases:
    def test_k_equals_n_samples(self):
        X = np.eye(4, dtype=float)
        model = KMedoids(n_clusters=4, random_state=0).fit(X)
        assert len(set(model.labels_)) == 4

    def test_single_cluster(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 2))
        model = KMedoids(n_clusters=1, random_state=0).fit(X)
        assert len(set(model.labels_)) == 1
        assert len(model.medoid_indices_) == 1
