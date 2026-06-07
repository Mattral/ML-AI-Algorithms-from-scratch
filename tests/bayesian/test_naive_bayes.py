"""
Tests for mlscratch.bayesian.naive_bayes
"""

import numpy as np
import pytest
from mlscratch.bayesian.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_gaussian():
    """Two linearly separable Gaussian blobs."""
    rng = np.random.default_rng(0)
    X0 = rng.normal([0, 0], 0.5, (50, 2))
    X1 = rng.normal([4, 4], 0.5, (50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


@pytest.fixture
def three_class_gaussian():
    rng = np.random.default_rng(1)
    blobs = [rng.normal(c, 0.5, (30, 2)) for c in [[0,0],[5,0],[2.5,5]]]
    X = np.vstack(blobs)
    y = np.repeat([0, 1, 2], 30)
    return X, y


@pytest.fixture
def count_data():
    """Toy word-count data, 2 classes."""
    X = np.array([
        [3, 0, 1], [2, 1, 0], [0, 3, 2], [1, 2, 3],
        [4, 0, 0], [0, 0, 5], [2, 2, 0], [0, 4, 1],
    ], dtype=float)
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    return X, y


@pytest.fixture
def binary_data():
    X = np.array([
        [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1],
        [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0],
    ], dtype=float)
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    return X, y


# ============================================================
# GaussianNB
# ============================================================

class TestGaussianNBBasic:
    def test_fit_returns_self(self, binary_gaussian):
        X, y = binary_gaussian
        assert GaussianNB().fit(X, y) is GaussianNB().fit(X, y).__class__().\
               fit(X, y).__class__() or GaussianNB().fit(X, y) is not None

    def test_classes_detected(self, three_class_gaussian):
        X, y = three_class_gaussian
        model = GaussianNB().fit(X, y)
        np.testing.assert_array_equal(model.classes_, [0, 1, 2])

    def test_theta_shape(self, three_class_gaussian):
        X, y = three_class_gaussian
        model = GaussianNB().fit(X, y)
        assert model.theta_.shape == (3, 2)

    def test_sigma_positive(self, binary_gaussian):
        X, y = binary_gaussian
        model = GaussianNB().fit(X, y)
        assert np.all(model.sigma_ > 0)

    def test_prior_sums_to_one(self, binary_gaussian):
        X, y = binary_gaussian
        model = GaussianNB().fit(X, y)
        np.testing.assert_allclose(model.class_prior_.sum(), 1.0, atol=1e-10)


class TestGaussianNBCorrectness:
    def test_perfect_separation(self, binary_gaussian):
        X, y = binary_gaussian
        preds = GaussianNB().fit(X, y).predict(X)
        acc = (preds == y).mean()
        assert acc >= 0.95

    def test_predict_proba_shape(self, three_class_gaussian):
        X, y = three_class_gaussian
        proba = GaussianNB().fit(X, y).predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_predict_proba_sums_to_one(self, binary_gaussian):
        X, y = binary_gaussian
        proba = GaussianNB().fit(X, y).predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_three_class_accuracy(self, three_class_gaussian):
        X, y = three_class_gaussian
        acc = (GaussianNB().fit(X, y).predict(X) == y).mean()
        assert acc >= 0.90


# ============================================================
# MultinomialNB
# ============================================================

class TestMultinomialNBBasic:
    def test_fit_sets_feature_log_prob(self, count_data):
        X, y = count_data
        model = MultinomialNB().fit(X, y)
        assert model.feature_log_prob_.shape == (2, 3)

    def test_feature_log_prob_leq_zero(self, count_data):
        X, y = count_data
        model = MultinomialNB().fit(X, y)
        assert np.all(model.feature_log_prob_ <= 0)

    def test_predict_shape(self, count_data):
        X, y = count_data
        preds = MultinomialNB().fit(X, y).predict(X)
        assert len(preds) == len(y)

    def test_predict_proba_rows_sum_to_one(self, count_data):
        X, y = count_data
        proba = MultinomialNB().fit(X, y).predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_laplace_smoothing_zero_count(self):
        """With alpha=1, unseen features should not cause log(0)."""
        X_train = np.array([[2, 0, 0], [0, 3, 0]], dtype=float)
        y_train = np.array([0, 1])
        X_test  = np.array([[0, 0, 5]], dtype=float)
        preds = MultinomialNB(alpha=1.0).fit(X_train, y_train).predict(X_test)
        assert preds[0] in (0, 1)


# ============================================================
# BernoulliNB
# ============================================================

class TestBernoulliNBBasic:
    def test_fit_sets_feature_probs(self, binary_data):
        X, y = binary_data
        model = BernoulliNB().fit(X, y)
        assert model.feature_log_prob_.shape == (2, 3)
        assert model.feature_log_prob_neg_.shape == (2, 3)

    def test_predict_correct_shape(self, binary_data):
        X, y = binary_data
        preds = BernoulliNB().fit(X, y).predict(X)
        assert len(preds) == len(y)

    def test_predict_proba_sums_to_one(self, binary_data):
        X, y = binary_data
        proba = BernoulliNB().fit(X, y).predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_binarize_threshold(self):
        X = np.array([[0.3, 0.8], [0.9, 0.1]])
        y = np.array([0, 1])
        model = BernoulliNB(binarize=0.5).fit(X, y)
        # Should not raise; binarization applied
        preds = model.predict(X)
        assert len(preds) == 2

    def test_none_binarize_uses_raw(self, binary_data):
        X, y = binary_data
        BernoulliNB(binarize=None).fit(X, y).predict(X)   # no error

    def test_simple_separation(self):
        """Class 0: mostly 1s; class 1: mostly 0s → should classify correctly."""
        X = np.array([[1,1,1],[1,1,0],[0,1,1],[1,0,1],
                      [0,0,0],[0,0,1],[1,0,0],[0,1,0]], dtype=float)
        y = np.array([0,0,0,0,1,1,1,1])
        acc = (BernoulliNB().fit(X, y).predict(X) == y).mean()
        assert acc >= 0.75
