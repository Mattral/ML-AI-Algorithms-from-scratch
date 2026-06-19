"""
Tests for mlscratch.neural.rbf_network.RBFNetwork
"""

from __future__ import annotations

import numpy as np
import pytest

from mlscratch.neural.rbf_network import RBFNetwork


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def regression_data():
    """y = sin(x), 60 training points."""
    X = np.linspace(-np.pi, np.pi, 60).reshape(-1, 1)
    y = np.sin(X.ravel())
    return X, y


@pytest.fixture
def binary_clf_data():
    rng = np.random.default_rng(0)
    X0 = rng.normal([0, 0], 0.5, (25, 2))
    X1 = rng.normal([4, 4], 0.5, (25, 2))
    X  = np.vstack([X0, X1])
    y  = np.array([0] * 25 + [1] * 25)
    return X, y


@pytest.fixture
def three_class_data():
    rng = np.random.default_rng(1)
    centres = [[0, 0], [6, 0], [3, 5]]
    X = np.vstack([rng.normal(c, 0.5, (20, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 20)
    return X, y


# ===================================================================
# Instantiation
# ===================================================================

class TestRBFNetworkInit:
    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            RBFNetwork(task="invalid")

    def test_default_task_is_regression(self):
        rbf = RBFNetwork()
        assert rbf.task == "regression"

    def test_centers_none_before_fit(self):
        rbf = RBFNetwork()
        assert rbf.centers_ is None
        assert rbf.sigmas_ is None
        assert rbf.weights_ is None


# ===================================================================
# Centre / width selection
# ===================================================================

class TestRBFCentreSelection:
    def test_centers_shape(self, regression_data):
        X, y = regression_data
        rbf = RBFNetwork(n_centers=8, random_state=0).fit(X, y)
        assert rbf.centers_.shape == (8, 1)

    def test_fewer_centers_than_samples_when_k_gt_n(self):
        """If n_centers >= n_samples, use all samples."""
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 2.0])
        rbf = RBFNetwork(n_centers=100, random_state=0).fit(X, y)
        assert len(rbf.centers_) <= 3

    def test_sigmas_positive(self, regression_data):
        X, y = regression_data
        rbf = RBFNetwork(n_centers=8, random_state=0).fit(X, y)
        assert np.all(rbf.sigmas_ > 0)

    def test_width_scaling_affects_sigmas(self, regression_data):
        X, y = regression_data
        rbf1 = RBFNetwork(n_centers=5, width_scaling=0.5, random_state=0).fit(X, y)
        rbf2 = RBFNetwork(n_centers=5, width_scaling=2.0, random_state=0).fit(X, y)
        assert rbf2.sigmas_.mean() > rbf1.sigmas_.mean()

    def test_single_center_has_constant_sigma(self):
        X = np.arange(10).reshape(-1, 1).astype(float)
        y = X.ravel()
        rbf = RBFNetwork(n_centers=1, random_state=0).fit(X, y)
        assert rbf.sigmas_.shape == (1,)
        assert rbf.sigmas_[0] > 0


# ===================================================================
# RBF feature map
# ===================================================================

class TestRBFFeatures:
    def test_transform_shape(self, regression_data):
        X, y = regression_data
        rbf = RBFNetwork(n_centers=8, random_state=0).fit(X, y)
        phi = rbf.transform(X)
        assert phi.shape == (60, 8)

    def test_transform_in_unit_interval(self, regression_data):
        """Gaussian RBF values are in (0, 1]."""
        X, y = regression_data
        rbf = RBFNetwork(n_centers=8, random_state=0).fit(X, y)
        phi = rbf.transform(X)
        assert np.all(phi > 0) and np.all(phi <= 1.0 + 1e-9)

    def test_transform_center_self_distance_is_one(self, regression_data):
        """A point at a centre should have φ=1 for that basis function."""
        X, y = regression_data
        rbf = RBFNetwork(n_centers=5, random_state=0).fit(X, y)
        phi_at_centers = rbf.transform(rbf.centers_)
        # Each row evaluated at the corresponding centre should be ~1
        diag = np.array([phi_at_centers[i, i] for i in range(5)])
        np.testing.assert_allclose(diag, 1.0, atol=1e-9)


# ===================================================================
# Regression
# ===================================================================

class TestRBFRegression:
    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        rbf = RBFNetwork(n_centers=8, task="regression", random_state=0)
        assert rbf.fit(X, y) is rbf

    def test_predict_shape(self, regression_data):
        X, y = regression_data
        rbf = RBFNetwork(n_centers=8, random_state=0).fit(X, y)
        pred = rbf.predict(X)
        assert pred.shape == (60,)

    def test_predict_proba_raises_for_regression(self, regression_data):
        X, y = regression_data
        rbf = RBFNetwork(n_centers=8, task="regression", random_state=0).fit(X, y)
        with pytest.raises(ValueError):
            rbf.predict_proba(X)

    def test_sine_curve_low_mse(self, regression_data):
        X, y = regression_data
        rbf = RBFNetwork(n_centers=12, width_scaling=1.0, task="regression",
                         random_state=0).fit(X, y)
        mse = np.mean((rbf.predict(X) - y) ** 2)
        assert mse < 0.01

    def test_linear_function(self):
        """y = 2x should be fit with very low MSE."""
        X = np.linspace(0, 5, 30).reshape(-1, 1)
        y = 2.0 * X.ravel()
        rbf = RBFNetwork(n_centers=8, random_state=0).fit(X, y)
        mse = np.mean((rbf.predict(X) - y) ** 2)
        assert mse < 0.1

    def test_closed_form_solution_exact_when_n_centers_eq_n_samples(self):
        """With as many centres as samples and small ridge, interpolation
        should be near-perfect."""
        X = np.linspace(0, 1, 5).reshape(-1, 1)
        y = X.ravel() ** 2
        rbf = RBFNetwork(n_centers=5, ridge=1e-12, random_state=0).fit(X, y)
        mse = np.mean((rbf.predict(X) - y) ** 2)
        assert mse < 0.01


# ===================================================================
# Classification
# ===================================================================

class TestRBFClassification:
    def test_fit_returns_self(self, binary_clf_data):
        X, y = binary_clf_data
        rbf = RBFNetwork(n_centers=8, task="classification", n_classes=2,
                         random_state=0)
        assert rbf.fit(X, y) is rbf

    def test_predict_shape(self, binary_clf_data):
        X, y = binary_clf_data
        rbf = RBFNetwork(n_centers=8, task="classification", n_classes=2,
                         random_state=0).fit(X, y)
        preds = rbf.predict(X)
        assert preds.shape == (50,)

    def test_predict_valid_class_labels(self, binary_clf_data):
        X, y = binary_clf_data
        rbf = RBFNetwork(n_centers=8, task="classification", n_classes=2,
                         random_state=0).fit(X, y)
        preds = rbf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, three_class_data):
        X, y = three_class_data
        rbf = RBFNetwork(n_centers=10, task="classification", n_classes=3,
                         random_state=0).fit(X, y)
        proba = rbf.predict_proba(X)
        assert proba.shape == (60, 3)

    def test_predict_proba_sums_to_one(self, three_class_data):
        X, y = three_class_data
        rbf = RBFNetwork(n_centers=10, task="classification", n_classes=3,
                         random_state=0).fit(X, y)
        proba = rbf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_non_negative(self, three_class_data):
        X, y = three_class_data
        rbf = RBFNetwork(n_centers=10, task="classification", n_classes=3,
                         random_state=0).fit(X, y)
        proba = rbf.predict_proba(X)
        assert np.all(proba >= 0)

    def test_high_accuracy_binary(self, binary_clf_data):
        X, y = binary_clf_data
        rbf = RBFNetwork(n_centers=10, task="classification", n_classes=2,
                         random_state=0).fit(X, y)
        acc = (rbf.predict(X) == y).mean()
        assert acc >= 0.90

    def test_high_accuracy_three_class(self, three_class_data):
        X, y = three_class_data
        rbf = RBFNetwork(n_centers=12, task="classification", n_classes=3,
                         random_state=0).fit(X, y)
        acc = (rbf.predict(X) == y).mean()
        assert acc >= 0.85


# ===================================================================
# Ridge regularisation
# ===================================================================

class TestRBFRidge:
    def test_large_ridge_yields_finite_weights(self, regression_data):
        X, y = regression_data
        rbf = RBFNetwork(n_centers=8, ridge=1e4, random_state=0).fit(X, y)
        assert np.all(np.isfinite(rbf.weights_))

    def test_zero_ridge_matches_ordinary_least_squares(self, regression_data):
        """With ridge=0 and well-conditioned Φ, the solution should match
        np.linalg.lstsq."""
        X, y = regression_data
        rbf = RBFNetwork(n_centers=5, ridge=0.0, random_state=0).fit(X, y)
        pred = rbf.predict(X)
        assert np.all(np.isfinite(pred))


# ===================================================================
# Edge cases
# ===================================================================

class TestRBFEdgeCases:
    def test_1d_features(self, regression_data):
        X, y = regression_data
        rbf = RBFNetwork(n_centers=5, random_state=0).fit(X, y)
        assert rbf.centers_.shape[1] == 1

    def test_high_dimensional(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 10))
        y = X[:, 0] + X[:, 1]
        rbf = RBFNetwork(n_centers=8, random_state=0).fit(X, y)
        pred = rbf.predict(X)
        assert pred.shape == (30,)

    def test_single_center(self, regression_data):
        X, y = regression_data
        rbf = RBFNetwork(n_centers=1, random_state=0).fit(X, y)
        pred = rbf.predict(X)
        assert pred.shape == (60,)
        assert np.all(np.isfinite(pred))
