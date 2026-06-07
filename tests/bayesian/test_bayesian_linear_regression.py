"""
Tests for mlscratch.bayesian.bayesian_linear_regression.BayesianLinearRegression
"""

import numpy as np
import pytest
from mlscratch.bayesian.bayesian_linear_regression import BayesianLinearRegression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_1d():
    """y = 3x + 1.5 + ε"""
    rng = np.random.default_rng(42)
    X = np.linspace(-2, 2, 60).reshape(-1, 1)
    y = 3.0 * X.ravel() + 1.5 + rng.normal(0, 0.3, 60)
    return X, y


@pytest.fixture
def linear_2d():
    """y = 2x1 - x2 + 0.5 + ε"""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((80, 2))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.5 + rng.normal(0, 0.2, 80)
    return X, y


@pytest.fixture
def noise_free():
    """Exact linear relationship; posterior should concentrate sharply."""
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    y = 5.0 * X.ravel() + 2.0
    return X, y


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestBLRBasic:
    def test_fit_returns_self(self, linear_1d):
        X, y = linear_1d
        m = BayesianLinearRegression()
        assert m.fit(X, y) is m

    def test_posterior_mean_shape_with_intercept(self, linear_1d):
        X, y = linear_1d
        m = BayesianLinearRegression(fit_intercept=True).fit(X, y)
        # 1 feature + 1 bias
        assert m.m_N_.shape == (2,)

    def test_posterior_mean_shape_without_intercept(self, linear_1d):
        X, y = linear_1d
        m = BayesianLinearRegression(fit_intercept=False).fit(X, y)
        assert m.m_N_.shape == (1,)

    def test_posterior_cov_is_symmetric(self, linear_1d):
        X, y = linear_1d
        m = BayesianLinearRegression().fit(X, y)
        np.testing.assert_allclose(m.S_N_, m.S_N_.T, atol=1e-10)

    def test_posterior_cov_is_positive_definite(self, linear_1d):
        X, y = linear_1d
        m = BayesianLinearRegression().fit(X, y)
        eigvals = np.linalg.eigvalsh(m.S_N_)
        assert np.all(eigvals > 0)

    def test_predict_shape(self, linear_1d):
        X, y = linear_1d
        m = BayesianLinearRegression().fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_with_std_shapes(self, linear_1d):
        X, y = linear_1d
        m = BayesianLinearRegression().fit(X, y)
        mean, std = m.predict(X, return_std=True)
        assert mean.shape == (len(X),)
        assert std.shape == (len(X),)

    def test_predictive_std_positive(self, linear_1d):
        X, y = linear_1d
        m = BayesianLinearRegression().fit(X, y)
        _, std = m.predict(X, return_std=True)
        assert np.all(std > 0)

    def test_coef_shape(self, linear_2d):
        X, y = linear_2d
        m = BayesianLinearRegression(fit_intercept=True).fit(X, y)
        assert m.coef_.shape == (2,)

    def test_intercept_is_scalar(self, linear_2d):
        X, y = linear_2d
        m = BayesianLinearRegression(fit_intercept=True).fit(X, y)
        assert np.isscalar(m.intercept_)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestBLRCorrectness:
    def test_posterior_mean_recovers_true_slope(self, linear_1d):
        """Posterior mean should be close to true slope=3, intercept=1.5."""
        X, y = linear_1d
        m = BayesianLinearRegression(alpha=1e-3, beta=1.0 / 0.09).fit(X, y)
        assert abs(m.coef_[0] - 3.0) < 0.3
        assert abs(m.intercept_ - 1.5) < 0.3

    def test_predictive_mean_low_mse(self, linear_1d):
        X, y = linear_1d
        m = BayesianLinearRegression(alpha=1e-3, beta=10.0).fit(X, y)
        mse = np.mean((m.predict(X) - y) ** 2)
        assert mse < 0.5

    def test_uncertainty_wider_outside_training(self, linear_1d):
        """Predictive std should be wider for extrapolation points."""
        X, y = linear_1d
        m = BayesianLinearRegression().fit(X, y)
        X_train = X
        X_far = np.array([[100.0]])
        _, std_train = m.predict(X_train, return_std=True)
        _, std_far = m.predict(X_far, return_std=True)
        assert std_far[0] > std_train.mean()

    def test_noise_free_tight_posterior(self, noise_free):
        """With near-zero noise the posterior should be very tight."""
        X, y = noise_free
        m = BayesianLinearRegression(alpha=1e-6, beta=1e6).fit(X, y)
        _, std = m.predict(X, return_std=True)
        assert std.max() < 0.1

    def test_2d_coefficient_recovery(self, linear_2d):
        X, y = linear_2d
        m = BayesianLinearRegression(alpha=1e-4, beta=1.0 / 0.04).fit(X, y)
        assert abs(m.coef_[0] - 2.0) < 0.3
        assert abs(m.coef_[1] - (-1.0)) < 0.3

    def test_predict_without_std_equals_mean(self, linear_1d):
        X, y = linear_1d
        m = BayesianLinearRegression().fit(X, y)
        mean_only = m.predict(X, return_std=False)
        mean_with, _ = m.predict(X, return_std=True)
        np.testing.assert_allclose(mean_only, mean_with, atol=1e-12)


# ---------------------------------------------------------------------------
# Hyperparameter optimisation
# ---------------------------------------------------------------------------

class TestBLRHyperparams:
    def test_evidence_optimisation_runs(self, linear_1d):
        X, y = linear_1d
        m = BayesianLinearRegression(
            optimize_hyperparams=True, max_iter=50
        ).fit(X, y)
        assert m.alpha > 0
        assert m.beta > 0

    def test_evidence_optimisation_improves_fit(self, linear_1d):
        """After evidence approx, MSE should not be catastrophically worse
        than a well-tuned fixed model."""
        X, y = linear_1d
        m_fixed = BayesianLinearRegression(alpha=1.0, beta=10.0).fit(X, y)
        m_opt = BayesianLinearRegression(
            optimize_hyperparams=True, max_iter=100
        ).fit(X, y)
        mse_fixed = np.mean((m_fixed.predict(X) - y) ** 2)
        mse_opt = np.mean((m_opt.predict(X) - y) ** 2)
        # Optimised should be within 2× of fixed
        assert mse_opt < mse_fixed * 2.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestBLREdgeCases:
    def test_single_feature_single_sample(self):
        X = np.array([[1.0]])
        y = np.array([2.0])
        m = BayesianLinearRegression().fit(X, y)
        pred = m.predict(X)
        assert pred.shape == (1,)

    def test_no_intercept_zero_coef_for_zero_data(self):
        X = np.zeros((10, 2))
        y = np.zeros(10)
        m = BayesianLinearRegression(fit_intercept=False).fit(X, y)
        np.testing.assert_allclose(m.coef_, 0.0, atol=1e-6)

    def test_predict_new_points(self, linear_1d):
        X, y = linear_1d
        m = BayesianLinearRegression().fit(X, y)
        X_new = np.array([[0.0], [1.0], [-1.0]])
        preds = m.predict(X_new)
        assert preds.shape == (3,)
