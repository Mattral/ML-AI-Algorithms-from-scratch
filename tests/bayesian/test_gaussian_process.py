"""
Tests for mlscratch.bayesian.gaussian_process.GaussianProcessRegressor
and the four kernel classes.
"""

import numpy as np
import pytest
from mlscratch.bayesian.gaussian_process import (
    GaussianProcessRegressor,
    RBFKernel,
    Matern52Kernel,
    LinearKernel,
    PeriodicKernel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_1d():
    """y = sin(x) with small noise."""
    rng = np.random.default_rng(0)
    X = np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)
    y = np.sin(X.ravel()) + rng.normal(0, 0.05, 30)
    return X, y


@pytest.fixture
def linear_1d():
    rng = np.random.default_rng(1)
    X = np.linspace(-1, 1, 20).reshape(-1, 1)
    y = 2.0 * X.ravel() + rng.normal(0, 0.1, 20)
    return X, y


@pytest.fixture
def test_points():
    return np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------

class TestKernels:
    """Kernels must be symmetric, positive semi-definite, and correct shaped."""

    @pytest.mark.parametrize("kernel", [
        RBFKernel(),
        Matern52Kernel(),
        LinearKernel(),
        PeriodicKernel(),
    ])
    def test_kernel_shape(self, kernel):
        X = np.random.default_rng(0).standard_normal((5, 1))
        K = kernel(X, X)
        assert K.shape == (5, 5)

    @pytest.mark.parametrize("kernel", [
        RBFKernel(),
        Matern52Kernel(),
        PeriodicKernel(),
    ])
    def test_kernel_symmetry(self, kernel):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((6, 2))
        K = kernel(X, X)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    @pytest.mark.parametrize("kernel", [
        RBFKernel(),
        Matern52Kernel(),
        PeriodicKernel(),
    ])
    def test_kernel_psd(self, kernel):
        """K(X,X) must be positive semi-definite."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((8, 1))
        K = kernel(X, X)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-8)

    def test_rbf_diagonal_equals_signal_variance_squared(self):
        k = RBFKernel(signal_variance=2.0)
        X = np.array([[0.0], [1.0], [2.0]])
        K = k(X, X)
        np.testing.assert_allclose(np.diag(K), 4.0, atol=1e-10)

    def test_rbf_far_points_near_zero(self):
        k = RBFKernel(length_scale=0.1)
        X1 = np.array([[0.0]])
        X2 = np.array([[100.0]])
        assert k(X1, X2)[0, 0] < 1e-6

    def test_rbf_same_point_max_covariance(self):
        k = RBFKernel(length_scale=1.0, signal_variance=1.0)
        X = np.array([[1.0]])
        np.testing.assert_allclose(k(X, X)[0, 0], 1.0, atol=1e-10)

    def test_matern52_diagonal(self):
        k = Matern52Kernel(signal_variance=3.0)
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        K = k(X, X)
        np.testing.assert_allclose(np.diag(K), 9.0, atol=1e-10)

    def test_cross_kernel_rectangle_shape(self):
        k = RBFKernel()
        X1 = np.random.default_rng(0).standard_normal((4, 2))
        X2 = np.random.default_rng(1).standard_normal((7, 2))
        K = k(X1, X2)
        assert K.shape == (4, 7)

    def test_periodic_kernel_periodic_behaviour(self):
        k = PeriodicKernel(period=2 * np.pi, length_scale=1.0)
        X1 = np.array([[0.0]])
        X2 = np.array([[2 * np.pi]])   # one period away
        k_val = k(X1, X2)[0, 0]
        k_self = k(X1, X1)[0, 0]
        np.testing.assert_allclose(k_val, k_self, atol=1e-6)


# ---------------------------------------------------------------------------
# GPR basic API
# ---------------------------------------------------------------------------

class TestGPRBasic:
    def test_fit_returns_self(self, sine_1d):
        X, y = sine_1d
        gp = GaussianProcessRegressor()
        assert gp.fit(X, y) is gp

    def test_predict_shape(self, sine_1d, test_points):
        X, y = sine_1d
        mean = GaussianProcessRegressor().fit(X, y).predict(test_points)
        assert mean.shape == (len(test_points),)

    def test_predict_with_std_shapes(self, sine_1d, test_points):
        X, y = sine_1d
        mean, std = GaussianProcessRegressor().fit(X, y).predict(
            test_points, return_std=True
        )
        assert mean.shape == (len(test_points),)
        assert std.shape == (len(test_points),)

    def test_predictive_std_non_negative(self, sine_1d, test_points):
        X, y = sine_1d
        _, std = GaussianProcessRegressor().fit(X, y).predict(
            test_points, return_std=True
        )
        assert np.all(std >= 0)

    def test_alpha_shape(self, sine_1d):
        X, y = sine_1d
        gp = GaussianProcessRegressor().fit(X, y)
        assert gp.alpha_.shape == (len(y),)

    def test_1d_input_auto_reshape(self):
        X = np.linspace(0, 1, 10)  # 1-D input
        y = np.sin(X)
        gp = GaussianProcessRegressor().fit(X, y)
        preds = gp.predict(X)
        assert preds.shape == (10,)


# ---------------------------------------------------------------------------
# GPR correctness
# ---------------------------------------------------------------------------

class TestGPRCorrectness:
    def test_interpolation_at_training_points(self):
        """With near-zero noise, GPR should nearly interpolate training data."""
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = np.sin(X.ravel())
        gp = GaussianProcessRegressor(
            kernel=RBFKernel(length_scale=0.3), noise_variance=1e-8
        ).fit(X, y)
        pred = gp.predict(X)
        np.testing.assert_allclose(pred, y, atol=1e-4)

    def test_uncertainty_higher_far_from_data(self):
        """Predictive uncertainty should be larger far from training data."""
        X_train = np.linspace(0, 1, 10).reshape(-1, 1)
        y_train = np.sin(X_train.ravel())
        gp = GaussianProcessRegressor(
            kernel=RBFKernel(length_scale=0.3), noise_variance=1e-6
        ).fit(X_train, y_train)
        X_near = np.array([[0.5]])
        X_far  = np.array([[10.0]])
        _, std_near = gp.predict(X_near, return_std=True)
        _, std_far  = gp.predict(X_far, return_std=True)
        assert std_far[0] > std_near[0]

    def test_sine_curve_low_mse(self, sine_1d, test_points):
        X, y = sine_1d
        gp = GaussianProcessRegressor(
            kernel=RBFKernel(length_scale=1.0), noise_variance=0.01
        ).fit(X, y)
        mean = gp.predict(test_points)
        true = np.sin(test_points.ravel())
        mse = np.mean((mean - true) ** 2)
        assert mse < 0.1

    def test_linear_kernel_recovers_linear_trend(self, linear_1d):
        X, y = linear_1d
        gp = GaussianProcessRegressor(
            kernel=LinearKernel(signal_variance=2.0), noise_variance=0.05
        ).fit(X, y)
        X_test = np.array([[0.0], [0.5]])
        preds = gp.predict(X_test)
        # Should follow y ≈ 2x
        assert abs(preds[1] - preds[0]) > 0.2  # increasing trend

    @pytest.mark.parametrize("kernel", [
        RBFKernel(), Matern52Kernel(), PeriodicKernel(period=2 * np.pi)
    ])
    def test_all_kernels_fit_and_predict(self, sine_1d, test_points, kernel):
        X, y = sine_1d
        gp = GaussianProcessRegressor(kernel=kernel).fit(X, y)
        mean = gp.predict(test_points)
        assert mean.shape == (len(test_points),)
        assert not np.any(np.isnan(mean))


# ---------------------------------------------------------------------------
# Posterior sampling
# ---------------------------------------------------------------------------

class TestGPRSampling:
    def test_sample_shape(self, sine_1d):
        X, y = sine_1d
        gp = GaussianProcessRegressor().fit(X, y)
        X_test = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1)
        samples = gp.sample_posterior(X_test, n_samples=5, random_state=0)
        assert samples.shape == (5, 20)

    def test_samples_finite(self, sine_1d):
        X, y = sine_1d
        gp = GaussianProcessRegressor().fit(X, y)
        X_test = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        samples = gp.sample_posterior(X_test, n_samples=3, random_state=1)
        assert not np.any(np.isnan(samples))
        assert not np.any(np.isinf(samples))

    def test_sample_reproducibility(self, sine_1d):
        X, y = sine_1d
        gp = GaussianProcessRegressor().fit(X, y)
        X_test = np.linspace(0, 1, 5).reshape(-1, 1)
        s1 = gp.sample_posterior(X_test, n_samples=3, random_state=42)
        s2 = gp.sample_posterior(X_test, n_samples=3, random_state=42)
        np.testing.assert_allclose(s1, s2, atol=1e-12)
