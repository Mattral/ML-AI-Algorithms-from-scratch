"""
Tests for mlscratch.bayesian.bayesian_nn.BayesianNeuralNetwork

Note: BNNs are stochastic and computationally expensive.  Tests focus on
contracts, output shapes, uncertainty properties, and gross sanity checks
on tiny problems with few epochs.  We deliberately keep datasets tiny
and epoch counts low so the test suite runs in seconds.
"""

import numpy as np
import pytest
from mlscratch.bayesian.bayesian_nn import (
    BayesianNeuralNetwork,
    BayesianLayer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def xor_data():
    """XOR — a minimal non-linear binary classification problem."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y


@pytest.fixture
def linearly_separable():
    rng = np.random.default_rng(42)
    X0 = rng.normal([-2, 0], 0.4, (20, 2))
    X1 = rng.normal([ 2, 0], 0.4, (20, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 20 + [1] * 20)
    return X, y


@pytest.fixture
def three_class():
    rng = np.random.default_rng(7)
    centres = [[0, 0], [5, 0], [2.5, 4]]
    X = np.vstack([rng.normal(c, 0.5, (15, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 15)
    return X, y


# ---------------------------------------------------------------------------
# BayesianLayer unit tests
# ---------------------------------------------------------------------------

class TestBayesianLayer:
    def test_forward_output_shape(self):
        rng = np.random.default_rng(0)
        layer = BayesianLayer(4, 3, rng=rng)
        X = rng.standard_normal((10, 4))
        out = layer.forward(X)
        assert out.shape == (10, 3)

    def test_weights_sampled_differ_between_calls(self):
        rng = np.random.default_rng(0)
        layer = BayesianLayer(3, 2, rng=rng)
        X = np.ones((5, 3))
        _ = layer.forward(X)
        W1 = layer.W_sample.copy()
        _ = layer.forward(X)
        W2 = layer.W_sample.copy()
        assert not np.allclose(W1, W2)

    def test_kl_divergence_positive(self):
        rng = np.random.default_rng(0)
        layer = BayesianLayer(4, 3, prior_std=1.0, rng=rng)
        kl = layer.kl_divergence()
        assert kl >= 0.0

    def test_kl_zero_when_posterior_equals_prior(self):
        """If mu=0 and sigma=prior_std, KL should be ~0."""
        rng = np.random.default_rng(0)
        layer = BayesianLayer(2, 2, prior_std=1.0, rng=rng)
        layer.mu_W = np.zeros_like(layer.mu_W)
        layer.mu_b = np.zeros_like(layer.mu_b)
        layer.log_sigma_W = np.zeros_like(layer.log_sigma_W)  # sigma = exp(0) = 1
        layer.log_sigma_b = np.zeros_like(layer.log_sigma_b)
        np.testing.assert_allclose(layer.kl_divergence(), 0.0, atol=1e-6)

    def test_kl_increases_with_larger_mean(self):
        rng = np.random.default_rng(0)
        layer_small = BayesianLayer(3, 2, rng=np.random.default_rng(0))
        layer_large = BayesianLayer(3, 2, rng=np.random.default_rng(0))
        layer_large.mu_W = layer_large.mu_W + 10.0
        assert layer_large.kl_divergence() > layer_small.kl_divergence()

    def test_sigma_W_always_positive(self):
        rng = np.random.default_rng(0)
        layer = BayesianLayer(4, 3, rng=rng)
        assert np.all(layer.sigma_W > 0)
        assert np.all(layer.sigma_b > 0)


# ---------------------------------------------------------------------------
# BNN basic API — binary
# ---------------------------------------------------------------------------

class TestBNNBinaryBasic:
    def test_fit_returns_self(self, linearly_separable):
        X, y = linearly_separable
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[8], task="binary",
            n_epochs=3, batch_size=16, random_state=0
        )
        assert bnn.fit(X, y) is bnn

    def test_layers_built_after_fit(self, linearly_separable):
        X, y = linearly_separable
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[8], task="binary", n_epochs=2, random_state=0
        ).fit(X, y)
        assert len(bnn.layers_) > 0

    def test_losses_recorded(self, linearly_separable):
        X, y = linearly_separable
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[8], task="binary", n_epochs=5, random_state=0
        ).fit(X, y)
        assert len(bnn.losses_) == 5

    def test_predict_shape(self, linearly_separable):
        X, y = linearly_separable
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[8], task="binary", n_epochs=3, random_state=0
        ).fit(X, y)
        preds = bnn.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_proba_shape(self, linearly_separable):
        X, y = linearly_separable
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[8], task="binary", n_epochs=3, random_state=0
        ).fit(X, y)
        proba = bnn.predict_proba(X, n_samples=10)
        assert proba.shape == (len(X),)

    def test_predict_proba_in_unit_interval(self, linearly_separable):
        X, y = linearly_separable
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[8], task="binary", n_epochs=3, random_state=0
        ).fit(X, y)
        proba = bnn.predict_proba(X, n_samples=5)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_binary_labels(self, linearly_separable):
        X, y = linearly_separable
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[8], task="binary", n_epochs=3, random_state=0
        ).fit(X, y)
        preds = bnn.predict(X)
        assert set(preds).issubset({0, 1})

    def test_no_nan_in_predictions(self, linearly_separable):
        X, y = linearly_separable
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[4], task="binary", n_epochs=3, random_state=0
        ).fit(X, y)
        assert not np.any(np.isnan(bnn.predict_proba(X, n_samples=5)))


# ---------------------------------------------------------------------------
# BNN basic API — multiclass
# ---------------------------------------------------------------------------

class TestBNNMulticlassBasic:
    def test_predict_shape_multiclass(self, three_class):
        X, y = three_class
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[8], task="multiclass", n_classes=3,
            n_epochs=3, random_state=0
        ).fit(X, y)
        preds = bnn.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_proba_shape_multiclass(self, three_class):
        X, y = three_class
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[8], task="multiclass", n_classes=3,
            n_epochs=3, random_state=0
        ).fit(X, y)
        proba = bnn.predict_proba(X, n_samples=5)
        assert proba.shape == (len(X), 3)

    def test_predict_proba_sums_to_one(self, three_class):
        X, y = three_class
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[8], task="multiclass", n_classes=3,
            n_epochs=3, random_state=0
        ).fit(X, y)
        proba = bnn.predict_proba(X, n_samples=5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_labels_in_valid_range(self, three_class):
        X, y = three_class
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[8], task="multiclass", n_classes=3,
            n_epochs=3, random_state=0
        ).fit(X, y)
        preds = bnn.predict(X)
        assert set(preds).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# Uncertainty / Bayesian property tests
# ---------------------------------------------------------------------------

class TestBNNUncertainty:
    def test_mc_samples_variance_nonzero(self, linearly_separable):
        """Different MC samples must produce different predictions (stochastic)."""
        X, y = linearly_separable
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[8], task="binary", n_epochs=2, random_state=0
        ).fit(X, y)
        # Collect individual sample predictions
        samples = np.stack([
            bnn._forward(X[:5]) for _ in range(20)
        ])
        # Variance across samples should be > 0 for at least some points
        assert samples.var(axis=0).max() > 0

    def test_more_mc_samples_smoother_proba(self, linearly_separable):
        """Predictive probability variance should decrease with more MC samples."""
        X, y = linearly_separable
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[8], task="binary", n_epochs=3, random_state=0
        ).fit(X, y)
        # Variance across repeated predict_proba calls
        p_small = np.stack([bnn.predict_proba(X[:5], n_samples=2) for _ in range(10)])
        p_large = np.stack([bnn.predict_proba(X[:5], n_samples=50) for _ in range(10)])
        assert p_large.var(axis=0).mean() <= p_small.var(axis=0).mean() + 0.1


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestBNNArchitecture:
    def test_deep_network(self, linearly_separable):
        X, y = linearly_separable
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[16, 8, 4], task="binary", n_epochs=2, random_state=0
        ).fit(X, y)
        # 3 hidden layers + 1 output layer = 4 layers
        assert len(bnn.layers_) == 4

    def test_single_hidden_unit(self, linearly_separable):
        X, y = linearly_separable
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[1], task="binary", n_epochs=2, random_state=0
        ).fit(X, y)
        preds = bnn.predict(X)
        assert set(preds).issubset({0, 1})

    def test_full_batch_mode(self, linearly_separable):
        X, y = linearly_separable
        bnn = BayesianNeuralNetwork(
            hidden_sizes=[4], task="binary", n_epochs=3,
            batch_size=None, random_state=0
        ).fit(X, y)
        assert len(bnn.losses_) == 3
