"""
Tests for mlscratch.neural.perceptron
Covers: SingleLayerPerceptron, MultiLayerPerceptron
"""

from __future__ import annotations

import numpy as np
import pytest

from mlscratch.neural.perceptron import SingleLayerPerceptron, MultiLayerPerceptron


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def binary_clf_data():
    rng = np.random.default_rng(0)
    X0 = rng.normal([0, 0], 0.5, (30, 2))
    X1 = rng.normal([4, 4], 0.5, (30, 2))
    X  = np.vstack([X0, X1])
    y  = np.array([0] * 30 + [1] * 30)
    return X, y


@pytest.fixture
def three_class_data():
    rng = np.random.default_rng(1)
    centres = [[0, 0], [6, 0], [3, 5]]
    X = np.vstack([rng.normal(c, 0.4, (20, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 20)
    return X, y


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(2)
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    y = 3.0 * X.ravel() + 1.0 + rng.normal(0, 0.1, 50)
    return X, y


# ===================================================================
# SingleLayerPerceptron — Basic API
# ===================================================================

class TestSLPBasic:
    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            SingleLayerPerceptron(input_size=2, task="invalid")

    def test_fit_returns_self(self, binary_clf_data):
        X, y = binary_clf_data
        model = SingleLayerPerceptron(input_size=2, epochs=10)
        assert model.fit(X, y) is model

    def test_weights_shape(self, binary_clf_data):
        X, y = binary_clf_data
        model = SingleLayerPerceptron(input_size=2, epochs=10).fit(X, y)
        assert model.weights_.shape == (2,)

    def test_bias_is_scalar(self, binary_clf_data):
        X, y = binary_clf_data
        model = SingleLayerPerceptron(input_size=2, epochs=10).fit(X, y)
        assert np.isscalar(model.bias_)

    def test_losses_recorded(self, binary_clf_data):
        X, y = binary_clf_data
        model = SingleLayerPerceptron(input_size=2, epochs=15).fit(X, y)
        assert len(model.losses_) == 15

    def test_predict_classification_binary(self, binary_clf_data):
        X, y = binary_clf_data
        model = SingleLayerPerceptron(input_size=2, epochs=10).fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_regression_continuous(self, regression_data):
        X, y = regression_data
        model = SingleLayerPerceptron(input_size=1, task="regression", epochs=10).fit(X, y)
        preds = model.predict(X)
        assert preds.dtype.kind == "f"

    def test_predict_proba_classification(self, binary_clf_data):
        X, y = binary_clf_data
        model = SingleLayerPerceptron(input_size=2, epochs=10).fit(X, y)
        proba = model.predict_proba(X)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_predict_proba_raises_for_regression(self, regression_data):
        X, y = regression_data
        model = SingleLayerPerceptron(input_size=1, task="regression", epochs=5).fit(X, y)
        with pytest.raises(ValueError):
            model.predict_proba(X)

    def test_reproducible_with_seed(self, binary_clf_data):
        X, y = binary_clf_data
        m1 = SingleLayerPerceptron(input_size=2, epochs=20, random_state=42).fit(X, y)
        m2 = SingleLayerPerceptron(input_size=2, epochs=20, random_state=42).fit(X, y)
        np.testing.assert_allclose(m1.weights_, m2.weights_)


# ===================================================================
# SingleLayerPerceptron — Correctness
# ===================================================================

class TestSLPCorrectness:
    def test_classification_accuracy_high(self, binary_clf_data):
        X, y = binary_clf_data
        model = SingleLayerPerceptron(
            input_size=2, learning_rate=0.1, epochs=300, random_state=0
        ).fit(X, y)
        acc = (model.predict(X) == y).mean()
        assert acc >= 0.9

    def test_regression_recovers_slope(self, regression_data):
        X, y = regression_data
        model = SingleLayerPerceptron(
            input_size=1, task="regression", learning_rate=0.05,
            epochs=500, random_state=0
        ).fit(X, y)
        assert abs(model.weights_[0] - 3.0) < 0.3
        assert abs(model.bias_ - 1.0) < 0.3

    def test_loss_decreases_over_training(self, binary_clf_data):
        X, y = binary_clf_data
        model = SingleLayerPerceptron(
            input_size=2, learning_rate=0.1, epochs=200, random_state=0
        ).fit(X, y)
        assert model.losses_[-1] < model.losses_[0]

    def test_classification_loss_is_bce(self, binary_clf_data):
        X, y = binary_clf_data
        model = SingleLayerPerceptron(input_size=2, epochs=1, random_state=0)
        model.fit(X, y)
        # BCE loss must be non-negative
        assert model.losses_[0] >= 0

    def test_regression_loss_is_mse(self, regression_data):
        X, y = regression_data
        model = SingleLayerPerceptron(
            input_size=1, task="regression", epochs=1, random_state=0
        )
        model.fit(X, y)
        assert model.losses_[0] >= 0


# ===================================================================
# SingleLayerPerceptron — Edge cases
# ===================================================================

class TestSLPEdgeCases:
    def test_single_sample(self):
        X = np.array([[1.0, 2.0]])
        y = np.array([1])
        model = SingleLayerPerceptron(input_size=2, epochs=5).fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (1,)

    def test_single_feature(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        model = SingleLayerPerceptron(input_size=1, epochs=50, random_state=0).fit(X, y)
        assert model.weights_.shape == (1,)

    def test_zero_epochs(self, binary_clf_data):
        X, y = binary_clf_data
        model = SingleLayerPerceptron(input_size=2, epochs=0).fit(X, y)
        assert len(model.losses_) == 0
        # Weights are still initialised
        assert model.weights_ is not None


# ===================================================================
# MultiLayerPerceptron — Basic API
# ===================================================================

class TestMLPBasic:
    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            MultiLayerPerceptron(task="invalid")

    def test_fit_returns_self(self, binary_clf_data):
        X, y = binary_clf_data
        model = MultiLayerPerceptron(hidden_sizes=[8], epochs=2, random_state=0)
        assert model.fit(X, y) is model

    def test_weights_built_after_fit(self, binary_clf_data):
        X, y = binary_clf_data
        model = MultiLayerPerceptron(hidden_sizes=[8, 4], epochs=2, random_state=0).fit(X, y)
        assert len(model.weights_) == 3   # input->h1, h1->h2, h2->output
        assert len(model.biases_)  == 3

    def test_weight_shapes(self, binary_clf_data):
        X, y = binary_clf_data
        model = MultiLayerPerceptron(
            hidden_sizes=[8, 4], task="classification", n_classes=2,
            epochs=1, random_state=0
        ).fit(X, y)
        assert model.weights_[0].shape == (2, 8)
        assert model.weights_[1].shape == (8, 4)
        assert model.weights_[2].shape == (4, 2)

    def test_losses_recorded(self, binary_clf_data):
        X, y = binary_clf_data
        model = MultiLayerPerceptron(hidden_sizes=[8], epochs=10, random_state=0).fit(X, y)
        assert len(model.losses_) == 10

    def test_predict_classification_shape(self, three_class_data):
        X, y = three_class_data
        model = MultiLayerPerceptron(
            hidden_sizes=[16], task="classification", n_classes=3,
            epochs=5, random_state=0
        ).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_proba_shape(self, three_class_data):
        X, y = three_class_data
        model = MultiLayerPerceptron(
            hidden_sizes=[16], task="classification", n_classes=3,
            epochs=5, random_state=0
        ).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_predict_proba_sums_to_one(self, three_class_data):
        X, y = three_class_data
        model = MultiLayerPerceptron(
            hidden_sizes=[16], task="classification", n_classes=3,
            epochs=5, random_state=0
        ).fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_raises_for_regression(self, regression_data):
        X, y = regression_data
        model = MultiLayerPerceptron(
            hidden_sizes=[8], task="regression", epochs=2, random_state=0
        ).fit(X, y)
        with pytest.raises(ValueError):
            model.predict_proba(X)

    def test_full_batch_mode(self, binary_clf_data):
        X, y = binary_clf_data
        model = MultiLayerPerceptron(
            hidden_sizes=[8], epochs=3, batch_size=None, random_state=0
        ).fit(X, y)
        assert len(model.losses_) == 3


# ===================================================================
# MultiLayerPerceptron — Correctness
# ===================================================================

class TestMLPCorrectness:
    def test_binary_classification_accuracy(self, binary_clf_data):
        X, y = binary_clf_data
        model = MultiLayerPerceptron(
            hidden_sizes=[16], task="classification", n_classes=2,
            learning_rate=0.05, epochs=100, batch_size=16, random_state=0
        ).fit(X, y)
        acc = (model.predict(X) == y).mean()
        assert acc >= 0.9

    def test_three_class_accuracy(self, three_class_data):
        X, y = three_class_data
        model = MultiLayerPerceptron(
            hidden_sizes=[16], task="classification", n_classes=3,
            learning_rate=0.05, epochs=150, batch_size=16, random_state=0
        ).fit(X, y)
        acc = (model.predict(X) == y).mean()
        assert acc >= 0.85

    def test_regression_low_mse(self, regression_data):
        X, y = regression_data
        model = MultiLayerPerceptron(
            hidden_sizes=[16], task="regression",
            learning_rate=0.01, epochs=200, batch_size=16, random_state=0
        ).fit(X, y)
        mse = np.mean((model.predict(X) - y) ** 2)
        assert mse < 0.5

    def test_loss_decreases_during_training(self, binary_clf_data):
        X, y = binary_clf_data
        model = MultiLayerPerceptron(
            hidden_sizes=[16], epochs=50, batch_size=16,
            learning_rate=0.05, random_state=0
        ).fit(X, y)
        assert model.losses_[-1] < model.losses_[0]

    def test_xor_nonlinear_separability(self):
        """MLP must solve XOR — impossible for a single linear layer."""
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y = np.array([0,1,1,0])
        model = MultiLayerPerceptron(
            hidden_sizes=[8], task="classification", n_classes=2,
            learning_rate=0.1, epochs=500, batch_size=4, random_state=0
        ).fit(X, y)
        preds = model.predict(X)
        assert (preds == y).all(), f"XOR not solved: {preds} vs {y}"


# ===================================================================
# MultiLayerPerceptron — Edge cases
# ===================================================================

class TestMLPEdgeCases:
    def test_single_hidden_unit(self, binary_clf_data):
        X, y = binary_clf_data
        model = MultiLayerPerceptron(
            hidden_sizes=[1], epochs=5, batch_size=8, random_state=0
        ).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_deep_network(self, binary_clf_data):
        X, y = binary_clf_data
        model = MultiLayerPerceptron(
            hidden_sizes=[16, 8, 4], epochs=5, batch_size=8, random_state=0
        ).fit(X, y)
        assert len(model.weights_) == 4   # 3 hidden + 1 output

    def test_batch_size_larger_than_dataset(self, binary_clf_data):
        X, y = binary_clf_data
        model = MultiLayerPerceptron(
            hidden_sizes=[8], epochs=3, batch_size=1000, random_state=0
        ).fit(X, y)
        assert len(model.losses_) == 3

    def test_no_nan_after_training(self, binary_clf_data):
        X, y = binary_clf_data
        model = MultiLayerPerceptron(
            hidden_sizes=[16], epochs=20, batch_size=8, random_state=0
        ).fit(X, y)
        proba = model.predict_proba(X)
        assert not np.any(np.isnan(proba))
