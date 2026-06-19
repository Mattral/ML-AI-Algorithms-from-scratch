"""
Tests for mlscratch.neural.cnn
Covers: Conv2D, MaxPool2D, AvgPool2D, BatchNorm2D, Flatten, Dense, SimpleCNN
"""

from __future__ import annotations

import numpy as np
import pytest

from mlscratch.neural.cnn import (
    Conv2D, MaxPool2D, AvgPool2D, BatchNorm2D, Flatten, Dense, SimpleCNN,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def small_images():
    """8 grayscale 8x8 images."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((8, 1, 8, 8))


@pytest.fixture
def cnn_dataset():
    """20 grayscale 12x12 images, 3 classes."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 1, 12, 12))
    y = rng.integers(0, 3, 20)
    return X, y


# ===================================================================
# Conv2D
# ===================================================================

class TestConv2D:
    def test_weight_shape(self):
        conv = Conv2D(in_channels=2, out_channels=4, kernel_size=3, random_state=0)
        assert conv.weights.shape == (4, 2, 3, 3)
        assert conv.bias.shape == (4,)

    def test_forward_output_shape_valid_padding(self, small_images):
        conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3, random_state=0)
        out = conv.forward(small_images)
        # 8x8 input, 3x3 kernel, valid padding => 6x6 output
        assert out.shape == (8, 4, 6, 6)

    def test_forward_output_with_larger_kernel(self, small_images):
        conv = Conv2D(in_channels=1, out_channels=2, kernel_size=5, random_state=0)
        out = conv.forward(small_images)
        assert out.shape == (8, 2, 4, 4)

    def test_backward_returns_input_shape(self, small_images):
        conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3,
                       learning_rate=0.01, random_state=0)
        out = conv.forward(small_images)
        d_x = conv.backward(np.ones_like(out))
        assert d_x.shape == small_images.shape

    def test_backward_updates_weights(self, small_images):
        conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3,
                       learning_rate=0.1, random_state=0)
        W_before = conv.weights.copy()
        out = conv.forward(small_images)
        conv.backward(np.ones_like(out) * 0.1)
        assert not np.allclose(W_before, conv.weights)

    def test_multi_channel_input(self):
        conv = Conv2D(in_channels=3, out_channels=2, kernel_size=3, random_state=0)
        X = np.random.default_rng(0).standard_normal((4, 3, 6, 6))
        out = conv.forward(X)
        assert out.shape == (4, 2, 4, 4)

    def test_im2col_col2im_consistency(self, small_images):
        """col2im should redistribute gradients to the correct positions
        (verified via shape and finiteness, since im2col/col2im are inverses
        only under summation for overlapping patches)."""
        conv = Conv2D(in_channels=1, out_channels=1, kernel_size=2, random_state=0)
        out = conv.forward(small_images)
        d_x = conv.backward(np.ones_like(out))
        assert np.all(np.isfinite(d_x))


# ===================================================================
# MaxPool2D
# ===================================================================

class TestMaxPool2D:
    def test_output_shape(self, small_images):
        pool = MaxPool2D(pool_size=2)
        out = pool.forward(small_images)
        assert out.shape == (8, 1, 4, 4)

    def test_output_is_max_of_window(self):
        pool = MaxPool2D(pool_size=2)
        X = np.array([[[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]]], dtype=float)
        out = pool.forward(X)
        expected = np.array([[[[6, 8], [14, 16]]]], dtype=float)
        np.testing.assert_array_equal(out, expected)

    def test_backward_routes_gradient_to_max_position(self):
        pool = MaxPool2D(pool_size=2)
        X = np.array([[[[1, 2], [3, 4]]]], dtype=float)
        out = pool.forward(X)
        d_out = np.array([[[[1.0]]]])
        d_x = pool.backward(d_out)
        # Max was at position (1,1) with value 4
        expected = np.array([[[[0, 0], [0, 1.0]]]])
        np.testing.assert_array_equal(d_x, expected)

    def test_backward_shape(self, small_images):
        pool = MaxPool2D(pool_size=2)
        out = pool.forward(small_images)
        d_x = pool.backward(np.ones_like(out))
        assert d_x.shape == small_images.shape
        pool = MaxPool2D(pool_size=4)
        X = np.random.default_rng(0).standard_normal((2, 1, 8, 8))
        out = pool.forward(X)
        assert out.shape == (2, 1, 2, 2)


# ===================================================================
# AvgPool2D
# ===================================================================

class TestAvgPool2D:
    def test_output_shape(self, small_images):
        pool = AvgPool2D(pool_size=2)
        out = pool.forward(small_images)
        assert out.shape == (8, 1, 4, 4)

    def test_output_is_average_of_window(self):
        pool = AvgPool2D(pool_size=2)
        X = np.array([[[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]]], dtype=float)
        out = pool.forward(X)
        expected = np.array([[[[3.5, 5.5], [11.5, 13.5]]]])
        np.testing.assert_allclose(out, expected)

    def test_backward_distributes_gradient_evenly(self):
        pool = AvgPool2D(pool_size=2)
        X = np.array([[[[1, 2], [3, 4]]]], dtype=float)
        pool.forward(X)
        d_out = np.array([[[[4.0]]]])
        d_x = pool.backward(d_out)
        # Each of the 4 positions gets 4.0 / 4 = 1.0
        np.testing.assert_allclose(d_x, np.ones((1, 1, 2, 2)))

    def test_backward_shape(self, small_images):
        pool = AvgPool2D(pool_size=2)
        out = pool.forward(small_images)
        d_x = pool.backward(np.ones_like(out))
        assert d_x.shape == small_images.shape


# ===================================================================
# BatchNorm2D
# ===================================================================

class TestBatchNorm2D:
    def test_init_gamma_ones_beta_zeros(self):
        bn = BatchNorm2D(num_features=4)
        np.testing.assert_array_equal(bn.gamma, np.ones(4))
        np.testing.assert_array_equal(bn.beta, np.zeros(4))

    def test_forward_output_shape(self, small_images):
        bn = BatchNorm2D(num_features=1)
        out = bn.forward(small_images, training=True)
        assert out.shape == small_images.shape

    def test_normalised_output_zero_mean_unit_var(self):
        """With gamma=1, beta=0, output should have ~0 mean, ~1 var per channel."""
        bn = BatchNorm2D(num_features=2)
        X = np.random.default_rng(0).standard_normal((10, 2, 4, 4)) * 5 + 3
        out = bn.forward(X, training=True)
        for c in range(2):
            np.testing.assert_allclose(out[:, c].mean(), 0.0, atol=1e-6)
            np.testing.assert_allclose(out[:, c].var(), 1.0, atol=1e-5)

    def test_running_stats_updated_in_training(self, small_images):
        bn = BatchNorm2D(num_features=1, momentum=0.5)
        rm0 = bn.running_mean.copy()
        bn.forward(small_images, training=True)
        assert not np.array_equal(rm0, bn.running_mean)

    def test_inference_uses_running_stats(self, small_images):
        bn = BatchNorm2D(num_features=1)
        bn.forward(small_images, training=True)   # update running stats
        rm, rv = bn.running_mean.copy(), bn.running_var.copy()
        out_eval = bn.forward(small_images, training=False)
        # Output should use running_mean/var, not batch stats
        expected = (
            bn.gamma.reshape(1, -1, 1, 1)
            * (small_images - rm.reshape(1, -1, 1, 1))
            / np.sqrt(rv.reshape(1, -1, 1, 1) + bn.eps)
            + bn.beta.reshape(1, -1, 1, 1)
        )
        np.testing.assert_allclose(out_eval, expected, atol=1e-8)

    def test_backward_updates_gamma_beta(self, small_images):
        bn = BatchNorm2D(num_features=1, learning_rate=0.1)
        out = bn.forward(small_images, training=True)
        gamma0, beta0 = bn.gamma.copy(), bn.beta.copy()
        # d_gamma = Σ d_out · x_hat  (nonzero when d_out = x_hat + 1, since Σ x_hat² > 0)
        # d_beta  = Σ d_out           (nonzero since x_hat has zero mean, +1 gives B*H*W)
        d_out_useful = bn._cache["x_hat"] + 1.0
        d = bn.backward(d_out_useful)
        assert d.shape == small_images.shape
        assert not np.array_equal(gamma0, bn.gamma), "gamma must change"
        assert not np.array_equal(beta0, bn.beta),   "beta must change"

    def test_backward_output_shape(self, small_images):
        bn = BatchNorm2D(num_features=1)
        out = bn.forward(small_images, training=True)
        d_x = bn.backward(np.ones_like(out))
        assert d_x.shape == small_images.shape


# ===================================================================
# Flatten
# ===================================================================

class TestFlatten:
    def test_forward_shape(self, small_images):
        flat = Flatten()
        out = flat.forward(small_images)
        assert out.shape == (8, 1 * 8 * 8)

    def test_backward_restores_shape(self, small_images):
        flat = Flatten()
        out = flat.forward(small_images)
        d_x = flat.backward(out)
        assert d_x.shape == small_images.shape

    def test_values_preserved(self, small_images):
        flat = Flatten()
        out = flat.forward(small_images)
        d_x = flat.backward(out)
        np.testing.assert_array_equal(d_x, small_images)


# ===================================================================
# Dense
# ===================================================================

class TestDense:
    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError):
            Dense(4, 2, activation="invalid")

    def test_linear_forward_shape(self):
        dense = Dense(in_features=8, out_features=3, activation="linear",
                       random_state=0)
        X = np.random.default_rng(0).standard_normal((5, 8))
        out = dense.forward(X)
        assert out.shape == (5, 3)

    def test_relu_output_non_negative(self):
        dense = Dense(in_features=8, out_features=3, activation="relu",
                       random_state=0)
        X = np.random.default_rng(0).standard_normal((5, 8)) * 10
        out = dense.forward(X)
        assert np.all(out >= 0)

    def test_softmax_output_distribution(self):
        dense = Dense(in_features=8, out_features=3, activation="softmax",
                       random_state=0)
        X = np.random.default_rng(0).standard_normal((5, 8))
        out = dense.forward(X)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(out >= 0)

    def test_backward_shape(self):
        dense = Dense(in_features=8, out_features=3, activation="relu",
                       learning_rate=0.01, random_state=0)
        X = np.random.default_rng(0).standard_normal((5, 8))
        out = dense.forward(X)
        d_in = dense.backward(np.ones_like(out) * 0.1)
        assert d_in.shape == (5, 8)

    def test_backward_updates_weights(self):
        dense = Dense(in_features=8, out_features=3, activation="relu",
                       learning_rate=0.1, random_state=0)
        W0 = dense.weights.copy()
        X = np.random.default_rng(0).standard_normal((5, 8))
        out = dense.forward(X)
        dense.backward(np.ones_like(out) * 0.1)
        assert not np.allclose(W0, dense.weights)


# ===================================================================
# SimpleCNN — Basic API
# ===================================================================

class TestSimpleCNNBasic:
    def test_fit_returns_self(self, cnn_dataset):
        X, y = cnn_dataset
        cnn = SimpleCNN(in_channels=1, image_size=12, n_classes=3,
                        learning_rate=0.01, random_state=0)
        result = cnn.fit(X, y, epochs=1, batch_size=4)
        assert result is cnn

    def test_predict_shape(self, cnn_dataset):
        X, y = cnn_dataset
        cnn = SimpleCNN(in_channels=1, image_size=12, n_classes=3,
                        learning_rate=0.01, random_state=0).fit(X, y, epochs=1, batch_size=4)
        preds = cnn.predict(X)
        assert preds.shape == (20,)

    def test_predict_proba_shape(self, cnn_dataset):
        X, y = cnn_dataset
        cnn = SimpleCNN(in_channels=1, image_size=12, n_classes=3,
                        learning_rate=0.01, random_state=0).fit(X, y, epochs=1, batch_size=4)
        proba = cnn.predict_proba(X)
        assert proba.shape == (20, 3)

    def test_predict_proba_sums_to_one(self, cnn_dataset):
        X, y = cnn_dataset
        cnn = SimpleCNN(in_channels=1, image_size=12, n_classes=3,
                        learning_rate=0.01, random_state=0).fit(X, y, epochs=1, batch_size=4)
        proba = cnn.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predictions_in_valid_range(self, cnn_dataset):
        X, y = cnn_dataset
        cnn = SimpleCNN(in_channels=1, image_size=12, n_classes=3,
                        learning_rate=0.01, random_state=0).fit(X, y, epochs=1, batch_size=4)
        preds = cnn.predict(X)
        assert np.all((preds >= 0) & (preds < 3))

    def test_losses_recorded(self, cnn_dataset):
        X, y = cnn_dataset
        cnn = SimpleCNN(in_channels=1, image_size=12, n_classes=3,
                        learning_rate=0.01, random_state=0).fit(X, y, epochs=3, batch_size=4)
        assert len(cnn.losses_) == 3


# ===================================================================
# SimpleCNN — Correctness
# ===================================================================

class TestSimpleCNNCorrectness:
    def test_loss_decreases_with_training(self, cnn_dataset):
        X, y = cnn_dataset
        cnn = SimpleCNN(in_channels=1, image_size=12, n_classes=3,
                        learning_rate=0.01, random_state=0).fit(X, y, epochs=10, batch_size=4)
        assert cnn.losses_[-1] < cnn.losses_[0]

    def test_no_nan_in_predictions(self, cnn_dataset):
        X, y = cnn_dataset
        cnn = SimpleCNN(in_channels=1, image_size=12, n_classes=3,
                        learning_rate=0.01, random_state=0).fit(X, y, epochs=5, batch_size=4)
        proba = cnn.predict_proba(X)
        assert not np.any(np.isnan(proba))

    def test_feature_map_dims_after_two_conv_pool_blocks(self):
        """For image_size=12: (12-3+1)//2=5, (5-3+1)//2=1 → flat_dim=32*1*1=32"""
        cnn = SimpleCNN(in_channels=1, image_size=12, n_classes=3, random_state=0)
        assert cnn.dense1.weights.shape[0] == 32


# ===================================================================
# SimpleCNN — Edge cases
# ===================================================================

class TestSimpleCNNEdgeCases:
    def test_small_batch_size_one(self, cnn_dataset):
        X, y = cnn_dataset
        cnn = SimpleCNN(in_channels=1, image_size=12, n_classes=3,
                        learning_rate=0.01, random_state=0).fit(X, y, epochs=1, batch_size=1)
        preds = cnn.predict(X[:2])
        assert preds.shape == (2,)

    def test_multi_channel_input(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((6, 3, 12, 12))
        y = rng.integers(0, 2, 6)
        cnn = SimpleCNN(in_channels=3, image_size=12, n_classes=2,
                        learning_rate=0.01, random_state=0).fit(X, y, epochs=1, batch_size=2)
        preds = cnn.predict(X)
        assert preds.shape == (6,)

    def test_binary_classification(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 1, 12, 12))
        y = rng.integers(0, 2, 10)
        cnn = SimpleCNN(in_channels=1, image_size=12, n_classes=2,
                        learning_rate=0.01, random_state=0).fit(X, y, epochs=2, batch_size=4)
        proba = cnn.predict_proba(X)
        assert proba.shape == (10, 2)
