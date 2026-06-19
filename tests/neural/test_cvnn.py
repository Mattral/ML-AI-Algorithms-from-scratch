"""
Tests for mlscratch.neural.cvnn
Covers: ComplexDense, ComplexValuedNN, complex activations
"""

from __future__ import annotations

import numpy as np
import pytest

from mlscratch.neural.cvnn import (
    ComplexDense,
    ComplexValuedNN,
    _complex_tanh,
    _complex_tanh_grad,
    _mod_relu,
    _mod_relu_grad,
    _z_relu,
    _z_relu_grad,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def complex_batch():
    """8 complex samples, 4 features."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal((8, 4)) + 1j * rng.standard_normal((8, 4)))


@pytest.fixture
def signal_data():
    """
    Toy signal-reconstruction task: learn y = 2·x (complex scaling).
    30 samples, 2 complex features.
    """
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((30, 2)) + 1j * rng.standard_normal((30, 2))) * 0.5
    y = X * 2.0
    return X, y


# ===================================================================
# Complex activation functions
# ===================================================================

class TestComplexActivations:
    # ── complex_tanh ──────────────────────────────────────────────

    def test_ctanh_preserves_shape(self, complex_batch):
        out = _complex_tanh(complex_batch)
        assert out.shape == complex_batch.shape

    def test_ctanh_output_is_complex(self, complex_batch):
        out = _complex_tanh(complex_batch)
        assert out.dtype == complex

    def test_ctanh_real_part_bounded(self, complex_batch):
        out = _complex_tanh(complex_batch)
        assert np.all(out.real >= -1.0) and np.all(out.real <= 1.0)

    def test_ctanh_imag_part_bounded(self, complex_batch):
        out = _complex_tanh(complex_batch)
        assert np.all(out.imag >= -1.0) and np.all(out.imag <= 1.0)

    def test_ctanh_zero_maps_to_zero(self):
        z = np.array([0.0 + 0.0j])
        np.testing.assert_allclose(_complex_tanh(z), 0.0, atol=1e-10)

    def test_ctanh_grad_shape(self, complex_batch):
        grad = _complex_tanh_grad(complex_batch)
        assert grad.shape == complex_batch.shape

    def test_ctanh_grad_real_part_in_01(self, complex_batch):
        """1 - tanh²(Re z) is in (0, 1]."""
        grad = _complex_tanh_grad(complex_batch)
        assert np.all(grad.real > 0) and np.all(grad.real <= 1.0)

    # ── mod_relu ──────────────────────────────────────────────────

    def test_modrelu_shape(self, complex_batch):
        bias = np.zeros(complex_batch.shape[-1])
        out = _mod_relu(complex_batch, bias)
        assert out.shape == complex_batch.shape

    def test_modrelu_with_zero_bias_preserves_direction(self, complex_batch):
        """modReLU(z, 0) should preserve the phase of z where |z| > 0."""
        bias = np.zeros(complex_batch.shape[-1])
        out = _mod_relu(complex_batch, bias)
        # Phase (angle) should not change for nonzero inputs
        nz = np.abs(complex_batch) > 1e-6
        np.testing.assert_allclose(
            np.angle(out[nz]), np.angle(complex_batch[nz]), atol=1e-6
        )

    def test_modrelu_large_negative_bias_zeros_output(self):
        """With very large negative bias, all units should be off (zero output)."""
        z = np.array([0.1 + 0.1j, 0.2 + 0.3j])
        bias = np.full(2, -1e6)
        out = _mod_relu(z, bias)
        np.testing.assert_allclose(np.abs(out), 0.0, atol=1e-6)

    def test_modrelu_grad_is_binary_mask(self, complex_batch):
        bias = np.zeros(complex_batch.shape[-1])
        active = _mod_relu_grad(complex_batch, bias)
        assert set(np.unique(active)).issubset({0.0, 1.0})

    # ── zrelu ─────────────────────────────────────────────────────

    def test_zrelu_shape(self, complex_batch):
        out = _z_relu(complex_batch)
        assert out.shape == complex_batch.shape

    def test_zrelu_zeros_third_quadrant(self):
        """z with Re < 0 and Im < 0 should be zeroed."""
        z = np.array([-1.0 - 1.0j, -1.0 + 1.0j, 1.0 - 1.0j, 1.0 + 1.0j])
        out = _z_relu(z)
        assert out[0] == 0.0j                       # 3rd quadrant: zeroed
        assert out[1] == 0.0j                       # 2nd quadrant: zeroed
        assert out[2] == 0.0j                       # 4th quadrant: zeroed
        np.testing.assert_allclose(out[3], z[3])    # 1st quadrant: passed

    def test_zrelu_grad_matches_active_region(self):
        z = np.array([1.0 + 1.0j, -1.0 - 1.0j])
        active = _z_relu_grad(z)
        assert active[0] == 1.0   # 1st quadrant: active
        assert active[1] == 0.0   # 3rd quadrant: off


# ===================================================================
# ComplexDense — Basic API
# ===================================================================

class TestComplexDenseBasic:
    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError):
            ComplexDense(in_features=4, out_features=2, activation="relu")

    @pytest.mark.parametrize("activation", ["ctanh", "modrelu", "zrelu", "linear"])
    def test_all_activations_forward_shape(self, complex_batch, activation):
        layer = ComplexDense(in_features=4, out_features=3, activation=activation,
                              learning_rate=0.01, random_state=0)
        out = layer.forward(complex_batch)
        assert out.shape == (8, 3)

    @pytest.mark.parametrize("activation", ["ctanh", "modrelu", "zrelu", "linear"])
    def test_all_activations_output_is_complex(self, complex_batch, activation):
        layer = ComplexDense(in_features=4, out_features=3, activation=activation,
                              random_state=0)
        out = layer.forward(complex_batch)
        assert out.dtype == complex

    def test_weight_dtype_complex(self):
        layer = ComplexDense(in_features=4, out_features=3, random_state=0)
        assert layer.W.dtype == complex
        assert layer.b.dtype == complex

    def test_weight_shape(self):
        layer = ComplexDense(in_features=4, out_features=6, random_state=0)
        assert layer.W.shape == (4, 6)
        assert layer.b.shape == (6,)

    @pytest.mark.parametrize("activation", ["ctanh", "modrelu", "zrelu", "linear"])
    def test_backward_input_gradient_shape(self, complex_batch, activation):
        layer = ComplexDense(in_features=4, out_features=3, activation=activation,
                              learning_rate=0.01, random_state=0)
        out = layer.forward(complex_batch)
        d_in = layer.backward(np.ones_like(out) * (0.01 + 0.01j))
        assert d_in.shape == complex_batch.shape

    @pytest.mark.parametrize("activation", ["ctanh", "modrelu", "zrelu", "linear"])
    def test_backward_updates_weights(self, complex_batch, activation):
        layer = ComplexDense(in_features=4, out_features=3, activation=activation,
                              learning_rate=0.1, random_state=0)
        W0 = layer.W.copy()
        out = layer.forward(complex_batch)
        layer.backward(np.ones_like(out) * (0.1 + 0.1j))
        assert not np.allclose(W0, layer.W)

    def test_modrelu_has_bias_attribute(self):
        layer = ComplexDense(in_features=4, out_features=3, activation="modrelu",
                              random_state=0)
        assert hasattr(layer, "mod_bias")
        assert layer.mod_bias.shape == (3,)


# ===================================================================
# ComplexDense — Correctness
# ===================================================================

class TestComplexDenseCorrectness:
    def test_linear_layer_output_matches_manual(self):
        """With activation='linear', output = z_in @ W + b."""
        rng = np.random.default_rng(42)
        D_in, D_out = 4, 3
        layer = ComplexDense(in_features=D_in, out_features=D_out,
                              activation="linear", random_state=42)
        z = (rng.standard_normal((5, D_in)) + 1j * rng.standard_normal((5, D_in)))
        out = layer.forward(z)
        expected = z @ layer.W + layer.b
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_ctanh_output_consistent_with_activation(self, complex_batch):
        layer = ComplexDense(in_features=4, out_features=3, activation="ctanh",
                              random_state=0)
        out = layer.forward(complex_batch)
        z = complex_batch @ layer.W + layer.b
        expected = _complex_tanh(z)
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_gradient_not_nan(self, complex_batch):
        for activation in ["ctanh", "modrelu", "zrelu", "linear"]:
            layer = ComplexDense(in_features=4, out_features=3,
                                  activation=activation, random_state=0)
            out = layer.forward(complex_batch)
            d_in = layer.backward(np.ones_like(out) * 0.01)
            assert not np.any(np.isnan(d_in.real))
            assert not np.any(np.isnan(d_in.imag))


# ===================================================================
# ComplexValuedNN — Basic API
# ===================================================================

class TestCVNNBasic:
    def test_fit_returns_self(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 2], epochs=2, batch_size=8,
                               random_state=0)
        assert cvnn.fit(X, y) is cvnn

    def test_layers_count(self):
        cvnn = ComplexValuedNN(layer_sizes=[4, 8, 4, 2], random_state=0)
        assert len(cvnn.layers) == 3    # 3 transitions

    def test_last_layer_is_linear(self):
        cvnn = ComplexValuedNN(layer_sizes=[4, 8, 2], random_state=0)
        assert cvnn.layers[-1].activation == "linear"

    def test_forward_output_shape(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 2], epochs=1, random_state=0)
        cvnn.fit(X, y)
        pred = cvnn.predict(X)
        assert pred.shape == y.shape

    def test_forward_output_is_complex(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 2], epochs=1, random_state=0)
        cvnn.fit(X, y)
        pred = cvnn.predict(X)
        assert pred.dtype == complex

    def test_predict_magnitude_shape(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 2], epochs=1, random_state=0)
        cvnn.fit(X, y)
        mag = cvnn.predict_magnitude(X)
        assert mag.shape == (30, 2)
        assert mag.dtype.kind == "f"    # real

    def test_predict_magnitude_non_negative(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 2], epochs=1, random_state=0)
        cvnn.fit(X, y)
        mag = cvnn.predict_magnitude(X)
        assert np.all(mag >= 0)

    def test_predict_phase_shape(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 2], epochs=1, random_state=0)
        cvnn.fit(X, y)
        phase = cvnn.predict_phase(X)
        assert phase.shape == (30, 2)

    def test_predict_phase_in_pi_range(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 2], epochs=1, random_state=0)
        cvnn.fit(X, y)
        phase = cvnn.predict_phase(X)
        assert np.all(phase >= -np.pi) and np.all(phase <= np.pi)

    def test_losses_recorded(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 2], epochs=5, batch_size=8,
                               random_state=0).fit(X, y)
        assert len(cvnn.losses_) == 5


# ===================================================================
# ComplexValuedNN — Correctness
# ===================================================================

class TestCVNNCorrectness:
    def test_loss_decreases_with_training(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 16, 2],
                               hidden_activation="ctanh",
                               learning_rate=0.01, epochs=80,
                               batch_size=8, random_state=0).fit(X, y)
        assert cvnn.losses_[-1] < cvnn.losses_[0]

    def test_all_losses_finite(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 2], epochs=20, batch_size=8,
                               random_state=0).fit(X, y)
        assert all(np.isfinite(l) for l in cvnn.losses_)

    def test_no_nan_in_predictions(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 2], epochs=10, batch_size=8,
                               random_state=0).fit(X, y)
        pred = cvnn.predict(X)
        assert not np.any(np.isnan(pred.real))
        assert not np.any(np.isnan(pred.imag))

    @pytest.mark.parametrize("activation", ["ctanh", "modrelu"])
    def test_different_activations_train(self, signal_data, activation):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 2],
                               hidden_activation=activation,
                               epochs=5, batch_size=8,
                               random_state=0).fit(X, y)
        pred = cvnn.predict(X)
        assert pred.shape == y.shape
        assert all(np.isfinite(l) for l in cvnn.losses_)

    def test_weights_change_after_training(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 2], epochs=5, batch_size=8,
                               random_state=0)
        W0 = [l.W.copy() for l in cvnn.layers]
        cvnn.fit(X, y)
        assert any(not np.allclose(w0, l.W) for w0, l in zip(W0, cvnn.layers))

    def test_phase_recovery_of_doubled_signal(self, signal_data):
        """y = 2·x means |y| = 2|x| and arg(y) = arg(x).
        After sufficient training the predicted phase should correlate
        strongly with the true phase."""
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 32, 2],
                               hidden_activation="ctanh",
                               learning_rate=0.01, epochs=100,
                               batch_size=8, random_state=0).fit(X, y)
        pred = cvnn.predict(X)
        true_phase = np.angle(y.ravel())
        pred_phase = np.angle(pred.ravel())
        # Correlation of predicted vs true phase should be positive
        corr = np.corrcoef(true_phase, pred_phase)[0, 1]
        assert corr > 0.5


# ===================================================================
# Edge cases
# ===================================================================

class TestCVNNEdgeCases:
    def test_single_hidden_layer(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 4, 2], epochs=3, batch_size=8,
                               random_state=0).fit(X, y)
        pred = cvnn.predict(X)
        assert pred.shape == y.shape

    def test_deep_network(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 8, 4, 2], epochs=3, batch_size=8,
                               random_state=0).fit(X, y)
        assert len(cvnn.layers) == 4
        pred = cvnn.predict(X)
        assert pred.shape == y.shape

    def test_single_sample(self):
        rng = np.random.default_rng(0)
        X = (rng.standard_normal((1, 2)) + 1j * rng.standard_normal((1, 2)))
        y = X * 2.0
        cvnn = ComplexValuedNN(layer_sizes=[2, 4, 2], epochs=2, batch_size=1,
                               random_state=0).fit(X, y)
        pred = cvnn.predict(X)
        assert pred.shape == (1, 2)

    def test_full_batch_mode(self, signal_data):
        X, y = signal_data
        cvnn = ComplexValuedNN(layer_sizes=[2, 8, 2], epochs=3, batch_size=None,
                               random_state=0).fit(X, y)
        assert len(cvnn.losses_) == 3
