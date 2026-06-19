"""
Complex-Valued Neural Network (CVNN)
======================================
A feedforward network whose weights, biases, and activations are complex
numbers (z = a + bi), useful for signal-processing tasks (audio spectra,
radar, MRI, wireless communications) where phase information is meaningful
and would be lost by treating real/imaginary parts as separate real channels.

Complex linear layer
----------------------
    z_out = W z_in + b,    W, b, z ∈ ℂ

Complex activation functions
-------------------------------
``modReLU`` (Arjovsky et al., 2016):
    modReLU(z) = ReLU(|z| + b) · (z / |z|)     if |z| + b > 0, else 0

``complex tanh`` (split, applied to real and imaginary parts independently):
    ctanh(z) = tanh(Re(z)) + i·tanh(Im(z))

``zReLU``:
    zReLU(z) = z   if Re(z) > 0 and Im(z) > 0, else 0

Backpropagation
-----------------
Implemented via Wirtinger calculus / the CR-calculus convention: for a
real-valued loss L(z, z̄), gradients are computed with respect to the
conjugate ∂L/∂z̄, and parameter updates use:

    W ← W - η · ∂L/∂W̄

For the layers and activations implemented here, this reduces to applying
the same real-valued backprop formulas independently to the real and
imaginary parts, which is the standard split-complex-backprop technique
and is exact for holomorphic-friendly activations like the ones above.

References
----------
Arjovsky, M., Shah, A., & Bengio, Y. (2016). Unitary evolution recurrent
neural networks. ICML.

Trabelsi, C. et al. (2018). Deep Complex Networks. ICLR.

Only numpy is used.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Complex activations
# ============================================================

def _complex_tanh(z: np.ndarray) -> np.ndarray:
    """ctanh(z) = tanh(Re z) + i tanh(Im z)."""
    return np.tanh(z.real) + 1j * np.tanh(z.imag)


def _complex_tanh_grad(z: np.ndarray) -> np.ndarray:
    """Derivative w.r.t. real and imaginary parts (split form)."""
    return (1 - np.tanh(z.real) ** 2) + 1j * (1 - np.tanh(z.imag) ** 2)


def _mod_relu(z: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """modReLU(z) = ReLU(|z| + b) · (z / |z|)."""
    mag = np.abs(z)
    scale = np.maximum(mag + bias, 0.0) / (mag + 1e-8)
    return z * scale


def _mod_relu_grad(z: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Approximate split-real/imag gradient of modReLU.
    Returns the multiplicative factor applied to incoming gradients
    (1 where active, 0 where the unit is "off").
    """
    mag = np.abs(z)
    active = (mag + bias) > 0
    return active.astype(float)


def _z_relu(z: np.ndarray) -> np.ndarray:
    """zReLU(z) = z if Re(z) > 0 and Im(z) > 0 else 0."""
    mask = (z.real > 0) & (z.imag > 0)
    return z * mask


def _z_relu_grad(z: np.ndarray) -> np.ndarray:
    mask = (z.real > 0) & (z.imag > 0)
    return mask.astype(float)


# ============================================================
# Complex Dense Layer
# ============================================================

class ComplexDense:
    """
    Complex-valued fully-connected layer.

    Parameters
    ----------
    in_features : int
    out_features : int
    activation : str
        ``'modrelu'``, ``'ctanh'``, ``'zrelu'``, or ``'linear'``.
    learning_rate : float
    random_state : int or None
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "ctanh",
        learning_rate: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        if activation not in {"modrelu", "ctanh", "zrelu", "linear"}:
            raise ValueError("activation must be 'modrelu', 'ctanh', 'zrelu', or 'linear'.")
        self.in_features  = in_features
        self.out_features = out_features
        self.activation   = activation
        self.learning_rate = learning_rate

        rng   = np.random.default_rng(random_state)
        scale = np.sqrt(1.0 / in_features)

        # Complex weights: independent real and imaginary Gaussian parts
        self.W = (rng.normal(0, scale, (in_features, out_features))
                  + 1j * rng.normal(0, scale, (in_features, out_features)))
        self.b = np.zeros(out_features, dtype=complex)

        if activation == "modrelu":
            self.mod_bias = np.zeros(out_features)

        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z_in: np.ndarray) -> np.ndarray:
        """
        z_in : (B, in_features) complex array
        Returns (B, out_features) complex array
        """
        z = z_in @ self.W + self.b

        if self.activation == "ctanh":
            a = _complex_tanh(z)
        elif self.activation == "modrelu":
            a = _mod_relu(z, self.mod_bias)
        elif self.activation == "zrelu":
            a = _z_relu(z)
        else:
            a = z

        self._cache = {"z_in": z_in, "z": z}
        return a

    # ------------------------------------------------------------------
    # Backward (split real/imaginary backprop)
    # ------------------------------------------------------------------

    def backward(self, d_a: np.ndarray) -> np.ndarray:
        """
        d_a : (B, out_features) complex gradient of loss w.r.t. activation output
        Returns d_z_in : (B, in_features) complex gradient w.r.t. layer input
        """
        z_in, z = self._cache["z_in"], self._cache["z"]
        n = len(z_in)

        if self.activation == "ctanh":
            grad = _complex_tanh_grad(z)
            d_z  = d_a.real * grad.real + 1j * (d_a.imag * grad.imag)
        elif self.activation == "modrelu":
            active = _mod_relu_grad(z, self.mod_bias)
            d_z    = d_a * active
            # mod_bias gradient
            d_bias = (d_a.real * active).mean(axis=0)
            self.mod_bias -= self.learning_rate * d_bias
        elif self.activation == "zrelu":
            active = _z_relu_grad(z)
            d_z    = d_a * active
        else:
            d_z = d_a

        # Gradients w.r.t. W and b (split real/imag, standard complex backprop)
        d_W = np.conj(z_in).T @ d_z / n
        d_b = d_z.mean(axis=0)

        self.W -= self.learning_rate * d_W
        self.b -= self.learning_rate * d_b

        return d_z @ np.conj(self.W).T


# ============================================================
# Complex-Valued Neural Network (multi-layer)
# ============================================================

class ComplexValuedNN:
    """
    Multi-layer complex-valued feedforward network for regression on
    complex-valued data (e.g. signal reconstruction).

    The final layer is linear (no activation) so outputs can take any
    complex value; loss is mean squared modulus error.

    Parameters
    ----------
    layer_sizes : list[int]
        e.g. [n_in, hidden1, hidden2, n_out]
    hidden_activation : str
        ``'ctanh'``, ``'modrelu'``, or ``'zrelu'`` for hidden layers.
    learning_rate : float
    epochs : int
    batch_size : int or None
    random_state : int or None
    """

    def __init__(
        self,
        layer_sizes: list[int],
        hidden_activation: str = "ctanh",
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int | None = 32,
        random_state: int | None = None,
    ) -> None:
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.batch_size    = batch_size

        self.layers: list[ComplexDense] = []
        for i in range(len(layer_sizes) - 1):
            is_last = (i == len(layer_sizes) - 2)
            act = "linear" if is_last else hidden_activation
            seed = (random_state or 0) + i
            self.layers.append(
                ComplexDense(layer_sizes[i], layer_sizes[i + 1], act,
                             learning_rate, seed)
            )

        self.losses_: list[float] = []

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward(self, z: np.ndarray) -> np.ndarray:
        """z : (B, n_in) complex → (B, n_out) complex"""
        a = z
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def _backward(self, d_out: np.ndarray) -> None:
        d = d_out
        for layer in reversed(self.layers):
            d = layer.backward(d)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ComplexValuedNN":
        """
        Train on complex data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_in), complex dtype
        y : ndarray of shape (n_samples, n_out), complex dtype

        Returns
        -------
        self
        """
        n = len(X)
        bs = self.batch_size or n
        rng = np.random.default_rng(0)
        self.losses_ = []

        for _ in range(self.epochs):
            idx = rng.permutation(n)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, bs):
                mb = idx[start:start + bs]
                Xb, yb = X[mb], y[mb]

                y_hat = self.forward(Xb)
                diff  = y_hat - yb
                loss  = float(np.mean(np.abs(diff) ** 2))
                epoch_loss += loss
                n_batches  += 1

                # d/d(z̄) of |y_hat - y|² is (y_hat - y); gradient direction:
                d_out = diff / len(mb)
                self._backward(d_out)

            self.losses_.append(epoch_loss / n_batches)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return complex-valued predictions."""
        return self.forward(X)

    def predict_magnitude(self, X: np.ndarray) -> np.ndarray:
        """Return |ŷ| (magnitude of predictions) — often the quantity of interest."""
        return np.abs(self.forward(X))

    def predict_phase(self, X: np.ndarray) -> np.ndarray:
        """Return arg(ŷ) (phase of predictions) in radians."""
        return np.angle(self.forward(X))
