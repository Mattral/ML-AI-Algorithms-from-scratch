"""
Convolutional Neural Network (CNN)
====================================
Building blocks for 2-D spatial feature extraction:

    Conv2D      — learnable filters, valid convolution, forward + backward
    MaxPool2D   — spatial downsampling, forward + backward (max-index mask)
    AvgPool2D   — average pooling, forward + backward
    BatchNorm2D — channel-wise normalisation with learnable γ, β
    Flatten     — reshape (B, C, H, W) → (B, C*H*W)
    Dense       — fully-connected layer with optional activation
    SimpleCNN   — pre-wired model for quick experiments

All layers expose ``forward(x)`` / ``backward(grad)`` with weight
updates performed in-place, matching the style of the repo.

Reference
----------
LeCun et al. (1998). Gradient-based learning applied to document
recognition. Proceedings of the IEEE, 86(11), 2278–2324.

Only numpy is used (no PIL / scipy dependency in this file).
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Helper activations
# ============================================================

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ============================================================
# Conv2D
# ============================================================

class Conv2D:
    """
    2-D Convolutional layer (valid padding, stride=1).

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
        Square kernel side length.
    learning_rate : float
    random_state : int or None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        learning_rate: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.learning_rate = learning_rate

        rng   = np.random.default_rng(random_state)
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        # Shape: (out_channels, in_channels, kH, kW)
        self.weights = rng.normal(0, scale,
                                   (out_channels, in_channels, kernel_size, kernel_size))
        self.bias    = np.zeros(out_channels)

        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : (B, C_in, H, W)

        Returns
        -------
        out : (B, C_out, H_out, W_out)
        """
        B, C_in, H, W = x.shape
        K              = self.kernel_size
        H_out          = H - K + 1
        W_out          = W - K + 1
        C_out          = self.out_channels

        # im2col: extract every K×K patch
        # col shape: (B, C_in*K*K, H_out*W_out)
        col = self._im2col(x, K, H_out, W_out)          # (B, C_in*K²,  H_out*W_out)
        W_flat = self.weights.reshape(C_out, -1)         # (C_out, C_in*K²)

        # (B, C_out, H_out*W_out) → (B, C_out, H_out, W_out)
        out = (W_flat @ col).reshape(B, C_out, H_out, W_out)
        out += self.bias.reshape(1, -1, 1, 1)

        self._cache = {"x": x, "col": col, "H_out": H_out, "W_out": W_out}
        return out

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        d_out : (B, C_out, H_out, W_out)

        Returns
        -------
        d_x : (B, C_in, H, W)
        """
        x, col = self._cache["x"], self._cache["col"]
        B, C_in, H, W = x.shape
        K  = self.kernel_size
        C_out = self.out_channels
        H_out, W_out = self._cache["H_out"], self._cache["W_out"]

        W_flat = self.weights.reshape(C_out, -1)         # (C_out, C_in*K²)

        # d_out: (B, C_out, H_out, W_out) → (B, C_out, H_out*W_out)
        d_out_flat = d_out.reshape(B, C_out, -1)

        # Gradient w.r.t. weights: sum over batch and spatial
        d_W_flat = np.einsum("bci,bki->ck", d_out_flat, col) / B  # (C_out, C_in*K²)
        d_B      = d_out_flat.mean(axis=(0, 2))

        # Gradient w.r.t. col (input patches)
        d_col = np.einsum("ck,bci->bki", W_flat, d_out_flat)      # (B, C_in*K², H_out*W_out)

        # col2im
        d_x = self._col2im(d_col, x.shape, K, H_out, W_out)

        self.weights -= self.learning_rate * d_W_flat.reshape(self.weights.shape)
        self.bias    -= self.learning_rate * d_B

        return d_x

    # ------------------------------------------------------------------
    # im2col / col2im helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _im2col(
        x: np.ndarray, K: int, H_out: int, W_out: int
    ) -> np.ndarray:
        B, C, H, W = x.shape
        col = np.zeros((B, C * K * K, H_out * W_out))
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, :, i:i+K, j:j+K]          # (B, C, K, K)
                col[:, :, idx] = patch.reshape(B, -1)
                idx += 1
        return col

    @staticmethod
    def _col2im(
        d_col: np.ndarray, x_shape: tuple,
        K: int, H_out: int, W_out: int
    ) -> np.ndarray:
        B, C, H, W = x_shape
        d_x = np.zeros(x_shape)
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                patch = d_col[:, :, idx].reshape(B, C, K, K)
                d_x[:, :, i:i+K, j:j+K] += patch
                idx += 1
        return d_x


# ============================================================
# MaxPool2D
# ============================================================

class MaxPool2D:
    """
    2-D Max Pooling layer with backward support.

    Parameters
    ----------
    pool_size : int
        Square pooling window side length.
    """

    def __init__(self, pool_size: int = 2) -> None:
        self.pool_size = pool_size
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        out : (B, C, H//pool, W//pool)
        """
        B, C, H, W = x.shape
        P           = self.pool_size
        H_out, W_out = H // P, W // P
        out  = np.zeros((B, C, H_out, W_out))
        mask = np.zeros_like(x)

        for i in range(H_out):
            for j in range(W_out):
                region = x[:, :, i*P:(i+1)*P, j*P:(j+1)*P]   # (B,C,P,P)
                max_val = region.max(axis=(2, 3), keepdims=True)
                out[:, :, i, j]              = max_val.squeeze((2, 3))
                mask[:, :, i*P:(i+1)*P, j*P:(j+1)*P] = (region == max_val)

        self._cache = {"mask": mask, "x_shape": x.shape}
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Route gradients back through the max positions."""
        mask    = self._cache["mask"]
        x_shape = self._cache["x_shape"]
        B, C, H, W = x_shape
        P           = self.pool_size
        H_out, W_out = H // P, W // P
        d_x     = np.zeros(x_shape)

        for i in range(H_out):
            for j in range(W_out):
                d_region = d_out[:, :, i, j][:, :, np.newaxis, np.newaxis]
                d_x[:, :, i*P:(i+1)*P, j*P:(j+1)*P] += (
                    mask[:, :, i*P:(i+1)*P, j*P:(j+1)*P] * d_region
                )
        return d_x


# ============================================================
# AvgPool2D
# ============================================================

class AvgPool2D:
    """2-D Average Pooling."""

    def __init__(self, pool_size: int = 2) -> None:
        self.pool_size = pool_size
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, C, H, W = x.shape
        P           = self.pool_size
        H_out, W_out = H // P, W // P
        out = np.zeros((B, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                out[:, :, i, j] = x[:, :, i*P:(i+1)*P, j*P:(j+1)*P].mean(axis=(2, 3))
        self._cache = {"x_shape": x.shape}
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        B, C, H, W = self._cache["x_shape"]
        P           = self.pool_size
        H_out, W_out = H // P, W // P
        d_x = np.zeros((B, C, H, W))
        for i in range(H_out):
            for j in range(W_out):
                d_x[:, :, i*P:(i+1)*P, j*P:(j+1)*P] += (
                    d_out[:, :, i, j][:, :, np.newaxis, np.newaxis] / (P * P)
                )
        return d_x


# ============================================================
# BatchNorm2D
# ============================================================

class BatchNorm2D:
    """
    Batch Normalisation over channel dimension.

    Normalises each channel independently across (B, H, W),
    then applies learnable scale γ and shift β.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        learning_rate: float = 1e-3,
    ) -> None:
        self.num_features = num_features
        self.eps          = eps
        self.momentum     = momentum
        self.learning_rate = learning_rate

        self.gamma = np.ones(num_features)
        self.beta  = np.zeros(num_features)

        self.running_mean = np.zeros(num_features)
        self.running_var  = np.ones(num_features)
        self._cache: dict = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        x : (B, C, H, W)
        """
        B, C, H, W = x.shape

        if training:
            mean = x.mean(axis=(0, 2, 3))            # (C,)
            var  = x.var(axis=(0, 2, 3))             # (C,)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean = self.running_mean
            var  = self.running_var

        x_hat = (x - mean.reshape(1, -1, 1, 1)) / np.sqrt(var.reshape(1, -1, 1, 1) + self.eps)
        out   = self.gamma.reshape(1, -1, 1, 1) * x_hat + self.beta.reshape(1, -1, 1, 1)

        self._cache = {"x": x, "x_hat": x_hat, "mean": mean, "var": var}
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        x, x_hat = self._cache["x"], self._cache["x_hat"]
        var       = self._cache["var"]
        B, C, H, W = x.shape
        N          = B * H * W

        d_gamma = (d_out * x_hat).sum(axis=(0, 2, 3))
        d_beta  = d_out.sum(axis=(0, 2, 3))

        self.gamma -= self.learning_rate * d_gamma
        self.beta  -= self.learning_rate * d_beta

        d_x_hat = d_out * self.gamma.reshape(1, -1, 1, 1)
        inv_std = 1.0 / np.sqrt(var.reshape(1, -1, 1, 1) + self.eps)
        d_x = (
            inv_std / N * (
                N * d_x_hat
                - d_x_hat.sum(axis=(0, 2, 3), keepdims=True)
                - x_hat * (d_x_hat * x_hat).sum(axis=(0, 2, 3), keepdims=True)
            )
        )
        return d_x


# ============================================================
# Flatten
# ============================================================

class Flatten:
    """Reshape (B, C, H, W) → (B, C*H*W)."""

    def __init__(self) -> None:
        self._input_shape: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return d_out.reshape(self._input_shape)


# ============================================================
# Dense
# ============================================================

class Dense:
    """
    Fully-connected layer with optional ReLU activation.

    Parameters
    ----------
    in_features : int
    out_features : int
    activation : str
        ``'relu'``, ``'softmax'``, or ``'linear'`` (identity).
    learning_rate : float
    random_state : int or None
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "linear",
        learning_rate: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        if activation not in {"relu", "softmax", "linear"}:
            raise ValueError("activation must be 'relu', 'softmax', or 'linear'.")
        self.activation   = activation
        self.learning_rate = learning_rate

        rng   = np.random.default_rng(random_state)
        scale = np.sqrt(2.0 / in_features)
        self.weights = rng.normal(0, scale, (in_features, out_features))
        self.bias    = np.zeros(out_features)
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x : (B, in_features)  →  (B, out_features)"""
        z = x @ self.weights + self.bias
        if self.activation == "relu":
            out = _relu(z)
        elif self.activation == "softmax":
            out = _softmax(z)
        else:
            out = z
        self._cache = {"x": x, "z": z}
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        x, z = self._cache["x"], self._cache["z"]
        if self.activation == "relu":
            d_out = d_out * _relu_grad(z)
        # softmax gradient is handled externally via (ŷ - y) delta

        d_W = x.T @ d_out / len(x)
        d_b = d_out.mean(axis=0)
        d_x = d_out @ self.weights.T

        self.weights -= self.learning_rate * d_W
        self.bias    -= self.learning_rate * d_b
        return d_x


# ============================================================
# SimpleCNN  (pre-wired model)
# ============================================================

class SimpleCNN:
    """
    Pre-wired CNN for small grayscale image classification.

    Architecture:
        Conv2D(in, 16, 3) → ReLU → MaxPool(2)
        Conv2D(16, 32, 3) → ReLU → MaxPool(2)
        Flatten → Dense(flat_dim, 128, relu) → Dense(128, n_classes, softmax)

    Parameters
    ----------
    in_channels : int
    image_size : int
        Height (and width) of the square input image.
    n_classes : int
    learning_rate : float
    random_state : int or None
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 28,
        n_classes: int = 10,
        learning_rate: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        self.n_classes     = n_classes
        self.learning_rate = learning_rate

        # Compute feature-map size after two conv+pool blocks
        def _after_conv_pool(size: int, k: int = 3, p: int = 2) -> int:
            return (size - k + 1) // p

        h = _after_conv_pool(_after_conv_pool(image_size))
        flat_dim = 32 * h * h

        self.conv1  = Conv2D(in_channels,  16, 3, learning_rate, random_state)
        self.pool1  = MaxPool2D(2)
        self.conv2  = Conv2D(16,           32, 3, learning_rate, random_state)
        self.pool2  = MaxPool2D(2)
        self.flat   = Flatten()
        self.dense1 = Dense(flat_dim, 128, "relu",    learning_rate, random_state)
        self.dense2 = Dense(128, n_classes, "softmax", learning_rate, random_state)

        self.losses_: list[float] = []

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray) -> tuple:
        """Return (output, intermediate activations) for backprop."""
        z1 = self.conv1.forward(X)
        a1 = _relu(z1)
        a2 = self.pool1.forward(a1)
        z2 = self.conv2.forward(a2)
        a3 = _relu(z2)
        a4 = self.pool2.forward(a3)
        a5 = self.flat.forward(a4)
        a6 = self.dense1.forward(a5)
        a7 = self.dense2.forward(a6)
        self._fwd_cache = {"z1": z1, "z2": z2}
        return a7, (X, a1, a2, a3, a4, a5, a6, a7)

    def _backward(self, y_hot: np.ndarray, a7: np.ndarray) -> None:
        """Back-propagate cross-entropy loss through all layers."""
        n  = len(y_hot)
        d  = (a7 - y_hot) / n          # softmax + cross-entropy delta

        d  = self.dense2.backward(d)
        d  = self.dense1.backward(d)
        d  = self.flat.backward(d)
        d  = self.pool2.backward(d)
        d  = d * _relu_grad(self._fwd_cache["z2"])
        d  = self.conv2.backward(d)
        d  = self.pool1.backward(d)
        d  = d * _relu_grad(self._fwd_cache["z1"])
        d  = self.conv1.backward(d)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> "SimpleCNN":
        """
        Train the CNN.

        Parameters
        ----------
        X : ndarray (n_samples, in_channels, H, W)
        y : ndarray (n_samples,)  — integer class labels
        epochs : int
        batch_size : int

        Returns
        -------
        self
        """
        n = len(X)
        rng = np.random.default_rng(0)

        # One-hot targets
        y_hot = np.zeros((n, self.n_classes))
        y_hot[np.arange(n), y.astype(int)] = 1.0

        for _ in range(epochs):
            idx = rng.permutation(n)
            ep_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                mb   = idx[start:start + batch_size]
                Xb   = X[mb]
                yb   = y_hot[mb]

                out, _ = self._forward(Xb)
                loss   = float(-np.mean(np.sum(yb * np.log(out + 1e-8), axis=1)))
                ep_loss += loss
                n_batches += 1

                self._backward(yb, out)

            self.losses_.append(ep_loss / n_batches)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions for X."""
        out, _ = self._forward(X)
        return np.argmax(out, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax class probabilities."""
        out, _ = self._forward(X)
        return out
