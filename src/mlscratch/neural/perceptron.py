"""
Perceptrons — Single-Layer and Multi-Layer
===========================================
The foundational building blocks of neural networks.

SingleLayerPerceptron
---------------------
A single layer of neurons with a configurable activation function.
Supports binary classification (sigmoid + binary cross-entropy) and
regression (linear + MSE), making the original two scripts a single
clean class with a ``task`` switch.

    z   = X W + b
    ŷ   = σ(z)                           # classification
    ŷ   = z                              # regression

Multi-Layer Perceptron
-----------------------
Fully-connected feedforward network with:
  - Arbitrary depth via ``hidden_sizes``
  - ReLU hidden activations
  - Softmax output for multi-class classification
  - Linear output for regression
  - Mini-batch SGD with momentum
  - He weight initialisation

References
----------
Rosenblatt, F. (1958). The perceptron: a probabilistic model for information
storage and organization in the brain. Psychological Review, 65(6), 386–408.

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning
representations by back-propagating errors. Nature, 323, 533–536.

Only numpy is used.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Activations (module-level helpers)
# ============================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ============================================================
# Single-Layer Perceptron
# ============================================================

class SingleLayerPerceptron:
    """
    Single-Layer Perceptron for binary classification or regression.

    Parameters
    ----------
    input_size : int
        Number of input features.
    task : str
        ``'classification'`` (sigmoid + binary cross-entropy) or
        ``'regression'`` (linear + MSE).
    learning_rate : float
        Gradient-descent step size.
    epochs : int
        Number of full passes over the training data.
    random_state : int or None
        Seed for reproducible weight initialisation.
    """

    def __init__(
        self,
        input_size: int,
        task: str = "classification",
        learning_rate: float = 0.01,
        epochs: int = 1000,
        random_state: int | None = None,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        self.input_size    = input_size
        self.task          = task
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self._rng          = np.random.default_rng(random_state)

        # Parameters (initialised in fit)
        self.weights_: np.ndarray | None = None
        self.bias_: float | None = None
        self.losses_: list[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _activate(self, z: np.ndarray) -> np.ndarray:
        return _sigmoid(z) if self.task == "classification" else z

    def _loss(self, y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-8) -> float:
        if self.task == "classification":
            return float(-np.mean(
                y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
            ))
        return float(np.mean((y_hat - y) ** 2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SingleLayerPerceptron":
        """
        Train the perceptron on (X, y).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        n_samples = len(X)
        scale = np.sqrt(2.0 / self.input_size)
        self.weights_ = self._rng.normal(0, scale, self.input_size)
        self.bias_    = 0.0
        self.losses_  = []

        for _ in range(self.epochs):
            z     = X @ self.weights_ + self.bias_
            y_hat = self._activate(z)

            self.losses_.append(self._loss(y, y_hat))

            # Gradient (identical form for both tasks)
            error = y_hat - y
            dw    = X.T @ error / n_samples
            db    = error.mean()

            self.weights_ -= self.learning_rate * dw
            self.bias_    -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (classification) or values (regression).

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        z     = X @ self.weights_ + self.bias_
        y_hat = self._activate(z)
        if self.task == "classification":
            return (y_hat >= 0.5).astype(int)
        return y_hat

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return sigmoid probabilities (classification only).

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")
        return _sigmoid(X @ self.weights_ + self.bias_)


# ============================================================
# Multi-Layer Perceptron
# ============================================================

class MultiLayerPerceptron:
    """
    Multi-Layer Perceptron (fully-connected feedforward network).

    Parameters
    ----------
    hidden_sizes : list[int]
        Sizes of the hidden layers (e.g. [64, 64]).
    task : str
        ``'classification'`` (softmax + cross-entropy) or
        ``'regression'`` (linear + MSE).
    n_classes : int
        Number of output classes (ignored for regression).
    learning_rate : float
    momentum : float
        Momentum coefficient for SGD (0 = vanilla SGD).
    epochs : int
    batch_size : int or None
        Mini-batch size.  None = full-batch.
    random_state : int or None
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        task: str = "classification",
        n_classes: int = 2,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        epochs: int = 200,
        batch_size: int | None = 32,
        random_state: int | None = None,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        self.hidden_sizes  = hidden_sizes or [64, 64]
        self.task          = task
        self.n_classes     = n_classes
        self.learning_rate = learning_rate
        self.momentum      = momentum
        self.epochs        = epochs
        self.batch_size    = batch_size
        self._rng          = np.random.default_rng(random_state)

        # Built in fit()
        self.weights_: list[np.ndarray] = []
        self.biases_:  list[np.ndarray] = []
        self.losses_:  list[float]      = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self, n_features: int) -> None:
        n_out = 1 if self.task == "regression" else self.n_classes
        sizes = [n_features] + list(self.hidden_sizes) + [n_out]

        self.weights_ = []
        self.biases_  = []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])     # He initialisation
            self.weights_.append(self._rng.normal(0, scale, (sizes[i], sizes[i + 1])))
            self.biases_.append(np.zeros(sizes[i + 1]))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray) -> tuple[list, list]:
        """Return (pre_activations, activations) for backprop."""
        pre_acts, acts = [], [X]
        a = X
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = a @ W + b
            pre_acts.append(z)
            if i < len(self.weights_) - 1:
                a = _relu(z)
            else:
                a = _softmax(z) if self.task == "classification" else z
            acts.append(a)
        return pre_acts, acts

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _loss(self, y_hot: np.ndarray, y_hat: np.ndarray, eps: float = 1e-8) -> float:
        if self.task == "classification":
            return float(-np.mean(np.sum(y_hot * np.log(y_hat + eps), axis=1)))
        return float(np.mean((y_hat.ravel() - y_hot.ravel()) ** 2))

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def _backward(
        self,
        pre_acts: list,
        acts: list,
        y_hot: np.ndarray,
        vel_w: list,
        vel_b: list,
    ) -> None:
        n = len(y_hot)
        y_hat = acts[-1]

        # Output delta
        if self.task == "classification":
            delta = (y_hat - y_hot) / n
        else:
            delta = 2.0 * (y_hat - y_hot) / n

        for i in reversed(range(len(self.weights_))):
            dW = acts[i].T @ delta
            db = delta.sum(axis=0)

            # Momentum update
            vel_w[i] = self.momentum * vel_w[i] + self.learning_rate * dW
            vel_b[i] = self.momentum * vel_b[i] + self.learning_rate * db

            self.weights_[i] -= vel_w[i]
            self.biases_[i]  -= vel_b[i]

            if i > 0:
                delta = (delta @ self.weights_[i].T) * _relu_grad(pre_acts[i - 1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiLayerPerceptron":
        """
        Train the MLP.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) — integer class labels or floats

        Returns
        -------
        self
        """
        n_samples = len(X)
        self._build(X.shape[1])

        # One-hot encode targets for classification
        if self.task == "classification":
            n_cls = self.n_classes
            y_hot = np.zeros((n_samples, n_cls))
            y_hot[np.arange(n_samples), y.astype(int)] = 1.0
        else:
            y_hot = y.reshape(-1, 1).astype(float)

        # Velocity buffers for momentum
        vel_w = [np.zeros_like(w) for w in self.weights_]
        vel_b = [np.zeros_like(b) for b in self.biases_]

        bs = self.batch_size or n_samples
        self.losses_ = []

        for _ in range(self.epochs):
            idx = self._rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n_samples, bs):
                mb  = idx[start:start + bs]
                Xb  = X[mb]
                yb  = y_hot[mb]

                pre_acts, acts = self._forward(Xb)
                epoch_loss    += self._loss(yb, acts[-1])
                n_batches     += 1

                self._backward(pre_acts, acts, yb, vel_w, vel_b)

            self.losses_.append(epoch_loss / n_batches)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (classification) or values (regression).

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        _, acts = self._forward(X)
        y_hat   = acts[-1]
        if self.task == "classification":
            return np.argmax(y_hat, axis=1)
        return y_hat.ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return softmax probabilities (classification only).

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")
        _, acts = self._forward(X)
        return acts[-1]
