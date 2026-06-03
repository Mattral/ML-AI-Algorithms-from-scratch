"""
Logistic Regression
===================

A from-scratch binary classifier using gradient descent and a sigmoid link.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _validate_classification_inputs(
    X: ArrayLike, y: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).flatten()
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}."
        )
    if not np.all(np.isin(y_arr, [0.0, 1.0])):
        raise ValueError("y must contain only binary labels 0 and 1.")
    return X_arr, y_arr


class LogisticRegression:
    r"""Binary logistic regression using gradient descent.

    The model is:

    .. math::
        p(y=1 \mid x) = \sigma(w^\top x + b),
        \quad \sigma(z) = \frac{1}{1 + e^{-z}}

    The loss is the binary cross-entropy:

    .. math::
        L = -\frac{1}{n} \sum_{i=1}^n 
            \left[y_i \log \sigma(z_i) + (1-y_i) \log (1-\sigma(z_i))\right]
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 1000,
        batch_size: int = 32,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.weights_: FloatArray | None = None
        self.bias_: float | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        """Fit the logistic regression model to binary data."""
        X_arr, y_arr = _validate_classification_inputs(X, y)
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X_arr.shape
        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0
        self.loss_history_ = []

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_arr[perm[start:end]]
                y_batch = y_arr[perm[start:end]]
                z = X_batch @ self.weights_ + self.bias_
                predictions = self._sigmoid(z)
                errors = predictions - y_batch
                grad_w = X_batch.T @ errors / X_batch.shape[0]
                grad_b = np.mean(errors)
                self.weights_ -= self.learning_rate * grad_w
                self.bias_ -= self.learning_rate * grad_b

            loss = self._binary_cross_entropy(y_arr, self.predict_proba(X_arr))
            self.loss_history_.append(float(loss))
            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch} loss={loss:.6f}")
        return self

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Return probability estimates for the positive class."""
        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return self._sigmoid(X_arr @ self.weights_ + self.bias_)

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Return binary predictions for the input data."""
        return (self.predict_proba(X) >= 0.5).astype(np.int64)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy on the given dataset."""
        X_arr, y_arr = _validate_classification_inputs(X, y)
        y_pred = self.predict(X_arr)
        return float(np.mean(y_pred == y_arr))

    def _sigmoid(self, z: FloatArray) -> FloatArray:
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _binary_cross_entropy(self, y_true: FloatArray, y_prob: FloatArray) -> float:
        y_prob = np.clip(y_prob, 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))
