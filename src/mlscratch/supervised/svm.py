r"""
Linear Support Vector Machine Classifier
=======================================

A linear SVM trained with stochastic sub-gradient descent on the hinge loss.

The objective is:

.. math::
    \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \max(0, 1 - y_i w^\top x_i)

Where labels are mapped to :math:`y_i \in \{-1, +1\}`.

Complexity
----------
- Training: O(n d \cdot n\_epochs)
- Inference: O(d)
- Space:    O(d)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _validate_classification_inputs(
    X: ArrayLike, y: ArrayLike
) -> tuple[FloatArray, IntArray]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int).flatten()
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}."
        )
    unique = np.unique(y_arr)
    if unique.size != 2:
        raise ValueError("LinearSVMClassifier supports only binary classification.")
    return X_arr, y_arr


class LinearSVMClassifier:
    """Linear binary SVM classifier trained with hinge-loss SGD.

    Parameters
    ----------
    learning_rate : float, default=1e-3
        Step size for weight updates.
    n_epochs : int, default=1000
        Number of passes over the training data.
    C : float, default=1.0
        Regularization strength.
    random_state : int | None, default=None
        Seed for SGD shuffling.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        n_epochs: int = 1000,
        C: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.C = float(C)
        self.random_state = random_state
        self.w_: FloatArray | None = None
        self.classes_: IntArray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearSVMClassifier":
        X_arr, y_arr = _validate_classification_inputs(X, y)
        self.n_features_in_ = X_arr.shape[1]
        self.classes_ = np.unique(y_arr)

        signed_labels = np.where(y_arr == self.classes_[0], -1, 1)
        X_aug = np.hstack([np.ones((X_arr.shape[0], 1), dtype=float), X_arr])
        self.w_ = np.zeros(X_aug.shape[1], dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_epochs):
            indices = rng.permutation(X_aug.shape[0])
            for i in indices:
                xi = X_aug[i]
                yi = signed_labels[i]
                margin = yi * np.dot(self.w_, xi)
                if margin >= 1.0:
                    gradient = np.concatenate(([0.0], self.w_[1:]))
                else:
                    gradient = np.concatenate(([0.0], self.w_[1:])) - self.C * yi * xi
                self.w_ -= self.learning_rate * gradient
        return self

    def decision_function(self, X: ArrayLike) -> FloatArray:
        if self.w_ is None:
            raise RuntimeError("Call fit() before decision_function().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return np.dot(np.hstack([np.ones((X_arr.shape[0], 1), dtype=float), X_arr]), self.w_)

    def predict(self, X: ArrayLike) -> IntArray:
        if self.w_ is None or self.classes_ is None:
            raise RuntimeError("Call fit() before predict().")
        scores = self.decision_function(X)
        labels = np.where(scores >= 0.0, self.classes_[1], self.classes_[0])
        return labels.astype(np.int64)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        X_arr, y_arr = _validate_classification_inputs(X, y)
        return float(np.mean(self.predict(X_arr) == y_arr))
