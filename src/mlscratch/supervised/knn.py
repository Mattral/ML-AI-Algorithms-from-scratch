"""
K-Nearest Neighbors
====================

A simple from-scratch k-nearest neighbors classifier using Euclidean distance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _validate_classification_inputs(
    X: ArrayLike, y: ArrayLike,
) -> tuple[FloatArray, NDArray[np.int64]]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int).flatten()
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}."
        )
    return X_arr, y_arr


class KNeighborsClassifier:
    """A k-nearest neighbors classifier.

    Parameters
    ----------
    n_neighbors : int, default=3
        Number of nearest neighbors to use for prediction.
    """

    def __init__(self, n_neighbors: int = 3) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        self.n_neighbors = n_neighbors
        self.X_train_: FloatArray | None = None
        self.y_train_: NDArray[np.int64] | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNeighborsClassifier":
        """Store the training dataset."""
        X_arr, y_arr = _validate_classification_inputs(X, y)
        self.X_train_ = X_arr
        self.y_train_ = y_arr
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Predict class labels for the input samples."""
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        distances = self._pairwise_distances(X_arr, self.X_train_)
        nearest_indices = np.argsort(distances, axis=1)[:, : self.n_neighbors]
        nearest_labels = self.y_train_[nearest_indices]
        return np.array([np.bincount(row).argmax() for row in nearest_labels], dtype=np.int64)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy on the given data."""
        X_arr, y_arr = _validate_classification_inputs(X, y)
        y_pred = self.predict(X_arr)
        return float(np.mean(y_pred == y_arr))

    def _pairwise_distances(self, X: FloatArray, Y: FloatArray) -> FloatArray:
        """Compute the Euclidean distance matrix between X and Y."""
        X_norm = np.sum(X**2, axis=1)[:, None]
        Y_norm = np.sum(Y**2, axis=1)[None, :]
        cross = X @ Y.T
        distances = np.sqrt(np.maximum(X_norm + Y_norm - 2.0 * cross, 0.0))
        return distances
