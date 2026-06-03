r"""
K-Means Clustering
===================

A classic unsupervised clustering algorithm using Lloyd's iteration with
K-Means++ initialization.

The objective minimized is:

.. math::
    J = \sum_{i=1}^n \min_{1 \leq k \leq K} \|x_i - \mu_k\|^2

Complexity
----------
- Training: O(n K d \cdot n\_iter)
- Inference: O(n K d)
- Space:    O(K d)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _validate_input(X: ArrayLike) -> FloatArray:
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    return X_arr


class KMeans:
    """K-Means clustering with optional K-Means++ initialization.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, default=1e-4
        Convergence tolerance. The algorithm stops when centroid movement is
        less than this threshold.
    random_state : int | None, default=None
        Seed for centroid initialization.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.cluster_centers_: FloatArray | None = None
        self.labels_: NDArray[np.int64] | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def fit(self, X: ArrayLike) -> "KMeans":
        X_arr = _validate_input(X)
        n_samples, n_features = X_arr.shape
        if self.n_clusters <= 0 or self.n_clusters > n_samples:
            raise ValueError("n_clusters must be between 1 and n_samples.")

        rng = np.random.default_rng(self.random_state)
        centers = self._initialize_centroids(X_arr, rng)

        for iteration in range(1, self.max_iter + 1):
            labels = self._assign_clusters(X_arr, centers)
            new_centers = self._compute_centers(X_arr, labels, n_features)

            shift = np.linalg.norm(centers - new_centers, axis=1).max()
            centers = new_centers
            if shift <= self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(self._compute_inertia(X_arr, centers, labels))
        self.n_iter_ = iteration
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_arr = _validate_input(X)
        if X_arr.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X has a different number of features than the training data.")
        return self._assign_clusters(X_arr, self.cluster_centers_)

    def _initialize_centroids(self, X: FloatArray, rng: np.random.Generator) -> FloatArray:
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        first_idx = rng.integers(X.shape[0])
        centers[0] = X[first_idx]

        distances = np.full(X.shape[0], np.inf, dtype=float)
        for i in range(1, self.n_clusters):
            squared_distances = np.sum((X - centers[i - 1]) ** 2, axis=1)
            distances = np.minimum(distances, squared_distances)
            probabilities = distances / distances.sum()
            cumulative = np.cumsum(probabilities)
            chosen = rng.random()
            centers[i] = X[np.searchsorted(cumulative, chosen)]

        return centers

    def _assign_clusters(self, X: FloatArray, centers: FloatArray) -> NDArray[np.int64]:
        distances = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1).astype(np.int64)

    def _compute_centers(
        self, X: FloatArray, labels: NDArray[np.int64], n_features: int
    ) -> FloatArray:
        centers = np.zeros((self.n_clusters, n_features), dtype=float)
        for cluster_index in range(self.n_clusters):
            members = X[labels == cluster_index]
            if members.size == 0:
                centers[cluster_index] = X[np.random.default_rng(self.random_state).integers(X.shape[0])]
            else:
                centers[cluster_index] = members.mean(axis=0)
        return centers

    def _compute_inertia(
        self, X: FloatArray, centers: FloatArray, labels: NDArray[np.int64]
    ) -> float:
        diff = X - centers[labels]
        return float(np.sum(diff ** 2))
